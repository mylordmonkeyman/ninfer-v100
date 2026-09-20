#!/usr/bin/env python3
"""Extract one Flash-Next routed-expert layer as exact raw NVFP4 bank payloads.

This reader intentionally uses only the Python standard library so the Phase-1
CPU architecture probe does not require PyTorch or the converter environment.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

_MAGIC = b"NINFER\x00\x02"
_PREFIX = struct.Struct("<8sQ")
_PAYLOAD_ALIGNMENT = 4096
_LAYOUT_ALIGNMENT = 256
_LAYOUT = "expert-blockscale-k16-m128x4-v1"
_CHUNK = 16 * 1024 * 1024

_GATE_UP_SHAPE = (512, 1280, 2560)
_DOWN_SHAPE = (512, 2560, 640)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _expected_nvfp4_bank_bytes(shape: tuple[int, int, int]) -> int:
    experts, rows, columns = shape
    if experts <= 0 or rows <= 0 or columns <= 0 or rows % 128 or columns % 64:
        raise ValueError(f"invalid NVFP4 expert-bank shape: {shape}")
    elements = experts * rows * columns
    code_bytes = elements // 2
    scale_offset = _align_up(code_bytes, _LAYOUT_ALIGNMENT)
    scale_bytes = elements // 16
    return scale_offset + scale_bytes + experts * 4


def _read_directory(source) -> tuple[int, list[dict[str, object]], int]:
    source.seek(0, 2)
    file_bytes = source.tell()
    source.seek(0)
    prefix = source.read(_PREFIX.size)
    if len(prefix) != _PREFIX.size:
        raise ValueError("artifact is shorter than the NInfer v2 prefix")
    magic, json_bytes = _PREFIX.unpack(prefix)
    if magic != _MAGIC:
        raise ValueError("artifact magic is not NInfer v2")
    if json_bytes <= 0:
        raise ValueError("artifact JSON directory is empty")

    metadata_end = _PREFIX.size + json_bytes
    payload_offset = _align_up(metadata_end, _PAYLOAD_ALIGNMENT)
    if payload_offset > file_bytes:
        raise ValueError("artifact JSON or payload start extends beyond the file")

    directory_bytes = source.read(json_bytes)
    if len(directory_bytes) != json_bytes:
        raise ValueError("artifact JSON directory is truncated")
    try:
        directory = json.loads(directory_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid artifact JSON directory: {exc}") from exc

    if not isinstance(directory, dict):
        raise ValueError("artifact JSON root is not an object")
    objects = directory.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("artifact directory has no object inventory")
    return payload_offset, objects, file_bytes


def _find_tensor(
    objects: list[dict[str, object]],
    name: str,
    expected_shape: tuple[int, int, int],
    payload_offset: int,
    file_bytes: int,
) -> tuple[int, int]:
    matches = [obj for obj in objects if isinstance(obj, dict) and obj.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one artifact object named {name!r}, found {len(matches)}")
    obj = matches[0]
    if obj.get("kind") != "tensor":
        raise ValueError(f"{name} is not a tensor")
    if obj.get("format") != "NVFP4" or obj.get("layout") != _LAYOUT:
        raise ValueError(
            f"{name} is {obj.get('format')}/{obj.get('layout')}, expected NVFP4/{_LAYOUT}"
        )

    raw_shape = obj.get("shape")
    if not isinstance(raw_shape, list) or any(type(dim) is not int for dim in raw_shape):
        raise ValueError(f"{name} has an invalid shape")
    shape = tuple(raw_shape)
    if shape != expected_shape:
        raise ValueError(f"{name} has shape {shape}, expected {expected_shape}")

    offset = obj.get("offset")
    payload_bytes = obj.get("bytes")
    if type(offset) is not int or offset < 0 or type(payload_bytes) is not int or payload_bytes <= 0:
        raise ValueError(f"{name} has invalid offset/byte metadata")
    if offset % _LAYOUT_ALIGNMENT:
        raise ValueError(f"{name} payload offset is not {_LAYOUT_ALIGNMENT}-byte aligned")

    expected_bytes = _expected_nvfp4_bank_bytes(expected_shape)
    if payload_bytes != expected_bytes:
        raise ValueError(f"{name} stores {payload_bytes} bytes, expected {expected_bytes}")

    absolute_begin = payload_offset + offset
    absolute_end = absolute_begin + payload_bytes
    if absolute_end > file_bytes:
        raise ValueError(f"{name} payload extends beyond the artifact file")
    return absolute_begin, payload_bytes


def _copy_payload(source, begin: int, payload_bytes: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.seek(begin)
    remaining = payload_bytes
    with destination.open("wb") as output:
        while remaining:
            chunk = source.read(min(_CHUNK, remaining))
            if not chunk:
                raise ValueError("artifact payload is truncated while copying")
            output.write(chunk)
            remaining -= len(chunk)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path, help="Qwen3.8 Flash-Next .ninfer artifact")
    parser.add_argument("--layer", type=int, default=0, choices=range(48))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    prefix = f"text/layers/{args.layer}/mlp/experts"
    tensors = (
        (f"{prefix}/gate_up", _GATE_UP_SHAPE, args.out_dir / f"layer{args.layer:02d}-gate_up.nvfp4.bin"),
        (f"{prefix}/down", _DOWN_SHAPE, args.out_dir / f"layer{args.layer:02d}-down.nvfp4.bin"),
    )

    with args.artifact.open("rb") as source:
        payload_offset, objects, file_bytes = _read_directory(source)
        for name, shape, destination in tensors:
            begin, payload_bytes = _find_tensor(
                objects, name, shape, payload_offset, file_bytes
            )
            _copy_payload(source, begin, payload_bytes, destination)
            print(f"{name.rsplit('/', 1)[-1]}={destination} bytes={payload_bytes}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
