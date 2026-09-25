"""Check selected loaded V100 weight planes against the pinned source recipe.

The real-model test exports device-resident planes only to runner-local scratch.
This verifier reads source safetensors directly and can compare either the
prepared artifact or loaded V100 planes. Do not upload weight bytes as an Actions
artifact.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

from tools.convert.qwen3_8_flash_next import recipe
from tools.convert.qwen3_8_flash_next.source import read_safetensor_directory


class ArtifactPayload:
    """Read selected NInfer v2 payload ranges without importing Torch."""

    def __init__(self, path: Path):
        self.handle = path.open("rb")
        try:
            magic, size = struct.unpack("<8sQ", self.handle.read(16))
            if magic != b"NINFER\x00\x02" or size < 1:
                raise ValueError("invalid NInfer v2 artifact header")
            directory = json.loads(self.handle.read(size))
            self.offset = (16 + size + 4095) // 4096 * 4096
            self.objects = {obj["name"]: obj for obj in directory["objects"]}
            if len(self.objects) != len(directory["objects"]):
                raise ValueError("duplicate artifact object names")
            self.file_bytes = path.stat().st_size
        except BaseException:
            self.handle.close()
            raise

    def read(self, name: str, offset: int, size: int, shape: tuple[int, ...],
             fmt: str, layout: str) -> bytes:
        obj = self.objects[name]
        if (obj["kind"], tuple(obj["shape"]), obj["format"], obj["layout"]) != (
                "tensor", shape, fmt, layout):
            raise ValueError(f"candidate tensor signature mismatch: {name}")
        if offset < 0 or size < 0 or offset + size > obj["bytes"]:
            raise ValueError(f"candidate tensor plane out of bounds: {name}")
        absolute = self.offset + obj["offset"] + offset
        if absolute + size > self.file_bytes:
            raise ValueError(f"candidate tensor extends beyond artifact: {name}")
        self.handle.seek(absolute)
        return self.handle.read(size)

    def close(self) -> None:
        self.handle.close()


def verify(source_root: Path, candidate_root: Path | None = None,
           candidate_artifact: Path | None = None) -> list[dict[str, object]]:
    if (candidate_root is None) == (candidate_artifact is None):
        raise ValueError("select exactly one candidate source")
    index = json.loads((source_root / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    headers = {}

    def source_bytes(name: str, dtype: str, shape: tuple[int, ...]) -> bytes:
        path = source_root / weight_map[name]
        if path not in headers:
            headers[path] = read_safetensor_directory(path)
        tensor = headers[path].tensors[name]
        if tensor.dtype != dtype or tensor.shape != shape:
            raise ValueError(f"source tensor signature mismatch: {name}")
        with path.open("rb") as handle:
            handle.seek(tensor.absolute_offset)
            payload = handle.read(tensor.bytes)
        if len(payload) != tensor.bytes:
            raise ValueError(f"short source tensor: {name}")
        return payload

    rows = []
    artifact = ArtifactPayload(candidate_artifact) if candidate_artifact else None
    try:
        for layer in (0, 2):
            rows.extend(verify_layer(layer, recipe, source_bytes, candidate_root, artifact))
    finally:
        if artifact is not None:
            artifact.close()
    return rows


def verify_layer(layer: int, recipe_module, source_bytes, candidate_root,
                 artifact) -> list[dict[str, object]]:
    rows = []
    prefix = f"text/layers/{layer}/"
    for suffix in ("mlp/router", "mlp/hyper_connection/norm",
                   "mlp/hyper_connection/input_mix/down",
                   "mlp/hyper_connection/input_mix/up",
                   "gdn/query_key_value_z", "gdn/output"):
        name = prefix + suffix
        selected = recipe_module.RECIPES_BY_OBJECT[name]
        if isinstance(selected, recipe_module.DirectRecipe):
            if selected.transform != "copy" or len(selected.sources) != 1:
                raise ValueError(f"unsupported direct weight transform: {name}")
            part = selected.sources[0]
            planes = {"bf16": source_bytes(part.name, part.dtype, part.shape)}
        elif isinstance(selected, recipe_module.Fp8Recipe):
            planes = {
                "codes": b"".join(source_bytes(m.name, m.dtype, m.shape)
                                  for m in selected.matrices),
                "scales": b"".join(source_bytes(m.name + "_scale", "F32",
                                               (m.shape[0],))
                                   for m in selected.matrices),
            }
        else:
            raise ValueError(f"unsupported weight recipe: {name}")
        for plane, expected in planes.items():
            if artifact is None:
                actual_path = candidate_root / f"{name}.{plane}.bin"
                actual = actual_path.read_bytes()
            else:
                if isinstance(selected, recipe_module.DirectRecipe):
                    shape = selected.sources[0].shape
                    fmt, layout = "BF16", "contiguous-le-v1"
                    offset = 0
                else:
                    shape = (sum(m.shape[0] for m in selected.matrices),
                             selected.matrices[0].shape[1])
                    fmt, layout = "FP8_E4M3FN_ROW_F32S", "row-scale-f32-v1"
                    code_bytes = shape[0] * shape[1]
                    offset = 0 if plane == "codes" else (code_bytes + 255) // 256 * 256
                actual = artifact.read(name, offset, len(expected), shape, fmt, layout)
            if len(actual) != len(expected):
                raise ValueError(f"{name}/{plane}: size {len(actual)} != {len(expected)}")
            if actual != expected:
                first = next(i for i, (a, b) in enumerate(zip(actual, expected))
                             if a != b)
                raise ValueError(f"{name}/{plane}: first byte mismatch at {first}")
            rows.append({"name": name, "plane": plane,
                         "bytes": len(expected), "exact": True})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    candidates = parser.add_mutually_exclusive_group(required=True)
    candidates.add_argument("--candidate-root", type=Path)
    candidates.add_argument("--candidate-artifact", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = verify(args.source_root, args.candidate_root, args.candidate_artifact)
    args.report.write_text(json.dumps({"status": "exact", "planes": rows}, indent=2) + "\n")
    origin = "device-resident" if args.candidate_root else "prepared-artifact"
    print(f"Matched {len(rows)} {origin} weight planes against source bytes")
    for row in rows:
        print(f"{row['name']} {row['plane']} {row['bytes']} bytes exact")


if __name__ == "__main__":
    main()
