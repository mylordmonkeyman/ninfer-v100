"""Add the fixed DFlash2 bundle to an existing registered Qwen3.8-27B artifact.

The target objects are copied byte for byte. Only the companion checkpoint is
converted; no target checkpoint download or requantization is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import BinaryIO, Iterator

from tools.artifact.container import (
    Artifact, ArtifactWriter, ObjectSpec, ResourceObject, ResourceSpec,
    TensorObject, TensorSpec,
)
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion

from . import convert as groupwise_convert
from . import dflash2_recipe, inventory, inventory_nvfp4
from .dflash2_inventory import DFLASH2_TENSOR_SPECS


RECIPE_ID = "qwen3_8_27b-dflash2-splice-v1"
CHUNK_BYTES = 64 * 1024 * 1024


def base_specs(source: Artifact) -> tuple[ObjectSpec, ...]:
    """Require the exact registered base and preserve its directory order."""

    profiles = {"groupwise-int": inventory, "nvfp4": inventory_nvfp4}
    profile = profiles.get(source.identity.weights_id)
    if source.identity.model_id != "qwen3.8-27b" or profile is None:
        raise ValueError("DFlash2 requires a registered Qwen3.8-27B target artifact")
    expected_tensors = {spec.name: spec for spec in profile.BASE_TENSOR_SPECS}
    expected_resources = {spec.name: spec for spec in profile.RESOURCE_SPECS}
    if {obj.name for obj in source.objects} != expected_tensors.keys() | expected_resources.keys():
        raise ValueError("target must contain the complete base inventory and no DFlash2 suffix")

    result: list[ObjectSpec] = []
    for obj in source.objects:
        if isinstance(obj, TensorObject):
            spec = expected_tensors.get(obj.name)
            if spec is None or (obj.shape, obj.format, obj.layout) != (
                spec.shape, spec.format, spec.layout
            ):
                raise ValueError(f"target descriptor differs from registered inventory: {obj.name}")
            result.append(TensorSpec(obj.name, obj.shape, obj.format, obj.layout))
        elif isinstance(obj, ResourceObject):
            spec = expected_resources.get(obj.name)
            if spec is None or obj.encoding != spec.encoding:
                raise ValueError(f"target resource differs from registered inventory: {obj.name}")
            with source.payload(obj) as payload:
                digest = hashlib.sha256(payload).hexdigest()
            if digest != groupwise_convert.OFFICIAL_RESOURCE_SHA256[obj.name]:
                raise ValueError(f"target frontend resource checksum mismatch: {obj.name}")
            result.append(ResourceSpec(obj.name, obj.encoding, obj.bytes))
    return tuple(result)


def _copy_chunks(stream: BinaryIO, offset: int, size: int, digest) -> Iterator[bytes]:
    stream.seek(offset)
    remaining = size
    while remaining:
        chunk = stream.read(min(remaining, CHUNK_BYTES))
        if not chunk:
            raise ValueError("target payload was truncated during copying")
        digest.update(chunk)
        yield chunk
        remaining -= len(chunk)


def splice_artifact(
    target_path: str | Path,
    dflash2_model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str = "cpu",
) -> Path:
    started = time.perf_counter()
    target = Path(target_path)
    draft = Path(dflash2_model_dir)
    output = Path(out_path)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise ValueError("output and partial output paths must not already exist")
    if target.resolve() in (output.resolve(), partial.resolve()):
        raise ValueError("output must be separate from the target artifact")

    config = dflash2_recipe.validate_config(conversion.load_json(draft / "config.json"))
    source_preflight = dflash2_recipe.preflight_sources(draft)
    resolved_device = pick_device(device)
    companion_specs = tuple(
        TensorSpec(spec.name, spec.shape, spec.format, spec.layout)
        for spec in DFLASH2_TENSOR_SPECS
    )
    checksums: dict[str, str] = {}
    with Artifact.open(target) as source:
        target_specs = base_specs(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with ArtifactWriter(partial, source.identity, target_specs + companion_specs) as writer:
                with target.open("rb") as source_file:
                    for obj in source.objects:
                        digest = hashlib.sha256()
                        writer.write(obj.name, _copy_chunks(
                            source_file, source.payload_offset + obj.offset, obj.bytes, digest
                        ))
                        checksums[obj.name] = digest.hexdigest()
                print(f"copied {len(source.objects)} target objects without requantization", flush=True)
                with ShardReader.from_file(draft / "model.safetensors") as reader:
                    for index, spec in enumerate(DFLASH2_TENSOR_SPECS, 1):
                        tensor = dflash2_recipe.materialize_tensor(spec.name, reader)
                        payload = conversion.encode_tensor_payload(tensor, spec, resolved_device)
                        del tensor
                        writer.write(spec.name, payload)
                        checksums[spec.name] = hashlib.sha256(payload).hexdigest()
                        del payload
                        print(f"[{index}/{len(companion_specs)}] {spec.name}", flush=True)

            # Read the completed container through the same public reader and
            # verify every copied or encoded payload before publishing its path.
            with Artifact.open(partial) as verified:
                if verified.identity != source.identity or len(verified.objects) != len(checksums):
                    raise ValueError("completed artifact identity or inventory differs from its plan")
                for obj in verified.objects:
                    with verified.payload(obj) as payload:
                        if hashlib.sha256(payload).hexdigest() != checksums[obj.name]:
                            raise ValueError(f"completed artifact payload mismatch: {obj.name}")
            partial.replace(output)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

        report = {
            "recipe_id": RECIPE_ID,
            "identity": {"model_id": source.identity.model_id, "weights_id": source.identity.weights_id},
            "source": {
                "target": {"path": str(target.resolve()), "bytes": target.stat().st_size},
                "dflash2": {
                    "repository": dflash2_recipe.REPOSITORY,
                    "revision": dflash2_recipe.REVISION,
                    "path": str(draft.resolve()),
                    "config": config,
                    "source_tensors": source_preflight.source_tensor_count,
                },
            },
            "output": {"path": str(output.resolve()), "bytes": output.stat().st_size},
            "device": str(resolved_device),
            "target_objects_preserved": len(target_specs),
            "dflash2_objects_added": len(companion_specs),
            "payloads_verified": len(checksums),
            "elapsed_seconds": time.perf_counter() - started,
            "object_sha256": checksums,
        }
    report_path = output.with_name(output.name + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"verified artifact: {output}; report: {report_path}", flush=True)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--dflash2-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    splice_artifact(args.target, args.dflash2_model, args.out, device=args.device)


if __name__ == "__main__":
    main()
