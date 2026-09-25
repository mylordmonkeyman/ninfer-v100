"""Check selected loaded V100 weight planes against the pinned source recipe.

The real-model test exports device-resident planes only to runner-local scratch.
This verifier reads the source safetensors directly, without Torch or a model
forward pass. Do not upload either set of weight bytes as an Actions artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.convert.qwen3_8_flash_next import recipe
from tools.convert.qwen3_8_flash_next.source import read_safetensor_directory


def verify(source_root: Path, candidate_root: Path) -> list[dict[str, object]]:
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
    for layer in (0, 2):
        prefix = f"text/layers/{layer}/"
        for suffix in ("mlp/router", "mlp/hyper_connection/norm",
                       "mlp/hyper_connection/input_mix/down",
                       "mlp/hyper_connection/input_mix/up",
                       "gdn/query_key_value_z", "gdn/output"):
            name = prefix + suffix
            selected = recipe.RECIPES_BY_OBJECT[name]
            if isinstance(selected, recipe.DirectRecipe):
                if selected.transform != "copy" or len(selected.sources) != 1:
                    raise ValueError(f"unsupported direct weight transform: {name}")
                part = selected.sources[0]
                planes = {"bf16": source_bytes(part.name, part.dtype, part.shape)}
            elif isinstance(selected, recipe.Fp8Recipe):
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
                actual_path = candidate_root / f"{name}.{plane}.bin"
                actual = actual_path.read_bytes()
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
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = verify(args.source_root, args.candidate_root)
    args.report.write_text(json.dumps({"status": "exact", "planes": rows}, indent=2) + "\n")
    print(f"Matched {len(rows)} loaded V100 weight planes against source bytes")
    for row in rows:
        print(f"{row['name']} {row['plane']} {row['bytes']} bytes exact")


if __name__ == "__main__":
    main()
