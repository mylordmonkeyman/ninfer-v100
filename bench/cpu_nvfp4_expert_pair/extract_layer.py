#!/usr/bin/env python3
"""Extract one Flash-Next routed-expert layer as exact raw NVFP4 bank payloads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.artifact.container import Artifact, TensorObject  # noqa: E402

_CHUNK = 16 * 1024 * 1024


def _copy_payload(artifact: Artifact, name: str, destination: Path) -> int:
    obj = artifact.find(name)
    if not isinstance(obj, TensorObject):
        raise TypeError(f"{name} is not a tensor")
    if obj.format != "NVFP4" or obj.layout != "expert-blockscale-k16-m128x4-v1":
        raise ValueError(
            f"{name} is {obj.format}/{obj.layout}, expected "
            "NVFP4/expert-blockscale-k16-m128x4-v1"
        )

    payload = artifact.payload(obj)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as output:
            for start in range(0, len(payload), _CHUNK):
                output.write(payload[start : start + _CHUNK])
    finally:
        payload.release()
    return obj.bytes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path, help="Qwen3.8 Flash-Next .ninfer artifact")
    parser.add_argument("--layer", type=int, default=0, choices=range(48))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    prefix = f"text/layers/{args.layer}/mlp/experts"
    gate_name = f"{prefix}/gate_up"
    down_name = f"{prefix}/down"
    gate_out = args.out_dir / f"layer{args.layer:02d}-gate_up.nvfp4.bin"
    down_out = args.out_dir / f"layer{args.layer:02d}-down.nvfp4.bin"

    with Artifact.open(args.artifact) as artifact:
        gate_bytes = _copy_payload(artifact, gate_name, gate_out)
        down_bytes = _copy_payload(artifact, down_name, down_out)

    print(f"gate_up={gate_out} bytes={gate_bytes}")
    print(f"down={down_out} bytes={down_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
