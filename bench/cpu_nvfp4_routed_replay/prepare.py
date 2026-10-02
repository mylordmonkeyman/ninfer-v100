"""Write exact model-bank offsets without copying or expanding expert weights."""
import argparse
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "extract", Path(__file__).parents[1] / "cpu_nvfp4_expert_pair/extract_layer.py")
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)

def prepare(artifact, output):
    with artifact.open("rb") as source, output.open("w") as dest:
        payload, objects, size = extract._read_directory(source)
        for layer in range(48):
            prefix = f"text/layers/{layer}/mlp/experts"
            gate, gate_bytes = extract._find_tensor(
                objects, prefix + "/gate_up", (512, 1280, 2560), payload, size)
            down, down_bytes = extract._find_tensor(
                objects, prefix + "/down", (512, 2560, 640), payload, size)
            dest.write(f"{gate} {gate_bytes} {down} {down_bytes}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.artifact, args.output)
