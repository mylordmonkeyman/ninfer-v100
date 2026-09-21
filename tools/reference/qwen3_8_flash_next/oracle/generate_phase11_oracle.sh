#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: generate_phase11_oracle.sh [--replace]

Build the frozen 4096-position Qwen3.8 Flash-Next Phase 11 CPU FP32 oracle.
The artifact is built in a temporary directory, fully validated, and then
atomically published to /srv/ninfer/oracle/phase11.

Environment overrides:
  NINFER_PHASE11_PYTHON       Python interpreter
  NINFER_PHASE11_MODEL_DIR    Mixed source checkpoint
  NINFER_PHASE11_PLE_DIR      PLE INT4 source directory
  NINFER_PHASE11_OUTPUT_DIR   Final oracle directory
  NINFER_PHASE11_LOGITS_CHUNK LM-head positions per write chunk
EOF
}

replace=0
case "${1:-}" in
  "") ;;
  --replace) replace=1 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"

python_bin="${NINFER_PHASE11_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  if [[ -x "$repo_root/venv-oracle/bin/python" ]]; then
    python_bin="$repo_root/venv-oracle/bin/python"
  elif [[ -x "$HOME/ninfer-v100/venv-oracle/bin/python" ]]; then
    # Supports running the generator from a clean temporary worktree while
    # reusing the already-provisioned oracle environment in the canonical checkout.
    python_bin="$HOME/ninfer-v100/venv-oracle/bin/python"
  else
    python_bin="$repo_root/venv-oracle/bin/python"
  fi
fi
model_dir="${NINFER_PHASE11_MODEL_DIR:-/srv/ninfer/source/mixed}"
ple_dir="${NINFER_PHASE11_PLE_DIR:-/srv/ninfer/source/ple/ples_int4}"
output_dir="${NINFER_PHASE11_OUTPUT_DIR:-/srv/ninfer/oracle/phase11}"
chunk_size="${NINFER_PHASE11_LOGITS_CHUNK:-8}"

for path in "$python_bin" "$model_dir" "$ple_dir"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 1
  fi
done
if [[ ! -x "$python_bin" ]]; then
  echo "Python interpreter is not executable: $python_bin" >&2
  exit 1
fi

oracle_rel="tools/reference/qwen3_8_flash_next/oracle"
for tracked in   "$oracle_rel/run_oracle.py"   "$oracle_rel/make_phase11_corpus.py"   "$oracle_rel/validate_phase11_oracle.py"; do
  if ! git -C "$repo_root" diff --quiet -- "$tracked"; then
    echo "refusing to build oracle with uncommitted changes in $tracked" >&2
    exit 1
  fi
done

parent_dir="$(dirname -- "$output_dir")"
mkdir -p "$parent_dir"
build_dir="$(mktemp -d "$parent_dir/.phase11.build.XXXXXX")"
published=0
cleanup() {
  if [[ $published -eq 0 && -d "$build_dir" ]]; then
    rm -rf -- "$build_dir"
  fi
}
trap cleanup EXIT

echo "Phase 11 oracle build"
echo "  repo:   $repo_root"
echo "  commit: $(git -C "$repo_root" rev-parse HEAD)"
echo "  model:  $model_dir"
echo "  PLE:    $ple_dir"
echo "  temp:   $build_dir"
echo "  final:  $output_dir"

"$python_bin" "$script_dir/make_phase11_corpus.py"   --model-dir "$model_dir"   --output "$build_dir/token_ids.json"

"$python_bin" "$script_dir/run_oracle.py"   --model-dir "$model_dir"   --ple-dir "$ple_dir"   --ids-file "$build_dir/token_ids.json"   --dump-logits "$build_dir"   --logits-chunk-size "$chunk_size"

"$python_bin" "$script_dir/validate_phase11_oracle.py"   --corpus "$build_dir/token_ids.json"   --manifest "$build_dir/manifest.json"   --minimum 4096

REPO_COMMIT="$(git -C "$repo_root" rev-parse HEAD)" MIXED_REVISION="a4e813ed3cfbbcc61e2929699eccb864a4dfa843" PLE_REVISION="da8b39586016d8325ac619be28ad77d6296625ec" BUILD_DIR="$build_dir" "$python_bin" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["BUILD_DIR"])
corpus = json.loads((root / "token_ids.json").read_text(encoding="utf-8"))
manifest_bytes = (root / "manifest.json").read_bytes()
payload = {
    "schema": "ninfer.phase11.oracle_provenance.v1",
    "repository_commit": os.environ["REPO_COMMIT"],
    "mixed_checkpoint": {
        "repo": "primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8",
        "revision": os.environ["MIXED_REVISION"],
    },
    "ple_checkpoint": {
        "repo": "primitive-ai/Qwen3.8-Flash-Next-PLE-quant",
        "revision": os.environ["PLE_REVISION"],
    },
    "positions": len(manifest_bytes) and len(
        json.loads(manifest_bytes.decode("utf-8"))["positions"]
    ),
    "token_ids_sha256": corpus["token_ids_sha256"],
    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
}
(root / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

if [[ -e "$output_dir" ]]; then
  if [[ $replace -ne 1 ]]; then
    echo "final output already exists: $output_dir" >&2
    echo "rerun with --replace only if you intend to replace it" >&2
    exit 1
  fi
  backup="${output_dir}.previous.$(date -u +%Y%m%dT%H%M%SZ)"
  mv -- "$output_dir" "$backup"
  echo "moved previous oracle to $backup"
fi

mv -- "$build_dir" "$output_dir"
published=1
trap - EXIT

echo "Published validated Phase 11 oracle at $output_dir"
echo "Next step: dispatch V100 Phase 11 Runtime with full_phase11=true"
