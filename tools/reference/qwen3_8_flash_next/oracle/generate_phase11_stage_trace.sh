#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: generate_phase11_stage_trace.sh [--replace]

Build a compact 7-position CPU FP32 stage oracle from the frozen Phase 11 corpus.
It is intended for layer/stage localization of the first full-oracle divergence.

Environment overrides:
  NINFER_PHASE11_PYTHON             Python interpreter
  NINFER_PHASE11_MODEL_DIR          Mixed source checkpoint
  NINFER_PHASE11_PLE_DIR            PLE INT4 source directory
  NINFER_PHASE11_CORPUS             Frozen Phase 11 token_ids.json
  NINFER_PHASE11_STAGE_TRACE_DIR    Final stage-trace directory
  NINFER_PHASE11_STAGE_POSITIONS    Prefix length (default 7)
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
    python_bin="$HOME/ninfer-v100/venv-oracle/bin/python"
  else
    python_bin="$repo_root/venv-oracle/bin/python"
  fi
fi

model_dir="${NINFER_PHASE11_MODEL_DIR:-/srv/ninfer/source/mixed}"
ple_dir="${NINFER_PHASE11_PLE_DIR:-/srv/ninfer/source/ple/ples_int4}"
corpus="${NINFER_PHASE11_CORPUS:-/srv/ninfer/oracle/phase11/token_ids.json}"
output_dir="${NINFER_PHASE11_STAGE_TRACE_DIR:-/srv/ninfer/oracle/phase11-stage-trace7}"
positions="${NINFER_PHASE11_STAGE_POSITIONS:-7}"

for path in "$python_bin" "$model_dir" "$ple_dir" "$corpus"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 1
  fi
done
if [[ ! -x "$python_bin" ]]; then
  echo "Python interpreter is not executable: $python_bin" >&2
  exit 1
fi
if ! [[ "$positions" =~ ^[1-9][0-9]*$ ]]; then
  echo "NINFER_PHASE11_STAGE_POSITIONS must be a positive integer" >&2
  exit 2
fi

parent_dir="$(dirname -- "$output_dir")"
mkdir -p "$parent_dir"
build_dir="$(mktemp -d "$parent_dir/.phase11-stage-trace.build.XXXXXX")"
published=0
cleanup() {
  if [[ $published -eq 0 && -d "$build_dir" ]]; then
    rm -rf -- "$build_dir"
  fi
}
trap cleanup EXIT

CORPUS="$corpus" OUT="$build_dir/token_ids.json" POSITIONS="$positions"   "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

root = json.loads(Path(os.environ["CORPUS"]).read_text(encoding="utf-8"))
ids = root["token_ids"] if isinstance(root, dict) else root
n = int(os.environ["POSITIONS"])
if len(ids) < n:
    raise SystemExit(f"corpus has only {len(ids)} tokens, need {n}")
Path(os.environ["OUT"]).write_text(
    json.dumps({"token_ids": [int(x) for x in ids[:n]]}, indent=2) + "\n",
    encoding="utf-8",
)
print(f"Prepared {n}-token stage-trace prefix")
PY

echo "Phase 11 CPU stage trace"
echo "  repo:      $repo_root"
echo "  commit:    $(git -C "$repo_root" rev-parse HEAD)"
echo "  positions: $positions"
echo "  temp:      $build_dir"
echo "  final:     $output_dir"

"$python_bin" "$script_dir/run_oracle.py"   --model-dir "$model_dir"   --ple-dir "$ple_dir"   --ids-file "$build_dir/token_ids.json"   --dump-states "$build_dir"

ROOT="$build_dir" POSITIONS="$positions" "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
positions = int(os.environ["POSITIONS"])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
records = manifest.get("positions", [])
if len(records) != positions:
    raise SystemExit(f"manifest has {len(records)} positions, expected {positions}")

required = [
    "embedding",
    "hyper_init",
    "L00_attn_block_input",
    "L00_attn_block_output",
    "L00_hyper_after_attn",
    "L00_mlp_block_input",
    "L00_mlp_block_output",
    "L00_hyper_after_mlp",
    "L03_qsa_projected",
    "L03_qsa_gate",
    "L03_qsa_value",
    "L03_qsa_gated",
    "final_hidden",
    "logits",
]
last = records[-1]
names = {t["name"] for t in last["tensors"]}
missing = [name for name in required if name not in names]
if missing:
    raise SystemExit(f"stage trace is missing required tensors: {missing}")
for record in records:
    for tensor in record["tensors"]:
        path = root / tensor["file"]
        if not path.is_file() or path.stat().st_size != int(tensor["bytes"]):
            raise SystemExit(f"invalid tensor file: {path}")
print(f"Validated {positions}-position CPU stage trace")
PY

cat > "$build_dir/provenance.json" <<EOF
{
  "schema": "ninfer.phase11.stage_trace.v1",
  "repository_commit": "$(git -C "$repo_root" rev-parse HEAD)",
  "positions": $positions,
  "source_corpus": "$corpus"
}
EOF

if [[ -e "$output_dir" ]]; then
  if [[ $replace -ne 1 ]]; then
    echo "final output already exists: $output_dir" >&2
    echo "rerun with --replace only if you intend to replace it" >&2
    exit 1
  fi
  backup="${output_dir}.previous.$(date -u +%Y%m%dT%H%M%SZ)"
  mv -- "$output_dir" "$backup"
  echo "moved previous trace to $backup"
fi

chmod -R a+rX -- "$build_dir"
mv -- "$build_dir" "$output_dir"
published=1
trap - EXIT

echo "Published Phase 11 stage trace at $output_dir"
