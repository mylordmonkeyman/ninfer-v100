"""Import the registered OrcaRouter Qwen3.8-27B NVFP4 checkpoint.

The selected source owns every base-model component. Embeddings and the full
output head remain BF16; the existing mixed matrix codes/scales are imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from tools.artifact.container import ArtifactIdentity, ArtifactWriter
from tools.artifact.layouts import encode_direct
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion, recipe as family_recipe
from tools.convert.qwen3_6_27b import convert as family_config, draft_head
from tools.convert.qwen3_6_27b import recipe as bf16_recipe
from . import convert as canonical, convert_nvfp4 as mixed
from . import dflash2_recipe, inventory_nvfp4 as inventory, recipe_nvfp4 as recipe
from .dflash2_inventory import DFLASH2_TENSOR_SPECS

MODEL_ID = "qwen3.8-27b-orcarouter"
WEIGHTS_ID = "nvfp4"
REPOSITORY = "orcarouter/Qwen3.8-27B-Uncensored-NVFP4"
REVISION = "69d21348b2d6c11439fb69368f40414c2256e44e"
RECIPE_ID = "qwen3_8_27b_orcarouter_nvfp4-v1"
RESOURCE_SHA256 = {
    **canonical.OFFICIAL_RESOURCE_SHA256,
    "frontend/tokenizer.json": "f399b3cd12fa270d51457bb749fb30863521e8359b8a27059c71b6c2f7d6dd6c",
    "frontend/tokenizer_config.json": "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87",
    "frontend/generation_config.json": "b8eb74d15e0a56623d00ccd14950a4bb87fabbf84b5cc030dcc904b899fb1eb5",
}
BF16_ENDPOINTS = frozenset(("text/token_embedding", "text/output_head"))
BASE_TENSOR_SPECS = tuple(
    inventory.tensor_spec(spec.name, spec.shape, inventory.BF16)
    if spec.name in BF16_ENDPOINTS else spec
    for spec in inventory.BASE_TENSOR_SPECS
)
OBJECT_SPECS = (*inventory.RESOURCE_SPECS, *BASE_TENSOR_SPECS, *DFLASH2_TENSOR_SPECS)
BF16_RECIPES = tuple(
    bf16_recipe.RECIPES_BY_NAME[spec.name]
    for spec in BASE_TENSOR_SPECS
    if spec.name in BF16_ENDPOINTS or spec.name in recipe.OFFICIAL_RECIPES_BY_NAME
)
BF16_BY_NAME = {item.object_name: item for item in BF16_RECIPES}
SOURCE_REQUIREMENTS = {
    name: signature for name, signature in recipe.SOURCE_REQUIREMENTS.items()
    if not name.startswith("lm_head.")
}
SOURCE_REQUIREMENTS.update({
    item.name: (item.shape, item.dtype)
    for item in family_recipe.source_requirements(BF16_RECIPES).values()
})


def validate_config(config: dict) -> dict:
    summary = family_config.validate_config(config)
    quant = config.get("quantization_config", {})
    conversion.check_members("quantization_config", quant, {
        "quant_method": "compressed-tensors", "quantization_status": "compressed",
        "format": "mixed-precision",
    })
    groups = quant.get("config_groups", {})
    if set(groups) != {"group_0", "group_1"}:
        raise ValueError("OrcaRouter source requires the two registered quantization groups")
    fp8_names = sorted(s.name for s in recipe.FP8_SOURCES if s.name != "lm_head")
    nvfp4_names = sorted(s.name for s in recipe.NVFP4_SOURCES)
    if sorted(groups["group_0"].get("targets", [])) != fp8_names:
        raise ValueError("OrcaRouter FP8 matrix allocation differs from the registered source")
    if sorted(groups["group_1"].get("targets", [])) != nvfp4_names:
        raise ValueError("OrcaRouter NVFP4 matrix allocation differs from the registered source")
    # Validate arithmetic/storage settings with the shared mixed-format rules;
    # the exact source allocation above is expressed as names instead of regexes.
    mixed._validate_float_group({**groups["group_0"], "targets": mixed._FP8_TARGETS})
    mixed._validate_nvfp4_group({**groups["group_1"], "targets": mixed._NVFP4_TARGETS})
    if "lm_head" not in quant.get("ignore", []):
        raise ValueError("OrcaRouter source must retain its BF16 output head")
    return summary


def load_resources(model: Path) -> tuple[conversion.ResourcePayload, ...]:
    result = []
    for name, expected in RESOURCE_SHA256.items():
        data = (model / name.removeprefix("frontend/")).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"{name}: resource differs from the pinned OrcaRouter source")
        result.append(conversion.ResourcePayload(name, data))
    return tuple(result)


def preflight_sources(reader: ShardReader) -> family_recipe.SourcePreflight:
    metadata = reader.metadata(reader.names)
    for name, signature in SOURCE_REQUIREMENTS.items():
        item = metadata.get(name)
        if item is None or (item.shape, item.dtype) != signature:
            raise ValueError(f"{name}: source tensor does not match {signature}")
    expected_quantized = {
        name for name, (_, dtype) in SOURCE_REQUIREMENTS.items()
        if dtype in ("U8", "F8_E4M3", "F32")
    }
    actual_quantized = {
        name for name, item in metadata.items() if item.dtype in ("U8", "F8_E4M3", "F32")
    }
    if expected_quantized != actual_quantized or "lm_head.weight_scale" in metadata:
        raise ValueError("OrcaRouter quantized allocation differs from the registered source")
    counts: dict[str, int] = {}
    for _, dtype in SOURCE_REQUIREMENTS.values():
        counts[dtype] = counts.get(dtype, 0) + 1
    return family_recipe.SourcePreflight(
        recipe_count=len(BASE_TENSOR_SPECS), source_tensor_count=len(SOURCE_REQUIREMENTS),
        source_shard_count=len({metadata[n].shard for n in SOURCE_REQUIREMENTS}),
        source_dtype_counts=counts,
    )


def convert(model_dir: Path, dflash2_model_dir: Path, output: Path, *, device: str = "cuda") -> Path:
    started = time.perf_counter()
    resolved_device = pick_device(device)
    mixed._validate_index(model_dir)
    summary = validate_config(conversion.load_json(model_dir / "config.json"))
    resources = load_resources(model_dir)
    draft_summary = dflash2_recipe.validate_config(conversion.load_json(dflash2_model_dir / "config.json"))
    dflash2_recipe.validate_base_compatibility(summary, draft_summary)
    dflash_source = dflash2_recipe.preflight_sources(dflash2_model_dir)
    with ShardReader(model_dir) as reader:
        source = preflight_sources(reader)
    resource_map = {r.name: r.data for r in resources}
    plan = conversion.build_object_plan(OBJECT_SPECS, resource_map)
    root = Path(__file__).resolve().parents[3]
    ranking = root / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, model_dir)
    derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_head.materialize_draft_head_token_ids(draft)}
    identity = ArtifactIdentity(MODEL_ID, WEIGHTS_ID)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"preflight complete: {len(plan.objects)} objects; preserving BF16 endpoints", flush=True)
    with ArtifactWriter(output, identity, plan.specs) as writer:
        for name, data in resource_map.items():
            writer.write(name, data)
        with ShardReader(model_dir) as reader:
            for i, spec in enumerate(BASE_TENSOR_SPECS):
                if spec.name in BF16_BY_NAME:
                    tensor = family_recipe.materialize_recipe(BF16_BY_NAME[spec.name], reader, derived)
                    payload = conversion.encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                elif spec.name in recipe.FP8_WEIGHTS_BY_NAME:
                    payload = mixed._encode_fp8_weight(spec, reader)
                elif spec.name in recipe.NVFP4_WEIGHTS_BY_NAME:
                    payload = mixed._encode_nvfp4_weight(spec, reader)
                elif spec.name in recipe.INPUT_DIVISORS_BY_NAME:
                    payload = encode_direct(recipe.materialize_input_divisor(recipe.INPUT_DIVISORS_BY_NAME[spec.name], reader), inventory.FP32)
                else:
                    tensor = recipe.materialize_quantized_direct(spec.name, reader)
                    payload = encode_direct(tensor, spec.format)
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(f"[{i + 1}/{len(BASE_TENSOR_SPECS)}] {spec.name}", flush=True)
        with ShardReader.from_file(dflash2_model_dir / "model.safetensors") as reader:
            for spec in DFLASH2_TENSOR_SPECS:
                tensor = dflash2_recipe.materialize_tensor(spec.name, reader)
                payload = conversion.encode_tensor_payload(tensor, spec, resolved_device)
                del tensor
                writer.write(spec.name, payload)
                del payload
    report = conversion.build_conversion_report(
        identity=identity, target_key=inventory.TARGET_KEY, recipe_id=RECIPE_ID,
        repo_root=root, model_dir=model_dir, out_path=output,
        arguments={"model": str(model_dir), "dflash2_model": str(dflash2_model_dir), "out": str(output), "device": device},
        config_summary={"base": summary, "dflash2": draft_summary},
        source_preflight=source, objects=plan.objects,
        elapsed_seconds=time.perf_counter() - started, final_bytes=output.stat().st_size,
        device=resolved_device, ranking_path=ranking,
    )
    report["source"] = {
        "base": {"repository": REPOSITORY, "revision": REVISION, "model_path": str(model_dir.resolve())},
        "dflash2": {"repository": dflash2_recipe.REPOSITORY, "revision": dflash2_recipe.REVISION,
                    "model_path": str(dflash2_model_dir.resolve()), "tensors": dflash_source.source_tensor_count},
    }
    report["bf16_endpoints"] = sorted(BF16_ENDPOINTS)
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"complete: {output.stat().st_size} bytes; report={report_path}", flush=True)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dflash2-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    convert(args.model, args.dflash2_model, args.out, device=args.device)


if __name__ == "__main__":
    main()
