"""Admission tests against independently fetched source config/tensor headers."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.artifact.container import Artifact, ArtifactIdentity
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_8_27b import convert_orcarouter_nvfp4 as converter
from tools.convert.qwen3_6.common.draft_head import read_special_ids

FIXTURE = Path(__file__).parents[2] / "fixtures/qwen3_8_27b_orcarouter_source.json"


@pytest.fixture
def source():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def reader(tensors):
    return SimpleNamespace(names=tuple(tensors), metadata=lambda names: {
        name: SimpleNamespace(shape=tuple(tensors[name]["shape"]), dtype=tensors[name]["dtype"], shard="source")
        for name in names
    })


def test_pinned_source_admitted(source):
    assert converter.validate_config(source["config"])["text"]["hidden_size"] == 5120
    converter.preflight_sources(reader(source["tensors"]))


@pytest.mark.parametrize("name,change", [
    ("lm_head.weight", {"dtype": "F8_E4M3", "shape": [248320, 5120]}),
    ("mtp.layers.0.mlp.down_proj.weight", None),
    ("model.language_model.layers.0.mlp.gate_proj.weight_scale", {"dtype": "BF16", "shape": [17408, 320]}),
])
def test_wrong_component_rejected(source, name, change):
    tensors = deepcopy(source["tensors"])
    if change is None:
        tensors.pop(name)
    else:
        tensors[name] = change
    with pytest.raises(ValueError, match="source tensor"):
        converter.preflight_sources(reader(tensors))


def test_changed_precision_allocation_rejected(source):
    config = source["config"]
    config["quantization_config"]["config_groups"]["group_0"]["targets"].append("lm_head")
    with pytest.raises(ValueError, match="FP8 matrix allocation"):
        converter.validate_config(config)


def test_special_tokens_without_decoder_metadata(tmp_path):
    (tmp_path / "tokenizer.json").write_text(json.dumps({"added_tokens": [
        {"id": 4, "content": "<|end|>", "special": True},
        {"id": 5, "content": "<think>", "special": False},
    ]}), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert read_special_ids(tmp_path) == (4,)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "added_tokens_decoder": {"6": {"special": True}}
    }), encoding="utf-8")
    assert read_special_ids(tmp_path) == (4, 6)


def test_real_artifact_preserves_source_endpoints_and_frontend():
    model = os.environ.get("NINFER_ORCAROUTER_MODEL_DIR")
    artifact = os.environ.get("NINFER_ORCAROUTER_ARTIFACT")
    if not model or not artifact:
        pytest.skip("requires explicitly selected OrcaRouter source and artifact")
    with ShardReader(model) as source, Artifact.open(artifact) as result:
        assert result.identity == ArtifactIdentity(converter.MODEL_ID, "nvfp4")
        import numpy as np
        payload = result.payload("text/draft_head_token_ids")
        shortlist = set(np.frombuffer(payload, dtype="<i4").tolist())
        payload.release()
        tokenizer = json.loads((Path(model) / "tokenizer.json").read_text(encoding="utf-8"))
        special_ids = {t["id"] for t in tokenizer["added_tokens"] if t["special"]}
        assert len(special_ids) == 21
        assert special_ids <= shortlist
        for output, input_name in [
            ("text/token_embedding", "model.language_model.embed_tokens.weight"),
            ("text/output_head", "lm_head.weight"),
        ]:
            represented = source.get(input_name)
            assert represented.dtype == torch.bfloat16
            raw = memoryview(represented.view(torch.uint8).numpy()).cast("B")
            payload = result.payload(output)
            assert len(payload) == len(raw)
            assert hashlib.sha256(payload).digest() == hashlib.sha256(raw).digest()
            payload.release()
            raw.release()
            del represented
        for resource in converter.load_resources(Path(model)):
            payload = result.payload(resource.name)
            assert bytes(payload) == resource.data
            payload.release()
