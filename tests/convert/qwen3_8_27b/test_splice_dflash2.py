from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from tools.artifact.container import (
    Artifact, ArtifactIdentity, ResourceSpec, TensorSpec, write_artifact,
)
from tools.artifact.layouts import encode_direct
from tools.convert.qwen3_8_27b import splice_dflash2 as splice


@pytest.fixture
def small_target(tmp_path, monkeypatch):
    # A small native container exercises actual directory/payload copying;
    # the registered descriptor catalog is reduced to avoid a 20GB fixture.
    resource = ResourceSpec("frontend/tokenizer.json", "raw-bytes-v1", 2)
    weight = TensorSpec("text/output_head", (2, 2), "BF16", "contiguous-le-v1")
    monkeypatch.setattr(splice.inventory_nvfp4, "RESOURCE_SPECS", (resource,))
    monkeypatch.setattr(splice.inventory_nvfp4, "BASE_TENSOR_SPECS", (weight,))
    monkeypatch.setattr(splice.groupwise_convert, "OFFICIAL_RESOURCE_SHA256", {
        resource.name: hashlib.sha256(b"{}").hexdigest(),
    })
    payload = encode_direct(torch.tensor([[1, -2], [3, 0]], dtype=torch.bfloat16), "BF16")
    path = tmp_path / "target.ninfer"
    write_artifact(path, ArtifactIdentity("qwen3.8-27b", "nvfp4"), [
        (resource, b"{}"), (weight, payload),
    ])
    return path, resource, weight, payload


def test_splice_preserves_target_and_adds_encoded_companion(small_target, tmp_path, monkeypatch):
    target, resource, weight, target_payload = small_target
    draft_spec = TensorSpec("dflash2/context_norm", (2,), "BF16", "contiguous-le-v1")
    draft_values = torch.tensor([0.5, -1.5], dtype=torch.bfloat16)
    monkeypatch.setattr(splice, "DFLASH2_TENSOR_SPECS", (draft_spec,))
    monkeypatch.setattr(splice.conversion, "load_json", lambda _: {})
    monkeypatch.setattr(splice.dflash2_recipe, "validate_config", lambda _: {})
    monkeypatch.setattr(splice.dflash2_recipe, "preflight_sources", lambda _: SimpleNamespace(source_tensor_count=1))
    monkeypatch.setattr(splice.dflash2_recipe, "materialize_tensor", lambda *_: draft_values)
    from contextlib import nullcontext
    monkeypatch.setattr(splice.ShardReader, "from_file", lambda _: nullcontext(None))

    before = target.read_bytes()
    output = tmp_path / "with-draft.ninfer"
    report_path = splice.splice_artifact(target, tmp_path, output)
    assert target.read_bytes() == before
    with Artifact.open(output) as result:
        assert result.identity == ArtifactIdentity("qwen3.8-27b", "nvfp4")
        assert [obj.name for obj in result.objects] == [resource.name, weight.name, draft_spec.name]
        assert bytes(result.payload(resource.name)) == b"{}"
        assert bytes(result.payload(weight.name)) == target_payload
        assert bytes(result.payload(draft_spec.name)) == encode_direct(draft_values, "BF16")
    report = json.loads(report_path.read_text())
    assert report["target_objects_preserved"] == 2
    assert report["dflash2_objects_added"] == 1
    assert report["payloads_verified"] == 3
    assert not output.with_name(output.name + ".partial").exists()


def test_splice_rejects_changed_target_descriptor(small_target, tmp_path):
    _, resource, weight, _ = small_target
    changed = TensorSpec(weight.name, (4,), weight.format, weight.layout)
    malformed = tmp_path / "wrong-shape.ninfer"
    write_artifact(malformed, ArtifactIdentity("qwen3.8-27b", "nvfp4"), [
        (resource, b"{}"), (changed, b"\x00" * 8),
    ])
    with Artifact.open(malformed) as source:
        with pytest.raises(ValueError, match="descriptor"):
            splice.base_specs(source)


def test_splice_refuses_to_overwrite_target(small_target, tmp_path):
    target = small_target[0]
    before = target.read_bytes()
    with pytest.raises(ValueError, match="must not already exist"):
        splice.splice_artifact(target, tmp_path, target)
    assert target.read_bytes() == before
