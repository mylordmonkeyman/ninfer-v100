"""Focused CPU checks for persistent rounding and distribution comparisons."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] /
                       "tools/reference/qwen3_8_flash_next/oracle"))

from precision_profile import PROFILES, materialize_persistent_states, round_to_bf16
from precision_metrics import compare_logits


class PrecisionProfileTest(unittest.TestCase):
    def test_history_cast_changes_later_step(self):
        # A small value rounded into the stored history must be read back by
        # the next token; the FP32 control retains its original value.
        history = torch.tensor([1.001], dtype=torch.float32)
        ssm = torch.tensor([1.001], dtype=torch.float32)
        layer = SimpleNamespace(conv_states={0: history}, recurrent_states={0: ssm})
        cache = SimpleNamespace(layers=[layer])
        model = SimpleNamespace(layers=[SimpleNamespace(linear_attn=object(), ple=None)])
        materialize_persistent_states(cache, model, PROFILES["v100-phase11-storage"])
        self.assertEqual(history.item(), round_to_bf16(torch.tensor([1.001])).item())
        self.assertNotEqual(history.item(), 1.001)
        self.assertEqual(ssm.item(), torch.tensor([1.001]).item())

    def test_integer_conv_metadata_is_preserved(self):
        history = torch.tensor([1.001], dtype=torch.float32)
        position = torch.tensor([7], dtype=torch.long)
        layer = SimpleNamespace(conv_states={0: history, "position": position},
                                recurrent_states={})
        cache = SimpleNamespace(layers=[layer])
        model = SimpleNamespace(layers=[SimpleNamespace(linear_attn=object(), ple=None)])
        materialize_persistent_states(cache, model, PROFILES["v100-phase11-storage"])
        self.assertEqual(history.item(), round_to_bf16(torch.tensor([1.001])).item())
        self.assertEqual(position.item(), 7)

    def test_qsa_cache_cast(self):
        cache_layer = SimpleNamespace(keys=torch.tensor([1.001]),
                                      values=torch.tensor([1.002]))
        model_layer = SimpleNamespace(ple=None)
        materialize_persistent_states(SimpleNamespace(layers=[cache_layer]),
                                      SimpleNamespace(layers=[model_layer]),
                                      PROFILES["v100-phase11-storage"])
        self.assertEqual(cache_layer.keys.item(), round_to_bf16(torch.tensor([1.001])).item())

    def test_kl_invariant_to_logit_shift(self):
        oracle = np.array([0.5, 1.0, -0.3], dtype=np.float32)
        shifted = oracle + 2.0
        comparison = compare_logits(oracle, shifted)
        self.assertLess(comparison["kl"], 1e-12)
        self.assertTrue(comparison["top1_agree"])


if __name__ == "__main__":
    unittest.main()
