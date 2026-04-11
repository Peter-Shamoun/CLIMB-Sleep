import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import unittest
from src.synaptic_homeostasis import apply_synaptic_homeostasis, _is_target_param


class MockModel(nn.Module):
    """Mimics RoBERTa layer naming so _is_target_param matches."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleDict({"layer": nn.ModuleList([self._make_layer()])})
        self.embeddings = nn.Embedding(100, 8)

    def _make_layer(self):
        return nn.ModuleDict({
            "attention": nn.ModuleDict({"self": nn.ModuleDict({"query": nn.Linear(8, 8, bias=False)})}),
            "intermediate": nn.ModuleDict({"dense": nn.Linear(8, 16, bias=False)}),
            "output": nn.ModuleDict({"dense": nn.Linear(16, 8, bias=False), "LayerNorm": nn.LayerNorm(8)}),
        })


class TestSynapticHomeostasis(unittest.TestCase):
    def _get_model(self):
        m = MockModel()
        torch.manual_seed(0)
        return m

    def test_weights_scaled(self):
        m = self._get_model()
        q = m.encoder["layer"][0]["attention"]["self"]["query"].weight
        q.data.fill_(1.0)
        apply_synaptic_homeostasis(m, decay_factor=0.9, protect_percentile=0.0, zero_threshold=0.0)
        self.assertTrue(torch.allclose(q.data, torch.full_like(q.data, 0.9)))

    def test_high_magnitude_protected(self):
        m = self._get_model()
        q = m.encoder["layer"][0]["attention"]["self"]["query"].weight
        q.data.fill_(1.0)
        q.data[0, 0] = 100.0
        apply_synaptic_homeostasis(m, decay_factor=0.5, protect_percentile=99.0, zero_threshold=0.0)
        self.assertEqual(q.data[0, 0].item(), 100.0)
        self.assertAlmostEqual(q.data[1, 1].item(), 0.5, places=5)

    def test_below_threshold_zeroed(self):
        m = self._get_model()
        q = m.encoder["layer"][0]["attention"]["self"]["query"].weight
        q.data.fill_(0.05)
        apply_synaptic_homeostasis(m, decay_factor=0.9, protect_percentile=0.0, zero_threshold=0.05)
        self.assertTrue((q.data == 0.0).all())

    def test_layernorm_untouched(self):
        m = self._get_model()
        ln = m.encoder["layer"][0]["output"]["LayerNorm"]
        before = ln.weight.data.clone()
        apply_synaptic_homeostasis(m, decay_factor=0.5)
        self.assertTrue(torch.equal(ln.weight.data, before))

    def test_embeddings_untouched(self):
        m = self._get_model()
        before = m.embeddings.weight.data.clone()
        apply_synaptic_homeostasis(m, decay_factor=0.5)
        self.assertTrue(torch.equal(m.embeddings.weight.data, before))

    def test_stats_returned(self):
        m = self._get_model()
        stats = apply_synaptic_homeostasis(m)
        self.assertIn("n_params_scaled", stats)
        self.assertIn("n_params_protected", stats)
        self.assertIn("n_params_zeroed", stats)
        self.assertGreater(stats["n_params_scaled"], 0)

    def test_filter_fn(self):
        self.assertTrue(_is_target_param("encoder.layer.0.attention.self.query.weight"))
        self.assertTrue(_is_target_param("encoder.layer.0.intermediate.dense.weight"))
        self.assertTrue(_is_target_param("encoder.layer.0.output.dense.weight"))
        self.assertFalse(_is_target_param("encoder.layer.0.output.LayerNorm.weight"))
        self.assertFalse(_is_target_param("embeddings.word_embeddings.weight"))
        self.assertFalse(_is_target_param("encoder.layer.0.attention.self.query.bias"))


if __name__ == "__main__":
    unittest.main()
