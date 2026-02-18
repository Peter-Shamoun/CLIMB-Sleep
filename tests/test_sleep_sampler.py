import os
import random
import sys
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

# Allow `import src...` when running from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_curriculum.sleep_sampler import SleepSampler


def _base_collate_fn(samples):
    # Minimal collate to avoid importing project config dependencies
    joined = {}
    for s in samples:
        for k, v in s.items():
            joined.setdefault(k, []).append(torch.tensor(v))
    return {k: torch.stack(v) for k, v in joined.items()}


class _DummyDataset(Dataset):
    def __init__(self, n: int, seq_len: int = 8):
        self.samples = []
        for i in range(n):
            self.samples.append(
                {
                    "input_ids": [0] + [i + 3] * (seq_len - 2) + [2],
                    "attention_mask": [1] * seq_len,
                    "indices": i,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SleepSamplerTests(unittest.TestCase):
    def test_switch_to_sleep_populates_topk(self):
        ds = _DummyDataset(10)
        sampler = SleepSampler(ds, batch_size=2, replay_ratio=0.2, n_phases=1)

        idxs = list(range(10))
        losses = [float(i) for i in range(10)]
        sampler.add_to_buffer(idxs, losses)
        sampler.switch_phase("SLEEP")

        # top 20% of 10 => 2 items: indices 9 and 8
        self.assertEqual(sampler.phase, "SLEEP")
        self.assertEqual(set(sampler.replay_buffer), {9, 8})

    def test_dataloader_uses_sampler_indices(self):
        ds = _DummyDataset(6)
        sampler = SleepSampler(ds, batch_size=2, replay_ratio=0.5, n_phases=1)
        sampler.switch_phase("WAKE")

        dl = DataLoader(ds, sampler=sampler, batch_size=2, collate_fn=_base_collate_fn)
        batch = next(iter(dl))
        self.assertIn("indices", batch)
        self.assertEqual(batch["indices"].shape[0], 2)

    def test_sleep_sampling_is_loss_weighted(self):
        random.seed(0)
        ds = _DummyDataset(3)
        sampler = SleepSampler(ds, batch_size=1, replay_ratio=1.0, n_phases=1)

        sampler.wake_candidates = {0: 1.0, 1: 1.0, 2: 10.0}
        sampler.replay_buffer = [0, 1, 2]
        sampler.phase = "SLEEP"

        draws = [next(iter(sampler)) for _ in range(50)]
        # highest-loss sample should show up more than either low-loss sample
        self.assertGreater(draws.count(2), draws.count(0))
        self.assertGreater(draws.count(2), draws.count(1))


if __name__ == "__main__":
    unittest.main()

