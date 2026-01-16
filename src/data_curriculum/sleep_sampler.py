import random
from typing import Iterator, List, Tuple, Sequence

from torch.utils.data import Dataset, Sampler


class SleepSampler(Sampler):
    """
    Sampler that manages data stream for Sleep-Consolidated Learning.
    Switches between WAKE (new data) and SLEEP (replay high-loss data) phases.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        replay_ratio: float = 0.1,
    ) -> None:
        """
        Args:
            dataset: The dataset to sample from.
            batch_size: Batch size
            replay_ratio: Percentage of high-loss samples to keep for replay (e.g. 0.1).
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.replay_ratio = replay_ratio

        self.phase = "WAKE"
        self.replay_buffer: List[int] = []  # Stores indices for sleep
        # Stores (index, loss) during wake to determine "hardness"
        self.wake_candidates: List[Tuple[int, float]] = []

        # WAKE phase state
        self.dataset_indices = list(range(len(dataset))) # type: ignore
        random.shuffle(self.dataset_indices)
        self.wake_pointer = 0