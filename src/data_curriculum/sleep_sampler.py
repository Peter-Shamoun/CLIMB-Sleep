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
        # Stores {index: loss} during wake to determine difficulty
        self.wake_candidates: dict[int, float] = {}

        # WAKE phase state
        self.dataset_indices = list(range(len(dataset))) # type: ignore
        random.shuffle(self.dataset_indices)
        self.wake_pointer = 0
        
    def __iter__(self):
        # implement logic of sampling here
        batch = []
        # If wake phase:
        #   Shuffle data in fold randomly
        #   append to batch until batch size is met
        # If sleep phase:
        #   Sample from replay buffer based on loss (higher loss = higher prob)
        if self.phase == "WAKE":
            for i in self.dataset_indices:
                batch.append(self.dataset[i])
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        elif self.phase == "SLEEP":
            if not self.replay_buffer:
                # No data to replay, switch back to WAKE
                self.phase = "WAKE"
                self.wake_pointer = 0
                return self.__iter__()

            # Sample from replay buffer based on loss
            losses = [self.wake_candidates[idx] for idx in self.replay_buffer]
            total_loss = sum(losses)
            probabilities = [loss / total_loss for loss in losses]

            sampled_indices = random.choices(
                self.replay_buffer,
                weights=probabilities,
                k=len(self.replay_buffer)
            )

            for idx in sampled_indices:
                batch.append(self.dataset[idx])
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

    def __len__(self):
        return len(self.dataset)