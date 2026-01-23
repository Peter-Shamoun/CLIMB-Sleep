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

    def add_to_buffer(self, indices: List[int], losses: List[float]):
        """
        Add indices and losses to the candidate buffer during WAKE phase.
        Args:
            indices: List of sample indices.
            losses: List of per-sample loss values.
        """
        for idx, loss in zip(indices, losses):
            # Store or update with max loss seen for this index
            loss_val = float(loss)
            if idx in self.wake_candidates:
                # Keep the higher loss if we've seen this sample before
                self.wake_candidates[idx] = max(self.wake_candidates[idx], loss_val)
            else:
                self.wake_candidates[idx] = loss_val

    def switch_phase(self, new_phase: str):
        """
        Toggle between WAKE and SLEEP modes.
        When switching to SLEEP, populates replay_buffer from wake_candidates.
        Args:
            new_phase: Either "WAKE" or "SLEEP"
        """
        if new_phase == self.phase:
            return

        if new_phase == "SLEEP":
            # Transitioning WAKE -> SLEEP
            # Process wake_candidates to fill replay_buffer
            if self.wake_candidates:
                # Sort by loss descending (hardest first)
                sorted_candidates = sorted(
                    self.wake_candidates.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Keep top N%
                num_keep = int(len(sorted_candidates) * self.replay_ratio)
                num_keep = max(1, num_keep)  # Keep at least 1

                # Extract just the indices for replay buffer
                self.replay_buffer = [idx for idx, loss in sorted_candidates[:num_keep]]
            else:
                self.replay_buffer = []

        elif new_phase == "WAKE":
            # Transitioning SLEEP -> WAKE
            # Clear replay buffer and reset for new wake cycle
            self.replay_buffer = []
            self.wake_candidates = {}
            # Reset dataset pointer for next wake phase
            random.shuffle(self.dataset_indices)
            self.wake_pointer = 0

        self.phase = new_phase

    def __len__(self):
        return len(self.dataset)