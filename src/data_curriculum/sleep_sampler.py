import random
from typing import Iterator, List, Tuple, Sequence

import torch
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
        n_phases: int = 5,
        n_augmentations = 40,
        max_seq_length: int = 128,
        contextualize_sleep: bool = True,
        replay_strategy: str = "loss"
    ) -> None:
        """
        Args:
            dataset: The dataset to sample from.
            batch_size: Batch size
            replay_ratio: Percentage of high-loss samples to keep for replay (e.g. 0.1).
            n_phases: Number of wake-sleep cycles
            n_augmentations: Number of augmentations for shuffling in Replay Buffer
            max_seq_length: Maximum sequence length for concatenation during sleep
            contextualize_sleep: Whether to apply contextualization during sleep,
            replay_strategy: Type of strategy to apply for Replay Buffer
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Sleep hyperparameters
        self.n_phases = n_phases
        self.n_augmentations = n_augmentations
        self.max_seq_length = max_seq_length
        self.contextualize_sleep = contextualize_sleep
        self.contextualized_chunks: List[int] = []

        # Replay buffer
        self.replay_ratio = replay_ratio
        self.replay_strategy: str = replay_strategy # choice of "loss", "random", "loss_weighted"
        self.replay_buffer: List[int] = []  # Stores indices for sleep
        # Stores {index: loss} during wake to determine difficulty

        self.phase = "WAKE"
        self.wake_candidates: dict[int, float] = {}

        self.dataset_indices = list(range(len(dataset))) # type: ignore
        random.shuffle(self.dataset_indices)
        
        # Split indices into n_phases folds
        self.curr_fold = 0
        self.fold_size = len(self.dataset_indices) // self.n_phases
        self.folds = [
            self.dataset_indices[i: i + self.fold_size]
            for i in range(0, len(self.dataset_indices), self.fold_size)
        ]
        
        self.wake_pointer = 0
        
    def __iter__(self):
        # If wake phase:
        #   Shuffle data in fold randomly
        #   append to batch until batch size is met
        if self.phase == "WAKE":
            for i in self.folds[self.curr_fold]:
                yield i
        # If sleep phase:
        elif self.phase == "SLEEP":
            if not self.replay_buffer:
                # No data to replay, switch back to WAKE
                print("Replay buffer empty, switching back to WAKE phase.")
                self.phase = "WAKE"
                self.wake_pointer = 0
                return self.__iter__()
            
            # use contextualized chunks if available, otherwise use replay buffer
            indices_to_sample = (
                self.contextualized_chunks 
                if self.contextualize_sleep and self.contextualized_chunks 
                else self.replay_buffer
            )

            for i in indices_to_sample:
                yield i
            

    def add_to_candidates(self, indices: List[int], losses: List[float]):
        """
        Add indices and losses to the candidate buffer during WAKE phase.
        Args:
            indices: List of sample indices.
            losses: List of per-sample loss values.
        """
        if self.phase != "WAKE": # can only add during WAKE phase
            return
        for idx, loss in zip(indices, losses):
            # Store or update with max loss seen for this index
            loss_val = float(loss)
            if idx in self.wake_candidates:
                # Keep the lower loss if we've seen this sample before
                self.wake_candidates[idx] = min(self.wake_candidates[idx], loss_val)
            else:
                self.wake_candidates[idx] = loss_val

    def switch_phase(self, new_phase: str):
        """
        Toggle between WAKE and SLEEP modes.
        When switching to SLEEP, populates replay_buffer from wake_candidates.
        When switching to WAKE, clears replay_buffer and wake_candidates and 
        advances the fold for the next phase.
        Args:
            new_phase: Either "WAKE" or "SLEEP"
        """
        if new_phase == self.phase:
            return

        if new_phase == "SLEEP":
            # Transitioning WAKE -> SLEEP
            # Process wake_candidates to fill replay_buffer
            if self.wake_candidates:
                self.update_replay_buffer()
                if self.contextualize_sleep:
                    self.contextualized_chunks = self.contextualize_buffer()
            else:
                self.replay_buffer = []

        elif new_phase == "WAKE":
            # Transitioning SLEEP -> WAKE
            # Clear replay buffer and reset for new wake cycle
            self.replay_buffer = []
            self.wake_candidates = {}
            self.contextualized_chunks = []
            # Reset dataset pointer for next wake phase
            self.curr_fold = (self.curr_fold + 1) % self.n_phases
            self.wake_pointer = 0

        self.phase = new_phase

    def __len__(self):
        return len(self.dataset)

    def contextualize_buffer(self) -> List[int]:
        """
        "Contextualizes" replay buffer before sleep phase to make it more abstract

        Returns:
            List of shuffled orderings (each ordering is a list of indices)
        """
        if not self.replay_buffer:
            return []
        
        all_orderings = []

        # create n_augmentations different shuffled orderings
        for _ in range(self.n_augmentations):
            # shuffle replay buffer differently each time
            shuffled = self.replay_buffer.copy()
            random.shuffle(shuffled)
            all_orderings.append(shuffled)

        # flatten: convert list of orderings into individual indices
            # will allow __iter__ to yield them sequentially
        flattened = []
        for ordering in all_orderings:
            flattened.extend(ordering)
        
        return flattened
    
    def update_replay_buffer(self):
        """
        Updates the replay buffer with high-loss samples from the wake phase.
        """
        num_replay = int(len(self.wake_candidates.keys()) * self.replay_ratio)
        if self.replay_strategy == "loss":
            # Sort wake candidates by loss
            sorted_candidates = sorted(
                self.wake_candidates.items(),
                key=lambda item: item[1],
                reverse=True
            )
            self.replay_buffer = [idx for idx, loss in sorted_candidates[:num_replay]]
        elif self.replay_strategy == "loss_weighted":
            # Sample from wake candidates weighted by loss
            candidate_indices = list(self.wake_candidates.keys())
            candidate_losses = torch.tensor(list(self.wake_candidates.values()))
            sampled_indices = torch.multinomial(
                candidate_losses,
                num_samples=num_replay,
                replacement=False
            ).tolist()
            self.replay_buffer = [candidate_indices[i] for i in sampled_indices]
        elif self.replay_strategy == "random":
            # Randomly sample from wake candidates
            candidate_indices = list(self.wake_candidates.keys())
            self.replay_buffer = random.sample(candidate_indices, num_replay)
        if self.contextualize_sleep:
            self.replay_buffer = self.contextualize_buffer()