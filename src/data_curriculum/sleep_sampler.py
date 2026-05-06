from numpy import random
import math
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
        replay_ratio: float=0.1,
        n_phases: int=5,
        n_augmentations=40,
        max_seq_length: int=128,
        decay_rate: float=0.7,
        contextualize_sleep: bool=True,
        replay_strategy: str="weighted"
    ) -> None:
        """
        Args:
            dataset: The dataset to sample from.
            batch_size: Batch size
            replay_ratio: Percentage of high-loss samples to keep for replay (e.g. 0.1).
            n_phases: Number of wake-sleep cycles
            n_augmentations: Number of augmentations for shuffling in Replay Buffer
            max_seq_length: Maximum sequence length for concatenation during sleep
            decay_rate: controls how fast the chance of replay for older samples
                is decayed
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
        # Stores {index: replay score} during wake to determine difficulty
        self.wake_candidates: dict[int, float] = {}
        self.decay_rate = decay_rate # decays replay chance so that older data is less likely to be sampled

        self.phase = "WAKE"
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
        
        # Save max feasible steps for wake phase in case user-defined max exceeds
        # data in folds
        self.wake_max_steps = self.get_wake_max_steps()
        
        # print(f"MAX STEPS: {len(self.dataset) // self.batch_size}")
        
    def __iter__(self):
        # If wake phase:
        #   Shuffle data in fold randomly
        #   append to batch until batch size is met
        while True:
            if self.phase == "WAKE":
                for i in self.folds[self.curr_fold]:
                    assert i < len(self.dataset), f"Index {i} out of range of dataset"
                    yield i
            # If sleep phase:
            elif self.phase == "SLEEP":
                # use contextualized chunks if available, otherwise use replay buffer
                indices_to_sample = (
                    self.contextualized_chunks if self.contextualize_sleep and self.contextualized_chunks else self.replay_buffer
                )

                for i in indices_to_sample:
                    assert i < len(self.dataset), f"Index {i} out of range of dataset"
                    yield i
            

    def add_to_candidates(self, indices: List[int], losses: List[float]):
        """
        Add indices and losses to the candidate buffer during WAKE phase.
        Args:
            indices: List of sample indices.
            losses: List of per-sample loss values.
        """
        #Can only add during WAKE phase
        assert self.phase == "WAKE", "Attempted to update wake candidates during sleep phase."
        
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
                raise ValueError("Replay Buffer is Empty")

        elif new_phase == "WAKE":
            # Transitioning SLEEP -> WAKE
            # Clear replay buffer and reset for new wake cycle
            self.replay_buffer = []
            self.decay_wake_candidates()
            self.contextualized_chunks = []
            # Increment fold tracker and re-init max steps
            self.curr_fold = (self.curr_fold + 1) % self.n_phases
            self.wake_max_steps = self.get_wake_max_steps()
            
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
        if self.replay_strategy == "strict":
            # Sort wake candidates by loss
            sorted_candidates = sorted(
                self.wake_candidates.items(),
                key=lambda item: item[1],
                reverse=True
            )
            self.replay_buffer = [idx for idx, loss in sorted_candidates[:num_replay]]
        elif self.replay_strategy == "weighted":
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
            self.replay_buffer = list(random.choice(candidate_indices,
                                               size=num_replay,
                                               replace=False))
        if self.contextualize_sleep:
            self.contextualized_chunks = self.contextualize_buffer()
    
    def get_wake_max_steps(self):
        """
        Returns the max steps possible for this wake phase based on the size of the buffer.
        """
        return math.ceil(len(self.folds[self.curr_fold]) / self.batch_size)
    
    def decay_wake_candidates(self):
        for idx, prob in self.wake_candidates.items():
            self.wake_candidates[idx] = prob * self.decay_rate
    
    def get_replay_samples(self, num_samples=-1):
        if len(self.replay_buffer) < 1:
            return None
        if num_samples < 0:
            num_samples = len(self.replay_buffer)
        return_idxs = list(random.choice(self.replay_buffer, num_samples))
        result = [self.dataset[int(idx)] for idx in return_idxs]
        return result