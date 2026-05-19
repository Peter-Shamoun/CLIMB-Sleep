import numpy as np
import math
from typing import List, Optional

import torch
from numpy import random
from torch.utils.data import Dataset, Sampler

from src.data_curriculum.utility_scoring import sample_utility_indices


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
        max_seq_length: int = 128,
        decay_rate: float = 0.3,
        min_decay_factor: float = 0.2,
        contextualize_sleep: bool = True,
        replay_strategy: str = "weighted",
        utility_temperature_gain: float = 1.0,
        utility_temperature_need: float = 1.0,
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
        self.replay_strategy: str = (
            replay_strategy  # choice of "loss", "random", "loss_weighted"
        )
        self.replay_buffer: List[int] = []  # Stores indices for sleep
        # Stores {index: (replay score, decay_factor)} during wake to determine difficulty
        self.wake_candidates: dict[int, tuple(float, float)] = {}
        # Stores {index: squared gradient norm} during wake for Gain x Need replay.
        # Populated by add_to_candidates when the trainer computes per-sample grads.
        self.curr_wake_candidates: List[int] = []
        # Need scores: set by the trainer at end of wake before switch_phase("SLEEP").
        # self.need_scores: dict[int, float] = {}
        # Last utility-distribution diagnostics, set by update_replay_buffer when
        # replay_strategy == "utility" so the trainer can log them to WandB.
        self.last_utility_diagnostics: Optional[dict] = None
        # Utility-replay temperatures (only consulted when replay_strategy == "utility")
        self.utility_temperature_gain = utility_temperature_gain
        self.utility_temperature_need = utility_temperature_need
        self.decay_rate = decay_rate # decays replay score so that older data is less likely to be sampled
        self.min_decay_factor = min_decay_factor # clamps decay rate so that older samples are not completely forgotten

        self.phase = "WAKE"
        self.dataset_indices = list(range(len(dataset)))  # type: ignore
        np.random.shuffle(self.dataset_indices)

        # Split indices into n_phases folds
        self.curr_fold = 0
        self.fold_size = len(self.dataset_indices) // self.n_phases
        self.folds = [
            self.dataset_indices[i : i + self.fold_size]
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
                    assert i < len(
                        self.dataset
                    ), f"Index {i} out of range of dataset"
                    yield i
            # If sleep phase:
            elif self.phase == "SLEEP":
                # use contextualized chunks if available, otherwise use replay buffer
                indices_to_sample = (
                    self.contextualized_chunks
                    if self.contextualize_sleep and self.contextualized_chunks
                    else self.replay_buffer
                )

                for i in indices_to_sample:
                    assert i < len(
                        self.dataset
                    ), f"Index {i} out of range of dataset"
                    yield i

    def add_to_candidates(
        self,
        indices: List[int],
        scores: List[float],
        gain_norms: Optional[List[float]] = None,
    ):
        """
        Add indices, replay scores, and optional Gain scores to the candidate buffer during WAKE.
        Args:
            indices: List of sample indices.
            scores: List of per-sample loss values.
            gain_norms: Optional list of per-sample squared gradient norms (Gain).
                Required when replay_strategy == "utility"; ignored otherwise.
        """
        # Can only add during WAKE phase
        assert (
            self.phase == "WAKE"
        ), "Attempted to update wake candidates during sleep phase."
        if gain_norms is not None:
            assert len(gain_norms) == len(
                indices
            ), "gain_norms must have same length as indices"
        # store indices of candidates from this phase
        self.curr_wake_candidates.extend(indices)
        for idx, score in zip(indices, scores):
            # Store or update with max loss seen for this index
            score = float(score)
            if idx in self.wake_candidates:
                # Keep the lower loss if we've seen this sample before
                curr_score, curr_decay_factor = self.wake_candidates[idx]
                self.wake_candidates[idx] = (min(curr_score, score), curr_decay_factor)
            else:
                self.wake_candidates[idx] = (score, 1.0)
            # if gain_norms is not None:
            #     # Latest-seen Gain. Repeats within a single wake cycle are rare
            #     # (each fold is iterated at most once per cycle).
            #     self.wake_gain[idx] = float(gain_norms[i])

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
            self.curr_wake_candidates = []
            self.decay_wake_candidates()
            # Gain values are model-state-dependent; stale norms from earlier cycles
            # would bias selection. Clear fully rather than decay.
            # self.wake_gain = {}
            # self.need_scores = {}
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
            np.random.shuffle(shuffled)
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
            # Sort wake candidates by replay score
            sorted_candidates = sorted(
                self.wake_candidates.items(),
                key=lambda item: item[1][0] * item[1][1],
                reverse=True,
            )
            self.replay_buffer = [
                idx for idx, score in sorted_candidates[:num_replay]
            ]
        elif self.replay_strategy == "weighted":
            # Sample from wake candidates weighted by replay score
            candidate_indices = list(self.wake_candidates.keys())
            candidate_replay_scores = torch.prod(torch.tensor(
                list(zip(*self.wake_candidates.values()))
            ), dim=0)
            sampled_indices = torch.multinomial(
                candidate_replay_scores, num_samples=num_replay, replacement=False
            ).tolist()
            self.replay_buffer = [
                candidate_indices[i] for i in sampled_indices
            ]
        elif self.replay_strategy == "random":
            # Uniformly randomly sample from wake candidates
            candidate_indices = list(self.wake_candidates.keys())
            self.replay_buffer = list(
                np.random.choice(
                    candidate_indices, size=num_replay, replace=False
                )
            )
        # elif self.replay_strategy == "utility":
        #     assert (
        #         self.wake_gain
        #     ), "Gain dict empty; trainer should populate via add_to_candidates"
        #     assert (
        #         self.need_scores
        #     ), "Need dict empty; trainer should set before switch_phase('SLEEP')"
        #     # Utility pool is wake_gain (current-fold samples with fresh Gain),
        #     # not wake_candidates (which carries decayed losses across cycles).
        #     num_replay = int(len(self.wake_gain) * self.replay_ratio)
        #     sampled, diagnostics = sample_utility_indices(
        #         gain_scores=self.wake_gain,
        #         need_scores=self.need_scores,
        #         num_replay=num_replay,
        #         temperature_gain=self.utility_temperature_gain,
        #         temperature_need=self.utility_temperature_need,
        #     )
        #     self.replay_buffer = sampled
        #     self.last_utility_diagnostics = diagnostics
        if self.contextualize_sleep:
            self.contextualized_chunks = self.contextualize_buffer()

    def get_wake_max_steps(self):
        """
        Returns the max steps possible for this wake phase based on the size of the buffer.
        """
        return math.ceil(len(self.folds[self.curr_fold]) / self.batch_size)

    def decay_wake_candidates(self):
        for idx, candidate in self.wake_candidates.items():
            curr_loss, curr_decay_factor = candidate
            new_decay_factor = max(curr_decay_factor * (1 - self.decay_rate), self.min_decay_factor)
            self.wake_candidates[idx] = (curr_loss, new_decay_factor)

    def get_replay_samples(self, num_samples=-1):
        """Returns a random selection of samples from the replay buffer for analysis.

        Args:
            num_samples (int, optional): Number of samples to return. Defaults to -1: return entire buffer.

        Returns:
            List(Tuple(Dict, float)): List of samples and losses. Samples
                contain lists of ints with keys input_ids, special_tokens_mask,
                and attention_mask.
        """
        if len(self.replay_buffer) < 1:
            return None
        if num_samples < 0:
            return_idxs = self.replay_buffer
        else:
            return_idxs = list(np.random.choice(self.replay_buffer, num_samples))
        result = [
            (self.dataset[int(idx)], self.wake_candidates[idx])
            for idx in return_idxs
        ]
        return result
    
    def update_utility_scores(self, need_scores: dict[int, float]):
        """Updates stored gain scores to become utility scores with calculated need scores.

        Args:
            need_scores (dict[int, float]): mapping of sample idx to need score.
        """
        gains, _ = zip(*self.wake_candidates.values())
        gain_softmax_denom = np.exp(np.array(list(gains))).sum()
        needs_softmax_denom = np.exp(np.array(list(need_scores.values()))).sum()
        # calculate softmaxes to put both on the same scale
        for idx, need in need_scores.items():
            gain, decay_factor = self.wake_candidates[idx]
            sm_gain = np.exp(gain) / gain_softmax_denom
            sm_need = np.exp(need) / needs_softmax_denom
            utility_score = sm_gain * sm_need
            self.wake_candidates[idx] = (utility_score, decay_factor)
                