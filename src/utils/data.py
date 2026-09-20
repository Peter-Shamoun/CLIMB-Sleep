"""Class for preprocessing the data, including tokenization, etc."""

# typing imports
import string
from collections import defaultdict

# typing imports
from typing import Dict, List, Tuple

import torch
from torch.utils.data.sampler import Sampler
from transformers import PreTrainedTokenizerFast

from src.config import BabyLMConfig

def base_collate_fn(_samples: List[Dict[str, List[Tuple[int, float]]]]):
    joined_batch = defaultdict(list)
    for sample in _samples:
        # print("SAMPLE:", sample)
        for key, val in sample.items():
            if type(val) != list:
                continue
            joined_batch[key].append(torch.tensor(val))

    batch = {}

    for key, val in joined_batch.items():
        batch[key] = torch.stack(val)
    
    return batch


class SequentialSubsetSampler(Sampler):
    """
    Samples elements sequentially from a set of indices, always in the same order.
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class DatasetPreprocessor(object):
    def __init__(self, cfg: BabyLMConfig, tokenizer: PreTrainedTokenizerFast):
        """
        Args:
            cfg (BabyLMConfig): hydra config object
            tokenizer (PreTrainedTokenizer): instantiated tokenizer object
        """

        # data processing params
        self.include_punctuation = cfg.data_preprocessing.include_punctuation
        self.max_input_length = cfg.data_preprocessing.max_input_length
        self.join_sentences = cfg.data_preprocessing.join_sentences
        self.callback_functions = cfg.data_preprocessing.callback_functions
        self.dataset_subconfig = cfg.dataset.subconfig
        self.tokenizer = tokenizer

    ### --- Callback functions --- ###

    # NOTE: The function names of callbacks must match the names in the data preprocessing
    # callback_functions list (sepcified in the config file)

    ### --- Callback functions --- ###

    def __call__(self, examples):
        if not self.include_punctuation:
            examples["text"] = [
                line.translate(str.maketrans("", "", string.punctuation))
                for line in examples["text"]
            ]

        batch = {
            "input_ids": [],
            "special_tokens_mask": [],
            "attention_mask": [],
            "filename": [],
        }

        full_tokenized_inputs = {
            "input_ids": [],
            "special_tokens_mask": [],
            "attention_mask": [],
            "filename": [],
        }

        for example in range(len(examples["text"])):
            text = examples["text"][example]
            filename = examples["filename"][example]

            tokenized_inputs = self.tokenizer(
                text,
                pad_to_multiple_of=self.max_input_length
                if not self.join_sentences
                else None,
                padding="longest" if not self.join_sentences else "do_not_pad",
                max_length=self.max_input_length
                if not self.join_sentences
                else None,
                truncation=False,
                return_special_tokens_mask=True,
                return_offsets_mapping=False,
            )

            if self.join_sentences:
                full_tokenized_inputs["input_ids"].extend(
                    tokenized_inputs["input_ids"]
                )
                full_tokenized_inputs["special_tokens_mask"].extend(
                    tokenized_inputs["special_tokens_mask"]
                )
                full_tokenized_inputs["attention_mask"].extend(
                    tokenized_inputs["attention_mask"]
                )
                full_tokenized_inputs["filename"].extend(
                    [filename] * len(tokenized_inputs["input_ids"])
                )
            else:
                # Split into multiple examples if the input is too long
                for i in range(
                    0,
                    len(tokenized_inputs["input_ids"]),
                    self.max_input_length,
                ):
                    # Check if the final example would contain only special tokens and if so, don't include it
                    if (
                        sum(
                            tokenized_inputs["special_tokens_mask"][
                                i : i + self.max_input_length
                            ]
                        )
                        == self.max_input_length
                    ):
                        break
                    batch["input_ids"].append(
                        tokenized_inputs["input_ids"][i : i + self.max_input_length]  # type: ignore
                    )
                    batch["special_tokens_mask"].append(
                        tokenized_inputs["special_tokens_mask"][i : i + self.max_input_length]  # type: ignore
                    )
                    batch["attention_mask"].append(
                        tokenized_inputs["attention_mask"][i : i + self.max_input_length]  # type: ignore
                    )
                    batch["filename"].append(filename)

        if self.join_sentences:
            # NOTE: We drop the last batch if it's not full. This is just to 
            # ensure every example is the same length which makes things easier.
            truncated_length = (
                len(full_tokenized_inputs["input_ids"])
                // self.max_input_length
            ) * self.max_input_length

            for i in range(0, truncated_length, self.max_input_length):
                batch["input_ids"].append(
                    full_tokenized_inputs["input_ids"][i : i + self.max_input_length]  # type: ignore
                )
                batch["special_tokens_mask"].append(
                    full_tokenized_inputs["special_tokens_mask"][i : i + self.max_input_length]  # type: ignore
                )
                batch["attention_mask"].append(
                    full_tokenized_inputs["attention_mask"][i : i + self.max_input_length]  # type: ignore
                )
                batch["filename"].append(full_tokenized_inputs["filename"][i])

        if self.callback_functions:
            for callback_function in self.callback_functions:
                examples[callback_function] = getattr(self, callback_function)(
                    examples
                )

        return batch
