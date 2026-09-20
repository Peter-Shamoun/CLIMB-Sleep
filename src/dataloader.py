""" Custom Dataloading comptaible with Curriculum Learning """

import logging

# typing imports
from typing import Dict, List, Optional, Any, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data._utils.pin_memory import pin_memory as _torch_pin_memory
from torch.utils.data.dataloader import _BaseDataLoaderIter, _DatasetKind
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe
from transformers import PreTrainedTokenizerFast

# from src.objective_curriculum import ObjectiveCurriculum, StackedCollator
from src.data_curriculum.contextualize_collate import context_augmented_collate
from transformers import DataCollatorForLanguageModeling
from src.utils.data import base_collate_fn
# from src.vocabulary_curriculum.vocabulary_map import BaseVocabularyMap

logger = logging.getLogger(__name__)

class SleepDataLoader(DataLoader):
    def __init__(
        self,
        tokenizer,
        config,
        ignore_columns: Optional[List[str]] = None,
        num_workers: int = 0,
        **kwargs,
    ) -> None:

        self.ignore_columns = ignore_columns
        self.tokenizer = tokenizer
        self.cfg = config
        if num_workers != 0:
            # NOTE: No rush on this, the default Trainer uses 0 workers anyway and runs
            # very fast.
            logger.warning(
                "Multi-process dataloading is not supported yet - using 0 workers."
            )

        super().__init__(num_workers=0, **kwargs)

    def __iter__(self):
        return _SleepSingleProcessDataLoaderIter(self)
    
class _SleepSingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader: SleepDataLoader):
        super().__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        self.loader = loader
        self.config = loader.cfg

        if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            raise NotImplementedError(
                "IterDataPipe and MapDataPipe are not supported yet"
            )
            
        mlm = self.config.task.task == "mlm"

        sleep_cfg = getattr(self.config, "sleep_mechanism", None)
        max_seq_length = getattr(sleep_cfg, "max_seq_length", 128) if sleep_cfg else 128
        self._collate_fn = SleepCollatorForLanguageModeling(
            sampler=loader.sampler,
            tokenizer=loader.tokenizer,
            mlm=mlm,
            mlm_probability=self.config.task.optional_kwargs['mask_probability'] if mlm else None,
            max_seq_length=max_seq_length,
        )
        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind,
            self._dataset,
            self._auto_collation,
            self._collate_fn,
            self._drop_last,
        )

    def _next_index(self):
        idx = next(self._sampler_iter)
        return idx

    def _next_data(self):
        """
        Returns next data from this iterator.
        """
        
        index = self._next_index()  # may raise StopIteration

        data: Dict[str, Tensor] = self._dataset_fetcher.fetch(
            index
        )  # may raise StopIteration
        # add indices to data for sleep mechanism tracking
        if isinstance(index, list):
            data["indices"] = torch.tensor(index)
        else:
            data["indices"] = torch.tensor([index])
            
        if self._pin_memory:
            data = _torch_pin_memory(data, self._pin_memory_device)  # type: ignore[arg-type]

        # remove ignored columns
        # print(f"Ignore columns: {self.loader.ignore_columns}")
        if self.loader.ignore_columns is not None:
            for ignore_column in self.loader.ignore_columns:
                data.pop(ignore_column, None)

        return data
    
class SleepCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, sampler, *args, max_seq_length: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = sampler
        # Chunk length used when the replay buffer is repacked during SLEEP;
        # threaded from SleepMechanismParams.max_seq_length by the dataloader.
        self.max_seq_length = max_seq_length
        
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]], *args, **kwargs) -> Dict[str, Any]:
        if self.sampler.phase == "SLEEP":
            examples = self.context_augment(examples)
        for ex in examples:
            ids = ex["input_ids"]
            if isinstance(ids, torch.Tensor):
                max_id = ids.max().item()
                min_id = ids.min().item()
            else:
                max_id = max(ids)
                min_id = min(ids)

            assert min_id >= 0, f"Negative token id: {min_id}"
        result = super().torch_call(examples, *args, **kwargs)
        return result
    
    def context_augment(
            self,
            examples: List[List[int]],
            max_seq_length: Optional[int] = None,
        ) -> List[Dict[str, List[int]]]:
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        # print("examples to contextualize:", examples)
        
        pad_token_id = self.tokenizer.pad_token_id
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        # extract all tokens from all samples and concatenate
        all_sentences = []

        for batch in examples:
            if 'input_ids' in batch:
                sample = batch['input_ids']
            else:
                raise ValueError("No input ids in batch")
            if isinstance(sample, torch.Tensor):
                tokens = sample.tolist()
            else:
                tokens = sample
            
            # remove special tokens + padding, want only content tokens
            tokens = [t for t in tokens if t not in [cls_token_id, pad_token_id]]
            all_sentences.extend(tokens)

        # pack sentences into chunks, by max_seq_length
        chunks = []

        current_chunk = [cls_token_id]

        for token in all_sentences:
            assert isinstance(token, int), f"Non-integer token ID found: {token}, {type(token)}"
            if len(current_chunk) == 1 and token == sep_token_id:
                continue
            current_chunk.append(token)
            if len(current_chunk) == max_seq_length:
                chunks.append({"input_ids": current_chunk})
                current_chunk = [cls_token_id]
        
        # finalize last chunk. A chunk holding only the leading <s> (exact fill
        # of the previous chunk, or no content at all) carries no target and is
        # dropped rather than emitted as <s> + pads.
        if len(current_chunk) > 1:
            if len(current_chunk) < max_seq_length:
                padding_len = max_seq_length - len(current_chunk)
                current_chunk.extend([pad_token_id] * padding_len)
            chunks.append({"input_ids": current_chunk})

        # if no valid chunks created, return a single padded chunk so the batch
        # keeps its shape (all labels are ignored by the collator).
        if len(chunks) == 0:
            chunks = [{"input_ids": [cls_token_id] + [pad_token_id] * (max_seq_length - 1)}]
        
        # convert to tensors
        # input_ids_tensor = torch.tensor(chunks, dtype=torch.long)

        return chunks