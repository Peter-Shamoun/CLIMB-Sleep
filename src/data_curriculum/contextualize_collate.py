import torch
from typing import List, Dict, Any

def context_augmented_collate(
    batch: List[Dict[str, Any]], 
    max_seq_length: int = 128,
    pad_token_id: int = 1,
    cls_token_id: int = 0,
    sep_token_id: int = 2
) -> Dict[str, torch.Tensor]:
    
    # extract all tokens from all samples and concatenate
    all_sentences = []

    for sample in batch:
        # get input_ids
        if "input_ids" in sample:
            input_ids = sample["input_ids"]
        else:
            continue
    
        if isinstance(input_ids, torch.Tensor):
            tokens = input_ids.tolist()
        else:
            tokens = input_ids
        
        # remove special tokens + padding, want only content tokens
        tokens = [t for t in tokens if t not in [cls_token_id, pad_token_id]]

        # split by SEP token to get individual sentences
        current_sentence = []
        for token in tokens:
            if token == sep_token_id:
                if len(current_sentence) > 0:
                    all_sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(token)
        
        # add last sentence if it exists
        if len(current_sentence) > 0:
            all_sentences.append(current_sentence)

    # pack sentences into chunks, by max_seq_length
    chunks = []

    # keep 1 position for [CLS] at start
    chunk_capacity = max_seq_length - 1

    current_chunk = []
    current_length = 0

    for sentence in all_sentences:
        sentence_with_sep = sentence + [sep_token_id]
        sentence_length = len(sentence_with_sep)

        # if adding this sentence would exceed capacity, complete current chunk
        if current_length + sentence_length > chunk_capacity:
            if len(current_chunk) > 0:
                # finalize current chunk
                chunk_with_cls = [cls_token_id] + current_chunk

                # pad to max_seq_length:
                if len(chunk_with_cls) < max_seq_length:
                    padding_len = max_seq_length - len(chunk_with_cls)
                    chunk_with_cls.extend([pad_token_id] * padding_len)
                chunks.append(chunk_with_cls)

                # start new chunk
                current_chunk = []
                current_length = 0
        
        # add sentence to current chunk
        current_chunk.extend(sentence_with_sep)
        current_length += sentence_length
    
    # finalize last chunk
    if len(current_chunk) > 0:
        chunk_with_cls = [cls_token_id] + current_chunk
        if len(chunk_with_cls) < max_seq_length:
            padding_len = max_seq_length - len(chunk_with_cls)
            chunk_with_cls.extend([pad_token_id] * padding_len)
        chunks.append(chunk_with_cls)
    
    # if no valid chunks created, return single padded chunk
    if len(chunks) == 0:
        chunks = [[cls_token_id] + [pad_token_id] * chunk_capacity]
    
    # convert to tensors
    input_ids_tensor = torch.tensor(chunks, dtype=torch.long)
    # create attention mask (1 for real tok, 0 for pad)
    attention_mask = (input_ids_tensor != pad_token_id).long()
    # create labels for MLM (copy/clone of input_ids)
    labels = input_ids_tensor.clone()

    # create result dict
    result = {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask,
        "input_ids_mlm": input_ids_tensor.clone(),
        "labels_mlm": labels,
    }

    return result