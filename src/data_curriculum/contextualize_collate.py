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
    all_tokens = []

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
        tokens = [t for t in tokens if t not in [cls_token_id, sep_token_id, pad_token_id]]

        if len(tokens) > 0:
            all_tokens.extend(tokens)
            all_tokens.append(sep_token_id)

    # rechunk concatenated tokens at max_seq_length --> truncation at diverse positions
    chunks = []

    # keep 1 position for [CLS] at start
    chunk_capacity = max_seq_length - 1

    for i in range(0, len(all_tokens), chunk_capacity):
        # split at every chunk_capacity
        chunk_tokens = all_tokens[i: i + chunk_capacity]

        # add [CLS] to start
        chunk_w_CLS = [cls_token_id] + chunk_tokens

        # pad to max_seq_length if needed
        if len(chunk_w_CLS) < max_seq_length:
            padding_len = max_seq_length - len(chunk_w_CLS)
            chunk_w_CLS.extend([pad_token_id] * padding_len)
        
        chunks.append(chunk_w_CLS)

    # if no valid chunks created:
    if len(chunks) == 0:
        # return single padded chunk
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