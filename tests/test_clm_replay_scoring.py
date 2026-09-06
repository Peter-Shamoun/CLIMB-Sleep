"""CLM path of the loss-criterion replay score and the sleep collator.

Written while diagnosing random > loss under CLM (Sep 2026). These tests
pin down what is NOT wrong: padding is excluded from the per-sample score,
wake chunks carry no padding at all (join_sentences packs to a fixed
length), and the sleep collator repacks the buffer without cross-chunk
targets. The ordering itself is explained by over-replay (see
paper_results.md, "CLM criterion ordering"), not by these code paths.
"""
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_curriculum.sleep_sampler import SleepSampler  # noqa: E402
from src.dataloader import SleepCollatorForLanguageModeling  # noqa: E402
from src.utils.replay_score import per_sample_mean_token_loss  # noqa: E402

CLS, PAD, SEP = 0, 1, 2  # <s>, <pad>, </s> as in the cbt tokenizers


@pytest.fixture(scope="module")
def tokenizer():
    vocab = {"<s>": CLS, "<pad>": PAD, "</s>": SEP, "<unk>": 3}
    for i in range(4, 64):
        vocab[f"w{i}"] = i
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, cls_token="<s>", sep_token="</s>",
        pad_token="<pad>", unk_token="<unk>", bos_token="<s>", eos_token="</s>",
    )


def _collator(tokenizer, phase):
    sampler = SimpleNamespace(phase=phase)
    return SleepCollatorForLanguageModeling(sampler=sampler, tokenizer=tokenizer, mlm=False)


def _clm_shift(labels):
    return labels[:, 1:].contiguous()


def test_wake_clm_batch_has_no_ignored_labels_when_lengths_match(tokenizer):
    col = _collator(tokenizer, "WAKE")
    ex = [{"input_ids": [CLS, 10, 11, SEP, 12, 13, 14, SEP]}, {"input_ids": [CLS, 20, 21, 22, 23, 24, 25, SEP]}]
    batch = col.torch_call(ex)
    assert torch.equal(batch["labels"], batch["input_ids"])
    assert not (batch["labels"] == -100).any()


def test_padding_is_ignored_in_labels_and_in_the_replay_score(tokenizer):
    col = _collator(tokenizer, "WAKE")
    ex = [{"input_ids": [CLS, 10, 11, 12, 13, 14, 15, SEP]}, {"input_ids": [CLS, 20, 21, SEP]}]
    batch = col.torch_call(ex)
    labels = batch["labels"]
    assert labels.shape == (2, 8)
    assert (labels[1, 4:] == -100).all() and (labels[1, :4] != -100).all()
    shifted = _clm_shift(labels)
    assert (shifted[1, 3:] == -100).all()

    # Score must be the mean over real tokens only: build logits whose
    # per-token loss is constant on real positions and huge on pads.
    vocab = len(tokenizer)
    logits = torch.zeros(2, vocab, 7)
    score = per_sample_mean_token_loss(logits, shifted)
    expected = torch.log(torch.tensor(float(vocab)))  # uniform logits
    assert torch.allclose(score, expected.expand(2), atol=1e-5)
    # Sabotage the pad positions: the score must not move.
    logits[1, :, 3:] = torch.randn(vocab, 4) * 50
    assert torch.allclose(per_sample_mean_token_loss(logits, shifted), expected.expand(2), atol=1e-5)


def test_sleep_collator_repacks_buffer_into_cls_led_chunks_without_cross_chunk_targets(tokenizer):
    col = _collator(tokenizer, "SLEEP")
    L = 128
    # Three "wake" chunks of the production shape (join_sentences: exactly 128
    # tokens, <s>/</s> in the middle, no padding).
    body = [t for t in range(4, 64)]
    ex = []
    for k in range(3):
        toks = [CLS] + [body[(k * 7 + i) % len(body)] for i in range(L - 1)]
        toks[40] = SEP
        ex.append({"input_ids": toks})
    batch = col.torch_call(ex)
    ids, labels = batch["input_ids"], batch["labels"]
    # 3 x 127 content tokens repack into 3 full chunks of 128 with a leading
    # <s>. Quirk: because the packer opens a new [<s>] chunk after every full
    # one, an exact fill leaves a trailing all-pad chunk (<s> + 127 pads),
    # which carries no loss (labels -100) but occupies a batch slot.
    assert ids.shape in ((3, L), (4, L))
    assert (ids[:, 0] == CLS).all()
    assert (ids[:, 1:] != CLS).all()
    assert not (ids[:3] == PAD).any() and not (labels[:3] == -100).any()
    if ids.shape[0] == 4:
        assert (ids[3, 1:] == PAD).all() and (labels[3, 1:] == -100).all()
    # Content order is preserved across the concatenation, so the only
    # "boundary" a next-token target crosses is the same sentence boundary
    # the wake packing already contains; chunk k+1 restarts from <s>.
    content_in = [t for e in ex for t in e["input_ids"] if t != CLS]
    content_out = [t for row in ids.tolist() for t in row if t not in (CLS, PAD)]
    assert content_out == content_in


def test_sleep_collator_pads_only_the_last_chunk(tokenizer):
    col = _collator(tokenizer, "SLEEP")
    ex = [{"input_ids": [CLS] + [10] * 127}, {"input_ids": [CLS] + [11] * 60}]
    batch = col.torch_call(ex)
    ids, labels = batch["input_ids"], batch["labels"]
    assert ids.shape == (2, 128)
    assert not (ids[0] == PAD).any()
    n_pad = int((ids[1] == PAD).sum())
    assert n_pad == 128 - 1 - 60
    assert int((labels[1] == -100).sum()) == n_pad
    assert (labels[1][ids[1] == PAD] == -100).all()


class _Dataset:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"input_ids": [CLS, 10 + (i % 50), SEP]}


def test_strict_buffer_grows_with_all_seen_folds_and_keeps_stale_high_loss_samples():
    """Strict top-k ranks every sample seen so far by score x decay.

    Scores are recorded once, when the sample's fold is the wake fold, and
    never refreshed, so early high-loss samples stay in the buffer for
    several consecutive sleeps while the sleep length stays constant.
    """
    n, phases, ratio = 100, 5, 0.1
    s = SleepSampler(_Dataset(n), batch_size=10, replay_ratio=ratio, n_phases=phases,
                     n_augmentations=1, decay_rate=0.05, min_decay_factor=0.2,
                     contextualize_sleep=False, replay_strategy="strict")
    fold0 = list(s.folds[0])
    # Cycle 0: fold 0 seen at high loss (model is untrained).
    s.add_to_candidates(fold0, [10.0] * len(fold0))
    s.switch_phase("SLEEP")
    assert len(s.replay_buffer) == int(len(fold0) * ratio)
    first_buffer = set(s.replay_buffer)
    s.switch_phase("WAKE")
    # Cycle 1: fold 1 seen at much lower loss.
    fold1 = list(s.folds[1])
    s.add_to_candidates(fold1, [1.0] * len(fold1))
    s.switch_phase("SLEEP")
    assert len(s.replay_buffer) == int((len(fold0) + len(fold1)) * ratio)
    # Every cycle-0 pick is re-selected in cycle 1 (10 * 0.95 > 1.0).
    assert first_buffer <= set(s.replay_buffer)
    assert all(i in fold0 for i in s.replay_buffer)
