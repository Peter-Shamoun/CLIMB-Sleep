#!/usr/bin/env python3
"""Stage a run's trainer checkpoints for the BabyLM AoA evaluation.

The pipeline's AoA_word module loads one model per "step" and, with the
scripts/eval/aoa_local.patch applied, accepts a JSON list of local
checkpoint directories. This script builds that list for one run:

    python scripts/eval/aoa_prepare.py <run_dir> <staging_dir> [--tokens-per-word 1.409]

For every <run_dir>/checkpoint-<step>/ it creates <staging_dir>/<step>/ with
config.json + model.safetensors (+ generation_config.json) from the checkpoint
and the tokenizer files from <run_dir>/lm_model/, then writes
<staging_dir>/steps.json:

    [{"name": "step-<step>", "path": "<staging_dir>/<step>", "word_count": <int>}, ...]

word_count = step * batch * seq_len / tokens_per_word = tokens processed
(replay included; for a sleep run this is throughput, not unique data). The
strict 100M corpus packs to 1,100,608 x 128 tokens = 1.409 tokens per word,
so checkpoint 42,992 ~ 125M words and 343,936 ~ 1.0B words.
"""
import argparse
import json
import os
import re
import shutil
import sys

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
MODEL_FILES = ("config.json", "model.safetensors", "generation_config.json")


def checkpoints(run_dir):
    out = []
    for d in os.listdir(run_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", d)
        if m and os.path.isfile(os.path.join(run_dir, d, "model.safetensors")):
            out.append((int(m.group(1)), os.path.join(run_dir, d)))
    return sorted(out)


def stage(run_dir, staging_dir, tokens_per_word, batch, seq_len, only_final=False):
    ckpts = checkpoints(run_dir)
    if not ckpts:
        sys.exit(f"no checkpoint-<step>/model.safetensors under {run_dir}")
    # save_steps = max_steps // 8 produces checkpoint-343936 and the final
    # save checkpoint-343940 four steps apart; keep the final one only.
    dedup = []
    for step, path in ckpts:
        if dedup and step - dedup[-1][0] < 100:
            dedup[-1] = (step, path)
        else:
            dedup.append((step, path))
    if only_final:
        dedup = dedup[-1:]
    entries = []
    for step, src in dedup:
        dst = os.path.join(staging_dir, str(step))
        os.makedirs(dst, exist_ok=True)
        for f in MODEL_FILES:
            if os.path.exists(os.path.join(src, f)):
                shutil.copy(os.path.join(src, f), dst)
        for f in TOKENIZER_FILES:
            p = os.path.join(run_dir, "lm_model", f)
            if os.path.exists(p):
                shutil.copy(p, dst)
        entries.append({
            "name": f"step-{step}",
            "path": os.path.abspath(dst),
            "word_count": int(round(step * batch * seq_len / tokens_per_word)),
            "step": step,
        })
    with open(os.path.join(staging_dir, "steps.json"), "w") as f:
        json.dump(entries, f, indent=1)
    return entries


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("staging_dir")
    ap.add_argument("--tokens-per-word", type=float, default=1_100_608 * 128 / 1e8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--only-final", action="store_true")
    a = ap.parse_args(argv)
    entries = stage(a.run_dir, a.staging_dir, a.tokens_per_word, a.batch, a.seq_len, a.only_final)
    for e in entries:
        print(f"{e['name']:<14} words={e['word_count']:>13,}  {e['path']}")
    print(f"wrote {os.path.join(a.staging_dir, 'steps.json')} ({len(entries)} checkpoints)")


if __name__ == "__main__":
    main()
