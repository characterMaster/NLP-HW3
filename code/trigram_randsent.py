#!/usr/bin/env python3
# coding: utf-8
"""
trigram_randsent.py — Q6: sample text from a trained trigram LM (minimal CLI).

Usage:
  python ./code/trigram_randsent.py path/to/model.model 5 --max_length 50 --seed 42
"""

import argparse
import math
import random
from pathlib import Path
import torch

def extract_vocab_tokens(lm) -> list[str]:
    """Return the token list from lm.vocab (tries common layouts)."""
    v = getattr(lm, "vocab", None)
    if v is None:
        raise RuntimeError("Model has no `vocab` attribute.")
    for attr in ("id2word", "itos", "words", "tokens"):
        if hasattr(v, attr):
            arr = getattr(v, attr)
            if isinstance(arr, (list, tuple)) and arr and isinstance(arr[0], str):
                return list(arr)
    if hasattr(v, "word2id") and isinstance(v.word2id, dict):
        return [w for w, _ in sorted(v.word2id.items(), key=lambda kv: kv[1])]
    if isinstance(v, dict):
        try:
            return [w for w, _ in sorted(v.items(), key=lambda kv: kv[1])]
        except Exception:
            pass
    return list(v)

def renorm(ps):
    z = sum(ps)
    if z <= 0 or not math.isfinite(z):
        n = len(ps)
        return [1.0 / n] * n
    return [p / z for p in ps]

def sample_from_dist(tokens, probs):
    # r = random.random()
    # s = 0.0
    # for t, p in zip(tokens, probs):
    #     s += p
    #     if r <= s:
    #         return t
    # return tokens[-1]
    probs_tensor = torch.tensor(probs, dtype=torch.float32)
    probs_tensor = probs_tensor / probs_tensor.sum()  # normalize to 1
    idx = torch.multinomial(probs_tensor, num_samples=1).item()
    return tokens[idx]

def next_token_distribution(lm, vocab_list, prev2, prev1):
    """Return list of P(w | prev2, prev1) over vocab_list (robust to different APIs)."""
    out = []
    for w in vocab_list:
        p = None
        if hasattr(lm, "prob"):
            try:
                p = lm.prob((prev2, prev1, w))
            except TypeError:
                p = lm.prob(prev2, prev1, w)
        if p is None and hasattr(lm, "log_prob"):
            lp = lm.log_prob((prev2, prev1, w))
            p = 0.0 if not math.isfinite(lp) else 2.0 ** lp
        if p is None and hasattr(lm, "cond_prob"):
            p = lm.cond_prob(prev2, prev1, w)
        if p is None or not math.isfinite(p) or p < 0:
            p = 0.0
        out.append(p)
    return out

def unigram_probs(lm, vocab_list):
    """Try to get unigram-ish probs; fallback to uniform."""
    cnts = getattr(getattr(lm, "vocab", None), "counts", None)
    if isinstance(cnts, dict):
        arr = [max(cnts.get(w, 0), 0) for w in vocab_list]
        if sum(arr) > 0:
            return renorm(arr)
    return [1.0 / len(vocab_list)] * len(vocab_list)

ENDERS_CANDIDATES = {".", "!", "?"}  # common sentence enders

def sample_one(lm, vocab_list, max_length=50):
    start_token = "BOS"
    end_token = "EOS"
    # enders = list(ENDERS_CANDIDATES & set(vocab_list))
    # uni = unigram_probs(lm, vocab_list)

    # Cold start: first two tokens from unigram/uniform
    seq = [start_token, start_token]

    # Trigram sampling from the 3rd token
    while len(seq) < max_length+2:
        w2, w1 = seq[-2], seq[-1]
        probs = renorm(next_token_distribution(lm, vocab_list, w2, w1))
        nxt = sample_from_dist(vocab_list, probs)
        if nxt == end_token:
            break
        seq.append(nxt)
    seq = seq[2:]
    if seq[-1] not in ENDERS_CANDIDATES:
        return " ".join(seq)+' ...'
    return " ".join(seq)


def main():
    ap = argparse.ArgumentParser(description="Sample from a trigram LM without BOS/EOS.")
    ap.add_argument("model", type=Path, help="path to trained model (.model)")
    ap.add_argument("num", type=int, default=5, help="number of samples (default: 5)")
    ap.add_argument("--max_length", type=int, default=50, help="max tokens per sample (default: 50)")
    ap.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    lm = torch.load(args.model, map_location="cpu")
    vocab_list = extract_vocab_tokens(lm)
    if "UNK" not in vocab_list:
        vocab_list.append("UNK")
    print(f"INFO: model={args.model.name}  num={args.num}  max_length={args.max_length}")
    for i in range(1, args.num + 1):
        s = sample_one(lm, vocab_list, max_length=args.max_length)
        print(f"{i:02d}: {s}")

if __name__ == "__main__":
    main()