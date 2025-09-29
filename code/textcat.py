#!/usr/bin/env python3
"""
Text categorization via Bayes' Theorem using two smoothed trigram LMs.

Usage:
  python ./code/textcat.py GEN.model SPAM.model 0.7 path/to/files/*
Output format (per spec): spam.model foo.txt

  X files were more probably from gen.model (P%)
  Y files were more probably from spam.model (Q%)
"""
import argparse
import logging
import math
import sys
from pathlib import Path
import torch
import glob

from probs import Wordtype, LanguageModel, read_trigrams  # starter code APIs

log = logging.getLogger(Path(__file__).stem)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model1", type=Path, help="path to model #1 (e.g., gen.model)")
    p.add_argument("model2", type=Path, help="path to model #2 (e.g., spam.model)")
    p.add_argument("prior", type=float,
                   help="prior probability for the FIRST model (e.g., 0.7 for gen)")
    p.add_argument("test_files", type=Path, nargs="+", help="files to classify")
    p.add_argument("--device", type=str, default="cpu",
                   choices=["cpu", "cuda", "mps"],
                   help="device for PyTorch tensors")
    # verbosity
    p.set_defaults(logging_level=logging.INFO)
    g = p.add_mutually_exclusive_group()
    g.add_argument("-v", "--verbose", dest="logging_level",
                   action="store_const", const=logging.DEBUG)
    g.add_argument("-q", "--quiet", dest="logging_level",
                   action="store_const", const=logging.WARNING)
    return p.parse_args()

def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """Natural-log total probability of a file (one sentence per line)."""
    log_prob = 0.0
    x: Wordtype; y: Wordtype; z: Wordtype
    for (x, y, z) in read_trigrams(file, lm.vocab):
        lp = lm.log_prob(x, y, z)    # ln p(z | xy)
        log_prob += lp
        if log_prob == -math.inf:
            break
    return log_prob

def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # device checks (mirrors fileprob.py style)
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                log.critical("MPS not available: PyTorch not built with MPS.")
            else:
                log.critical("MPS not available on this macOS/device.")
            sys.exit(1)
    torch.set_default_device(args.device)

    # prior sanity
    if not (0.0 < args.prior < 1.0):
        raise ValueError("Prior must be in (0,1), e.g., 0.7")

    # load models
    lm_gen  = LanguageModel.load(args.model1, device=args.device)
    lm_spam = LanguageModel.load(args.model2, device=args.device)

    # vocab equality sanity check (required by spec)
    try:
        same_vocab = (lm_gen.vocab == lm_spam.vocab)
    except Exception:
        # fall back to string repr if __eq__ isn't defined
        same_vocab = (repr(lm_gen.vocab) == repr(lm_spam.vocab))
    if not same_vocab:
        log.critical("The two models do not share the SAME vocabulary. "
                     "Rebuild with a shared vocab (≥3 threshold over union of corpora, plus OOV/EOS).")
        sys.exit(1)

    log_prior_gen  = math.log(args.prior)
    log_prior_spam = math.log(1.0 - args.prior)

    # gather test files (expand globs, recurse into dirs)
    test_paths: list[Path] = []
    for pat in args.test_files:
        # glob first (PowerShell/Windows may not expand * on its own)
        matches = glob.glob(str(pat))
        if not matches:
            matches = [str(pat)]
        for m in matches:
            p = Path(m)
            if p.is_dir():
                # collect all regular files under this dir (recursive)
                test_paths.extend([q for q in p.rglob("*") if q.is_file()])
            else:
                test_paths.append(p)

    # classify each file
    gen_name  = str(args.model1)
    spam_name = str(args.model2)
    gen_count = 0
    spam_count = 0
    total = 0

    for f in test_paths:
        lp_gen  = file_log_prob(f, lm_gen)  + log_prior_gen
        lp_spam = file_log_prob(f, lm_spam) + log_prior_spam

        if lp_gen >= lp_spam:
            print(f"{gen_name} {f}")
            gen_count += 1
        else:
            print(f"{spam_name} {f}")
            spam_count += 1
        total += 1

    # summary (exact format requested)
    if total == 0:
        print("0 files were more probably from {} (0.00%)".format(gen_name))
        print("0 files were more probably from {} (0.00%)".format(spam_name))
        return

    gen_pct  = 100.0 * gen_count  / total
    spam_pct = 100.0 * spam_count / total
    print(f"{gen_count} files were more probably from {gen_name} ({gen_pct:.2f}%)")
    print(f"{spam_count} files were more probably from {spam_name} ({spam_pct:.2f}%)")

if __name__ == "__main__":
    main()
