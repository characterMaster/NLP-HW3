"""
speechrec.py : Q9(a) speech recognition rescoring with Bayes rule.
Choose among 9 candidates in each utterance file using:
    argmax_w [ log2 P(u|w) + log2 P_LM(w) ].
Report the WER of the chosen candidate (the file's 1st column),
and finally the micro-averaged overall WER.

Usage:
    python code/speechrec.py <lm.model> <dev_file> <dev_file> ...

Output (match assignment format):
    0.125   easy025
    0.037   easy034
    0.057   OVERALL
"""

import argparse
from pathlib import Path
import math
import sys
from typing import List

from probs import LanguageModel, Wordtype, OOV, BOS, EOS  # use your existing utilities

# LM scoring of a single candidate sentence
def lm_log2prob_sentence(lm: LanguageModel, tokens: List[str]) -> float:
    """
    Compute log2 P_LM(w_1..w_T) with trigram LM like the rest of the HW:
      start with BOS,BOS; step through tokens; then append EOS.
    The input tokens may contain literal "<s>" and "</s>" -- we ignore them.
    OOV tokens are mapped to the model's OOV symbol.
    """
    # drop literal boundary tokens if present in the candidate line
    toks: List[Wordtype] = [t for t in tokens if t not in ("<s>", "</s>")]

    # map to vocab or OOV
    mapped: List[Wordtype] = [t if t in lm.vocab else OOV for t in toks]

    # trigram walk from BOS,BOS to EOS
    x, y = BOS, BOS
    log2p = 0.0
    for z in mapped + [EOS]:
        # lm.log_prob() returns natural log; convert to log2
        log2p += lm.log_prob(x, y, z) / math.log(2.0)
        x, y = y, z
    return log2p

# ---------- parsing helpers ----------
def parse_ref_length(line: str) -> int:
    """
    First line: just read the initial integer (reference length) and ignore the rest.
    (Per assignment warning, the rest of this line may not be token-count faithful.)
    """
    first = line.strip().split(maxsplit=1)[0]
    return int(first)

def parse_candidate_line(line: str):
    """
    Candidate lines: columns are
        WER  AM_log2P  LEN  <s> ... </s>
    We must not use WER or LEN for choosing; we only use AM_log2P and text.
    Returns: (wer_float, am_log2p_float, tokens_list[str])
    """
    parts = line.rstrip("\n").split()
    if len(parts) < 4:
        raise ValueError(f"Malformed candidate line: {line}")
    wer = float(parts[0])
    am_log2p = float(parts[1])
    # length = int(parts[2])  # not needed for selection
    tokens = parts[3:]       # the rest is <s> ... </s>
    return wer, am_log2p, tokens

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lm_model", type=Path, help="trained LM model (e.g., swsmall_backoff_add0.1.model)")
    ap.add_argument("utt_files", type=Path, nargs="+", help="utterance files (e.g., data/speech/dev/easy/easy025)")
    args = ap.parse_args()

    lm = LanguageModel.load(args.lm_model, device="cpu")

    total_ref_words = 0.0
    total_err_words = 0.0

    for utt in args.utt_files:
        lines = utt.read_text(encoding="utf-8").splitlines()
        if len(lines) < 10:
            print(f"WARNING: {utt} has fewer than 10 lines; skipping.", file=sys.stderr)
            continue

        # first line: reference length only (do NOT use it for selecting)
        ref_len = parse_ref_length(lines[0])

        # consider 9 candidates (lines 2..10)
        best_sum = -float("inf")
        best_wer = None

        for i in range(1, min(10, len(lines))):
            wer, am_log2p, tokens = parse_candidate_line(lines[i])
            # LM log2 probability of the candidate
            lm_log2p = lm_log2prob_sentence(lm, tokens)
            s = am_log2p + lm_log2p
            if s > best_sum:
                best_sum = s
                best_wer = wer

        # accumulate micro-average WER
        if best_wer is None:
            continue
        total_ref_words += ref_len
        total_err_words += best_wer * ref_len

        # per-file output: WER (3 decimals) + basename (e.g., "easy025")
        print(f"{best_wer:0.3f}\t{utt.name}")

    # final overall WER (micro average)
    overall = (total_err_words / total_ref_words) if total_ref_words > 0 else float("nan")
    print(f"{overall:0.3f}\tOVERALL")

if __name__ == "__main__":
    main()
