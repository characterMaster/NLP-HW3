# scan_lambda.py  — robust to different fileprob outputs; computes combined dev via own token counts
import sys, re, glob, subprocess
from pathlib import Path

# ========= Paths (absolute) =========
ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"

TRAIN_PY     = [sys.executable, str(CODE / "train_lm.py")]
FILEPROB_PY  = [sys.executable, str(CODE / "fileprob.py")]

# Spam detection dataset (word trigram)
VOCAB          = ROOT / "vocab-genspam.txt"           # build once: build_vocab.py data/gen_spam/train --threshold 3
TRAIN_GEN      = ROOT / "data/gen_spam/train/gen"
TRAIN_SPAM     = ROOT / "data/gen_spam/train/spam"
DEV_GEN_GLOB   = str(ROOT / "data/gen_spam/dev/gen/*")
DEV_SPAM_GLOB  = str(ROOT / "data/gen_spam/dev/spam/*")

OUTDIR = ROOT / "scan_out"
OUTDIR.mkdir(exist_ok=True)

LAMBDAS = [5, 0.5, 0.05, 0.005, 0.0005]

# ======== Parsers for different fileprob styles ========
RE_PERLINE    = re.compile(r"logprob=\s*(-?\d+(?:\.\d+)?)\s+tokens=\s*(\d+)", re.I)
RE_TOTAL_1    = re.compile(r"Total[^-\d]*(-?\d+(?:\.\d+)?)[^\d]+(?:tokens|num\s*tokens)[^\d]*[:=]?\s*(\d+)", re.I)
RE_TOTAL_2    = re.compile(r"(?:sum_)?logprob[^-\d]*[:=]?\s*(-?\d+(?:\.\d+)?).*(?:tokens|num\s*tokens)[^\d]*[:=]?\s*(\d+)", re.I)
RE_TOTAL_3    = re.compile(r"(-?\d+(?:\.\d+)?)\s+(?:TOTAL|total)\b.*?\b(\d+)\s*(?:tokens?)", re.I)
RE_XENT_LINE  = re.compile(r"Overall\s+cross-entropy:\s*([0-9]+(?:\.[0-9]+)?)\s*bits\s*per\s*token", re.I)

def run_cmd(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(
            f"Command failed ({out.returncode}): {' '.join(map(str, cmd))}\n"
            f"STDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr}"
        )
    return out.stdout.splitlines()

def expand_files(*patterns) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            pp = Path(p)
            if pp.is_file():
                files.append(pp)
    return files

def count_tokens_word(files: list[Path]) -> int:
    """Approximate num_tokens(): sum of (len(words)+1 EOS) per nonempty line."""
    total = 0
    for fp in files:
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                total += len(line.split()) + 1  # +1 EOS per line
    return total

def run_fileprob(model_path: Path, files: list[Path]) -> list[str]:
    return run_cmd([*FILEPROB_PY, str(model_path), *map(str, files)])

def parse_totals(lines: list[str]) -> tuple[float, int] | None:
    # Style A: many "logprob=... tokens=..." lines -> sum them.
    lp_sum = 0.0
    tok_sum = 0
    matched_any = False
    for ln in lines:
        m = RE_PERLINE.search(ln)
        if m:
            matched_any = True
            lp_sum += float(m.group(1))
            tok_sum += int(m.group(2))
    if matched_any and tok_sum > 0:
        return lp_sum, tok_sum

    # Footer "Total ..." styles
    for ln in reversed(lines):
        for pat in (RE_TOTAL_1, RE_TOTAL_2, RE_TOTAL_3):
            m = pat.search(ln)
            if m:
                return float(m.group(1)), int(m.group(2))
    return None

def bits_per_token(model_path: Path, *patterns):
    """Return (bits_per_token, sum_logprob, tokens_sum). If sum/tokens unavailable from output,
       compute bits from 'Overall cross-entropy' and tokens by reading files.
    """
    files = expand_files(*patterns)
    if not files:
        raise FileNotFoundError(f"No files matched: {patterns}")
    lines = run_fileprob(model_path, files)

    # Try totals (preferred)
    totals = parse_totals(lines)
    if totals:
        lp_sum, tok_sum = totals
        return -lp_sum / tok_sum, lp_sum, tok_sum

    # Fallback: parse "Overall cross-entropy: X bits per token"
    xent = None
    for ln in lines:
        m = RE_XENT_LINE.search(ln)
        if m:
            xent = float(m.group(1))
            break
    if xent is not None:
        tok_sum = count_tokens_word(files)
        # If you ever need sum_logprob: lp_sum = -xent * tok_sum
        return xent, -xent * tok_sum, tok_sum

    # Nothing matched — show preview to debug
    preview = "\n".join(lines[:40])
    raise RuntimeError(f"Parsed 0 tokens from fileprob output.\nPreview:\n{preview}")

def train_model(vocab: Path, lam: float, train_dir: Path, out_path: Path):
    cmd = [*TRAIN_PY, str(vocab), "add_lambda", str(train_dir),
           "--lambda", str(lam), "--output", str(out_path)]
    run_cmd(cmd)

def main():
    print("lambda scan start...\n")

    sep_csv  = (OUTDIR / "dev_scan_sep.csv").open("w", encoding="utf-8")
    comb_csv = (OUTDIR / "dev_scan_combined.csv").open("w", encoding="utf-8")
    sep_csv.write("lambda,split,bits_per_token,logprob_sum,tokens_sum\n")
    comb_csv.write("lambda,combined_bits_per_token,logprob_sum,tokens_sum\n")

    best_gen  = (None, float("inf"))
    best_spam = (None, float("inf"))
    best_comb = (None, float("inf"))

    for lam in LAMBDAS:
        print(f"[lambda={lam}] train gen / spam...")
        gen_model  = OUTDIR / f"gen_{lam}.model"
        spam_model = OUTDIR / f"spam_{lam}.model"
        train_model(VOCAB, lam, TRAIN_GEN,  gen_model)
        train_model(VOCAB, lam, TRAIN_SPAM, spam_model)

        print(f"[lambda={lam}] evaluate dev/gen ...")
        gen_bits, gen_lp, gen_tok = bits_per_token(gen_model, DEV_GEN_GLOB)
        print(f"  gen/dev bits/token = {gen_bits:.6f}")
        sep_csv.write(f"{lam},gen,{gen_bits:.6f},{gen_lp:.3f},{gen_tok}\n")
        if gen_bits < best_gen[1]:
            best_gen = (lam, gen_bits)

        print(f"[lambda={lam}] evaluate dev/spam ...")
        spam_bits, spam_lp, spam_tok = bits_per_token(spam_model, DEV_SPAM_GLOB)
        print(f"  spam/dev bits/token = {spam_bits:.6f}")
        sep_csv.write(f"{lam},spam,{spam_bits:.6f},{spam_lp:.3f},{spam_tok}\n")
        if spam_bits < best_spam[1]:
            best_spam = (lam, spam_bits)

        # Combined dev = 加权平均（按 token 数）
        print(f"[lambda={lam}] merge(dev/gen+dev/spam) ...")
        lp_sum = gen_lp + spam_lp
        tok_sum = gen_tok + spam_tok
        combined_bits = -lp_sum / tok_sum
        print(f"  combined bits/token = {combined_bits:.6f}\n")
        comb_csv.write(f"{lam},{combined_bits:.6f},{lp_sum:.3f},{tok_sum}\n")
        if combined_bits < best_comb[1]:
            best_comb = (lam, combined_bits)

    sep_csv.close(); comb_csv.close()

    print("=== finished ===")
    print(f"best lambda on dev/gen   -> {best_gen[0]}  ({best_gen[1]:.6f} bits/token)")
    print(f"best lambda on dev/spam  -> {best_spam[0]} ({best_spam[1]:.6f} bits/token)")
    print(f"best lambda on combined  -> {best_comb[0]} ({best_comb[1]:.6f} bits/token)")
    print(f"CSV written to: {OUTDIR/'dev_scan_sep.csv'} and {OUTDIR/'dev_scan_combined.csv'}")

if __name__ == "__main__":
    main()
