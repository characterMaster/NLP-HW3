# coding: utf-8
# curve.py — Q3(f): performance vs file length on dev (gen_spam)
# Usage:
#   python ./code/curve.py <gen_model> <spam_model> --prior 0.7
# Outputs:
#   scan_out/length_curve.csv
#   scan_out/length_curve_error.png
#   scan_out/length_curve_bits.png

import argparse, subprocess, re, glob, csv
from collections import defaultdict, Counter
from pathlib import Path
from turtle import width

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
OUTDIR = ROOT / "scan_out"
OUTDIR.mkdir(exist_ok=True)

TEXTCAT = [Path().resolve()]  # placeholder; replaced by sys.executable below
FILEPROB = [Path().resolve()] # placeholder

import sys
TEXTCAT = [sys.executable, str(CODE / "textcat.py")]
FILEPROB = [sys.executable, str(CODE / "fileprob.py")]

DEV_GEN_GLOB  = str(ROOT / "data/gen_spam/dev/gen/*")
DEV_SPAM_GLOB = str(ROOT / "data/gen_spam/dev/spam/*")

NAME_RE = re.compile(r"^(gen|spam)\.(\d+)\.(\d+)\.txt$", re.I)
RE_PERLINE   = re.compile(r"logprob=\s*(-?\d+(?:\.\d+)?)\s+tokens=\s*(\d+)", re.I)
RE_TOTAL_1   = re.compile(r"Total[^-\d]*(-?\d+(?:\.\d+)?)[^\d]+(?:tokens|num\s*tokens)[^\d]*[:=]?\s*(\d+)", re.I)
RE_TOTAL_2   = re.compile(r"(?:sum_)?logprob[^-\d]*[:=]?\s*(-?\d+(?:\.\d+)?).*(?:tokens|num\s*tokens)[^\d]*[:=]?\s*(\d+)", re.I)
RE_TOTAL_3   = re.compile(r"(-?\d+(?:\.\d+)?)\s+(?:TOTAL|total)\b.*?\b(\d+)\s*(?:tokens?)", re.I)
RE_XENT_LINE = re.compile(r"Overall\s+cross-entropy:\s*([0-9]+(?:\.[0-9]+)?)\s*bits\s*per\s*token", re.I)

def run_cmd(cmd: list[str]) -> list[str]:
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(
            f"Command failed ({out.returncode}): {' '.join(map(str, cmd))}\n"
            f"STDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr}"
        )
    return out.stdout.splitlines()

def expand_files(pattern: str) -> list[Path]:
    return [Path(p) for p in glob.glob(pattern, recursive=True) if Path(p).is_file()]

def get_len(path: Path) -> int | None:
    m = NAME_RE.match(path.name)
    return int(m.group(2)) if m else None

def textcat_predict(gen_model: Path, spam_model: Path, prior: float, files: list[Path]) -> dict[Path, str]:
    """Return predicted model basename for each file."""
    lines = run_cmd([*TEXTCAT, str(gen_model), str(spam_model), str(prior), *map(str, files)])
    pred = {}
    for ln in lines:
        # format: "<model_path> <file_path>"
        parts = ln.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pred_model_path, file_path = parts
        pred[Path(file_path)] = Path(pred_model_path).name  # just basename for match
    return pred

def parse_totals(lines: list[str]) -> tuple[float, int] | None:
    lp_sum = 0.0; tok_sum = 0; matched = False
    for ln in lines:
        m = RE_PERLINE.search(ln)
        if m:
            matched = True
            lp_sum += float(m.group(1))
            tok_sum += int(m.group(2))
    if matched and tok_sum > 0:
        return lp_sum, tok_sum
    for ln in reversed(lines):
        for pat in (RE_TOTAL_1, RE_TOTAL_2, RE_TOTAL_3):
            m = pat.search(ln)
            if m:
                return float(m.group(1)), int(m.group(2))
    # fallback: overall cross-entropy
    for ln in lines:
        m = RE_XENT_LINE.search(ln)
        if m:
            xent = float(m.group(1))
            return -(xent), 1  # caller will multiply by token count properly（我们不会用这条，保留兜底）
    return None

def fileprob_totals(model_path: Path, files: list[Path]) -> tuple[float, int]:
    """Return (sum_logprob, tokens_sum) for a file list."""
    if not files:
        return 0.0, 0
    lines = run_cmd([*FILEPROB, str(model_path), *map(str, files)])
    totals = parse_totals(lines)
    if totals:
        return totals
    raise RuntimeError("Could not parse totals from fileprob output.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_model", type=Path)
    ap.add_argument("spam_model", type=Path)
    ap.add_argument("--prior", type=float, default=0.7)
    ap.add_argument("--bin-width", type=int, default=1000,
                help="histogram bin width over document length (default: 1000)")
    args = ap.parse_args()

    gen_model = args.gen_model.resolve()
    spam_model = args.spam_model.resolve()

    # 1) Collect dev files and group by length
    gen_files = expand_files(DEV_GEN_GLOB)
    spam_files = expand_files(DEV_SPAM_GLOB)
    if not gen_files or not spam_files:
        raise FileNotFoundError("No dev files found under data/gen_spam/dev/{gen,spam}")

    files_by_len = defaultdict(lambda: {"gen": [], "spam": []})
    for p in gen_files:
        L = get_len(p)
        if L is not None:
            files_by_len[L]["gen"].append(p)
    for p in spam_files:
        L = get_len(p)
        if L is not None:
            files_by_len[L]["spam"].append(p)

    # 2) use textcat predictions
    pred_gen  = textcat_predict(gen_model, spam_model, args.prior, gen_files)
    pred_spam = textcat_predict(gen_model, spam_model, args.prior, spam_files)

    gen_name  = gen_model.name
    spam_name = spam_model.name

    # 3)  0/1 error rate & combined bits/token
    rows = []
    lengths = sorted(files_by_len.keys())
    for L in lengths:
        g_list = files_by_len[L]["gen"]
        s_list = files_by_len[L]["spam"]
        if not g_list and not s_list:
            continue

        # 0/1 error rate（true gen→pred spam；true spam→pred gen）
        wrong = 0
        total = len(g_list) + len(s_list)
        for p in g_list:
            if pred_gen.get(p, "") == spam_name:
                wrong += 1
        for p in s_list:
            if pred_spam.get(p, "") == gen_name:
                wrong += 1
        err_pct = 100.0 * wrong / total if total > 0 else 0.0

        # combined bits/token
        lp_g, tok_g = fileprob_totals(gen_model, g_list)
        lp_s, tok_s = fileprob_totals(spam_model, s_list)
        tok_sum = tok_g + tok_s
        bits = (-(lp_g + lp_s) / tok_sum) if tok_sum > 0 else float("nan")

        rows.append((L, err_pct, wrong, total, bits, tok_g, tok_s))

    # 4) CSV
    csv_path = OUTDIR / "length_curve.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["length", "err_percent", "errors", "total", "combined_bits", "tokens_gen", "tokens_spam"])
        for r in rows:
            w.writerow(r)

    print(f"[save] CSV -> {csv_path}")


    # 5) histogram plots (binned by length)
    bin_w = args.bin_width
    # rows: (L, err_pct, wrong, total, bits, tok_g, tok_s)
    if not rows:
        print("No rows to plot."); return
    L_list = [r[0] for r in rows]
    max_len  = max(L_list)
    max_edge = ((max_len // bin_w) + 1) * bin_w 
    edges    = list(range(0, max_edge + 1, bin_w))
    K = len(edges) - 1
    # create bins [b, b+bin_w)
    bins = [f"{edges[i]}-{edges[i+1]-1}" for i in range(K)]
    # aggregator
    import math
    bin_err_wrong   = [0] * K
    bin_err_total   = [0] * K
    bin_tok_sum     = [0] * K
    bin_bits_weight = [0.0] * K  # = sum( bits_L * tok_L )

    def bin_index(L):
        # find the bin [bins[i], bins[i+1]) that L belongs to
        i = L // bin_w
        if i >= K:
            i = K-1
        return int(i)

    for L, err_pct, wrong, total, bits, tok_g, tok_s in rows:
        i = bin_index(L)
        bin_err_wrong[i]   += int(wrong)
        bin_err_total[i]   += int(total)
        tokL = int(tok_g) + int(tok_s)
        bin_tok_sum[i]     += tokL
        if not math.isnan(bits) and tokL > 0:
            bin_bits_weight[i] += float(bits) * tokL

    # calculate metrics for each bin
    bin_err_pct_all = [(100.0 * bin_err_wrong[i] / bin_err_total[i]) if bin_err_total[i] > 0 else float("nan")
                   for i in range(K)]
    bin_bits_all    = [(bin_bits_weight[i] / bin_tok_sum[i]) if bin_tok_sum[i] > 0 else float("nan")
                   for i in range(K)]

    # remove empty bins to avoid clutter
    keep = [i for i in range(K) if bin_err_total[i] > 0]
    bin_labels = [bins[i]  for i in keep]
    bin_err_pct = [bin_err_pct_all[i] for i in keep]
    bin_bits    = [bin_bits_all[i]    for i in keep]

    # plot bar chart: error rate
    import matplotlib.pyplot as plt
    xpos = list(range(len(keep)))
    width = 0.9 
    plt.figure()
    plt.bar(xpos, bin_err_pct, width=width, align="center")
    plt.xlabel("Document length (from filename)")
    plt.ylabel("Dev 0/1 error (%)")
    plt.title("Error vs. length (λ* fixed, binned)")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.xticks(xpos, bin_labels, rotation=45, ha="right")
    plt.tight_layout()
    err_png = OUTDIR / "length_curve_error.png"
    plt.savefig(err_png, dpi=150)

    plt.figure()    
    plt.bar(xpos, bin_bits, width=width, align="center")
    plt.xlabel("Document length (from filename)")
    plt.ylabel("Combined cross-entropy (bits/token)")
    plt.title("Cross-entropy vs. length (λ* fixed, binned)")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.xticks(xpos, bin_labels, rotation=45, ha="right")
    plt.tight_layout()
    bits_png = OUTDIR / "length_curve_bits.png"
    plt.savefig(bits_png, dpi=150)

    print(f"[save] plots -> {err_png} , {bits_png}")


if __name__ == "__main__":
    main()
