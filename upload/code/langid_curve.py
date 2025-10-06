# coding: utf-8
# curve_langid.py — Language ID (english_spanish) Q3(g)-style: performance vs. length buckets
# Usage:
#   python ./code/langid_curve.py <english_model> <spanish_model> --prior 0.7
# Outputs:
#   scan_out/langid_length_curve.csv
#   scan_out/langid_length_curve_error.png
#   scan_out/langid_length_curve_bits.png
import argparse, subprocess, re, glob, csv, math, sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
OUTDIR = ROOT / "scan_out"; OUTDIR.mkdir(exist_ok=True)

# Dev roots (already bucketed by length)
DEV_EN = ROOT / "data/english_spanish/dev/english"
DEV_SP = ROOT / "data/english_spanish/dev/spanish"

# tools
TEXTCAT  = [sys.executable, str(CODE / "textcat.py")]
FILEPROB = [sys.executable, str(CODE / "fileprob.py")]

# filename patterns: en.10.00(.txt)? / sp.10.00(.txt)? ; also parse parent directory length-10
RE_ENSP  = re.compile(r"^(en|sp)\.(\d+)\.(\d+)(?:\.txt)?$", re.I)
RE_LEN_DIR = re.compile(r"^length-(\d+)$", re.I)

# robust fileprob parsing (several flavors)
RE_PERLINE   = re.compile(r"logprob=\s*(-?\d+(?:\.\d+)?)\s+tokens=\s*(\d+)", re.I)
RE_TOTAL_1   = re.compile(r"Total[^-\d]*(-?\d+(?:\.\d+)?)[^\d]+(?:tokens|num\s*tokens)[^\d]*[:=]?\s*(\d+)", re.I)
RE_TOTAL_2   = re.compile(r"(?:sum_)?logprob[^-\d]*[:=]?\s*(-?\d+(?:\.\d+)?).*(?:tokens|num\s*tokens)[^\d]*[:=]?\s*(\d+)", re.I)
RE_TOTAL_3   = re.compile(r"(-?\d+(?:\.\d+)?)\s+(?:TOTAL|total)\b.*?\b(\d+)\s*(?:tokens?)", re.I)
RE_NUM_PATH  = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s+(.+)$")
RE_XENT_LINE = re.compile(r"Overall\s+cross-entropy:\s*([0-9]+(?:\.[0-9]+)?)\s*bits\s*per\s*token", re.I)

def run_cmd(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(
            f"Command failed ({out.returncode}): {' '.join(map(str, cmd))}\n"
            f"STDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr}"
        )
    return out.stdout.splitlines()

def collect_files_by_length(dev_root: Path) -> dict[int, list[Path]]:
    """Return {length: [files]} by reading length-* subdirs; fallback to filename if needed."""
    byL = defaultdict(list)
    for p in dev_root.rglob("*"):
        if not p.is_file(): 
            continue
        L = None
        # 1) try parent dir like length-10
        mdir = RE_LEN_DIR.match(p.parent.name)
        if mdir:
            L = int(mdir.group(1))
        else:
            # 2) fallback: filename en.10.00(.txt)?
            m = RE_ENSP.match(p.name)
            if m:
                L = int(m.group(2))
        if L is not None:
            byL[L].append(p)
    return byL

def textcat_predict(en_model: Path, sp_model: Path, prior: float, files):
    """Return predicted model basename for each file path."""
    lines = run_cmd([*TEXTCAT, str(en_model), str(sp_model), str(prior), *map(str, files)])
    pred = {}
    for ln in lines:
        parts = ln.strip().split(None, 1)
        if len(parts)==2:
            pred[Path(parts[1])] = Path(parts[0]).name
    return pred

def count_tokens_word(files):
    """Fallback token counter: words per line + EOS (for plain text)."""
    total = 0
    for fp in files:
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s: 
                    continue
                total += len(s.split()) + 1
    return total

def fileprob_totals(model: Path, files):
    """Return (sum_logprob, tokens_sum) robustly across fileprob styles."""
    if not files: 
        return 0.0, 0
    lines = run_cmd([*FILEPROB, str(model), *map(str, files)])

    # A) sum per-line 'logprob=... tokens=...'
    lp_sum = 0.0; tok_sum = 0; matched = False
    for ln in lines:
        m = RE_PERLINE.search(ln)
        if m:
            matched = True
            lp_sum += float(m.group(1))
            tok_sum += int(m.group(2))
    if matched and tok_sum > 0:
        return lp_sum, tok_sum

    # B) footer Total ... tokens ...
    for ln in reversed(lines):
        for pat in (RE_TOTAL_1, RE_TOTAL_2, RE_TOTAL_3):
            m = pat.search(ln)
            if m:
                return float(m.group(1)), int(m.group(2))

    # C) per-file '<float> <path>' only -> count tokens ourselves
    lp_sum = 0.0; matched = False
    for ln in lines:
        m = RE_NUM_PATH.match(ln)
        if m:
            matched = True
            lp_sum += float(m.group(1))
    if matched:
        tok_sum = count_tokens_word(files)
        return lp_sum, tok_sum

    # D) only overall cross-entropy -> count tokens, back out logprob
    for ln in lines:
        m = RE_XENT_LINE.search(ln)
        if m:
            xent = float(m.group(1))
            tok_sum = count_tokens_word(files)
            return -xent * tok_sum, tok_sum

    preview = "\n".join(lines[:30])
    raise RuntimeError(f"Could not parse totals from fileprob output.\nPreview:\n{preview}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("english_model", type=Path)
    ap.add_argument("spanish_model", type=Path)
    ap.add_argument("--prior", type=float, default=0.7, help="P(english)")
    args = ap.parse_args()

    en_model = args.english_model.resolve()
    sp_model = args.spanish_model.resolve()

    # 1) bucketed dev files
    en_byL = collect_files_by_length(DEV_EN)
    sp_byL = collect_files_by_length(DEV_SP)
    lengths = sorted(set(en_byL.keys()) | set(sp_byL.keys()))
    if not lengths:
        raise FileNotFoundError("No dev files found under data/english_spanish/dev/{english,spanish}/length-*")

    # 2) predictions (run once per language list)
    en_all = [f for L in lengths for f in en_byL.get(L, [])]
    sp_all = [f for L in lengths for f in sp_byL.get(L, [])]
    pred_en = textcat_predict(en_model, sp_model, args.prior, en_all)
    pred_sp = textcat_predict(en_model, sp_model, args.prior, sp_all)
    en_name, sp_name = en_model.name, sp_model.name

    # 3) per-length metrics
    rows = []
    for L in lengths:
        e_files = en_byL.get(L, [])
        s_files = sp_byL.get(L, [])
        total = len(e_files) + len(s_files)
        if total == 0:
            continue

        wrong = sum(1 for p in e_files if pred_en.get(p,"")==sp_name) + \
                sum(1 for p in s_files if pred_sp.get(p,"")==en_name)
        err_pct = 100.0 * wrong / total

        # combined bits/token (token-weighted)
        lp_e, tok_e = fileprob_totals(en_model, e_files)
        lp_s, tok_s = fileprob_totals(sp_model, s_files)
        tok_sum = tok_e + tok_s
        bits = (-(lp_e + lp_s) / tok_sum) if tok_sum > 0 else float("nan")

        rows.append((L, err_pct, wrong, total, bits, tok_e, tok_s))

    # 4) CSV
    csv_path = OUTDIR / "langid_length_curve.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["length_bucket","err_percent","errors","total","combined_bits","tokens_en","tokens_sp"])
        for r in rows:
            w.writerow(r)
    print(f"[save] CSV -> {csv_path}")

    # 5) Bar charts (one bar per length bucket)
    x = [r[0] for r in rows]
    err_vals = [r[1] for r in rows]
    bits_vals= [r[4] for r in rows]
    labels = [f"{L}" for L in x]

    # Error
    plt.figure()
    plt.bar(x, err_vals, width=0.8*min(max(x)-min(x), 20) if len(x)>1 else 5)
    plt.xlabel("Document length (bucket from directory name)")
    plt.ylabel("Dev 0/1 error (%)")
    plt.title("Language ID: Error vs. length (λ* fixed)")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.xticks(x, labels, rotation=0)
    plt.tight_layout()
    err_png = OUTDIR / "langid_length_curve_error.png"
    plt.savefig(err_png, dpi=150)

    # Bits/token
    plt.figure()
    plt.bar(x, bits_vals, width=0.8*min(max(x)-min(x), 20) if len(x)>1 else 5)
    plt.xlabel("Document length (bucket from directory name)")
    plt.ylabel("Combined cross-entropy (bits/token)")
    plt.title("Language ID: Cross-entropy vs. length (λ* fixed)")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.xticks(x, labels, rotation=0)
    plt.tight_layout()
    bits_png = OUTDIR / "langid_length_curve_bits.png"
    plt.savefig(bits_png, dpi=150)

    print(f"[save] plots -> {err_png} , {bits_png}")

if __name__ == "__main__":
    main()
