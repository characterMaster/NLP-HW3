# coding: utf-8
# 3i_curve.py — Language ID learning curve (en.{1K..50K} vs sp.{1K..50K})
# Usage:
#   python ./code/3i_curve.py --lambda 0.005 --prior 0.7

import argparse
import subprocess
from pathlib import Path
from glob import glob
import csv
import sys
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
OUT  = ROOT / "scan_out"; OUT.mkdir(exist_ok=True)

# === training sets ===
TRAIN_ROOT   = ROOT / "data/english_spanish/train"
TRAIN_SCALES = [
    ("1K",  TRAIN_ROOT / "en.1K",  TRAIN_ROOT / "sp.1K"),
    ("2K",  TRAIN_ROOT / "en.2K",  TRAIN_ROOT / "sp.2K"),
    ("5K",  TRAIN_ROOT / "en.5K",  TRAIN_ROOT / "sp.5K"),
    ("10K", TRAIN_ROOT / "en.10K", TRAIN_ROOT / "sp.10K"),
    ("20K", TRAIN_ROOT / "en.20K", TRAIN_ROOT / "sp.20K"),
    ("50K", TRAIN_ROOT / "en.50K", TRAIN_ROOT / "sp.50K"),
]

DEV_EN_DIR = ROOT / "data/english_spanish/dev/english"
DEV_SP_DIR = ROOT / "data/english_spanish/dev/spanish"

PY        = sys.executable
TRAIN_LM  = [PY, str(CODE / "train_lm.py")]
TEXTCAT   = [PY, str(CODE / "textcat.py")]
BUILD_VOC = [PY, str(CODE / "build_vocab.py")]

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(map(str, cmd))}\n"
            f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )
    return r.stdout

# ---- helpers ----
def files_in_path(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    if p.is_dir():
        return [q for q in p.iterdir() if q.is_file()]
    raise FileNotFoundError(f"Path not found: {p}")

def files_in_tree(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    return [Path(x) for x in glob(str(p / "**" / "*"), recursive=True) if Path(x).is_file()]

def ensure_vocab_for_size(tag: str, en_path: Path, sp_path: Path, threshold: int = 3) -> Path:
    vocab_path = ROOT / f"vocab-en_sp-{tag}.txt"
    if vocab_path.exists():
        return vocab_path

    en_files = files_in_path(en_path)
    sp_files = files_in_path(sp_path)
    if not en_files or not sp_files:
        raise FileNotFoundError(f"No train files in {en_path} or {sp_path}")

    cmd = [*BUILD_VOC, *map(str, en_files + sp_files),
           "--output", str(vocab_path), "--threshold", str(threshold)]
    print(f"[vocab-{tag}] building from {len(en_files)+len(sp_files)} files -> {vocab_path.name}")
    run(cmd)
    return vocab_path

def overall_error(en_model: Path, sp_model: Path, prior: float):
    en_files = files_in_tree(DEV_EN_DIR)   # dev 有 length-* 子目录
    sp_files = files_in_tree(DEV_SP_DIR)

    out_en = run([*TEXTCAT, str(en_model), str(sp_model), str(prior), *map(str, en_files)])
    out_sp = run([*TEXTCAT, str(en_model), str(sp_model), str(prior), *map(str, sp_files)])

    en_prefix = str(en_model)
    sp_prefix = str(sp_model)
    wrong_en = sum(1 for ln in out_en.splitlines() if ln.startswith(sp_prefix + " "))
    wrong_sp = sum(1 for ln in out_sp.splitlines() if ln.startswith(en_prefix  + " "))
    total  = len(en_files) + len(sp_files)
    errors = wrong_en + wrong_sp
    err_pct = 100.0 * errors / total if total else float("nan")
    return err_pct, errors, total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda", dest="lam", type=float, default=0.005, help="add-lambda (λ*)")
    ap.add_argument("--prior", type=float, default=0.7, help="P(english) prior for textcat")
    args = ap.parse_args()

    rows = []
    for tag, en_path, sp_path in TRAIN_SCALES:
        if not en_path.exists() or not sp_path.exists():
            print(f"[skip] {tag}: {en_path} / {sp_path} not found")
            continue

        print(f"\n=== Train size {tag} ===")
        vocab = ensure_vocab_for_size(tag, en_path, sp_path, threshold=3)

        en_model = OUT / f"en_{tag}.model"
        sp_model = OUT / f"sp_{tag}.model"

        print(" train EN ...")
        run([*TRAIN_LM, str(vocab), "add_lambda", str(en_path),
             "--lambda", str(args.lam), "--output", str(en_model)])

        print(" train SP ...")
        run([*TRAIN_LM, str(vocab), "add_lambda", str(sp_path),
             "--lambda", str(args.lam), "--output", str(sp_model)])

        print(" evaluate dev error ...")
        err_pct, errors, total = overall_error(en_model, sp_model, args.prior)
        print(f"  error = {err_pct:.2f}%  ({errors}/{total})")
        rows.append((tag, err_pct, errors, total))

    # CSV
    csv_path = OUT / "langid_learning_curve.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["train_size","err_percent","errors","total"]); w.writerows(rows)
    print(f"\n[save] CSV -> {csv_path}")

    # Plot
    xs = [r[0] for r in rows]; ys = [r[1] for r in rows]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Training size (1K, 2K, 5K, 10K, 20K, 50K)")
    plt.ylabel("Dev 0/1 error (%)")
    plt.title("Language ID learning curve (add-λ*, prior=0.7)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    png = OUT / "langid_learning_curve.png"
    plt.savefig(png, dpi=150)
    print(f"[save] plot -> {png}")

if __name__ == "__main__":
    main()
