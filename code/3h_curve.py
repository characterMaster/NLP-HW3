# coding: utf-8
# learning_curve_sizes.py — Q3(h): error vs training size for gen_spam with your directory layout
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

# Paths that match your tree exactly
TRAIN_ROOT   = ROOT / "data/gen_spam/train"
TRAIN_SCALES = [
    ("1x",  TRAIN_ROOT/"gen",         TRAIN_ROOT/"spam"),
    ("2x",  TRAIN_ROOT/"gen-times2",  TRAIN_ROOT/"spam-times2"),
    ("4x",  TRAIN_ROOT/"gen-times4",  TRAIN_ROOT/"spam-times4"),
    ("8x",  TRAIN_ROOT/"gen-times8",  TRAIN_ROOT/"spam-times8"),
]
DEV_GEN_DIR  = ROOT / "data/gen_spam/dev/gen"
DEV_SPAM_DIR = ROOT / "data/gen_spam/dev/spam"

# Tools
PY         = sys.executable
TRAIN_LM   = [PY, str(CODE/"train_lm.py")]
TEXTCAT    = [PY, str(CODE/"textcat.py")]
BUILD_VOC  = [PY, str(CODE/"build_vocab.py")]

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str,cmd))}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    return r.stdout

def ensure_vocab(vocab_path: Path, threshold: int = 3):
    if vocab_path.exists():
        return
    # build from BOTH classes to share vocab
    print(f"[vocab] building at {vocab_path} (threshold={threshold}) ...")
    run([*BUILD_VOC, str(TRAIN_ROOT), "--output", str(vocab_path), "--threshold", str(threshold)])
    print("[vocab] done.")

def expand_files(dirpath: Path):
    return [p for p in dirpath.iterdir() if p.is_file()]

def overall_error(gen_model: Path, spam_model: Path, prior: float) -> tuple[float,int,int]:
    """Run textcat on dev/gen and dev/spam, count misclassified files."""
    gen_files  = expand_files(DEV_GEN_DIR)
    spam_files = expand_files(DEV_SPAM_DIR)
    # call textcat with explicit file lists (no wildcards)
    out_gen = run([*TEXTCAT, str(gen_model), str(spam_model), str(prior), *map(str, gen_files)])
    out_spa = run([*TEXTCAT, str(gen_model), str(spam_model), str(prior), *map(str, spam_files)])

    gen_prefix  = str(gen_model)
    spam_prefix = str(spam_model)
    wrong_gen = sum(1 for ln in out_gen.splitlines() if ln.startswith(spam_prefix + " "))
    wrong_spa = sum(1 for ln in out_spa.splitlines() if ln.startswith(gen_prefix  + " "))
    total = len(gen_files) + len(spam_files)
    errors = wrong_gen + wrong_spa
    err_pct = 100.0 * errors / total if total else float("nan")
    return err_pct, errors, total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda", dest="lam", type=float, default=0.005, help="add-lambda value (λ*)")
    ap.add_argument("--prior", type=float, default=0.7, help="P(gen) prior used by textcat")
    ap.add_argument("--vocab", type=Path, default=ROOT/"vocab-genspam.txt", help="shared vocab file")
    args = ap.parse_args()

    ensure_vocab(args.vocab)

    rows = []
    for tag, gen_dir, spam_dir in TRAIN_SCALES:
        if not gen_dir.exists() or not spam_dir.exists():
            print(f"[skip] {tag}: {gen_dir} / {spam_dir} not found")
            continue

        print(f"\n=== Train size {tag} ===")
        gen_model  = OUT / f"gen_{tag}.model"
        spam_model = OUT / f"spam_{tag}.model"

        print(" train gen ...")
        run([*TRAIN_LM, str(args.vocab), "add_lambda", str(gen_dir),
             "--lambda", str(args.lam), "--output", str(gen_model)])
        print(" train spam ...")
        run([*TRAIN_LM, str(args.vocab), "add_lambda", str(spam_dir),
             "--lambda", str(args.lam), "--output", str(spam_model)])

        print(" evaluate dev error ...")
        err_pct, errors, total = overall_error(gen_model, spam_model, args.prior)
        print(f"  error = {err_pct:.2f}%  ({errors}/{total})")
        rows.append((tag, err_pct, errors, total))

    # Save CSV
    csv_path = OUT / "learning_curve_sizes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_size","err_percent","errors","total"])
        w.writerows(rows)
    print(f"\n[save] CSV -> {csv_path}")

    # Plot
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Training size (1x, 2x, 4x, 8x)")
    plt.ylabel("Dev 0/1 error (%)")
    plt.title("Learning curve (add-λ*, shared vocab, prior=0.7)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    png = OUT / "learning_curve_sizes.png"
    plt.savefig(png, dpi=150)
    print(f"[save] plot -> {png}")

if __name__ == "__main__":
    main()
