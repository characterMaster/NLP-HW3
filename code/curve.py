
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "scan_out" / "learning_curve.csv"
OUTDIR = ROOT 
OUTDIR.mkdir(exist_ok=True)

sizes = []
err_pct = []
bits = []

with CSV.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            n = int(row.get("size_per_class", "").strip())
        except Exception:
            continue
        sizes.append(n)
        try:
            e = float(row.get("err_percent", row.get("error_percent", "nan")))
        except Exception:
            e = float("nan")
        err_pct.append(e)

        # bits/token
        try:
            b = float(row.get("combined_bits", row.get("bits_token_combined", "nan")))
        except Exception:
            b = float("nan")
        bits.append(b)
# Sort by size
order = sorted(range(len(sizes)), key=lambda i: sizes[i])
sizes = [sizes[i] for i in order]
err_pct = [err_pct[i] for i in order]
bits = [bits[i] for i in order]

# curve.py: plot learning curves from CSV data
plt.figure()
plt.plot(sizes, err_pct, marker="o")
plt.xlabel("Training files per class (N)")
plt.ylabel("Dev 0/1 error (%)")
plt.title("Learning curve: dev error vs. training size")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
err_png = OUTDIR / "learning_curve_error.png"
plt.savefig(err_png, dpi=150)

# Cross-entropy curve
plt.figure()
plt.plot(sizes, bits, marker="o")
plt.xlabel("Training files per class (N)")
plt.ylabel("Combined cross-entropy (bits/token)")
plt.title("Learning curve: cross-entropy vs. training size")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
bits_png = OUTDIR / "learning_curve_bits.png"
plt.savefig(bits_png, dpi=150)

print(f"Saved:\n - {err_png}\n - {bits_png}")
