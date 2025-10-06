#Q9
import sys, subprocess, shlex
from pathlib import Path

# use existing fileprob.py to get log2 P_LM for a sentence
def lm_log2prob(model: Path, sentence: str) -> float:
    tmp = Path(".__tmp_lm_sentence.txt")
    tmp.write_text(sentence, encoding="utf-8")
    out = subprocess.run(
        [sys.executable, str(Path("code")/"fileprob.py"), str(model), str(tmp)],
        capture_output=True, text=True, check=False
    )
    tmp.unlink(missing_ok=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    # select the line containing the temp file name and take the first number
    for line in out.stdout.splitlines():
        if tmp.name in line:
            return float(line.split()[0])
    # fallback: find the first token that can be converted to float
    for line in out.stdout.splitlines():
        tok = line.strip().split()[0]
        try:
            return float(tok)
        except:
            pass
    raise RuntimeError("Could not parse LM log2-prob.")

def main():
    if len(sys.argv) != 3:
        print("usage: python code/choose_best_transcript.py <LM_MODEL> <DEV_FILE>", file=sys.stderr)
        sys.exit(1)
    lm_model = Path(sys.argv[1])
    dev_file = Path(sys.argv[2])

    lines = [l.rstrip("\n") for l in dev_file.read_text(encoding="utf-8").splitlines()]
    assert len(lines) >= 10, "expect 10 lines: 1 ref + 9 candidates"

    ref = lines[0]  
    cand_lines = lines[1:]  # 9 cand_lines

    best = (None, float("-inf"))  # (text, score)
    print(f"# file: {dev_file}")
    for i, row in enumerate(cand_lines, start=1):
        # each line：WER  AM_log2P  LEN  <s> ... </s>
        parts = row.split(maxsplit=3)
        if len(parts) < 4:
            print(f"skip malformed line {i+1}: {row}", file=sys.stderr)
            continue
        wer_str, am_lp_str, length_str, sent = parts
        am_lp = float(am_lp_str)           # second column: AM log2-prob
        lm_lp = lm_log2prob(lm_model, sent)  # use LM score
        # combined score: AM + LM
        score = am_lp + lm_lp
        print(f"cand {i:02d}: AM={am_lp:.2f}  LM={lm_lp:.2f}  SUM={score:.2f}  | {sent}")
        if score > best[1]:
            best = (sent, score)

    print("\n# Best by AM+LM:")
    print(best[0])

if __name__ == "__main__":
    main()
