## Q1：

tabel 1
|            | switchboard-small |               | switchboard |               |
|------------|------------------:|---------------|------------:|---------------|
|            | log-prob          | cross-entropy | log-prob    | cross-entropy |
| **Sample1** | -8282.07          | 7.85052       | -6819.01    | 6.46370       |
| **Sample2** | -5008.97          | 8.30622       | -4192.79    | 6.95278       |
| **Sample3** | -5085.45          | 8.29012       | -4195.70    | 6.83969       |

---

tabel 2
| perplexity | switchboard-small | switchboard |
|------------|------------------:|------------:|
| **Sample1** | 230.8             | 88.3        |
| **Sample2** | 316.5             | 123.9       |
| **Sample3** | 313.0             | 114.5       |


When trained on the larger switchboard corpus, the log₂-probabilities
become less negative and the perplexities are much lower. This is
because the larger dataset provides more n-gram evidence, reducing
sparsity and allowing the model to assign higher probabilities to test
sequences. As a result, the model's predictions are closer to the true
distribution, leading to lower cross-entropy and perplexity.

## Q2:

## Q3:
(a) The result is: 
247 files were more probably from gen.model (91.48%)
23 files were more probably from spam.model (8.52%)
Expected error rate on dev files: 0.2522 (270 files)
Average log-loss on dev files: 6.5242 bits per doc (270 files)
Actual 0/1 error rate on dev files: 0.2556 (270 files)

(b) The result is: 
115 files were more probably from en1k.model (48.12%)
124 files were more probably from sp1k.model (51.88%)
Expected error rate on dev files: 0.1017 (239 files)
Average log-loss on dev files: 1.1601 bits per doc (239 files)
Actual 0/1 error rate on dev files: 0.0962 (239 files)

(c) Minimum prior P(gen) to classify ALL dev as spam = 0.0000.
The result Minimum prior P(gen) = 0.0000 means there exists at least one dev document whose likelihood under the gen model is vastly higher than under spam (i.e., a very large delta = logp(d|gen)-logp(d|spam)). To force even that document to be labeled spam, you’d have to set P(gen) to an essentially zero prior, which rounds to 0.0000 at display precision. In practical terms, under any reasonable prior, the classifier will not label all dev files as spam.

(d)&(e) Results:
best lambda on dev/gen   -> 0.005  (9.046160 bits/token)
best lambda on dev/spam  -> 0.005 (9.095720 bits/token)
best lambda on combined  -> 0.005 (9.068397 bits/token)
Sweeping lambda in {5, 0.5, 0.05, 0.005, 0.0005}, the dev cross-entropy is minimized at lambda=0.005 (combined 9.068 bits/token). Larger lambda over-smooths toward uniform, while very small lambda under-smooths and overfits, so a mid-range lambda performs best.

(f)