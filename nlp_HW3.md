Q1：

### 表 1
|            | switchboard-small |               | switchboard |               |
|------------|------------------:|---------------|------------:|---------------|
|            | log-prob          | cross-entropy | log-prob    | cross-entropy |
| **Sample1** | -8282.07          | 7.85052       | -6819.01    | 6.46370       |
| **Sample2** | -5008.97          | 8.30622       | -4192.79    | 6.95278       |
| **Sample3** | -5085.45          | 8.29012       | -4195.70    | 6.83969       |

---

### 表 2
| perplexity | switchboard-small | switchboard |
|------------|------------------:|------------:|
| **Sample1** | 227.6             | 88.9        |
| **Sample2** | 315.9             | 122.9       |
| **Sample3** | 312.7             | 114.5       |


When trained on the larger switchboard corpus, the log₂-probabilities
become less negative and the perplexities are much lower. This is
because the larger dataset provides more n-gram evidence, reducing
sparsity and allowing the model to assign higher probabilities to test
sequences. As a result, the model's predictions are closer to the true
distribution, leading to lower cross-entropy and perplexity.

Q2:
