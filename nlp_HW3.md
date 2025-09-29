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
## Q4:
(a) For uniform estimate, every word, including 'oov', has a possibility of $\frac{1}{19999}$. However, this could be summed to $\frac{20000}{19999}$ and leaves out 'oov' with 0 probability. Model will then not be able to accept any new word.

Based on the add-$\lambda$ formula, we could have a trigram:
$$\hat{p}(z|xy)=\frac{c(xyz)+\lambda}{c(xy)+\lambda V}$$
Therefore, when $V = 19999$, if $\lambda =0$, the smooth is removed; if $\lambda>0$, the denominater would increase, and it falls into the same that the summation of all the probabilities is larger than 1. The 'oov' has 0 possibility.  

(b) If $\lambda =0$, then it is a maximum-likelihood estimate that model overfits the training corpus, assuming there is no new word. The bias is 0, yet the variance is high since unseen words could be severe noises which make the result unstable.  

(c) No, it doesn't. With smoothing, if $p(xyz)=0$, $\hat{p}(z|xy)$ could be backoffed to $\hat{p}(z|x)$, which might not be 0, and so is $\hat{p}(z'|xy)$. Therefore, they are not neccesarily equal.
Given the smooth formula, we have:
$$\hat{p}(z|xy)=\frac{c(xyz)+\lambda V\hat{p}(z|y)}{c(xy)+\lambda V}=\frac{\lambda V\hat{p}(z|y)}{c(xy)+\lambda V}$$

If $c(xyz)=c(xyz')=1$, we have:
$$\hat{p}(z|xy)=\frac{1+\lambda V\hat{p}(z|y)}{c(xy)+\lambda V}.$$
Since it is unknown whether $\hat{p}(z|y)=\hat{p}(z'|y)$, the answer is still not necessarily equal.

(d) When $\lambda$ increases, the ratio of the backed-off version model, (n-1)-gram, will become more weighted. Since model with less context (n gets smaller) could have larger possibilities for words, which introduces more smoothing, resulting in less variance and higher bias. Model tends to underfit the current corpus but to have a more stable result.  