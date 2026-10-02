## How Income-Insight works

Income-Insight predicts whether a person's annual income exceeds **\$50K** from
census attributes (UCI Adult, 1994). A deep multi-layer perceptron (MLP) turns
one row of features into a probability, and a threshold turns that probability
into a label.

### 1. From a row to a vector

Five numeric features are standardized (zero mean, unit variance on the training
split) and five categorical features are one-hot encoded. One person becomes a
vector $x \in \mathbb{R}^{d}$ with $d = 84$. **Sex and race are not inputs** by
default; they are kept only to audit the model.

### 2. Forward propagation in matrix form

For a mini-batch $X$ of $B$ rows and hidden width $H$ (default: 2 hidden
layers, $H = 32$, activation $f = \tanh$):

$$
\underbrace{A_1}_{B \times H} = f(\underbrace{X}_{B \times d}\,\underbrace{W_1}_{d \times H} + b_1), \qquad
\underbrace{A_2}_{B \times H} = f(A_1\,\underbrace{W_2}_{H \times H} + b_2), \qquad
\underbrace{z}_{B \times 1} = A_2\,\underbrace{W_3}_{H \times 1} + b_3, \qquad
p = \sigma(z)
$$

Each matrix multiplication mixes every input into every hidden unit; the
non-linearity $f$ between them is what lets stacked layers represent curved
decision boundaries (a stack of purely linear layers collapses to one linear
layer).

### 3. Cost: binary cross-entropy

$$
\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B}\Big[y_i \log p_i + (1-y_i)\log(1-p_i)\Big]
$$

### 4. Backward propagation

The chain rule runs the error back through the layers. Because the sigmoid and
cross-entropy fit together, the output error is simply $\frac{1}{B}(p - y)$;
each layer then passes back $\delta_{\ell} = (\delta_{\ell+1} W_{\ell+1}^{\top}) \odot f'(Z_\ell)$
and gets the gradient $\partial \mathcal{L} / \partial W_\ell = A_{\ell-1}^{\top}\, \delta_\ell$,
which has exactly the shape of $W_\ell$. Adam uses these gradients to update
every weight; one pass over the training split is an **epoch**.

The full step-by-step derivation with the shape of every weight and gradient,
and a worked **XOR** example, are in the engineering report. The XOR example is
runnable: `python -m scripts.xor_matrix_backprop` prints every matrix shape,
checks the gradients against finite differences, and trains a 2-2-1 network
that solves XOR.

### 5. From probability to label

$\hat{y} = 1$ (">50K") when $p \ge t$. The default threshold is $t = 0.5$;
raising it makes the model more conservative (higher precision, lower recall
for >50K). The **Score a Row** and **Score a CSV** tabs let you move $t$.

### 6. Is the probability trustworthy? (calibration)

A calibrated model is right about 70% of the time when it says $p = 0.7$. The
**Model Performance** tab plots predicted vs. observed rates (the reliability
diagram) and reports the Brier score and expected calibration error (ECE).

### 7. Is the model fair? (FPR / FNR)

- **False-positive rate** $= \frac{FP}{FP + TN}$: of people truly earning ≤50K, the share predicted >50K.
- **False-negative rate** $= \frac{FN}{FN + TP}$: of people truly earning >50K, the share the model misses.

The **Bias Audit** tab compares these rates across sex and race groups. Equal
accuracy can hide very unequal error rates.
