# Income-Insight — Engineering Report

CST-435 / AIT-204 · Project 2 · Luke Hoyle

**Contents:** 1. Problem statement · 2. Forward and backward propagation in matrix form ·
3. XOR worked example · 4. Algorithm of the solution · 5. Analysis of the findings ·
6. Ethical consideration · 7. References

---

## 1. Problem statement

Income-Insight answers one question about a person: **does their annual income
exceed \$50K?** It answers from ten census attributes (age, education, hours
worked, capital gains and losses, work class, marital status, occupation,
household relationship, and country of origin) using the UCI Adult dataset,
48,842 records from the 1994 US Current Population Survey.

A classifier that scores people has three obligations beyond accuracy, and the
product is built around them:

1. **It must be explainable.** We must be able to show, layer by layer, how an
   input becomes a probability and how training changes the weights (Sections 2
   and 3).
2. **Its probabilities must be trustworthy.** A "70%" should mean right about 70%
   of the time, so we measure calibration, not just accuracy (Section 5).
3. **Its mistakes must be visible.** Equal accuracy can hide unequal error rates
   between groups, so we audit false-positive and false-negative rates by sex and
   race (Sections 5 and 6).

A deep multi-layer perceptron (MLP) fits the problem: the inputs are a fixed
set of mixed numeric and categorical columns, the output is a single
probability, and the decision boundary is non-linear (marriage, occupation,
education, and capital gains interact).

---

## 2. Forward and backward propagation in matrix form

### 2.1 Notation and shapes

We process a **mini-batch** of $B$ people at once, one person per row.

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $B$ | batch size (256 in training) | scalar |
| $d$ | input features after preprocessing (5 standardized numeric + 79 one-hot) | $d = 84$ |
| $H$ | hidden-layer width | $H = 32$ |
| $X$ | input batch | $B \times d$ |
| $Y$ | true labels ($1$ = ">50K") | $B \times 1$ |
| $W_1, b_1$ | layer-1 weights and bias | $d \times H$, $1 \times H$ |
| $W_2, b_2$ | layer-2 weights and bias | $H \times H$, $1 \times H$ |
| $W_3, b_3$ | output weights and bias | $H \times 1$, $1 \times 1$ |
| $\mathbf{1}$ | column of ones (adds a bias to every row) | $B \times 1$ |
| $\odot$ | element-wise product | — |

Biases are row vectors; $\mathbf{1}\,b$ copies $b$ onto every row of the batch.
That is exactly what NumPy and PyTorch "broadcasting" do when we write `Z = X @ W + b`.

The network has $84 \cdot 32 + 32 + 32 \cdot 32 + 32 + 32 + 1 = 3{,}809$ trainable parameters.

### 2.2 Forward propagation

$$
\begin{aligned}
Z_1 &= X W_1 + \mathbf{1} b_1 &&(B \times d)(d \times H) \to B \times H \\
A_1 &= \tanh(Z_1) &&B \times H \\
Z_2 &= A_1 W_2 + \mathbf{1} b_2 &&(B \times H)(H \times H) \to B \times H \\
A_2 &= \tanh(Z_2) &&B \times H \\
Z_3 &= A_2 W_3 + \mathbf{1} b_3 &&(B \times H)(H \times 1) \to B \times 1 \\
P &= \sigma(Z_3) = \frac{1}{1 + e^{-Z_3}} &&B \times 1
\end{aligned}
$$

Each matrix product is legal because the inner dimensions match: the number of
columns of the left matrix (features coming in) equals the number of rows of the
weight matrix. Row $i$ of $P$ is the model's probability that person $i$ earns
more than \$50K.

### 2.3 Cost function: binary cross-entropy

$$
\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \Big[\, y_i \log p_i + (1 - y_i)\log(1 - p_i) \,\Big]
$$

If $y_i = 1$ only the first term is active, and the cost is $-\log p_i$: near zero
when $p_i \approx 1$, and very large when the model is confidently wrong
($p_i \approx 0$). The second term does the same for $y_i = 0$.

### 2.4 Backward propagation

Backpropagation applies the chain rule from the loss back to every weight. We
write $\delta_\ell = \partial \mathcal{L} / \partial Z_\ell$ for the error at the
pre-activation of layer $\ell$; it always has the same shape as $Z_\ell$.

**Step 1: output error.** For one example,

$$
\frac{\partial \mathcal{L}}{\partial p_i} = -\frac{1}{B}\left(\frac{y_i}{p_i} - \frac{1 - y_i}{1 - p_i}\right),
\qquad
\frac{\partial p_i}{\partial z_i} = \sigma(z_i)\big(1 - \sigma(z_i)\big) = p_i(1 - p_i).
$$

Multiplying (chain rule):

$$
\frac{\partial \mathcal{L}}{\partial z_i}
= -\frac{1}{B}\Big(y_i(1 - p_i) - (1 - y_i)\,p_i\Big)
= \frac{1}{B}\,(p_i - y_i).
$$

The $p_i(1-p_i)$ factors cancel, so sigmoid and cross-entropy together give the
simplest possible output error. For the whole batch:

$$
\delta_3 = \frac{1}{B}\,(P - Y) \qquad B \times 1
$$

**Step 2: output-layer gradients.** Element $(i)$ of $Z_3$ is
$\sum_k (A_2)_{ik} (W_3)_k + b_3$, so
$\partial \mathcal{L} / \partial (W_3)_k = \sum_i (\delta_3)_i (A_2)_{ik}$, a sum
over the batch, which is a matrix product with $A_2$ transposed:

$$
\frac{\partial \mathcal{L}}{\partial W_3} = A_2^{\top}\,\delta_3
\qquad (H \times B)(B \times 1) \to H \times 1,
\qquad
\frac{\partial \mathcal{L}}{\partial b_3} = \mathbf{1}^{\top} \delta_3 \qquad 1 \times 1
$$

The bias gradient is the column sum of $\delta_3$, because the same bias was
added to every row.

**Step 3: send the error back through $W_3$.** $A_2$ affects the loss only
through $Z_3$, so

$$
\frac{\partial \mathcal{L}}{\partial A_2} = \delta_3\, W_3^{\top}
\qquad (B \times 1)(1 \times H) \to B \times H
$$

**Step 4: through the tanh.** Since $\frac{d}{dz}\tanh z = 1 - \tanh^2 z$, and the
activation acts element-wise,

$$
\delta_2 = \frac{\partial \mathcal{L}}{\partial A_2} \odot \left(1 - A_2 \odot A_2\right) \qquad B \times H
$$

**Steps 5–7: repeat for layers 2 and 1.**

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial W_2} &= A_1^{\top}\,\delta_2 &&(H \times B)(B \times H) \to H \times H \\
\frac{\partial \mathcal{L}}{\partial b_2} &= \mathbf{1}^{\top}\delta_2 &&1 \times H \\
\delta_1 &= \left(\delta_2\, W_2^{\top}\right) \odot \left(1 - A_1 \odot A_1\right) &&(B \times H)(H \times H) \to B \times H \\
\frac{\partial \mathcal{L}}{\partial W_1} &= X^{\top}\,\delta_1 &&(d \times B)(B \times H) \to d \times H \\
\frac{\partial \mathcal{L}}{\partial b_1} &= \mathbf{1}^{\top}\delta_1 &&1 \times H
\end{aligned}
$$

**The general pattern.** For any layer with input $A_{\ell-1}$:

$$
\frac{\partial \mathcal{L}}{\partial W_\ell} = A_{\ell-1}^{\top}\,\delta_\ell,
\qquad
\frac{\partial \mathcal{L}}{\partial b_\ell} = \mathbf{1}^{\top}\delta_\ell,
\qquad
\delta_{\ell-1} = \left(\delta_\ell\, W_\ell^{\top}\right) \odot f'(Z_{\ell-1}).
$$

**Why every gradient has the shape of its weight.** $\partial \mathcal{L} / \partial W_\ell$
contains one number per entry of $W_\ell$ (how much the loss changes if that one
weight changes), so it must be the same shape. The transposes are what make this
happen: $A_{\ell-1}^{\top}$ is (features in $\times$ $B$) and $\delta_\ell$ is
($B \times$ features out), and multiplying them sums each weight's contribution
over the $B$ people in the batch.

### 2.5 Shape summary

| Forward | Shape | Backward | Shape |
|---------|-------|----------|-------|
| $X$ | $B \times 84$ | — | — |
| $W_1$, $b_1$ | $84 \times 32$, $1 \times 32$ | $\partial \mathcal{L}/\partial W_1 = X^\top \delta_1$, $\mathbf{1}^\top\delta_1$ | $84 \times 32$, $1 \times 32$ |
| $Z_1$, $A_1$ | $B \times 32$ | $\delta_1$ | $B \times 32$ |
| $W_2$, $b_2$ | $32 \times 32$, $1 \times 32$ | $\partial \mathcal{L}/\partial W_2 = A_1^\top \delta_2$, $\mathbf{1}^\top\delta_2$ | $32 \times 32$, $1 \times 32$ |
| $Z_2$, $A_2$ | $B \times 32$ | $\delta_2$ | $B \times 32$ |
| $W_3$, $b_3$ | $32 \times 1$, $1 \times 1$ | $\partial \mathcal{L}/\partial W_3 = A_2^\top \delta_3$, $\mathbf{1}^\top\delta_3$ | $32 \times 1$, $1 \times 1$ |
| $Z_3$, $P$ | $B \times 1$ | $\delta_3 = (P - Y)/B$ | $B \times 1$ |

### 2.6 Updating the weights

Plain gradient descent moves every weight a small step against its gradient:
$W_\ell \leftarrow W_\ell - \eta\, \partial \mathcal{L} / \partial W_\ell$ with
learning rate $\eta$. We use **Adam** (Kingma & Ba, 2015), which keeps running
averages of each gradient ($m$) and of its square ($v$) and divides by
$\sqrt{v}$, so every weight gets its own step size:

$$
m \leftarrow \beta_1 m + (1-\beta_1)\,g, \qquad
v \leftarrow \beta_2 v + (1-\beta_2)\,g^2, \qquad
W \leftarrow W - \eta\,\frac{\hat m}{\sqrt{\hat v} + \epsilon}
$$

where $g$ is the gradient, $\hat m, \hat v$ are bias-corrected, and we use
$\eta = 0.005$ with PyTorch's defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$.
One pass over every mini-batch of the training split is one **epoch**; we train
for 15 epochs.

### 2.7 Where this lives in the code

`api/training.py` implements exactly this loop:

```python
for _ in range(epochs):                        # multiple epochs
    for start in range(0, n, batch_size):      # mini-batches of B rows
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)          # forward propagation + BCE cost
        loss.backward()                        # backpropagation: every dL/dW
        optimizer.step()                       # Adam weight update
```

PyTorch's autograd computes the same gradients derived above. Two implementation
details: `nn.Linear` stores each weight as (out $\times$ in), i.e. $W_\ell^\top$,
and computes $X W_\ell + b$ from it; and `BCEWithLogitsLoss` folds the sigmoid
into the loss, which is the $\delta_3 = (P - Y)/B$ simplification computed in a
numerically stable way.

### 2.8 Matrix multiplication and expressive capacity

Without activation functions, depth adds nothing. Two linear layers collapse
into one:

$$
(X W_1 + \mathbf{1} b_1)\, W_2 + \mathbf{1} b_2 = X\,(W_1 W_2) + \mathbf{1}\,(b_1 W_2 + b_2),
$$

which is a single linear layer with weight $W_1 W_2$. A linear model can only
separate classes with a flat hyperplane. The non-linearity between the matrix
products is what gives depth its power: each matrix multiplication **re-combines**
the features into new directions, and each activation **bends** that space, so
the next layer sees a representation in which the classes are easier to
separate. With enough hidden units, one hidden layer can approximate any
continuous function on a bounded domain (Cybenko, 1989; Hornik, 1991). XOR is
the smallest problem where this is visible, so we work it out completely next.

---

## 3. XOR worked example

`scripts/xor_matrix_backprop.py` implements Section 2 in plain NumPy (no
autograd) for XOR. Run it with `python -m scripts.xor_matrix_backprop`; the
values below are its output (seed 1). `tests/test_xor.py` checks the gradients
and the result automatically.

### 3.1 Why XOR needs a hidden layer

| $x_1$ | $x_2$ | XOR |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

The two positive points sit on one diagonal of the unit square and the two
negative points on the other, so **no single straight line separates them**. A
logistic-regression model (one linear layer + sigmoid) fit to XOR outputs 0.5
for all four points: 50% accuracy, no better than guessing.

### 3.2 Network and shapes

A 2-2-1 network: two inputs, two sigmoid hidden units, one sigmoid output, BCE
cost, the whole truth table as one batch ($n = 4$). It is Section 2 with one
hidden layer and sigmoid hidden units, so the hidden derivative is
$\sigma'(Z_1) = A_1 \odot (1 - A_1)$.

| Forward | Shape | Backward | Shape |
|---------|-------|----------|-------|
| $X$ | $4 \times 2$ | — | — |
| $W_1$, $b_1$ | $2 \times 2$, $1 \times 2$ | $X^\top \delta_1$, $\mathbf{1}^\top \delta_1$ | $2 \times 2$, $1 \times 2$ |
| $Z_1 = XW_1 + \mathbf{1}b_1$, $A_1 = \sigma(Z_1)$ | $4 \times 2$ | $\delta_1 = (\delta_2 W_2^\top) \odot A_1 \odot (1 - A_1)$ | $4 \times 2$ |
| $W_2$, $b_2$ | $2 \times 1$, $1 \times 1$ | $A_1^\top \delta_2$, $\mathbf{1}^\top \delta_2$ | $2 \times 1$, $1 \times 1$ |
| $Z_2 = A_1 W_2 + b_2$, $P = \sigma(Z_2)$ | $4 \times 1$ | $\delta_2 = (P - Y)/4$ | $4 \times 1$ |

### 3.3 One full forward pass (initial weights)

$$
X = \begin{bmatrix} 0 & 0 \\ 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{bmatrix},\quad
W_1 = \begin{bmatrix} 0.3456 & 0.8216 \\ 0.3304 & -1.3032 \end{bmatrix},\quad
b_1 = \begin{bmatrix} 0 & 0 \end{bmatrix},\quad
W_2 = \begin{bmatrix} 0.9054 \\ 0.4464 \end{bmatrix},\quad
b_2 = 0
$$

$$
Z_1 = X W_1 = \begin{bmatrix} 0 & 0 \\ 0.3304 & -1.3032 \\ 0.3456 & 0.8216 \\ 0.6760 & -0.4815 \end{bmatrix},
\qquad
A_1 = \sigma(Z_1) = \begin{bmatrix} 0.5000 & 0.5000 \\ 0.5819 & 0.2136 \\ 0.5855 & 0.6946 \\ 0.6629 & 0.3819 \end{bmatrix}
$$

Check one entry: row 4 of $X$ is $[1, 1]$, so $(Z_1)_{4,1} = 0.3456 + 0.3304 = 0.6760$.

$$
Z_2 = A_1 W_2 = \begin{bmatrix} 0.6759 \\ 0.6222 \\ 0.8402 \\ 0.7706 \end{bmatrix},
\qquad
P = \sigma(Z_2) = \begin{bmatrix} 0.6628 \\ 0.6507 \\ 0.6985 \\ 0.6836 \end{bmatrix},
\qquad
\mathcal{L} = 0.7566
$$

The untrained network predicts about 0.67 for every input: it has not learned
anything yet.

### 3.4 One full backward pass

$$
\delta_2 = \frac{P - Y}{4} = \frac{1}{4}\begin{bmatrix} 0.6628 - 0 \\ 0.6507 - 1 \\ 0.6985 - 1 \\ 0.6836 - 0 \end{bmatrix}
= \begin{bmatrix} 0.1657 \\ -0.0873 \\ -0.0754 \\ 0.1709 \end{bmatrix}
$$

$$
\frac{\partial \mathcal{L}}{\partial W_2} = A_1^\top \delta_2 = \begin{bmatrix} 0.1012 \\ 0.0771 \end{bmatrix},
\qquad
\frac{\partial \mathcal{L}}{\partial b_2} = \textstyle\sum_i (\delta_2)_i = 0.1739
$$

$$
\delta_2 W_2^\top = \begin{bmatrix} 0.1500 & 0.0740 \\ -0.0791 & -0.0390 \\ -0.0682 & -0.0336 \\ 0.1547 & 0.0763 \end{bmatrix},
\qquad
\delta_1 = (\delta_2 W_2^\top) \odot A_1 \odot (1 - A_1) = \begin{bmatrix} 0.0375 & 0.0185 \\ -0.0192 & -0.0065 \\ -0.0166 & -0.0071 \\ 0.0346 & 0.0180 \end{bmatrix}
$$

Check one entry: $(\delta_1)_{1,1} = 0.1500 \times 0.5 \times 0.5 = 0.0375$.

$$
\frac{\partial \mathcal{L}}{\partial W_1} = X^\top \delta_1 = \begin{bmatrix} 0.0180 & 0.0109 \\ 0.0153 & 0.0115 \end{bmatrix},
\qquad
\frac{\partial \mathcal{L}}{\partial b_1} = \mathbf{1}^\top \delta_1 = \begin{bmatrix} 0.0363 & 0.0228 \end{bmatrix}
$$

Every gradient has the shape of its weight ($2 \times 1$, $1 \times 1$,
$2 \times 2$, $1 \times 2$), as Section 2.4 requires.

### 3.5 Gradient check

To confirm the derivation, the script also computes every gradient numerically
with central differences, nudging each weight by $\epsilon = 10^{-6}$:
$\frac{\mathcal{L}(w + \epsilon) - \mathcal{L}(w - \epsilon)}{2\epsilon}$.

| Gradient | Max \|analytic − numerical\| |
|----------|------------------------------|
| $\partial \mathcal{L}/\partial W_1$ | $3.1 \times 10^{-11}$ |
| $\partial \mathcal{L}/\partial b_1$ | $2.5 \times 10^{-11}$ |
| $\partial \mathcal{L}/\partial W_2$ | $4.2 \times 10^{-11}$ |
| $\partial \mathcal{L}/\partial b_2$ | $1.7 \times 10^{-11}$ |

Agreement to about $10^{-11}$ means the matrix-form backward equations are correct.

### 3.6 Training

Gradient descent with $\eta = 2.0$: one update already lowers the loss from
0.7566 to 0.6957.

| Epoch | 1 | 101 | 501 | 1,001 | 2,001 | 5,000 |
|-------|---|-----|-----|-------|-------|-------|
| BCE loss | 0.7566 | 0.6187 | 0.0161 | 0.0060 | 0.0026 | 0.0010 |

After 5,000 epochs every input is classified correctly:

| $x_1$ | $x_2$ | Target | $P(y = 1)$ | Predicted |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 | 0.0011 | 0 |
| 0 | 1 | 1 | 0.9991 | 1 |
| 1 | 0 | 1 | 0.9991 | 1 |
| 1 | 1 | 0 | 0.0010 | 0 |

### 3.7 What the hidden layer learned

$$
W_1 = \begin{bmatrix} -7.558 & 7.662 \\ 7.277 & -7.929 \end{bmatrix},\quad
b_1 = \begin{bmatrix} -3.935 & -4.130 \end{bmatrix},\quad
W_2 = \begin{bmatrix} 14.856 \\ 14.768 \end{bmatrix},\quad
b_2 = -7.299
$$

$$
A_1 = \begin{bmatrix} 0.019 & 0.016 \\ 0.966 & 0.000 \\ 0.000 & 0.972 \\ 0.015 & 0.012 \end{bmatrix}
\quad
\begin{matrix} \leftarrow (0,0) \\ \leftarrow (0,1) \\ \leftarrow (1,0) \\ \leftarrow (1,1) \end{matrix}
$$

- **Hidden unit 1** $\approx \sigma(-7.6\,x_1 + 7.3\,x_2 - 3.9)$ fires only for
  $(0, 1)$: it computes "$x_2$ AND NOT $x_1$".
- **Hidden unit 2** $\approx \sigma(7.7\,x_1 - 7.9\,x_2 - 4.1)$ fires only for
  $(1, 0)$: "$x_1$ AND NOT $x_2$".
- **The output** $\approx \sigma(14.9\,a_1 + 14.8\,a_2 - 7.3)$ fires when either
  hidden unit is on: an OR.

So XOR = ($x_2$ AND NOT $x_1$) OR ($x_1$ AND NOT $x_2$). In hidden space the
four points move to $(0,0)$, $(1,0)$, $(0,1)$, and $(0,0)$: both negatives land
on the same corner, and the line $a_1 + a_2 = 0.5$ now separates the classes.
**The first matrix multiplication plus the non-linearity re-mapped the input into
a space where the problem became linearly separable.** That is the expressive
capacity Section 2.8 described, and it is the same mechanism our 84-input
network uses on the census data with 32 hidden units per layer.

---

## 4. Algorithm of the solution

1. **Load the data.** `db/load_adult.py` downloads `adult.data` and `adult.test`
   from UCI, combines them (48,842 rows), maps the missing-value marker `?` to
   `Unknown`, normalizes the test file's `>50K.` labels, and drops `fnlwgt` (a
   census sampling weight) and `education` (a duplicate of `education_num`).
2. **Split once, permanently.** A stratified 70/15/15 split with seed 42 assigns
   every row to train (34,189), validation (7,326), or test (7,327), keeping the
   23.9% ">50K" rate in each. The split is stored in Supabase, so every run uses
   the same rows.
3. **Preprocess.** An sklearn `ColumnTransformer` standardizes the 5 numeric
   columns and one-hot encodes the 5 categorical ones, fit on the training rows
   only (no information leaks from validation or test): 84 features.
   Sex and race are excluded from the inputs.
4. **Train.** The deep MLP of Section 2 (2 × 32 tanh) is trained with Adam on BCE:
   lr 0.005, batch 256, 15 epochs, seed 0, on a 15,000-row sample of the training
   split (to fit Render's free-tier time limit). The seed fixes the weight
   initialization, batch order, and sample, so runs are reproducible.
5. **Check for divergence.** If the loss goes NaN/Inf or ends higher than it
   started, the run is stored with `status = 'diverged'` and NULL metrics, and is
   never served.
6. **Evaluate.** Accuracy on train/validation/test; test precision, recall, F1,
   and ROC-AUC; per-class scores; the confusion matrix; calibration (10-bin
   reliability diagram, Brier score, ECE); permutation importance on validation.
7. **Persist.** The API writes the run and all its metrics to `runs`, the fitted
   pipeline to `run_artifacts`, and one prediction per test row to `predictions`.
8. **Audit in SQL.** The `bias_audit` view joins those predictions to the true
   labels and protected attributes in `adult_income` and computes FPR and FNR per
   group; the `activation_comparison` view groups runs by their controls.
9. **Serve.** At startup the API loads the newest successful pipeline. Score a
   Row and Score a CSV apply it with a user-chosen threshold $t$:
   $\hat{y} = 1$ if $p \ge t$.

---

## 5. Analysis of the findings

All results are for the default configuration on the fixed test split
(7,327 rows); they are reproducible, and the live app shows the same values.

### 5.1 Overall performance

| Split | Accuracy |
|-------|----------|
| Train | 0.859 |
| Validation | 0.845 |
| Test | 0.847 |

The model generalizes well: test accuracy is within 1.2 points of training
accuracy, so it is not memorizing the training sample. ROC-AUC is **0.899**: given
one random high earner and one random low earner, the model ranks the high
earner higher about 90% of the time. For context, always guessing "≤50K" would
score 76.1% accuracy, so accuracy alone overstates the model; the per-class
numbers matter more.

### 5.2 The harder class

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| ≤50K | 0.886 | 0.917 | 0.901 | 5,574 |
| >50K | 0.702 | 0.625 | 0.661 | 1,753 |

**">50K" is clearly harder** (F1 0.661 vs 0.901), for two reasons. First, class
imbalance: high earners are 24% of the data, so minimizing average loss pulls
the model toward "≤50K". Second, heterogeneity: high earners reach that income
in different ways (executive jobs, professional degrees, capital gains, long
hours), while low earners look more alike. The confusion matrix shows where the
errors are: false negatives (missed high earners, 37.5% of them) outnumber false
positives. Lowering the threshold below 0.5 would catch more high earners at the
cost of precision; the threshold slider in the app shows this trade-off directly.

### 5.3 Calibration

Brier score 0.106 and expected calibration error (ECE) **0.023**: on average, the
predicted probability is within about 2 points of the observed rate. The
reliability diagram tracks the diagonal at the extremes, which contain most
rows, but the model is over-confident in the upper-middle range: rows predicted
around 0.65 are actually >50K about 55% of the time, and around 0.84 about 74%.
So the probabilities are usable for ranking and rough risk, but a "0.8" should
not be read literally as 80%.

### 5.4 Activation-function comparison

Three runs identical in every control (split, seed 0, 15,000 training rows,
2 × 32, lr 0.005, batch 256, 15 epochs), differing only in activation. The table
is generated by the `activation_comparison` SQL view over the `runs` table:

| Activation | Train acc | Val acc | Test acc | Test F1 (>50K) | ROC-AUC | Brier | ECE |
|------------|-----------|---------|----------|----------------|---------|-------|-----|
| relu | 0.8705 | 0.8444 | 0.8486 | 0.6498 | 0.8962 | 0.1069 | 0.0339 |
| **tanh** | 0.8594 | 0.8452 | 0.8469 | **0.6612** | 0.8987 | 0.1057 | 0.0227 |
| sigmoid | 0.8604 | 0.8471 | 0.8462 | 0.6533 | **0.8996** | **0.1051** | **0.0191** |

Accuracy is effectively tied (a 0.24-point spread), so we chose on the criteria
that matter for this product:

- **tanh** has the best F1 on the hard class.
- **ReLU** overfits most (train–validation gap 0.026 vs 0.014 for tanh) and is
  worst calibrated (ECE 0.034). Its unbounded output lets the network fit the
  15,000-row sample more aggressively.
- **Sigmoid** is best calibrated, but its outputs are not zero-centered and
  saturate near 0 and 1, which shrinks gradients; it had the highest final
  training loss (slowest learning).
- tanh is zero-centered with a stronger gradient near zero than sigmoid
  (maximum slope 1 vs 0.25), a good middle ground.

**Caveat:** this is one seed. Differences this small can flip with a different
seed; repeating the comparison over several seeds would turn the ranking into
evidence. The view supports this: each seed adds rows.

### 5.5 What drives the predictions

Permutation importance (drop in validation ROC-AUC when one column is shuffled):

| Feature | AUC drop |
|---------|----------|
| marital_status | 0.088 |
| capital_gain | 0.038 |
| age | 0.029 |
| hours_per_week | 0.026 |
| education_num | 0.023 |
| occupation | 0.019 |
| capital_loss | 0.006 |
| relationship | 0.006 |
| workclass | 0.002 |
| native_country | 0.001 |

**Marital status dominates.** Being married (`Married-civ-spouse`) is the
strongest single signal in this 1994 data. `relationship` scores low not because
it is uninformative, but because it duplicates marital status: shuffle one and
the other still carries the information. That redundancy matters for fairness,
because both columns encode sex (Husband/Wife), which is how sex information
reaches a model that never sees the `sex` column (Section 6).

### 5.6 Bias audit

| Sex | Test rows | True >50K rate | Predicted >50K rate | FPR | FNR |
|-----|-----------|----------------|---------------------|-----|-----|
| Female | 2,414 | 0.110 | 0.080 | 0.026 | **0.477** |
| Male | 4,913 | 0.303 | 0.278 | 0.119 | 0.357 |

- **FNR gap 0.12:** the model misses 47.7% of women who truly earn >50K, versus
  35.7% of men.
- **FPR gap 0.094:** men who earn ≤50K are wrongly flagged as high earners 4.6×
  as often as women.
- **Removing sex from the inputs did not remove the gap.** Retraining with sex and
  race as inputs (same seed) changes little: the FNR gap narrows only from 0.12
  to 0.09. The model learns sex through proxies (relationship, marital status),
  and the labels themselves reflect the 1994 wage gap.
- **By race,** high-earning Black (FNR 0.469) and Amer-Indian-Eskimo (0.429)
  people are missed more often than White (0.373) or Asian-Pac-Islander (0.294)
  people. Several race groups have fewer than 100 test rows, so these rates are
  uncertain.

---

## 6. Ethical consideration

### 6.1 The gap in a deployment context

Suppose a lender used Income-Insight to **pre-qualify loan applicants** by
predicted income. A false negative means a person who really earns more than
\$50K is scored as earning less: they are steered toward smaller loans or worse
terms, or denied outright. Our audit shows that this error falls on women far
more than men: **nearly half of high-earning women (47.7%) would be
under-scored, compared with about a third of men (35.7%).** The model never saw
anyone's sex, and its overall accuracy is 85%, so a lender looking only at the
headline metric would never see the problem. That is what makes disparate impact
dangerous: it hides inside numbers that look fair.

The false-positive gap creates a different harm: low-earning men are more often
over-scored, which could push them toward credit they cannot repay.

### 6.2 Two mitigations

1. **Per-group thresholds (equal opportunity).** Lower the decision threshold for
   women until their false-negative rate matches men's (Hardt et al., 2016). This
   directly closes the gap we measured and keeps the model unchanged, but it
   requires using the protected attribute at decision time, which lending law may
   restrict, and it lowers precision for the adjusted group.
2. **Reweighting the training data.** Increase the training weight of high-earning
   women so the loss pays more attention to the people the model currently
   misses. This keeps one threshold for everyone and does not use sex at decision
   time, but it must be re-audited: it can narrow the gap without closing it, and
   it can shift errors onto another group.

Neither mitigation replaces the most important safeguard: the score should never
make the decision alone. It belongs in front of a human reviewer, with the audit
published alongside it.

### 6.3 A Christian-worldview reflection on stewardship

Scripture describes the work of our hands as stewardship: what we are given,
including skill and tools, is entrusted to us, and we are accountable for how it
is used (Matthew 25:14–30). A model is such a tool. Building it well is not only
a matter of accuracy; it is a matter of whom it serves and whom it harms.

Every person scored by this model bears the image of God (Genesis 1:27), and that
dignity does not change with the group they belong to. A model that quietly
misses nearly half of the qualified women it evaluates treats those people as
less visible than others, even though no one designed it to. We need to approach
these models with the right level of respect to the people God has surrounded us
with, realizing everyone should be seen. Proverbs 31:8–9 calls us to speak up for
those who cannot speak for themselves and to judge fairly. Here, the people harmed
by a false negative usually never learn why they were turned away; the audit is
how their case gets heard.

Stewardship therefore asks three things of us as builders. First, **responsible
disclosure**: we publish the FPR and FNR gaps in the README, the model card, and a
dashboard tab, instead of reporting a single flattering accuracy number; "do not
have differing weights and measures" (Deuteronomy 25:13–16; compare Proverbs
11:1) applies to how we report a model's performance. Second, **humility about
limits**: the model is trained on 1994 data and is a teaching tool, and we say
plainly that it must not be used for real decisions. Third, **acting on what we
find**: naming mitigations and committing to re-audit after any change. Justice,
mercy, and humility (Micah 6:8) are not abstractions here; they are the
difference between a tool that hides who it fails and one that tells the truth
about it.

---

## 7. References

- Becker, B., & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems, 2*(4), 303–314.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of the 34th International Conference on Machine Learning*.
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *Advances in Neural Information Processing Systems, 29*.
- Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks, 4*(2), 251–257.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations*.
- Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems, 32*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- *The Holy Bible*. Genesis 1:27; Deuteronomy 25:13–16; Proverbs 11:1; Proverbs 31:8–9; Micah 6:8; Matthew 25:14–30.
