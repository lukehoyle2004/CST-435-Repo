# Model Card — Income-Insight

## Model details
- **Task:** binary classification — does a person's annual income exceed \$50K?
- **Architecture:** a deep multi-layer perceptron (PyTorch).
  `Input(84) → [Linear → tanh] × 2 (32 units each) → Linear(1) → sigmoid`.
  Depth, width, and activation (relu / leaky_relu / tanh / sigmoid / gelu) are
  configurable per run; tanh is the default, chosen by a matched comparison
  (see the README).
- **Preprocessing:** an sklearn `ColumnTransformer` — `StandardScaler` on 5
  numeric features, `OneHotEncoder` on 5 categorical features — fit on the
  training split only and serialized together with the network.
- **Training:** Adam, binary cross-entropy, lr 0.005, batch 256, 15 epochs,
  seed 0, on a 15,000-row sample of the training split (all settings are stored
  with each run in Supabase). Runs that diverge are stored with NULL metrics and
  never served.
- **Version / owners:** Income-Insight v2 · Luke Hoyle and partner, CST-435 / AIT-204, October 2026.

## Intended use
- **Primary:** a teaching and demonstration service for the full
  train → evaluate → calibrate → serve → audit loop of a tabular classifier
  across three clouds.
- **Users:** students, instructors, and reviewers exploring how an MLP behaves
  and how its errors are distributed.
- **Out of scope:** any real decision about a real person — hiring, lending,
  housing, insurance, benefits eligibility, or income verification. The model
  is trained on 1994 census data and shows unequal error rates by sex and race.

## Data
- **Source:** UCI Adult (1994 US Current Population Survey), 48,842 records,
  23.9% earn >50K. No names or direct identifiers.
- **Inputs:** age, education_num, capital_gain, capital_loss, hours_per_week,
  workclass, marital_status, occupation, relationship, native_country.
- **Protected attributes:** sex and race are stored for auditing and are **not**
  model inputs by default.
- **Split:** fixed, stratified 70 / 15 / 15 train / validation / test (seed 42),
  identical for every run.
- **Known data issues:** the data is over 30 years old, reflects the 1994 wage
  gap, uses a binary sex field and coarse race categories, has missing values
  (`?` → `Unknown`) concentrated in workclass / occupation, and thin coverage for
  some groups and countries.

## Performance (test split, 7,327 rows, threshold 0.5)

| Metric | Value |
|--------|-------|
| Accuracy (train / val / test) | 0.859 / 0.845 / 0.847 |
| ROC-AUC | 0.899 |
| `>50K` precision / recall / F1 | 0.702 / 0.625 / 0.661 |
| `<=50K` precision / recall / F1 | 0.886 / 0.917 / 0.901 |
| Brier score / ECE | 0.106 / 0.023 |

`>50K` is the harder class: about 37% of true high earners are missed. The
probabilities are reasonably calibrated but somewhat over-confident between
about 0.55 and 0.85. Live numbers for every run are in the Model Performance tab
and the `runs` table.

## Fairness statement
We audited false-positive and false-negative rates by sex and race on the test
split, using a SQL view over the logged predictions. Although the model never
sees `sex`, it **misses 48% of women who truly earn >50K versus 36% of men**
(FNR gap 0.12), and it wrongly flags ≤50K men about 4.6× as often as ≤50K women
(FPR 0.119 vs 0.026). Proxies such as `relationship` and `marital_status` carry
sex information, and the labels reflect a historical wage gap, so removing the
protected attribute does not make the model fair. We publish these gaps instead
of hiding them behind a single accuracy number. Any real use would require a
mitigation (for example per-group thresholds for equal opportunity, or
reweighting the training data), a re-audit, and a legal review — and should
never be fully automated.

## What drives predictions
Permutation importance (validation ROC-AUC drop): marital_status (0.088) ≫
capital_gain (0.038) > age (0.029) > hours_per_week (0.026) > education_num
(0.023) > occupation (0.019); the rest are below 0.01.

## Limitations
- Single fixed split and a single seed per configuration; small differences
  between configurations (< ~0.5 accuracy points) may be noise.
- Trained on a 15,000-row sample by default to fit Render's free-tier timeout.
- Shallow tabular MLP; gradient-boosted trees typically match or beat it on
  this dataset.
- 1994 data does not represent today's labor market; predictions about current
  incomes are not meaningful.
- Rates for small race groups (< 100 test rows) are statistically noisy.

## Ethical considerations
The score can look objective while encoding historical inequity. Users must be
told plainly that it is a demonstration on old census data, what its error rates
are, and who those errors fall on. Do not overstate its confidence, and do not
use it to make or justify decisions about people.

## What we log (and don't)
- **Logged:** run settings and all evaluation results (`runs`), the fitted
  preprocessor + weights (`run_artifacts`), and every prediction with its
  probability, label, threshold, and source (`predictions`).
- **Not logged:** uploaded CSV files themselves, user identities (there are no
  accounts), or any secrets. Rows typed into Score a Row / Score a CSV are stored
  as predictions, so do not enter real people's data.
