"""UCI Adult Income feature contract, cleaning, and splitting.

Lives in ``shared`` so the API tier (to train), the loader script (to fill the
Supabase ``adult_income`` table), and the tests all use the same column lists and
the same cleaning rules.

Data source: the UCI *Adult* census dataset (1994 Current Population Survey,
48,842 rows after combining ``adult.data`` and ``adult.test``). It contains no
names or direct identifiers, but it does contain protected attributes (``sex``,
``race``) that we keep for the bias audit.

``generate_adult_like`` produces small synthetic rows in the SAME schema so the
offline test suite (and CI) never needs network or database access.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult"
UCI_FILES = ("adult.data", "adult.test")

# Column order of the raw UCI files.
RAW_COLUMNS: List[str] = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]

# The feature contract shared across all three clouds. The API's sklearn
# ColumnTransformer and the UI's auto-generated form both read these lists.
# Dropped on purpose: fnlwgt (a census sampling weight, not a person attribute)
# and education (a text duplicate of education_num).
NUMERIC_COLS: List[str] = [
    "age", "education_num", "capital_gain", "capital_loss", "hours_per_week",
]
CATEGORICAL_COLS: List[str] = [
    "workclass", "marital_status", "occupation", "relationship", "native_country",
]
# Protected attributes: stored and audited, but NOT model inputs unless a run
# sets include_protected=True (see the bias-audit discussion in the README).
PROTECTED_COLS: List[str] = ["sex", "race"]

FEATURE_COLS: List[str] = NUMERIC_COLS + CATEGORICAL_COLS


def feature_columns(include_protected: bool = False) -> Tuple[List[str], List[str]]:
    """Return ``(numeric, categorical)`` model-input columns for a run."""
    categorical = CATEGORICAL_COLS + (PROTECTED_COLS if include_protected else [])
    return list(NUMERIC_COLS), categorical


UNKNOWN = "Unknown"  # replaces the raw "?" missing-value marker

CATEGORIES: Dict[str, List[str]] = {
    "workclass": [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
        "State-gov", "Without-pay", "Never-worked", UNKNOWN,
    ],
    "marital_status": [
        "Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed",
        "Married-spouse-absent", "Married-AF-spouse",
    ],
    "occupation": [
        "Prof-specialty", "Exec-managerial", "Craft-repair", "Adm-clerical", "Sales",
        "Other-service", "Machine-op-inspct", "Transport-moving", "Handlers-cleaners",
        "Farming-fishing", "Tech-support", "Protective-serv", "Priv-house-serv",
        "Armed-Forces", UNKNOWN,
    ],
    "relationship": [
        "Husband", "Wife", "Not-in-family", "Own-child", "Unmarried", "Other-relative",
    ],
    "native_country": [
        "United-States", "Mexico", "Philippines", "Germany", "Puerto-Rico", "Canada",
        "El-Salvador", "India", "Cuba", "England", "China", "South", "Jamaica",
        "Italy", "Dominican-Republic", "Japan", "Guatemala", "Poland", "Vietnam",
        "Columbia", "Haiti", "Portugal", "Taiwan", "Iran", "Greece", "Nicaragua",
        "Peru", "Ecuador", "France", "Ireland", "Hong", "Thailand", "Cambodia",
        "Trinadad&Tobago", "Outlying-US(Guam-USVI-etc)", "Yugoslavia", "Laos",
        "Scotland", "Honduras", "Hungary", "Holand-Netherlands", UNKNOWN,
    ],
    "sex": ["Female", "Male"],
    "race": ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
}

TARGET_NAME = "income"
TARGET_CLASSES = ["<=50K", ">50K"]  # index 0 / 1

SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42

# Columns stored in the Supabase adult_income table (besides id).
STORED_COLS: List[str] = (
    NUMERIC_COLS + CATEGORICAL_COLS + PROTECTED_COLS + ["income", "label", "split"]
)


def clean_adult(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw UCI rows into the stored schema (without ``split``/``id``).

    * strips whitespace, maps "?" to "Unknown",
    * normalizes the test file's ">50K." labels to ">50K",
    * adds ``label`` (1 == ">50K"), drops fnlwgt/education.
    """
    df = raw.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    df = df.replace("?", UNKNOWN)
    df["income"] = df["income"].str.rstrip(".")
    bad = ~df["income"].isin(TARGET_CLASSES)
    if bad.any():
        raise ValueError(f"unexpected income labels: {sorted(df.loc[bad, 'income'].unique())}")
    df["label"] = (df["income"] == ">50K").astype(int)
    for col in NUMERIC_COLS:
        df[col] = df[col].astype(int)
    return df[NUMERIC_COLS + CATEGORICAL_COLS + PROTECTED_COLS + ["income", "label"]]


def assign_splits(df: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Add a stratified, reproducible 70/15/15 ``split`` column and a stable ``id``.

    Stratifying on the label keeps the >50K share equal across splits; the fixed
    seed means every run (and every teammate) trains and tests on the same rows,
    so run-to-run comparisons are controlled.
    """
    rng = np.random.default_rng(seed)
    split = np.empty(len(df), dtype=object)
    for label in (0, 1):
        idx = np.flatnonzero(df["label"].to_numpy() == label)
        idx = rng.permutation(idx)
        n_train = int(round(len(idx) * SPLIT_FRACTIONS["train"]))
        n_val = int(round(len(idx) * SPLIT_FRACTIONS["val"]))
        split[idx[:n_train]] = "train"
        split[idx[n_train:n_train + n_val]] = "val"
        split[idx[n_train + n_val:]] = "test"
    out = df.reset_index(drop=True).copy()
    out["split"] = split
    out.insert(0, "id", np.arange(len(out)))
    return out


def generate_adult_like(n_rows: int, seed: int = 0) -> pd.DataFrame:
    """Synthetic rows in the stored ``adult_income`` schema, for offline tests.

    Income follows a known logistic rule (education, age, hours, capital gain,
    marriage, occupation) so tests can assert the MLP learns real signal. The
    rule also depends on ``sex`` through ``relationship`` (as in the real data),
    which gives the bias-audit tests a non-trivial gap to measure.
    """
    rng = np.random.default_rng(seed)
    sex = rng.choice(["Female", "Male"], n_rows, p=[0.33, 0.67])
    married = rng.random(n_rows) < np.where(sex == "Male", 0.6, 0.15)
    relationship = np.where(
        married, np.where(sex == "Male", "Husband", "Wife"),
        rng.choice(["Not-in-family", "Own-child", "Unmarried"], n_rows),
    )
    marital = np.where(
        married, "Married-civ-spouse",
        rng.choice(["Never-married", "Divorced", "Separated", "Widowed"], n_rows),
    )
    age = rng.integers(17, 76, n_rows)
    education_num = rng.integers(1, 17, n_rows)
    hours = rng.integers(10, 71, n_rows)
    capital_gain = ((rng.random(n_rows) < 0.08) * rng.integers(1000, 20001, n_rows)).astype(int)
    capital_loss = ((rng.random(n_rows) < 0.05) * rng.integers(100, 3001, n_rows)).astype(int)
    occupation = rng.choice(
        ["Prof-specialty", "Exec-managerial", "Craft-repair", "Adm-clerical",
         "Sales", "Other-service"], n_rows,
    )
    occ_offset = {
        "Prof-specialty": 0.8, "Exec-managerial": 0.9, "Craft-repair": 0.0,
        "Adm-clerical": -0.2, "Sales": 0.1, "Other-service": -0.9,
    }
    z = (
        0.04 * (age - 38)
        + 0.35 * (education_num - 10)
        + 0.03 * (hours - 40)
        + 0.0002 * capital_gain
        + 1.6 * married
        + np.array([occ_offset[o] for o in occupation])
        - 1.6
    )
    label = (rng.random(n_rows) < 1 / (1 + np.exp(-z))).astype(int)
    df = pd.DataFrame({
        "age": age,
        "education_num": education_num,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours,
        "workclass": rng.choice(["Private", "Self-emp-not-inc", "Local-gov"], n_rows),
        "marital_status": marital,
        "occupation": occupation,
        "relationship": relationship,
        "native_country": rng.choice(["United-States", "Mexico", "India"], n_rows,
                                     p=[0.9, 0.06, 0.04]),
        "sex": sex,
        "race": rng.choice(["White", "Black", "Asian-Pac-Islander"], n_rows,
                           p=[0.85, 0.1, 0.05]),
        "label": label,
    })
    df["income"] = np.where(df["label"] == 1, ">50K", "<=50K")
    return assign_splits(df[NUMERIC_COLS + CATEGORICAL_COLS + PROTECTED_COLS + ["income", "label"]],
                         seed=seed)
