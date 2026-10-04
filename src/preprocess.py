"""
preprocess.py – Shared preprocessing utilities and pipeline builder.

This module defines which columns are numeric vs categorical,
builds a ColumnTransformer that applies StandardScaler to numeric
features and OneHotEncoder to categorical features, and wraps
everything into a scikit-learn Pipeline so that the scaler/encoder
are fit **only** on training data (no data leakage).
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ─── Column Definitions ────────────────────────────────────────────

# All columns present in the raw CSV (minus the target G3)
ALL_FEATURES = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc",
    "health", "absences", "G1", "G2",
]

# Features for Experiment B (early-warning): G1 and G2 removed
EARLY_WARNING_DROP = ["G1", "G2"]

# Numeric and categorical split (used by ColumnTransformer)
NUMERIC_COLS_ALL = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2",
]

CATEGORICAL_COLS_ALL = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic",
]


def get_column_lists(include_grades: bool = True):
    """Return (numeric_cols, categorical_cols) for a given experiment.

    Parameters
    ----------
    include_grades : bool
        If True  → Experiment A (all features including G1, G2).
        If False → Experiment B (early-warning, no G1/G2).
    """
    if include_grades:
        return NUMERIC_COLS_ALL.copy(), CATEGORICAL_COLS_ALL.copy()
    else:
        num = [c for c in NUMERIC_COLS_ALL if c not in EARLY_WARNING_DROP]
        return num, CATEGORICAL_COLS_ALL.copy()


def build_preprocessor(numeric_cols: list, categorical_cols: list):
    """Build a ColumnTransformer with StandardScaler + OneHotEncoder.

    Parameters
    ----------
    numeric_cols : list of str
        Columns that should be standardised.
    categorical_cols : list of str
        Columns that should be one-hot encoded.

    Returns
    -------
    sklearn ColumnTransformer
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",  # safety: drop any unexpected columns
    )
    return preprocessor


def build_pipeline(model, include_grades: bool = True):
    """Create a full Pipeline: preprocessor → model.

    The preprocessor's scaler & encoder will be fit only when
    pipeline.fit(X_train, y_train) is called, preventing data leakage.

    Parameters
    ----------
    model : sklearn estimator
        Any scikit-learn regressor.
    include_grades : bool
        Whether to include G1/G2 columns.

    Returns
    -------
    sklearn Pipeline
    """
    num_cols, cat_cols = get_column_lists(include_grades)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def load_data(path: str = "data/student-mat.csv"):
    """Load the student-mat CSV and return (X, y).

    Parameters
    ----------
    path : str
        Path to the CSV file (semicolon-separated).

    Returns
    -------
    X : DataFrame  (all columns except G3)
    y : Series     (G3 – final grade)
    """
    df = pd.read_csv(path, sep=";")
    X = df.drop(columns=["G3"])
    y = df["G3"].astype(float)
    return X, y


def get_feature_names(pipeline, include_grades: bool = True):
    """Extract human-readable feature names after one-hot encoding.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
        Must already be fitted so the encoder knows its categories.
    include_grades : bool
        Must match the pipeline's configuration.

    Returns
    -------
    list of str
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    num_cols, cat_cols = get_column_lists(include_grades)

    # Numeric feature names stay the same
    feature_names = list(num_cols)

    # Categorical feature names come from the fitted OneHotEncoder
    ohe = preprocessor.named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    feature_names.extend(cat_feature_names)

    return feature_names
