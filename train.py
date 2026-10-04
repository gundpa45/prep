"""
train.py – Full training, evaluation, and model-saving pipeline.

Usage:
    python train.py

This script:
  1. Loads the UCI Student Performance dataset.
  2. Performs data understanding (shape, dtypes, nulls, duplicates).
  3. Runs EDA and saves all plots to reports/figures/.
  4. Splits 80/20 train/test BEFORE any fitting (no data leakage).
  5. Trains 6 models across two experiments:
       A – all features (including period grades G1, G2)
       B – early-warning (without G1, G2)
  6. Cross-validates and tunes hyper-parameters.
  7. Evaluates on the held-out test set and prints comparison tables.
  8. Runs diagnostics (residuals, learning curves, overfitting check).
  9. Computes feature importance and permutation importance.
  10. Converts regression predictions to at-risk classification.
  11. Saves the best models with joblib and a metrics.json.
"""

import os
import sys
import json
import warnings

# Fix Windows console encoding for special characters (box-drawing, emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score,
    GridSearchCV, learning_curve,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.inspection import permutation_importance

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from src.preprocess import (
    load_data, build_pipeline, get_column_lists, get_feature_names,
    EARLY_WARNING_DROP,
)

# ─── CONFIG ─────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "reports", "figures")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "student-mat.csv")

# Create directories
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Plotting style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def save_fig(name: str):
    """Save the current figure and close it."""
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path)
    plt.close()
    print(f"  📊 Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# STEP 1 – DATA UNDERSTANDING
# ═══════════════════════════════════════════════════════════════════

def step1_data_understanding(df: pd.DataFrame):
    """Print shape, dtypes, nulls, duplicates, and descriptive stats."""
    print("\n" + "=" * 70)
    print("STEP 1 – DATA UNDERSTANDING")
    print("=" * 70)

    print(f"\n▸ Shape: {df.shape}")
    print(f"\n▸ Data types:\n{df.dtypes.value_counts()}")
    print(f"\n▸ Missing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    if df.isnull().sum().sum() == 0:
        print("  ✅ No missing values.")
    print(f"\n▸ Duplicate rows: {df.duplicated().sum()}")
    print(f"\n▸ Descriptive statistics:\n{df.describe().round(2)}")

    # G3 distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["G3"], bins=21, kde=True, color="#4C72B0", edgecolor="white", ax=ax)
    ax.set_title("Distribution of Final Grade (G3)", fontsize=14, fontweight="bold")
    ax.set_xlabel("G3 (Final Grade)")
    ax.set_ylabel("Count")
    zero_count = (df["G3"] == 0).sum()
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)
    ax.text(0.5, ax.get_ylim()[1] * 0.9, f"G3=0: {zero_count} students",
            color="red", fontsize=11)
    save_fig("g3_distribution.png")

    print(f"\n▸ Students with G3 = 0: {zero_count} ({zero_count/len(df)*100:.1f}%)")

    # At-risk class balance
    at_risk = (df["G3"] < 10).astype(int)
    counts = at_risk.value_counts()
    print(f"\n▸ At-risk (G3 < 10) class balance:")
    print(f"  Not at risk (0): {counts.get(0, 0)}")
    print(f"  At risk     (1): {counts.get(1, 0)}")
    print(f"  Ratio: {counts.get(1,0)/len(df)*100:.1f}% at risk")

    fig, ax = plt.subplots(figsize=(6, 4))
    at_risk.value_counts().plot(kind="bar", color=["#55A868", "#C44E52"],
                                 edgecolor="white", ax=ax)
    ax.set_xticklabels(["Not At Risk (≥10)", "At Risk (<10)"], rotation=0)
    ax.set_title("At-Risk Class Balance", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    for i, v in enumerate(at_risk.value_counts().values):
        ax.text(i, v + 2, str(v), ha="center", fontweight="bold")
    save_fig("at_risk_balance.png")


# ═══════════════════════════════════════════════════════════════════
# STEP 2 – EDA
# ═══════════════════════════════════════════════════════════════════

def step2_eda(df: pd.DataFrame):
    """Run exploratory data analysis and save all figures."""
    print("\n" + "=" * 70)
    print("STEP 2 – EDA")
    print("=" * 70)

    # --- Correlation heatmap (numeric columns) ---
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(14, 10))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap (Numeric Features)", fontsize=14, fontweight="bold")
    save_fig("correlation_heatmap.png")
    print("  Insight: G1 and G2 are extremely highly correlated with G3 (>0.8),")
    print("  confirming they are near-proxies for the final grade.")

    # --- Boxplots: G3 vs key features ---
    box_features = ["failures", "studytime", "Medu", "Fedu", "higher", "schoolsup"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for i, feat in enumerate(box_features):
        ax = axes[i // 3, i % 3]
        sns.boxplot(data=df, x=feat, y="G3", palette="Set2", ax=ax)
        ax.set_title(f"G3 vs {feat}", fontweight="bold")
    plt.suptitle("G3 vs Key Categorical/Ordinal Features", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig("boxplots_g3_vs_features.png")
    print("  Insight: More failures → lower G3. Higher education aspiration → higher G3.")

    # --- Boxplot: G3 vs absences (binned) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    df_temp = df.copy()
    df_temp["absences_bin"] = pd.cut(df_temp["absences"], bins=[0, 2, 6, 15, 100],
                                      labels=["0-2", "3-6", "7-15", "16+"],
                                      include_lowest=True)
    sns.boxplot(data=df_temp, x="absences_bin", y="G3", palette="YlOrRd", ax=ax)
    ax.set_title("G3 vs Absences (binned)", fontweight="bold")
    save_fig("boxplot_g3_vs_absences.png")
    print("  Insight: Very high absences tend to reduce final grade slightly.")

    # --- Scatter: G1 vs G3 and G2 vs G3 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, grade in enumerate(["G1", "G2"]):
        ax = axes[i]
        ax.scatter(df[grade], df["G3"], alpha=0.5, edgecolors="white",
                   linewidth=0.5, s=50, color="#4C72B0")
        # Fit line
        z = np.polyfit(df[grade], df["G3"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[grade].min(), df[grade].max(), 100)
        ax.plot(x_line, p(x_line), "--", color="#C44E52", linewidth=2)
        r = df[grade].corr(df["G3"])
        ax.set_title(f"{grade} vs G3  (r = {r:.3f})", fontsize=13, fontweight="bold")
        ax.set_xlabel(grade)
        ax.set_ylabel("G3")
    plt.tight_layout()
    save_fig("scatter_g1_g2_vs_g3.png")
    print("  Insight: G1 and G2 are almost linearly related to G3.")
    print("  This makes them powerful predictors but problematic for early-warning.")


# ═══════════════════════════════════════════════════════════════════
# STEP 3/4 – SPLIT & EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════

def step3_4_split(X, y):
    """Split data 80/20 BEFORE any fitting. Return splits for both experiments."""
    print("\n" + "=" * 70)
    print("STEP 3 – PREPROCESSING / STEP 4 – EXPERIMENTS")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n▸ Train size: {len(X_train)}, Test size: {len(X_test)}")
    print("  ✅ Split done BEFORE any fitting – no data leakage.")

    print("\n▸ Experiment A: All features including G1 and G2.")
    print("▸ Experiment B: Early-warning – G1 and G2 DROPPED.")
    print("  ⚠️  Why drop G1/G2? These period grades are recorded during the")
    print("  school year. Using them to predict the final grade (G3) is almost")
    print("  circular – by the time G2 is available, the year is nearly over.")
    print("  For a genuine early-warning system we need to predict BEFORE any")
    print("  grades exist, using only demographic / behavioural features.")

    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════
# STEP 5 – MODELS
# ═══════════════════════════════════════════════════════════════════

def get_models():
    """Return dict of model_name → (estimator, param_grid_or_None)."""
    return {
        "Linear Regression": (LinearRegression(), None),
        "Ridge": (Ridge(random_state=RANDOM_STATE), None),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {"model__max_depth": [3, 5, 8, 12], "model__min_samples_leaf": [2, 5, 10]},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {"model__n_estimators": [100, 200], "model__max_depth": [5, 10, 15],
             "model__min_samples_leaf": [2, 5]},
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1],
             "model__max_depth": [3, 5]},
        ),
        "SVR": (
            SVR(),
            {"model__C": [0.1, 1, 10], "model__kernel": ["rbf"],
             "model__epsilon": [0.1, 0.5]},
        ),
    }


def evaluate_model(pipeline, X_test, y_test):
    """Return dict of R², MAE, RMSE on test data."""
    y_pred = pipeline.predict(X_test)
    return {
        "R2": round(r2_score(y_test, y_pred), 4),
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
    }


def step5_train_and_evaluate(X_train, X_test, y_train, y_test, include_grades: bool):
    """Train all models for one experiment, return results dict + best pipeline."""
    exp_name = "A (with G1/G2)" if include_grades else "B (early-warning)"
    print(f"\n{'─' * 60}")
    print(f"Experiment {exp_name}")
    print(f"{'─' * 60}")

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models = get_models()
    results = []
    best_score = -np.inf
    best_pipeline = None
    best_name = ""

    for name, (estimator, param_grid) in models.items():
        pipe = build_pipeline(estimator, include_grades=include_grades)

        # Cross-validation
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

        # Hyper-parameter tuning if grid is provided
        if param_grid is not None:
            gs = GridSearchCV(pipe, param_grid, scoring="r2", cv=kf,
                              n_jobs=-1, refit=True)
            gs.fit(X_train, y_train)
            pipe = gs.best_estimator_
            print(f"  {name}: CV R²={cv_mean:.4f}±{cv_std:.4f}  |  "
                  f"Best params: {gs.best_params_}")
        else:
            pipe.fit(X_train, y_train)
            print(f"  {name}: CV R²={cv_mean:.4f}±{cv_std:.4f}")

        # Train score (for overfitting check)
        train_score = r2_score(y_train, pipe.predict(X_train))

        # Test evaluation
        metrics = evaluate_model(pipe, X_test, y_test)
        metrics["CV_R2_mean"] = round(cv_mean, 4)
        metrics["CV_R2_std"] = round(cv_std, 4)
        metrics["Train_R2"] = round(train_score, 4)
        metrics["Model"] = name
        results.append(metrics)

        if metrics["R2"] > best_score:
            best_score = metrics["R2"]
            best_pipeline = pipe
            best_name = name

    # Results table
    df_results = pd.DataFrame(results).sort_values("R2", ascending=False)
    df_results = df_results[["Model", "CV_R2_mean", "CV_R2_std", "Train_R2",
                              "R2", "MAE", "RMSE"]]
    print(f"\n  📋 Experiment {exp_name} – Comparison Table (sorted by test R²):")
    print(df_results.to_string(index=False))
    print(f"\n  🏆 Best model: {best_name} (Test R² = {best_score:.4f})")

    return df_results, best_pipeline, best_name, results


# ═══════════════════════════════════════════════════════════════════
# STEP 6 – DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════

def step6_diagnostics(pipe, X_train, X_test, y_train, y_test, exp_label: str,
                       results: list):
    """Actual vs predicted, residuals, learning curve, overfit check."""
    print(f"\n{'─' * 60}")
    print(f"STEP 6 – DIAGNOSTICS ({exp_label})")
    print(f"{'─' * 60}")

    y_pred = pipe.predict(X_test)

    # --- Actual vs Predicted ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="white", linewidth=0.5,
               s=50, color="#4C72B0")
    ax.plot([0, 20], [0, 20], "--", color="#C44E52", linewidth=2)
    ax.set_xlabel("Actual G3")
    ax.set_ylabel("Predicted G3")
    ax.set_title(f"Actual vs Predicted ({exp_label})", fontweight="bold")

    # --- Residuals ---
    ax = axes[1]
    residuals = y_test - y_pred
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors="white", linewidth=0.5,
               s=50, color="#DD8452")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Predicted G3")
    ax.set_ylabel("Residual")
    ax.set_title(f"Residual Plot ({exp_label})", fontweight="bold")
    plt.tight_layout()
    save_fig(f"diagnostics_actual_vs_pred_{exp_label}.png")

    # --- Learning Curve ---
    fig, ax = plt.subplots(figsize=(10, 5))
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring="r2", n_jobs=-1, random_state=RANDOM_STATE,
    )
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training R²", color="#4C72B0")
    ax.fill_between(train_sizes,
                     train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1),
                     alpha=0.15, color="#4C72B0")
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation R²", color="#55A868")
    ax.fill_between(train_sizes,
                     val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1),
                     alpha=0.15, color="#55A868")
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("R² Score")
    ax.set_title(f"Learning Curve – Best Model ({exp_label})", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim([-0.1, 1.05])
    save_fig(f"learning_curve_{exp_label}.png")

    # --- Overfitting Check ---
    print("\n  Train vs Test R² (overfitting check):")
    for r in results:
        gap = r["Train_R2"] - r["R2"]
        flag = " ⚠️ OVERFIT" if gap > 0.15 else ""
        print(f"    {r['Model']:25s} Train={r['Train_R2']:.4f}  Test={r['R2']:.4f}  "
              f"Gap={gap:.4f}{flag}")


# ═══════════════════════════════════════════════════════════════════
# STEP 7 – EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════

def step7_explainability(pipe, X_train, X_test, y_test, include_grades: bool,
                          exp_label: str):
    """Feature importance + permutation importance for best tree model."""
    print(f"\n{'─' * 60}")
    print(f"STEP 7 – EXPLAINABILITY ({exp_label})")
    print(f"{'─' * 60}")

    feature_names = get_feature_names(pipe, include_grades)
    model = pipe.named_steps["model"]

    # --- Feature importance (tree-based) ---
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi = fi.sort_values("Importance", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=fi, x="Importance", y="Feature", palette="viridis", ax=ax)
        ax.set_title(f"Top 10 Feature Importances ({exp_label})", fontweight="bold")
        save_fig(f"feature_importance_{exp_label}.png")
        print(f"\n  Top 10 features ({exp_label}):")
        print(fi.to_string(index=False))
    else:
        print("  (Model does not have feature_importances_)")

    # --- Permutation importance ---
    # When computed on the full pipeline, permutation importance permutes
    # the *original* input columns, not the one-hot-encoded features.
    perm = permutation_importance(pipe, X_test, y_test, n_repeats=10,
                                   random_state=RANDOM_STATE, scoring="r2",
                                   n_jobs=-1)
    original_cols = list(X_test.columns)
    pi = pd.DataFrame({
        "Feature": original_cols,
        "Importance": perm.importances_mean,
    }).sort_values("Importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=pi, x="Importance", y="Feature", palette="magma", ax=ax)
    ax.set_title(f"Top 10 Permutation Importances ({exp_label})", fontweight="bold")
    save_fig(f"permutation_importance_{exp_label}.png")
    print(f"\n  Top 10 permutation importances ({exp_label}):")
    print(pi.to_string(index=False))

    if include_grades:
        print("\n  📝 Interpretation: G2 and G1 dominate. They carry most of the")
        print("  predictive signal since they are intermediate assessments.")
    else:
        print("\n  📝 Interpretation: Without G1/G2, features like failures, absences,")
        print("  family education (Medu/Fedu), and going-out frequency matter most.")


# ═══════════════════════════════════════════════════════════════════
# STEP 8 – AT-RISK CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

def step8_at_risk(pipe, X_test, y_test, exp_label: str):
    """Convert regression to binary at-risk classification."""
    print(f"\n{'─' * 60}")
    print(f"STEP 8 – AT-RISK CLASSIFICATION ({exp_label})")
    print(f"{'─' * 60}")

    y_pred = pipe.predict(X_test)
    y_pred_class = (y_pred < 10).astype(int)
    y_true_class = (y_test < 10).astype(int)

    acc = accuracy_score(y_true_class, y_pred_class)
    prec = precision_score(y_true_class, y_pred_class, zero_division=0)
    rec = recall_score(y_true_class, y_pred_class, zero_division=0)
    f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
    cm = confusion_matrix(y_true_class, y_pred_class)

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}  ← most important (don't miss at-risk students!)")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")
    print(f"\n  Full Report:\n{classification_report(y_true_class, y_pred_class, target_names=['Not At Risk', 'At Risk'])}")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not At Risk", "At Risk"],
                yticklabels=["Not At Risk", "At Risk"],
                ax=ax, cbar=False, annot_kws={"size": 16})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix – At Risk ({exp_label})", fontweight="bold")
    save_fig(f"confusion_matrix_{exp_label}.png")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║       STUDENT PERFORMANCE PREDICTION SYSTEM – TRAINING         ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # ── Load data ──
    try:
        X, y = load_data(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Dataset not found at {DATA_PATH}")
        print("   Please place student-mat.csv in the data/ folder.")
        sys.exit(1)

    df = pd.concat([X, y.rename("G3")], axis=1)

    # ── Step 1: Data Understanding ──
    step1_data_understanding(df)

    # ── Step 2: EDA ──
    step2_eda(df)

    # ── Step 3/4: Split & Experiments ──
    X_train, X_test, y_train, y_test = step3_4_split(X, y)

    # ── Step 5: Models ──
    print("\n" + "=" * 70)
    print("STEP 5 – MODEL TRAINING & EVALUATION")
    print("=" * 70)

    # Experiment A: with grades
    df_results_a, best_pipe_a, best_name_a, results_a = step5_train_and_evaluate(
        X_train, X_test, y_train, y_test, include_grades=True
    )

    # Experiment B: early-warning
    df_results_b, best_pipe_b, best_name_b, results_b = step5_train_and_evaluate(
        X_train, X_test, y_train, y_test, include_grades=False
    )

    # ── Step 6: Diagnostics ──
    print("\n" + "=" * 70)
    print("STEP 6 – DIAGNOSTICS")
    print("=" * 70)

    step6_diagnostics(best_pipe_a, X_train, X_test, y_train, y_test,
                       "ExpA_with_grades", results_a)
    step6_diagnostics(best_pipe_b, X_train, X_test, y_train, y_test,
                       "ExpB_early_warning", results_b)

    # ── Step 7: Explainability ──
    print("\n" + "=" * 70)
    print("STEP 7 – EXPLAINABILITY")
    print("=" * 70)

    step7_explainability(best_pipe_a, X_train, X_test, y_test,
                          include_grades=True, exp_label="ExpA_with_grades")
    step7_explainability(best_pipe_b, X_train, X_test, y_test,
                          include_grades=False, exp_label="ExpB_early_warning")

    # ── Step 8: At-Risk Classification ──
    print("\n" + "=" * 70)
    print("STEP 8 – AT-RISK CLASSIFICATION")
    print("=" * 70)

    risk_a = step8_at_risk(best_pipe_a, X_test, y_test, "ExpA_with_grades")
    risk_b = step8_at_risk(best_pipe_b, X_test, y_test, "ExpB_early_warning")

    # ── Step 9: Save models & metrics ──
    print("\n" + "=" * 70)
    print("STEP 9 – SAVING MODELS & METRICS")
    print("=" * 70)

    # Save best pipelines
    model_a_path = os.path.join(MODELS_DIR, "model_with_grades.joblib")
    model_b_path = os.path.join(MODELS_DIR, "model_early_warning.joblib")
    joblib.dump(best_pipe_a, model_a_path)
    joblib.dump(best_pipe_b, model_b_path)
    print(f"\n  💾 Saved: {model_a_path}  ({best_name_a})")
    print(f"  💾 Saved: {model_b_path}  ({best_name_b})")

    # Save metrics JSON
    metrics = {
        "experiment_A": {
            "best_model": best_name_a,
            "results": results_a,
            "at_risk_metrics": risk_a,
        },
        "experiment_B": {
            "best_model": best_name_b,
            "results": results_b,
            "at_risk_metrics": risk_b,
        },
    }
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  💾 Saved: {metrics_path}")

    # ── Final Summary ──
    print("\n" + "=" * 70)
    print("✅ FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  Experiment A (with G1/G2):")
    print(f"    Best model: {best_name_a}")
    best_a = [r for r in results_a if r["Model"] == best_name_a][0]
    print(f"    Test R²={best_a['R2']:.4f}  MAE={best_a['MAE']:.4f}  RMSE={best_a['RMSE']:.4f}")
    print(f"    At-risk recall: {risk_a['recall']:.4f}")

    print(f"\n  Experiment B (early-warning, no G1/G2):")
    print(f"    Best model: {best_name_b}")
    best_b = [r for r in results_b if r["Model"] == best_name_b][0]
    print(f"    Test R²={best_b['R2']:.4f}  MAE={best_b['MAE']:.4f}  RMSE={best_b['RMSE']:.4f}")
    print(f"    At-risk recall: {risk_b['recall']:.4f}")

    print("\n  All plots saved to: reports/figures/")
    print("  All models saved to: models/")
    print("\n  Done! 🎉")


if __name__ == "__main__":
    main()
