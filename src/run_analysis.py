"""Reproducible analysis for mini-project 36.

Downloads the official UCI dataset, trains comparable classifiers, tunes
Gradient Boosting without touching the final test set, and writes evidence
files/plots into results/.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
DATA_URL = "https://archive.ics.uci.edu/static/public/519/heart+failure+clinical+records.zip"
WEB_DATA_DIR = ROOT / "web" / "data"

FEATURE_DESCRIPTIONS = {
    "age": "Patient age in years.",
    "anaemia": "Binary anaemia indicator.",
    "creatinine_phosphokinase": "Blood creatinine phosphokinase level.",
    "diabetes": "Binary diabetes indicator.",
    "ejection_fraction": "Percentage of blood leaving the heart at each contraction.",
    "high_blood_pressure": "Binary hypertension indicator.",
    "platelets": "Platelet count.",
    "serum_creatinine": "Serum creatinine level.",
    "serum_sodium": "Serum sodium level.",
    "sex": "Binary sex indicator as supplied by the dataset.",
    "smoking": "Binary smoking indicator.",
    "time": "Follow-up period in days.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_data() -> tuple[pd.DataFrame, str]:
    zip_path = DATA_DIR / "heart-failure-clinical-records.zip"
    csv_path = DATA_DIR / "heart_failure_clinical_records_dataset.csv"
    if not csv_path.exists():
        print("Downloading official UCI archive...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            csv_member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
            archive.extract(csv_member, DATA_DIR)
            extracted = DATA_DIR / csv_member
            extracted.replace(csv_path)
    digest = sha256(csv_path)
    return pd.read_csv(csv_path), digest


def metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    """Calculate the operating-point metrics without changing model probabilities."""
    return metrics(y_true, (y_prob >= threshold).astype(int), y_prob)


def json_scalar(value: object) -> object:
    """Convert numpy scalars into values that can be written to dashboard JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def feature_metadata(df: pd.DataFrame, target: str) -> list[dict[str, object]]:
    metadata: list[dict[str, object]] = []
    for name in df.columns:
        if name == target:
            continue
        series = df[name]
        values = series.dropna()
        unique = sorted(values.unique().tolist())
        binary = set(unique).issubset({0, 1}) and len(unique) <= 2
        entry: dict[str, object] = {
            "name": name,
            "label": name.replace("_", " ").title(),
            "kind": "binary" if binary else "numeric",
            "description": FEATURE_DESCRIPTIONS.get(name, "Numeric predictor supplied by the dataset."),
            "missing": int(series.isna().sum()),
            "unique": int(values.nunique()),
            "min": json_scalar(values.min()),
            "max": json_scalar(values.max()),
            "mean": json_scalar(values.mean()),
            "median": json_scalar(values.median()),
        }
        if binary:
            entry["categories"] = [
                {"value": json_scalar(category), "count": int((values == category).sum())}
                for category in unique
            ]
        else:
            entry["quantiles"] = {
                "q1": json_scalar(values.quantile(.25)),
                "q3": json_scalar(values.quantile(.75)),
            }
        metadata.append(entry)
    return metadata


def tree_path_contributions(model: GradientBoostingClassifier, X: pd.DataFrame) -> np.ndarray:
    """Decompose each fitted tree's path correction by split feature.

    This is an exact additive decomposition of the fitted estimator's raw
    decision score along each sample's tree paths. It is exported as a model
    explanation, not as a causal or clinical effect.
    """
    contributions = np.zeros((len(X), X.shape[1]), dtype=float)
    values = X.to_numpy()
    for estimator in model.estimators_[:, 0]:
        tree = estimator.tree_
        for row_index, row in enumerate(values):
            node = 0
            while tree.children_left[node] != tree.children_right[node]:
                feature_index = int(tree.feature[node])
                next_node = tree.children_left[node] if row[feature_index] <= tree.threshold[node] else tree.children_right[node]
                delta = float(tree.value[next_node][0][0] - tree.value[node][0][0])
                contributions[row_index, feature_index] += model.learning_rate * delta
                node = next_node
    return contributions


def export_dashboard_data(
    df: pd.DataFrame,
    target: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    models: dict[str, object],
    probabilities: dict[str, np.ndarray],
    tuning: RandomizedSearchCV,
    digest: str,
) -> None:
    """Export actual pipeline outputs for the static explainability frontend."""
    feature_names = list(X_train.columns)
    best = models["Tuned Gradient Boosting"]
    best_probabilities = probabilities["Tuned Gradient Boosting"]
    best_contributions = tree_path_contributions(best, X_test)
    # Each tree's path contribution is measured relative to that tree's root
    # value. Include all root values in the baseline so the exported identity
    # is exact: decision_score = baseline + sum(feature contributions).
    tree_root_score = sum(float(estimator.tree_.value[0][0][0]) for estimator in best.estimators_[:, 0]) * best.learning_rate
    X_test_values = X_test.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        base_scores = np.asarray(best._raw_predict_init(X_test_values), dtype=float).reshape(-1) + tree_root_score
        decision_scores = np.asarray(best.decision_function(X_test_values), dtype=float).reshape(-1)
    feature_info = feature_metadata(df, target)
    model_payload: list[dict[str, object]] = []
    for name, probability in probabilities.items():
        model_payload.append({
            "name": name,
            "metrics_at_050": threshold_metrics(y_test, probability, .50),
            "metrics_at_035": threshold_metrics(y_test, probability, .35),
        })

    tree_summaries = []
    for index, estimator in enumerate(best.estimators_[:, 0], start=1):
        tree = estimator.tree_
        root_feature = int(tree.feature[0])
        tree_summaries.append({
            "index": index,
            "depth": int(estimator.get_depth()),
            "leaves": int(estimator.get_n_leaves()),
            "root_feature": feature_names[root_feature] if root_feature >= 0 else None,
            "root_threshold": json_scalar(tree.threshold[0]) if root_feature >= 0 else None,
        })

    test_predictions = []
    for position, row_index in enumerate(X_test.index):
        test_predictions.append({
            "row_index": int(row_index),
            "record_number": int(row_index) + 1,
            "features": {name: json_scalar(X_test.iloc[position][name]) for name in feature_names},
            "actual": int(y_test.iloc[position]),
            "probability": float(best_probabilities[position]),
            "decision_score": float(decision_scores[position]),
            "base_score": float(base_scores[position]),
            "contributions": {
                name: float(best_contributions[position, index])
                for index, name in enumerate(feature_names)
            },
            "all_model_probabilities": {
                name: float(probability[position])
                for name, probability in probabilities.items()
            },
        })

    dashboard_data = {
        "artifact": "actual-project-pipeline",
        "generated_by": "src/run_analysis.py",
        "dataset": {
            "source_url": DATA_URL,
            "sha256": digest,
            "rows": int(df.shape[0]),
            "usable_rows": int(df.dropna().shape[0]),
            "columns": int(df.shape[1]),
            "feature_count": len(feature_names),
            "target": target,
            "target_description": "1 indicates a recorded death event during follow-up; 0 indicates no recorded event.",
            "positive_events": int(y_train.sum() + y_test.sum()),
            "negative_events": int((1 - y_train).sum() + (1 - y_test).sum()),
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "features": feature_info,
            "numerical_features": [item["name"] for item in feature_info if item["kind"] == "numeric"],
            "categorical_features": [item["name"] for item in feature_info if item["kind"] == "binary"],
        },
        "preprocessing": {
            "target_separated": target,
            "missing_value_handling": "Checked; no missing values were found in the supplied dataset.",
            "duplicate_handling": "Checked; no duplicate rows were found in the supplied dataset.",
            "split": "train_test_split(test_size=0.20, stratify=y, random_state=42)",
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "gradient_boosting_transform": "Raw numeric predictors are passed to the tree ensemble; no scaling or one-hot encoding is applied.",
            "other_models": "Logistic Regression uses StandardScaler; Decision Tree and Gradient Boosting use raw numeric predictors.",
            "steps": [
                f"Separate {target} from the 12 predictors.",
                "Check missing values and duplicate rows.",
                "Create an 80/20 stratified train/test split with random seed 42.",
                "Keep the held-out test set untouched during Gradient Boosting tuning.",
                "Pass raw numeric predictors to the tree-based models.",
            ],
        },
        "model": {
            "name": "Tuned Gradient Boosting",
            "class": "sklearn.ensemble.GradientBoostingClassifier",
            "parameters": {key: json_scalar(value) for key, value in tuning.best_params_.items()},
            "feature_names": feature_names,
            "feature_importances": {
                name: float(importance)
                for name, importance in zip(feature_names, best.feature_importances_)
            },
            "estimator_count": int(best.n_estimators),
            "tree_summaries": tree_summaries,
            "concept_note": "Tree structure metadata is from the fitted ensemble. The sequential correction arrows in the UI are a conceptual explanation; this artifact does not claim a simplified tree score is the estimator's exact internal score.",
        },
        "models": model_payload,
        "selected_model": "Tuned Gradient Boosting",
        "thresholds": {"default": .50, "analysis": .35, "slider_min": .10, "slider_max": .90},
        "validation": {
            "tuning_method": "RandomizedSearchCV",
            "folds": 5,
            "stratified": True,
            "scoring": "recall",
            "best_cv_recall": float(tuning.best_score_),
            "test_set_used_during_tuning": False,
            "top_cv_results": [
                {
                    "rank": int(row.rank_test_score),
                    "mean_recall": float(row.mean_test_score),
                    "std_recall": float(row.std_test_score),
                    "parameters": {key: json_scalar(value) for key, value in row.params.items()},
                }
                for row in pd.DataFrame(tuning.cv_results_).sort_values("rank_test_score").head(10).itertuples()
            ],
        },
        "explainability": {
            "available": True,
            "method": "Exact tree-path additive decomposition of the fitted Gradient Boosting decision score.",
            "scale": "raw decision-score / log-odds scale, before the sigmoid probability transform",
            "identity": "For every exported record: decision_score = baseline_score + sum(feature_contributions). The baseline includes the fitted initial raw score and tree-root corrections.",
            "note": "A contribution shows how the fitted model's tree paths moved its decision score for this record. It is model behavior, not causation.",
        },
        "test_predictions": test_predictions,
    }
    serialized = json.dumps(dashboard_data, indent=2)
    RESULTS_DIR.joinpath("dashboard_data.json").write_text(serialized, encoding="utf-8")
    WEB_DATA_DIR.mkdir(exist_ok=True)
    WEB_DATA_DIR.joinpath("dashboard_data.json").write_text(serialized, encoding="utf-8")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    df, digest = get_data()
    target = "DEATH_EVENT"
    X = df.drop(columns=target)
    y = df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=RANDOM_STATE),
    }
    tuning = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_distributions={
            "n_estimators": [50, 75, 100, 150, 200],
            "learning_rate": [0.02, 0.05, 0.08, 0.1, 0.15],
            "max_depth": [1, 2, 3],
            "min_samples_leaf": [2, 4, 8, 12],
            "subsample": [0.7, 0.85, 1.0],
        },
        n_iter=24, scoring="recall", cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
        refit=True,
    )
    tuning.fit(X_train, y_train)
    models["Tuned Gradient Boosting"] = tuning.best_estimator_

    rows = []
    probabilities: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        probabilities[name] = prob
        row = metrics(y_test, pred, prob)
        row["model"] = name
        rows.append(row)
    metric_df = pd.DataFrame(rows).set_index("model")
    metric_df.to_csv(RESULTS_DIR / "metrics.csv")
    pd.DataFrame(tuning.cv_results_).sort_values("rank_test_score").head(10).to_csv(RESULTS_DIR / "tuning_results_top10.csv", index=False)

    summary = {
        "source_url": DATA_URL,
        "sha256": digest,
        "rows": int(df.shape[0]), "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "positive_events": int(y.sum()), "negative_events": int((1-y).sum()),
        "best_parameters": tuning.best_params_,
        "best_cv_recall": float(tuning.best_score_),
        "train_rows": int(X_train.shape[0]), "test_rows": int(X_test.shape[0]),
    }
    (RESULTS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS_DIR / "dataset_summary.txt").write_text(df.describe().to_string() + "\n\n" + json.dumps(summary, indent=2), encoding="utf-8")
    export_dashboard_data(df, target, X_train, X_test, y_train, y_test, models, probabilities, tuning, digest)

    # Class balance.
    plt.figure(figsize=(7, 5))
    sns.countplot(x=y.map({0: "Survived", 1: "Death event"}), hue=y.map({0: "Survived", 1: "Death event"}), legend=False, palette="Set2")
    plt.title("Target distribution in the 299-patient dataset")
    plt.xlabel("Outcome"); plt.ylabel("Number of patients"); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_balance.png", dpi=180); plt.close()

    # ROC and PR curves.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        precision, recall, _ = precision_recall_curve(y_test, prob)
        axes[0].plot(fpr, tpr, label=name)
        axes[1].plot(recall, precision, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=.5); axes[0].set_title("ROC curves"); axes[0].set_xlabel("False-positive rate"); axes[0].set_ylabel("Recall")
    axes[1].set_title("Precision–recall curves"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_pr_curves.png", dpi=180); plt.close()

    best = models["Tuned Gradient Boosting"]
    best_pred = best.predict(X_test)
    plt.figure(figsize=(6, 5)); sns.heatmap(confusion_matrix(y_test, best_pred), annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Tuned Gradient Boosting confusion matrix"); plt.xlabel("Predicted label"); plt.ylabel("True label"); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=180); plt.close()

    importance = pd.Series(best.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(8, 6)); importance.plot.barh(color="#4472c4"); plt.title("Gradient Boosting feature importance"); plt.xlabel("Relative importance"); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=180); plt.close()

    print("Analysis complete.")
    print(metric_df.round(3).to_string())
    print("Best parameters:", tuning.best_params_)
    print("Results written to", RESULTS_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        raise
