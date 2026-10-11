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
