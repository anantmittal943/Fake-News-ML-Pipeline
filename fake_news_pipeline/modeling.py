from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PLOTS_DIR,
    RANDOM_STATE,
)
from .transformers import SentinelToNaN

matplotlib.use("Agg")


def build_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    numeric_cols = [
        "word_count",
        "caps_ratio",
        "exclamation_count",
        "source_credibility_rank",
    ]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    year_pipe = Pipeline(
        steps=[
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("year", year_pipe, ["year"]),
            ("drop_id", "drop", ["id"]),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("fix_sentinel", SentinelToNaN(column="caps_ratio", sentinel=0.0)),
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Place 32_fake_news.csv in the project root or pass --data-path."
        )
    df = pd.read_csv(data_path)
    X = df.drop(columns=["fake"])
    y = df["fake"]
    return X, y


def evaluate_and_plot(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plots_dir: Path,
) -> dict[str, float]:
    plots_dir.mkdir(parents=True, exist_ok=True)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\nClassification report")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    metrics = {
        "macro_precision": precision_score(y_test, y_pred, average="macro"),
        "macro_recall": recall_score(y_test, y_pred, average="macro"),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    model = pipeline.named_steps["model"]
    year_ohe = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["year"]
        .named_steps["ohe"]
        .get_feature_names_out(["year"])
        .tolist()
    )
    feature_names = [
        "word_count",
        "caps_ratio",
        "exclamation_count",
        "source_credibility_rank",
    ] + year_ohe

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(importances)), importances[idx], color="steelblue")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right")
    ax.set_title("Feature Importances")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    fig.savefig(plots_dir / "feature_importances.png", dpi=150)
    plt.close(fig)

    return metrics


def train_and_evaluate(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    plots_dir: Path = DEFAULT_PLOTS_DIR,
    random_state: int = RANDOM_STATE,
) -> None:
    X, y = load_dataset(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    print(f"Loaded rows: {len(X):,}")
    print(f"Train/Test: {len(X_train):,}/{len(X_test):,}")

    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    print(f"5-fold macro-F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    metrics = evaluate_and_plot(pipeline, X_test, y_test, plots_dir)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")
    print(
        "Metrics: "
        + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate fake-news model pipeline")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train_and_evaluate(
        data_path=args.data_path,
        model_path=args.model_path,
        plots_dir=args.plots_dir,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
