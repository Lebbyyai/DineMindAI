
from pathlib import Path
from datetime import datetime
import shutil

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ==========================================
# FILE PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "datasets"
    / "raw"
    / "sentiment"
    / "yelp_labelled.txt"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "naive_bayes_sentiment_model.joblib"
)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ==========================================
# LOAD DATA
# ==========================================

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["review", "label"],
        encoding="utf-8",
    )

    df = df.dropna(subset=["review", "label"]).copy()
    df["review"] = df["review"].astype(str).str.strip()
    df["label"] = pd.to_numeric(
        df["label"], errors="raise"
    ).astype(int)

    df = df[df["review"].str.len() > 0]
    df = df[df["label"].isin([0, 1])]

    # Remove conflicting exact duplicates.
    conflicts = df.groupby("review")["label"].nunique()
    conflicting_reviews = set(
        conflicts[conflicts > 1].index
    )

    df = df[
        ~df["review"].isin(conflicting_reviews)
    ].copy()

    df = df.drop_duplicates(
        subset=["review", "label"]
    ).reset_index(drop=True)

    print("Dataset records:", len(df))
    print("Label distribution:")
    print(df["label"].value_counts())

    return df


# ==========================================
# MODEL SEARCH
# ==========================================

def build_search():
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", MultinomialNB()),
    ])

    grids = [
        {
            "vectorizer": [
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                ),
                CountVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                ),
            ],
            "vectorizer__ngram_range": [
                (1, 1),
                (1, 2),
            ],
            "vectorizer__min_df": [1, 2],
            "vectorizer__max_df": [1.0],
            "vectorizer__binary": [False, True],
            "classifier__alpha": [
                0.1, 0.5, 1.0, 2.0, 5.0
            ],
        }
    ]

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid=grids,
        scoring="f1_macro",
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
        verbose=1,
        error_score="raise",
    )


# ==========================================
# EVALUATION
# ==========================================

def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test, predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predictions": predictions,
    }


def print_metrics(title, evaluation, y_test):
    print(f"\n{'=' * 55}")
    print(title)
    print(f"{'=' * 55}")

    print(
        "Accuracy:",
        round(evaluation["accuracy"], 4)
    )

    print(
        "Macro F1:",
        round(evaluation["macro_f1"], 4)
    )

    print(
        "\nClassification report:\n",
        classification_report(
            y_test,
            evaluation["predictions"],
            labels=[0, 1],
            target_names=["Negative", "Positive"],
            zero_division=0,
        )
    )

    print(
        "Confusion matrix:\n",
        confusion_matrix(
            y_test,
            evaluation["predictions"],
            labels=[0, 1],
        )
    )


# ==========================================
# EXAMPLE REVIEWS
# ==========================================

def test_example_reviews(model):
    examples = [
        "The food was disgusting",
        "The food was absolutely disgusting.",
        "The service was terrible and the food tasted awful.",
        "I will never eat here again.",
        "The restaurant was dirty and the staff were rude.",
        "The food was delicious and the staff were friendly.",
        "The meal was excellent!",
        "The food was not good.",
        "The service was not bad.",
    ]

    print("\n" + "=" * 55)
    print("EXAMPLE REVIEW PREDICTIONS")
    print("=" * 55)

    probabilities = model.predict_proba(examples)
    predictions = model.predict(examples)

    for review, prediction, row in zip(
        examples, predictions, probabilities
    ):
        positive_index = list(
            model.classes_
        ).index(1)

        positive_probability = row[positive_index]

        sentiment = (
            "Positive"
            if int(prediction) == 1
            else "Negative"
        )

        print(f"\nReview: {review}")
        print(f"Prediction: {sentiment}")
        print(
            "Positive probability:",
            f"{positive_probability:.2%}"
        )

    print(
        "\nNote: These examples are diagnostic probes,"
        " not the model's test-set score."
    )


# ==========================================
# MAIN TRAINING PROGRAM
# ==========================================

def main():
    df = load_data()

    X = df["review"]
    y = df["label"]

    # Hold out 20% of the data.
    # Grid search never sees the held-out labels.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nSearching for better Naive Bayes settings...")

    search = build_search()
    search.fit(X_train, y_train)

    candidate = search.best_estimator_

    print("\nBest CV macro F1:", search.best_score_)
    print("Best settings:", search.best_params_)

    candidate_eval = evaluate(
        candidate,
        X_test,
        y_test,
    )

    print_metrics(
        "NEW CANDIDATE - HELD-OUT TEST",
        candidate_eval,
        y_test,
    )

    # Evaluate existing model for diagnostic comparison.
    # This comparison is descriptive if the existing
    # model has previously seen any of these test rows.
    old_eval = None

    if MODEL_PATH.exists():
        try:
            old_model = joblib.load(MODEL_PATH)
            old_eval = evaluate(
                old_model,
                X_test,
                y_test,
            )

            print_metrics(
                "EXISTING MODEL - SAME ROWS",
                old_eval,
                y_test,
            )
        except Exception as exc:
            print(
                "\nExisting model could not be evaluated:",
                str(exc),
            )

    # Save detailed test predictions for inspection.
    output = pd.DataFrame({
        "review": X_test.to_numpy(),
        "actual_label": y_test.to_numpy(),
        "predicted_label": candidate_eval["predictions"],
    })

    output.to_csv(
        RESULTS_DIR / "sentiment_candidate_predictions.csv",
        index=False,
    )

    test_example_reviews(candidate)

    # Conservative promotion:
    # require at least 0.80 macro F1 on the held-out test.
    # If a comparable incumbent was evaluated, also require
    # no degradation in both accuracy and macro F1.
    promote = candidate_eval["macro_f1"] >= 0.80

    if old_eval is not None:
        promote = (
            promote
            and candidate_eval["macro_f1"]
                >= old_eval["macro_f1"]
            and candidate_eval["accuracy"]
                >= old_eval["accuracy"]
        )

    if not promote:
        print("\nMODEL NOT REPLACED.")
        print(
            "Candidate did not meet the promotion checks."
        )
        print(
            "Your existing sentiment model is unchanged."
        )
        return

    if MODEL_PATH.exists():
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        backup_path = MODEL_PATH.with_name(
            f"naive_bayes_sentiment_backup_{timestamp}.joblib"
        )

        shutil.copy2(MODEL_PATH, backup_path)

        print("\nBackup saved:", backup_path)

    joblib.dump(candidate, MODEL_PATH)

    print("\nNEW MODEL SAVED!")
    print("Model path:", MODEL_PATH)
    print(
        "Candidate held-out accuracy:",
        f"{candidate_eval['accuracy']:.2%}"
    )
    print(
        "Candidate held-out macro F1:",
        f"{candidate_eval['macro_f1']:.4f}"
    )
    print("\nRestart FastAPI to load the new model.")


if __name__ == "__main__":
    main()
