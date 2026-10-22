
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from datasets import load_dataset

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

OLD_MODEL = ROOT / "models" / "naive_bayes_sentiment_model.joblib"
NEW_MODEL = ROOT / "models" / "naive_bayes_sentiment_candidate.joblib"

SMALL_DATA = (
    ROOT / "datasets" / "raw"
    / "sentiment" / "yelp_labelled.txt"
)

DATASET_NAME = "ajay232/yelp_polarity"


def get_reviews(split, samples_per_class, seed):
    print(f"\nLoading {split} reviews...")

    dataset = load_dataset(
        DATASET_NAME,
        split=split,
        streaming=True,
    )

    dataset = dataset.shuffle(
        seed=seed,
        buffer_size=5000,
    )

    counts = {0: 0, 1: 0}
    records = []

    for row in dataset:
        label = int(row["label"])
        review = str(row["text"]).strip()

        if label not in counts or not review:
            continue

        if counts[label] >= samples_per_class:
            continue

        records.append({
            "review": review,
            "label": label,
        })

        counts[label] += 1

        if all(
            count >= samples_per_class
            for count in counts.values()
        ):
            break

    if any(
        count < samples_per_class
        for count in counts.values()
    ):
        raise RuntimeError(
            f"Not enough records collected: {counts}"
        )

    result = pd.DataFrame(records)

    print("Collected:", len(result))
    print("Labels:", counts)

    return result


def clean_data(df):
    df = df.copy()
    df["review"] = df["review"].astype(str).str.strip()
    df["key"] = df["review"].str.casefold()

    df = df[df["review"].str.len() > 0]

    # Remove reviews with conflicting labels.
    conflicts = df.groupby("key")["label"].nunique()
    conflicting_keys = conflicts[conflicts > 1].index

    df = df[~df["key"].isin(conflicting_keys)]

    return df.drop_duplicates("key").reset_index(drop=True)


def main():
    # Larger Yelp dataset, separate official splits.
    train = get_reviews(
        "train",
        samples_per_class=10000,
        seed=42,
    )

    test = get_reviews(
        "test",
        samples_per_class=1000,
        seed=99,
    )

    # Preserve and incorporate the existing short reviews.
    original = pd.read_csv(
        SMALL_DATA,
        sep="\t",
        header=None,
        names=["review", "label"],
    )

    train = clean_data(
        pd.concat([train, original], ignore_index=True)
    )
    test = clean_data(test)

    # Prevent exact review overlap between train and test.
    train = train[
        ~train["key"].isin(set(test["key"]))
    ].reset_index(drop=True)

    print("\nTraining size:", len(train))
    print("Test size:", len(test))

    pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB()),
    ])

    parameters = {
        "vectorizer": [
            CountVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
            ),
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
        ],
        "classifier__alpha": [
            0.1, 0.5, 1.0, 2.0
        ],
    }

    search = GridSearchCV(
        pipeline,
        parameters,
        scoring="f1_macro",
        cv=StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=42,
        ),
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    print("\nTraining expanded Naive Bayes model...")

    search.fit(
        train["review"],
        train["label"],
    )

    model = search.best_estimator_

    print("\nBest settings:", search.best_params_)
    print("Cross-validation F1:", round(search.best_score_, 4))

    X_test = test["review"]
    y_test = test["label"]

    prediction = model.predict(X_test)

    new_accuracy = accuracy_score(y_test, prediction)
    new_f1 = f1_score(
        y_test, prediction, average="macro"
    )

    print("\nNEW MODEL")
    print("Accuracy:", f"{new_accuracy:.2%}")
    print("Macro F1:", f"{new_f1:.4f}")

    print(classification_report(
        y_test,
        prediction,
        target_names=["Negative", "Positive"],
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, prediction))

    if OLD_MODEL.exists():
        old = joblib.load(OLD_MODEL)
        old_prediction = old.predict(X_test)

        print("\nEXISTING MODEL - SAME TEST REVIEWS")
        print(
            "Accuracy:",
            f"{accuracy_score(y_test, old_prediction):.2%}"
        )
        print(
            "Macro F1:",
            f"{f1_score(y_test, old_prediction, average='macro'):.4f}"
        )

    examples = [
        "The food was disgusting",
        "The service was terrible.",
        "The restaurant was dirty.",
        "The food was delicious.",
        "The staff were excellent.",
        "The food was not good.",
    ]

    print("\nEXAMPLE PREDICTIONS")

    probabilities = model.predict_proba(examples)
    positive_index = list(model.classes_).index(1)

    for review, row in zip(examples, probabilities):
        positive = row[positive_index]
        label = "Positive" if positive >= 0.5 else "Negative"

        print(
            f"{review}\n"
            f"  {label} | Positive: {positive:.2%}"
        )

    # Save separately. Do not overwrite the live model.
    joblib.dump(model, NEW_MODEL)

    print("\nCandidate model saved:")
    print(NEW_MODEL)
    print("\nExisting production model was NOT replaced.")


if __name__ == "__main__":
    main()
