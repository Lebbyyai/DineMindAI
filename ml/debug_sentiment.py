
import joblib
import numpy as np

from pathlib import Path

# ============================================
# DINEMIND AI - SENTIMENT MODEL DIAGNOSTIC
# ============================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "models"
    / "naive_bayes_sentiment_model.joblib"
)

REVIEWS = [
    "The food was disgusting",
    "The food was disgusting.",
    "Disgusting food.",
    "The food was terrible.",
    "The service was awful.",
    "The food was delicious.",
    "The food was not good.",
    "The food was excellent.",
]

WORDS = [
    "disgusting",
    "terrible",
    "awful",
    "dirty",
    "bad",
    "delicious",
    "excellent",
    "good",
]

def main():
    print("=" * 60)
    print("DINEMIND AI - SENTIMENT DEBUGGER")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print("ERROR: Saved model not found!")
        print(MODEL_PATH)
        return

    model = joblib.load(MODEL_PATH)

    print("\nMODEL FILE:")
    print(MODEL_PATH)

    print("\nMODEL TYPE:")
    print(type(model).__name__)

    if not hasattr(model, "named_steps"):
        print(
            "\nThis saved object is not a standard "
            "scikit-learn Pipeline."
        )
        return

    print("\nPIPELINE STEPS:")
    for name, step in model.named_steps.items():
        print(f"  {name}: {type(step).__name__}")

    vectorizer = next(
        (
            step
            for step in model.named_steps.values()
            if hasattr(step, "vocabulary_")
        ),
        None,
    )

    classifier = next(
        (
            step
            for step in model.named_steps.values()
            if hasattr(step, "feature_log_prob_")
        ),
        None,
    )

    if vectorizer is None or classifier is None:
        print(
            "\nERROR: Could not locate the fitted "
            "vectorizer and Naive Bayes classifier."
        )
        return

    # ========================================
    # VOCABULARY CHECK
    # ========================================

    print("\n" + "=" * 60)
    print("WORD VOCABULARY CHECK")
    print("=" * 60)

    for word in WORDS:
        present = word in vectorizer.vocabulary_

        print(
            f"{word:<15} "
            f"{'FOUND' if present else 'NOT FOUND'}"
        )

    # ========================================
    # WORD CONTRIBUTIONS
    # ========================================

    print("\n" + "=" * 60)
    print("WORD CONTRIBUTIONS")
    print("=" * 60)

    classes = list(classifier.classes_)

    if 0 not in classes or 1 not in classes:
        print("Unexpected class labels:", classes)
        return

    negative_index = classes.index(0)
    positive_index = classes.index(1)

    for word in WORDS:
        index = vectorizer.vocabulary_.get(word)

        if index is None:
            print(f"{word:<15} Not in vocabulary")
            continue

        positive_log = classifier.feature_log_prob_[
            positive_index, index
        ]

        negative_log = classifier.feature_log_prob_[
            negative_index, index
        ]

        difference = positive_log - negative_log

        if difference > 0:
            direction = "POSITIVE"
        elif difference < 0:
            direction = "NEGATIVE"
        else:
            direction = "NEUTRAL"

        print(
            f"{word:<15} "
            f"Score: {difference:+.4f} "
            f"Direction: {direction}"
        )

    # ========================================
    # REVIEW PREDICTIONS
    # ========================================

    print("\n" + "=" * 60)
    print("REVIEW PREDICTIONS")
    print("=" * 60)

    probabilities = model.predict_proba(REVIEWS)
    predictions = model.predict(REVIEWS)

    model_classes = list(model.classes_)
    positive_index = model_classes.index(1)
    negative_index = model_classes.index(0)

    for review, prediction, probability in zip(
        REVIEWS,
        predictions,
        probabilities,
    ):
        positive = probability[positive_index] * 100
        negative = probability[negative_index] * 100

        label = (
            "POSITIVE"
            if int(prediction) == 1
            else "NEGATIVE"
        )

        print(f"\nReview: {review}")
        print(f"Prediction: {label}")
        print(f"Positive: {positive:.2f}%")
        print(f"Negative: {negative:.2f}%")

    # ========================================
    # FEATURE BREAKDOWN
    # ========================================

    print("\n" + "=" * 60)
    print("FEATURE BREAKDOWN")
    print("=" * 60)

    review = "The food was disgusting"

    transformed = vectorizer.transform([review])

    feature_names = vectorizer.get_feature_names_out()

    for index in transformed.nonzero()[1]:
        word = feature_names[index]
        weight = transformed[0, index]

        difference = (
            classifier.feature_log_prob_[positive_index, index]
            - classifier.feature_log_prob_[negative_index, index]
        )

        contribution = float(weight * difference)

        print(
            f"{word:<25} "
            f"Weight: {weight:.4f} "
            f"Contribution: {contribution:+.4f}"
        )

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

    print(
        "\nPositive contributions favor a positive"
        " classification; negative contributions"
        " favor a negative classification."
    )

    print(
        "\nThis script only inspects the model."
        " It does not modify or retrain anything."
    )


    # ========================================
    # TRAINING DATA WORD FREQUENCY
    # ========================================

    import pandas as pd

    data_path = (
        ROOT
        / "datasets"
        / "raw"
        / "sentiment"
        / "yelp_labelled.txt"
    )

    df = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=["review", "label"],
    )

    print("\n" + "=" * 60)
    print("TRAINING DATA WORD FREQUENCY")
    print("=" * 60)

    for word in [
        "disgusting",
        "terrible",
        "awful",
        "bad",
        "delicious",
        "excellent",
    ]:
        matching = df[
            df["review"].str.contains(
                rf"\b{word}\b",
                case=False,
                na=False,
                regex=True,
            )
        ]

        negative = int(
            (matching["label"] == 0).sum()
        )

        positive = int(
            (matching["label"] == 1).sum()
        )

        print(
            f"{word:<15} "
            f"Negative: {negative:<4} "
            f"Positive: {positive:<4}"
        )

    print("\nCLASS PRIOR PROBABILITIES")

    for label, log_prior in zip(
        classifier.classes_,
        classifier.class_log_prior_,
    ):
        sentiment = (
            "Positive"
            if int(label) == 1
            else "Negative"
        )

        print(
            f"{sentiment}: "
            f"{np.exp(log_prior):.2%}"
        )

if __name__ == "__main__":
    main(
        
    )