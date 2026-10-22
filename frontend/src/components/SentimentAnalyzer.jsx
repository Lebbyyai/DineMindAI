
import { useState } from "react";
import {
  ArrowLeft,
  BrainCircuit,
  CheckCircle2,
  AlertCircle,
  LoaderCircle,
} from "lucide-react";

const API_URL = "http://127.0.0.1:8000";

export default function SentimentAnalyzer({ onBack }) {
  const [review, setReview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function analyzeReview(event) {
    event.preventDefault();

    if (!review.trim()) {
      setError("Please enter a restaurant review.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        `${API_URL}/predict/sentiment`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ review: review.trim() }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Prediction request failed."
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err instanceof TypeError
          ? "Cannot connect to FastAPI. Check that the backend is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  function resetForm() {
    setReview("");
    setResult(null);
    setError("");
  }

  return (
    <section className="sentiment-page">
      <div className="sentiment-container">
        <button
          type="button"
          className="sentiment-back"
          onClick={onBack}
        >
          <ArrowLeft size={17} />
          Back to ML Solutions
        </button>

        <div className="sentiment-header">
          <span className="section-kicker">
            ML-02 • NAIVE BAYES
          </span>

          <h2>Customer Review Sentiment Analyzer</h2>

          <p>
            Enter a restaurant review and discover whether
            our trained machine learning model identifies
            the feedback as positive or negative.
          </p>
        </div>

        <div className="sentiment-layout">
          <div className="sentiment-card">
            <div className="sentiment-card-heading">
              <BrainCircuit size={22} />
              <h3>Analyze Customer Feedback</h3>
            </div>

            <form onSubmit={analyzeReview}>
              <label htmlFor="customer-review">
                Customer Review
              </label>

              <textarea
                id="customer-review"
                value={review}
                maxLength={3000}
                onChange={(event) => {
                  setReview(event.target.value);
                  setResult(null);
                  setError("");
                }}
                placeholder="Example: The food was delicious and the service was excellent!"
                rows={6}
              />

              <div className="review-count">
                {review.length}/3000 characters
              </div>

              {error && (
                <div className="sentiment-error" role="alert">
                  <AlertCircle size={18} />
                  {error}
                </div>
              )}

              <div className="sentiment-actions">
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={loading || !review.trim()}
                >
                  {loading ? (
                    <>
                      <LoaderCircle size={18} className="spin" />
                      Analyzing...
                    </>
                  ) : (
                    "Analyze Sentiment"
                  )}
                </button>

                <button
                  type="button"
                  className="sentiment-reset"
                  onClick={resetForm}
                >
                  Clear
                </button>
              </div>
            </form>
          </div>

          <div className="sentiment-card">
            <div className="sentiment-card-heading">
              <CheckCircle2 size={22} />
              <h3>Prediction Results</h3>
            </div>

            {!result ? (
              <div className="sentiment-empty">
                <BrainCircuit size={42} />
                <h4>Ready to Analyze</h4>
                <p>
                  Enter a customer review and click Analyze
                  Sentiment to see your prediction.
                </p>
              </div>
            ) : (
              <div aria-live="polite">
                <span className="result-label">
                  Predicted Sentiment
                </span>

                <div
                  className={
                    result.sentiment === "Positive"
                      ? "sentiment-result positive"
                      : "sentiment-result negative"
                  }
                >
                  {result.sentiment}
                </div>

                <h4>Model-estimated probabilities</h4>

                {[
                  ["Positive", result.probabilities.positive],
                  ["Negative", result.probabilities.negative],
                ].map(([name, probability]) => (
                  <div className="probability-item" key={name}>
                    <div className="probability-label">
                      <span>{name}</span>
                      <strong>
                        {(probability * 100).toFixed(2)}%
                      </strong>
                    </div>

                    <div className="probability-track">
                      <div
                        className={
                          name === "Positive"
                            ? "probability-fill positive-fill"
                            : "probability-fill negative-fill"
                        }
                        style={{
                          width: `${Math.max(
                            0,
                            Math.min(probability * 100, 100)
                          )}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}

                <p className="result-note">
                  This result was generated by our trained
                  Multinomial Naive Bayes model using
                  TF-IDF text features. Probabilities
                  are estimates, not guarantees.
                </p>
              </div>
            )}
          </div>
        </div>

        <p className="sentiment-disclaimer">
          Educational demonstration trained on historical
          English-language Yelp reviews. Not validated
          for Japanese-language customer feedback.
        </p>
      </div>
    </section>
  );
}
