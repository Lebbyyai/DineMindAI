
import { useEffect, useState } from "react";
import {
  ArrowLeft,
  HeartHandshake,
  BrainCircuit,
  AlertCircle,
  LoaderCircle,
  RotateCcw,
} from "lucide-react";

const API_URL = "http://127.0.0.1:8000";

export default function SatisfactionPredictor({ onBack }) {
  const [customers, setCustomers] = useState([]);
  const [restaurants, setRestaurants] = useState([]);
  const [userId, setUserId] = useState("");
  const [restaurantId, setRestaurantId] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingOptions, setLoadingOptions] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    async function loadOptions() {
      try {
        const response = await fetch(
          `${API_URL}/satisfaction/options`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error("Unable to load customers and restaurants.");
        }

        const data = await response.json();

        setCustomers(data.customers || []);
        setRestaurants(data.restaurants || []);
        setUserId(data.customers?.[0] || "");
        setRestaurantId(
          String(data.restaurants?.[0]?.placeID || "")
        );
      } catch (err) {
        if (err.name !== "AbortError") {
          setError(
            "Unable to load options. Check your FastAPI server."
          );
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoadingOptions(false);
        }
      }
    }

    loadOptions();
    return () => controller.abort();
  }, []);

  async function predictSatisfaction(event) {
    event.preventDefault();

    if (!userId || !restaurantId) {
      setError("Please select a customer and restaurant.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        `${API_URL}/predict/satisfaction`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            user_id: userId,
            restaurant_id: Number(restaurantId),
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Prediction failed."
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err instanceof TypeError
          ? "Cannot connect to the FastAPI backend."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  function resetPrediction() {
    setResult(null);
    setError("");
  }

  const selectedRestaurant = restaurants.find(
    (item) => String(item.placeID) === restaurantId
  );

  return (
    <section className="satisfaction-page">
      <div className="satisfaction-container">
        <button
          type="button"
          className="satisfaction-back"
          onClick={onBack}
        >
          <ArrowLeft size={18} />
          Back to ML Solutions
        </button>

        <div className="satisfaction-heading">
          <span className="section-kicker">
            ML-04 · SUPPORT VECTOR MACHINE
          </span>

          <h2>Customer Satisfaction Prediction</h2>

          <p>
            Explore how a Support Vector Machine uses
            historical customer ratings and restaurant
            characteristics to estimate satisfaction.
          </p>
        </div>

        <div className="satisfaction-layout">
          <div className="satisfaction-card">
            <div className="satisfaction-card-title">
              <HeartHandshake size={24} />
              <h3>Customer & Restaurant Selection</h3>
            </div>

            {loadingOptions ? (
              <p>Loading available records...</p>
            ) : (
              <form onSubmit={predictSatisfaction}>
                <label htmlFor="satisfaction-user">
                  Select Customer
                </label>

                <select
                  id="satisfaction-user"
                  value={userId}
                  onChange={(event) => {
                    setUserId(event.target.value);
                    resetPrediction();
                  }}
                >
                  {customers.map((id) => (
                    <option key={id} value={id}>
                      {id}
                    </option>
                  ))}
                </select>

                <label htmlFor="satisfaction-restaurant">
                  Select Restaurant
                </label>

                <select
                  id="satisfaction-restaurant"
                  value={restaurantId}
                  onChange={(event) => {
                    setRestaurantId(event.target.value);
                    resetPrediction();
                  }}
                >
                  {restaurants.map((restaurant) => (
                    <option
                      key={restaurant.placeID}
                      value={restaurant.placeID}
                    >
                      {restaurant.name} ({restaurant.placeID})
                    </option>
                  ))}
                </select>

                <div className="satisfaction-details">
                  <strong>Selected Restaurant</strong>
                  <p>
                    {selectedRestaurant?.name ||
                      "No restaurant selected"}
                  </p>
                  <small>
                    Historical customer satisfaction dataset
                  </small>
                </div>

                <div className="satisfaction-actions">
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={
                      loading ||
                      loadingOptions ||
                      !userId ||
                      !restaurantId
                    }
                  >
                    {loading ? (
                      <>
                        <LoaderCircle size={18} className="spin" />
                        Predicting...
                      </>
                    ) : (
                      <>
                        <BrainCircuit size={18} />
                        Predict Satisfaction
                      </>
                    )}
                  </button>

                  <button
                    type="button"
                    className="satisfaction-clear"
                    onClick={resetPrediction}
                  >
                    <RotateCcw size={16} />
                    Clear
                  </button>
                </div>
              </form>
            )}

            {error && (
              <div className="satisfaction-error" role="alert">
                <AlertCircle size={18} />
                {error}
              </div>
            )}
          </div>

          <div className="satisfaction-card">
            <div className="satisfaction-card-title">
              <BrainCircuit size={24} />
              <h3>SVM Prediction Results</h3>
            </div>

            {!result ? (
              <div className="satisfaction-empty">
                <HeartHandshake size={45} />
                <h4>Ready to Predict</h4>
                <p>
                  Choose a customer and restaurant to
                  generate a satisfaction prediction.
                </p>
              </div>
            ) : (
              <div aria-live="polite">
                <span className="satisfaction-result-label">
                  Predicted Satisfaction
                </span>

                <div
                  className={
                    result.prediction === "Satisfied"
                      ? "satisfaction-result is-satisfied"
                      : "satisfaction-result is-not-satisfied"
                  }
                >
                  {result.prediction}
                </div>

                <div className="satisfaction-summary">
                  <p>
                    <strong>Algorithm:</strong>{" "}
                    {result.algorithm}
                  </p>
                  <p>
                    <strong>Customer:</strong>{" "}
                    {result.user_id}
                  </p>
                  <p>
                    <strong>Restaurant:</strong>{" "}
                    {selectedRestaurant?.name ||
                      result.restaurant_id}
                  </p>
                </div>

                <p className="satisfaction-explanation">
                  The SVM classified this customer-restaurant
                  combination using patterns learned from
                  historical satisfaction ratings.
                  This is an estimate, not a confirmed
                  customer opinion.
                </p>
              </div>
            )}
          </div>
        </div>

        <p className="satisfaction-disclaimer">
          Educational demonstration using historical restaurant
          ratings from Mexico. This model has not been validated
          for Japanese restaurants or real-world operational use.
        </p>
      </div>
    </section>
  );
}
