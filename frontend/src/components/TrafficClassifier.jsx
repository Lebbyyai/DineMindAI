
import { useEffect, useState } from "react";
import {
  ArrowLeft,
  Users,
  BrainCircuit,
  CalendarDays,
  AlertCircle,
  LoaderCircle,
  RotateCcw,
  Database,
} from "lucide-react";

const API_URL = "http://127.0.0.1:8000";

const DEFAULT_RESTAURANT = "air_00a91d42b08b08d9";
const DEFAULT_DATE = "2017-03-15";

function normalizeRestaurants(payload) {
  const items = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.restaurants)
      ? payload.restaurants
      : [];

  return items
    .map((item) => {
      if (typeof item === "string") {
        return {
          id: item,
          label: item,
        };
      }

      const id =
        item.restaurant_id ??
        item.air_store_id ??
        item.id;

      if (!id) return null;

      const details = [
        item.air_genre_name,
        item.air_area_name,
      ].filter(Boolean);

      return {
        id: String(id),
        label: details.length
          ? `${details.join(" · ")} (${id})`
          : String(id),
      };
    })
    .filter(Boolean);
}

function getTrafficLabel(data) {
  const value =
    data.predicted_traffic ??
    data.prediction ??
    data.traffic_class ??
    data.predicted_class ??
    data.traffic_prediction;

  if (value === undefined || value === null) {
    return null;
  }

  return String(value);
}

export default function TrafficClassifier({ onBack }) {
  const [restaurants, setRestaurants] = useState([]);
  const [restaurantId, setRestaurantId] = useState(
    DEFAULT_RESTAURANT
  );
  const [predictionDate, setPredictionDate] = useState(
    DEFAULT_DATE
  );

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingRestaurants, setLoadingRestaurants] =
    useState(true);
  const [error, setError] = useState("");
  const [restaurantError, setRestaurantError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    async function loadRestaurants() {
      try {
        const response = await fetch(
          `${API_URL}/restaurants/japan`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error("Unable to load restaurants.");
        }

        const data = await response.json();
        const options = normalizeRestaurants(data);

        if (options.length === 0) {
          throw new Error("No restaurant records available.");
        }

        setRestaurants(options);

        if (!options.some((r) => r.id === DEFAULT_RESTAURANT)) {
          setRestaurantId(options[0].id);
        }
      } catch (err) {
        if (err.name !== "AbortError") {
          setRestaurantError(
            "Restaurant list unavailable. You can enter a valid restaurant ID manually."
          );
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoadingRestaurants(false);
        }
      }
    }

    loadRestaurants();

    return () => controller.abort();
  }, []);

  function clearPrediction() {
    setResult(null);
    setError("");
  }

  function resetForm() {
    setRestaurantId(
      restaurants.some((r) => r.id === DEFAULT_RESTAURANT)
        ? DEFAULT_RESTAURANT
        : restaurants[0]?.id || DEFAULT_RESTAURANT
    );
    setPredictionDate(DEFAULT_DATE);
    clearPrediction();
  }

  async function classifyTraffic(event) {
    event.preventDefault();

    if (!restaurantId.trim() || !predictionDate) {
      setError("Please select a restaurant and date.");
      return;
    }

    if (
      predictionDate < "2017-03-01" ||
      predictionDate > "2017-04-22"
    ) {
      setError(
        "Choose a date from March 1 to April 22, 2017."
      );
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        `${API_URL}/predict/traffic`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            restaurant_id: restaurantId.trim(),
            prediction_date: predictionDate,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        const message =
          typeof data.detail === "string"
            ? data.detail
            : "Traffic prediction failed.";

        throw new Error(message);
      }

      const traffic = getTrafficLabel(data);

      if (!traffic) {
        throw new Error(
          "The API responded, but no traffic classification was found."
        );
      }

      setResult({
        ...data,
        displayTraffic: traffic,
      });
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

  const trafficClass = result?.displayTraffic
    ?.toLowerCase()
    .includes("high")
      ? "high"
      : result?.displayTraffic
          ?.toLowerCase()
          .includes("medium")
        ? "medium"
        : "low";

  const selectedRestaurant = restaurants.find(
    (r) => r.id === restaurantId
  );

  const actualVisitors =
    result?.actual_visitors ??
    result?.actual_visitor_count ??
    result?.actual_count;

  return (
    <section className="traffic-page">
      <div className="traffic-container">
        <button
          type="button"
          className="traffic-back"
          onClick={onBack}
        >
          <ArrowLeft size={18} />
          Back to ML Solutions
        </button>

        <div className="traffic-heading">
          <span className="section-kicker">
            ML-05 · DECISION TREE
          </span>

          <h2>Restaurant Traffic Classification</h2>

          <p>
            Classify historical visitor demand as Low,
            Medium, or High using our trained Decision Tree
            machine learning model.
          </p>
        </div>

        <div className="traffic-layout">
          <div className="traffic-card">
            <div className="traffic-card-title">
              <CalendarDays size={24} />
              <h3>Restaurant & Date Selection</h3>
            </div>

            <form onSubmit={classifyTraffic}>
              <label htmlFor="traffic-restaurant">
                Japanese Restaurant
              </label>

              {restaurants.length > 0 ? (
                <select
                  id="traffic-restaurant"
                  value={restaurantId}
                  onChange={(event) => {
                    setRestaurantId(event.target.value);
                    clearPrediction();
                  }}
                  disabled={loadingRestaurants}
                >
                  {restaurants.map((restaurant) => (
                    <option
                      key={restaurant.id}
                      value={restaurant.id}
                    >
                      {restaurant.label}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  id="traffic-restaurant"
                  type="text"
                  value={restaurantId}
                  onChange={(event) => {
                    setRestaurantId(event.target.value);
                    clearPrediction();
                  }}
                  placeholder="Enter historical restaurant ID"
                  required
                />
              )}

              {restaurantError && (
                <p className="traffic-helper">
                  {restaurantError}
                </p>
              )}

              <label htmlFor="traffic-date">
                Historical Prediction Date
              </label>

              <input
                id="traffic-date"
                type="date"
                min="2017-03-01"
                max="2017-04-22"
                value={predictionDate}
                required
                onChange={(event) => {
                  setPredictionDate(event.target.value);
                  clearPrediction();
                }}
              />

              <div className="traffic-info">
                <Database size={19} />

                <div>
                  <strong>Historical test period</strong>
                  <p>
                    March 1, 2017 to April 22, 2017.
                    This tool analyzes past restaurant
                    records and does not forecast current
                    restaurant traffic.
                  </p>
                </div>
              </div>

              {error && (
                <div className="traffic-error" role="alert">
                  <AlertCircle size={18} />
                  {error}
                </div>
              )}

              <div className="traffic-actions">
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={loading || !restaurantId}
                >
                  {loading ? (
                    <>
                      <LoaderCircle
                        size={18}
                        className="spin"
                      />
                      Classifying...
                    </>
                  ) : (
                    <>
                      <BrainCircuit size={18} />
                      Classify Traffic
                    </>
                  )}
                </button>

                <button
                  type="button"
                  className="traffic-reset"
                  onClick={resetForm}
                >
                  <RotateCcw size={16} />
                  Reset
                </button>
              </div>
            </form>
          </div>

          <div className="traffic-card">
            <div className="traffic-card-title">
              <Users size={24} />
              <h3>Decision Tree Results</h3>
            </div>

            {!result ? (
              <div className="traffic-empty">
                <Users size={46} />
                <h4>Ready to Classify</h4>

                <p>
                  Select a historical restaurant and date,
                  then click Classify Traffic to see
                  the prediction.
                </p>
              </div>
            ) : (
              <div aria-live="polite">
                <span className="traffic-result-label">
                  Predicted Traffic Level
                </span>

                <div
                  className={`traffic-result ${trafficClass}`}
                >
                  {result.displayTraffic}
                </div>

                <div className="traffic-summary">
                  <p>
                    <strong>Algorithm:</strong>{" "}
                    Decision Tree
                  </p>

                  <p>
                    <strong>Restaurant:</strong>{" "}
                    {selectedRestaurant?.label || restaurantId}
                  </p>

                  <p>
                    <strong>Date:</strong>{" "}
                    {predictionDate}
                  </p>

                  {actualVisitors !== undefined &&
                    actualVisitors !== null && (
                      <p>
                        <strong>Actual Visitors:</strong>{" "}
                        {actualVisitors}
                      </p>
                    )}
                </div>

                <div className="traffic-explanation">
                  <strong>
                    What does this result mean?
                  </strong>

                  <p>
                    The Decision Tree uses historical
                    patterns to classify the restaurant's
                    visitor demand into a traffic category.
                    Its output is a model prediction,
                    not a guarantee of the actual
                    traffic level.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        <p className="traffic-disclaimer">
          Educational demonstration using the historical
          Japanese Recruit Restaurant Visitor Forecasting
          dataset. The model was evaluated on a historical
          holdout period and is not validated for current
          or future restaurant operations.
        </p>
      </div>
    </section>
  );
}
