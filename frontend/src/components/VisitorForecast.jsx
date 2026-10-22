
import { useEffect, useState } from "react";
import {
  ArrowLeft,
  ChartColumnIncreasing,
  BrainCircuit,
  CalendarDays,
  AlertCircle,
  LoaderCircle,
  RotateCcw,
  Database,
  Users,
} from "lucide-react";

const API_URL = "http://127.0.0.1:8000";
const DEFAULT_RESTAURANT = "air_00a91d42b08b08d9";
const DEFAULT_DATE = "2017-03-15";

function normalizeRestaurants(data) {
  const list = Array.isArray(data)
    ? data
    : Array.isArray(data?.restaurants)
      ? data.restaurants
      : [];

  return list
    .map((item) => {
      if (typeof item === "string") {
        return { id: item, label: item };
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

function getNumericValue(data, keys) {
  for (const key of keys) {
    const value = data?.[key];

    if (value !== undefined && value !== null && value !== "") {
      const number = Number(value);

      if (Number.isFinite(number)) {
        return number;
      }
    }
  }

  return null;
}

export default function VisitorForecast({ onBack }) {
  const [restaurants, setRestaurants] = useState([]);
  const [restaurantId, setRestaurantId] =
    useState(DEFAULT_RESTAURANT);
  const [predictionDate, setPredictionDate] =
    useState(DEFAULT_DATE);

  const [loadingRestaurants, setLoadingRestaurants] =
    useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
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
          throw new Error("Could not load restaurant records.");
        }

        const data = await response.json();
        const options = normalizeRestaurants(data);

        if (!options.length) {
          throw new Error("No restaurants were returned.");
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

  function clearResult() {
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
    clearResult();
  }

  async function forecastVisitors(event) {
    event.preventDefault();

    if (!restaurantId.trim() || !predictionDate) {
      setError("Please select a restaurant and a date.");
      return;
    }

    if (
      predictionDate < "2017-03-01" ||
      predictionDate > "2017-04-22"
    ) {
      setError(
        "Please choose a date between March 1 and April 22, 2017."
      );
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        `${API_URL}/predict/visitors`,
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
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Visitor prediction failed."
        );
      }

      const predicted = getNumericValue(data, [
        "predicted_visitors",
        "predicted_visitor_count",
        "prediction",
        "visitor_prediction",
      ]);

      const actual = getNumericValue(data, [
        "actual_visitors",
        "actual_visitor_count",
        "actual_count",
      ]);

      if (predicted === null) {
        throw new Error(
          "The API responded, but no numeric visitor prediction was found. Check the response field names in FastAPI."
        );
      }

      setResult({
        predicted: Math.max(0, predicted),
        actual,
        raw: data,
      });
    } catch (err) {
      setError(
        err instanceof TypeError
          ? "Cannot connect to FastAPI. Make sure your backend server is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  const selectedRestaurant = restaurants.find(
    (item) => item.id === restaurantId
  );

  const predicted = result?.predicted;
  const actual = result?.actual;

  const chartMaximum = result
    ? Math.max(predicted, actual ?? 0, 1)
    : 1;

  const predictedPercent = result
    ? (predicted / chartMaximum) * 100
    : 0;

  const actualPercent =
    result && actual !== null
      ? (actual / chartMaximum) * 100
      : 0;

  const absoluteError =
    result && actual !== null
      ? Math.abs(predicted - actual)
      : null;

  return (
    <section className="forecast-page">
      <div className="forecast-container">
        <button
          type="button"
          className="forecast-back"
          onClick={onBack}
        >
          <ArrowLeft size={18} />
          Back to ML Solutions
        </button>

        <div className="forecast-heading">
          <span className="section-kicker">
            ML-06 · ARTIFICIAL NEURAL NETWORK
          </span>

          <h2>Restaurant Visitor Forecasting</h2>

          <p>
            Use our trained Artificial Neural Network to
            estimate daily visitor counts for restaurants
            in a historical Japanese dataset.
          </p>
        </div>

        <div className="forecast-layout">
          <div className="forecast-card">
            <div className="forecast-card-title">
              <CalendarDays size={24} />
              <h3>Restaurant & Date</h3>
            </div>

            <form onSubmit={forecastVisitors}>
              <label htmlFor="forecast-restaurant">
                Select Japanese Restaurant
              </label>

              {restaurants.length > 0 ? (
                <select
                  id="forecast-restaurant"
                  value={restaurantId}
                  disabled={loadingRestaurants}
                  onChange={(event) => {
                    setRestaurantId(event.target.value);
                    clearResult();
                  }}
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
                  id="forecast-restaurant"
                  type="text"
                  value={restaurantId}
                  required
                  placeholder="Enter historical restaurant ID"
                  onChange={(event) => {
                    setRestaurantId(event.target.value);
                    clearResult();
                  }}
                />
              )}

              {restaurantError && (
                <p className="forecast-helper">
                  {restaurantError}
                </p>
              )}

              <label htmlFor="forecast-date">
                Historical Prediction Date
              </label>

              <input
                id="forecast-date"
                type="date"
                min="2017-03-01"
                max="2017-04-22"
                required
                value={predictionDate}
                onChange={(event) => {
                  setPredictionDate(event.target.value);
                  clearResult();
                }}
              />

              <div className="forecast-info">
                <Database size={19} />

                <div>
                  <strong>Historical evaluation period</strong>
                  <p>
                    March 1 to April 22, 2017. This
                    educational demo estimates visitor
                    counts for past dates, not future
                    restaurant demand.
                  </p>
                </div>
              </div>

              {error && (
                <div className="forecast-error" role="alert">
                  <AlertCircle size={18} />
                  {error}
                </div>
              )}

              <div className="forecast-actions">
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={loading || !restaurantId.trim()}
                >
                  {loading ? (
                    <>
                      <LoaderCircle
                        size={18}
                        className="spin"
                      />
                      Forecasting...
                    </>
                  ) : (
                    <>
                      <BrainCircuit size={18} />
                      Forecast Visitors
                    </>
                  )}
                </button>

                <button
                  type="button"
                  className="forecast-reset"
                  onClick={resetForm}
                >
                  <RotateCcw size={16} />
                  Reset
                </button>
              </div>
            </form>
          </div>

          <div className="forecast-card">
            <div className="forecast-card-title">
              <ChartColumnIncreasing size={24} />
              <h3>ANN Forecast Results</h3>
            </div>

            {!result ? (
              <div className="forecast-empty">
                <Users size={46} />
                <h4>Ready to Forecast</h4>
                <p>
                  Choose a restaurant and historical date
                  to generate a visitor estimate.
                </p>
              </div>
            ) : (
              <div aria-live="polite">
                <span className="forecast-result-label">
                  Predicted Daily Visitors
                </span>

                <div className="forecast-result-number">
                  {Math.round(predicted).toLocaleString()}
                </div>

                <p className="forecast-result-unit">
                  estimated visitors
                </p>

                <div className="forecast-comparison">
                  <h4>Predicted vs. Actual Visitors</h4>

                  <div className="forecast-bar-label">
                    <span>ANN Prediction</span>
                    <strong>{predicted.toFixed(1)}</strong>
                  </div>

                  <div className="forecast-bar-track">
                    <div
                      className="forecast-bar predicted"
                      style={{
                        width: `${predictedPercent}%`,
                      }}
                    />
                  </div>

                  {actual !== null ? (
                    <>
                      <div className="forecast-bar-label">
                        <span>Actual Visitors</span>
                        <strong>{actual.toFixed(0)}</strong>
                      </div>

                      <div className="forecast-bar-track">
                        <div
                          className="forecast-bar actual"
                          style={{
                            width: `${actualPercent}%`,
                          }}
                        />
                      </div>

                      <p className="forecast-error-value">
                        Absolute error for this example:{" "}
                        <strong>
                          {absoluteError.toFixed(1)} visitors
                        </strong>
                      </p>
                    </>
                  ) : (
                    <p className="forecast-helper">
                      Actual visitor count was not returned
                      for this restaurant and date.
                    </p>
                  )}
                </div>

                <div className="forecast-summary">
                  <p>
                    <strong>Model:</strong>{" "}
                    Artificial Neural Network
                  </p>

                  <p>
                    <strong>Restaurant:</strong>{" "}
                    {selectedRestaurant?.label || restaurantId}
                  </p>

                  <p>
                    <strong>Date:</strong>{" "}
                    {predictionDate}
                  </p>
                </div>

                <p className="forecast-explanation">
                  This estimate is based on historical
                  restaurant visitor patterns and
                  calendar-related features. Differences
                  between predicted and actual counts
                  show that the model is not perfectly
                  accurate.
                </p>
              </div>
            )}
          </div>
        </div>

        <p className="forecast-disclaimer">
          Academic demonstration using the historical
          Japanese Recruit Restaurant Visitor Forecasting
          dataset. These results are retrospective
          estimates and are not validated predictions
          for current or future restaurant operations.
        </p>
      </div>
    </section>
  );
}
