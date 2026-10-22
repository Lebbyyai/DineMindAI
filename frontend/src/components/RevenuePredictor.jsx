
import { useState } from "react";
import {
  ArrowLeft,
  TrendingUp,
  BrainCircuit,
  AlertCircle,
  LoaderCircle,
  RotateCcw,
  Info,
} from "lucide-react";

const API_URL = "http://127.0.0.1:8000";

const initialForm = {
  opening_year: "2010",
  opening_month: "5",
  city_group: "Big Cities",
  restaurant_type: "FC",
  P1: "4",
  P2: "5",
  P3: "4",
  P4: "4",
  P5: "2",
};

const months = [
  "January", "February", "March", "April",
  "May", "June", "July", "August",
  "September", "October", "November", "December",
];

export default function RevenuePredictor({ onBack }) {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function updateField(field, value) {
    setForm((previous) => ({
      ...previous,
      [field]: value,
    }));
    setResult(null);
    setError("");
  }

  function resetForm() {
    setForm(initialForm);
    setResult(null);
    setError("");
  }

  async function predictRevenue(event) {
    event.preventDefault();

    const numericalFields = [
      "opening_year",
      "opening_month",
      "P1", "P2", "P3", "P4", "P5",
    ];

    if (
      numericalFields.some(
        (field) =>
          form[field].trim() === "" ||
          !Number.isFinite(Number(form[field]))
      )
    ) {
      setError("Please enter valid numbers in all fields.");
      return;
    }

    const payload = {
      opening_year: Number(form.opening_year),
      opening_month: Number(form.opening_month),
      city_group: form.city_group,
      restaurant_type: form.restaurant_type,
      P1: Number(form.P1),
      P2: Number(form.P2),
      P3: Number(form.P3),
      P4: Number(form.P4),
      P5: Number(form.P5),
    };

    if (
      !Number.isInteger(payload.opening_year) ||
      payload.opening_year < 1900 ||
      payload.opening_year > 2026
    ) {
      setError("Opening year must be between 1900 and 2026.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        `${API_URL}/predict/revenue`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        const message =
          typeof data.detail === "string"
            ? data.detail
            : "Revenue prediction failed.";

        throw new Error(message);
      }

      if (
        typeof data.predicted_revenue !== "number" ||
        !Number.isFinite(data.predicted_revenue)
      ) {
        throw new Error("Invalid prediction returned by the API.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err instanceof TypeError
          ? "Cannot connect to FastAPI. Make sure your backend is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  const formattedPrediction =
    result?.predicted_revenue?.toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });

  return (
    <section className="revenue-page">
      <div className="revenue-container">

        <button
          type="button"
          className="revenue-back"
          onClick={onBack}
        >
          <ArrowLeft size={18} />
          Back to ML Solutions
        </button>

        <div className="revenue-heading">
          <span className="section-kicker">
            ML-01 · LINEAR REGRESSION
          </span>

          <h2>Restaurant Revenue Prediction</h2>

          <p>
            Explore how restaurant characteristics
            relate to estimated revenue using our
            trained Linear Regression model.
          </p>
        </div>

        <div className="revenue-layout">

          {/* INPUT FORM */}
          <div className="revenue-card">

            <div className="revenue-card-title">
              <TrendingUp size={24} />
              <h3>Restaurant Information</h3>
            </div>

            <form onSubmit={predictRevenue}>

              <div className="revenue-form-grid">

                <div className="revenue-field">
                  <label htmlFor="revenue-year">
                    Opening Year
                  </label>
                  <input
                    id="revenue-year"
                    type="number"
                    min="1900"
                    max="2026"
                    step="1"
                    required
                    value={form.opening_year}
                    onChange={(event) =>
                      updateField(
                        "opening_year",
                        event.target.value
                      )
                    }
                  />
                </div>

                <div className="revenue-field">
                  <label htmlFor="revenue-month">
                    Opening Month
                  </label>

                  <select
                    id="revenue-month"
                    value={form.opening_month}
                    onChange={(event) =>
                      updateField(
                        "opening_month",
                        event.target.value
                      )
                    }
                  >
                    {months.map((month, index) => (
                      <option
                        key={month}
                        value={index + 1}
                      >
                        {month}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="revenue-field">
                  <label htmlFor="revenue-city">
                    City Group
                  </label>

                  <select
                    id="revenue-city"
                    value={form.city_group}
                    onChange={(event) =>
                      updateField(
                        "city_group",
                        event.target.value
                      )
                    }
                  >
                    <option value="Big Cities">
                      Big Cities
                    </option>
                    <option value="Other">
                      Other
                    </option>
                  </select>
                </div>

                <div className="revenue-field">
                  <label htmlFor="revenue-type">
                    Restaurant Type
                  </label>

                  <select
                    id="revenue-type"
                    value={form.restaurant_type}
                    onChange={(event) =>
                      updateField(
                        "restaurant_type",
                        event.target.value
                      )
                    }
                  >
                    <option value="FC">FC</option>
                    <option value="IL">IL</option>
                    <option value="DT">DT</option>
                    <option value="MB">MB</option>
                  </select>
                </div>

              </div>

              <div className="revenue-feature-heading">
                <h4>Numerical Restaurant Features</h4>
                <p>
                  P1–P5 are anonymized attributes supplied
                  by the research dataset.
                </p>
              </div>

              <div className="revenue-form-grid">
                {["P1", "P2", "P3", "P4", "P5"].map(
                  (field) => (
                    <div
                      className="revenue-field"
                      key={field}
                    >
                      <label htmlFor={`revenue-${field}`}>
                        {field}
                      </label>

                      <input
                        id={`revenue-${field}`}
                        type="number"
                        step="any"
                        required
                        value={form[field]}
                        onChange={(event) =>
                          updateField(
                            field,
                            event.target.value
                          )
                        }
                      />
                    </div>
                  )
                )}
              </div>

              <div className="revenue-info">
                <Info size={18} />
                <p>
                  These fields are based on the original
                  dataset. Their complete business meanings
                  are not publicly disclosed.
                </p>
              </div>

              {error && (
                <div
                  className="revenue-error"
                  role="alert"
                >
                  <AlertCircle size={18} />
                  {error}
                </div>
              )}

              <div className="revenue-actions">

                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <LoaderCircle
                        size={18}
                        className="spin"
                      />
                      Predicting...
                    </>
                  ) : (
                    <>
                      <BrainCircuit size={18} />
                      Predict Revenue
                    </>
                  )}
                </button>

                <button
                  type="button"
                  className="revenue-clear"
                  onClick={resetForm}
                >
                  <RotateCcw size={16} />
                  Reset
                </button>

              </div>
            </form>
          </div>

          {/* PREDICTION RESULTS */}
          <div className="revenue-card">

            <div className="revenue-card-title">
              <BrainCircuit size={24} />
              <h3>Prediction Results</h3>
            </div>

            {!result ? (
              <div className="revenue-empty">
                <TrendingUp size={45} />

                <h4>Ready to Predict</h4>

                <p>
                  Enter the restaurant attributes,
                  then click Predict Revenue to
                  generate a model-based estimate.
                </p>
              </div>
            ) : (
              <div
                className="revenue-results"
                aria-live="polite"
              >
                <span className="revenue-result-label">
                  Estimated Restaurant Revenue
                </span>

                <div className="revenue-result-number">
                  {formattedPrediction}
                </div>

                <p className="revenue-unit">
                  {result.unit || "Dataset revenue units"}
                </p>

                <div className="revenue-summary">
                  <p>
                    <strong>Algorithm:</strong>{" "}
                    Linear Regression
                  </p>

                  <p>
                    <strong>City Group:</strong>{" "}
                    {form.city_group}
                  </p>

                  <p>
                    <strong>Restaurant Type:</strong>{" "}
                    {form.restaurant_type}
                  </p>

                  <p>
                    <strong>Opening:</strong>{" "}
                    {months[Number(form.opening_month) - 1]}{" "}
                    {form.opening_year}
                  </p>
                </div>

                <div className="revenue-warning">
                  <AlertCircle size={19} />

                  <div>
                    <strong>Experimental prediction</strong>
                    <p>
                      This model performed worse than
                      a mean-revenue baseline during
                      cross-validation. Its results
                      should not be used for actual
                      financial decisions.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <p className="revenue-disclaimer">
          Academic machine learning demonstration using
          the historical TFI restaurant revenue dataset.
          Revenue values are anonymized and must not
          be interpreted as Japanese yen, Philippine
          pesos, or actual financial forecasts.
        </p>
      </div>
    </section>
  );
}
