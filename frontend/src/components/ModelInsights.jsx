
import { useState } from "react";
import {
  ArrowLeft,
  BrainCircuit,
  Database,
  Info,
  CheckCircle2,
  AlertTriangle,
  BarChart3,
} from "lucide-react";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell,
} from "recharts";

// ============================================
// VERIFIED PROJECT EVALUATION RESULTS
// ============================================

// Metrics are historical results from this project's
// training and evaluation runs.
//
// Accuracy, F1, MAE, RMSE, and R2 measure different
// properties. Never treat them as interchangeable.

const modelData = [
  {
    id: "revenue",
    number: "01",
    name: "Revenue Prediction",
    algorithm: "Linear Regression",
    task: "Regression",
    dataset: "TFI Restaurant Revenue Prediction",
    region: "Turkey",
    purpose: "Estimate restaurant revenue from structured attributes.",
    metricName: "Mean Absolute Error",
    metricValue: "1,825,871.90",
    metrics: [
      ["MAE", "1,825,871.90"],
      ["RMSE", "2,581,812.46"],
      ["R²", "-0.0118"],
    ],
    explanation:
      "Linear Regression models the relationship between restaurant attributes and a numerical revenue target.",
    limitation:
      "Only 137 source training records were available. The model's cross-validated R² was negative, and its MAE was worse than a mean-value baseline. Revenue units should not be treated as a confirmed currency.",
    chartType: "regression",
    chartData: [
      { name: "Linear Regression", mae: 1825871.9 },
      { name: "Mean Baseline", mae: 1681468.23 },
    ],
    chartTitle: "Cross-validated MAE versus baseline",
    chartNote:
      "Lower MAE is better. The baseline performed better than Linear Regression.",
  },
  {
    id: "sentiment",
    number: "02",
    name: "Review Sentiment",
    algorithm: "Naive Bayes",
    task: "Binary classification",
    dataset:
      "Yelp Polarity and UCI Sentiment Labelled Sentences",
    region: "United States / English-language reviews",
    purpose: "Classify restaurant feedback as positive or negative.",
    metricName: "Test Accuracy",
    metricValue: "90.95%",
    metrics: [
      ["Accuracy", "90.95%"],
      ["Macro F1", "0.9095"],
      ["Test Reviews", "2,000"],
    ],
    explanation:
      "Multinomial Naive Bayes learns associations between review text features and positive or negative labels.",
    limitation:
      "Evaluation used a balanced sample of longer Yelp Polarity reviews. Performance may differ for short reviews, Japanese-language text, or restaurants outside the dataset. Probabilities are model estimates.",
    chartType: "confusion",
    confusion: [
      [925, 75],
      [106, 894],
    ],
    chartTitle: "Sentiment confusion matrix",
    chartNote:
      "Rows represent actual labels; columns represent predicted labels. 925 negative and 894 positive reviews were classified correctly.",
  },
  {
    id: "recommendations",
    number: "03",
    name: "Restaurant Recommendations",
    algorithm: "K-Nearest Neighbors",
    task: "Content-based recommendation",
    dataset: "UCI Restaurant & Consumer Data",
    region: "Mexico",
    purpose:
      "Recommend restaurants with similar cuisine, price, and ambience attributes.",
    metricName: "Source Restaurants",
    metricValue: "130",
    metrics: [
      ["Restaurants", "130"],
      ["Historical Ratings", "1,161"],
      ["Cuisine Entries", "916"],
    ],
    explanation:
      "A nearest-neighbor model finds restaurants whose encoded attributes are most similar to a user's selected preferences, using cosine similarity.",
    limitation:
      "No independently validated recommendation accuracy or ranking-quality metric has been established. A 100% attribute similarity score does not mean guaranteed customer satisfaction. Restaurant records are from Mexico, not Japan.",
    chartType: "none",
    chartTitle: "Recommendation model information",
  },
  {
    id: "satisfaction",
    number: "04",
    name: "Customer Satisfaction",
    algorithm: "Support Vector Machine",
    task: "Binary classification",
    dataset: "UCI Restaurant & Consumer Data",
    region: "Mexico",
    purpose:
      "Estimate whether a customer will be satisfied with a restaurant.",
    metricName: "Test Accuracy",
    metricValue: "71.24%",
    metrics: [
      ["Accuracy", "71.24%"],
      ["Correct Predictions", "166"],
      ["Test Records", "233"],
    ],
    explanation:
      "An RBF Support Vector Machine separates satisfied and not-satisfied examples using restaurant attributes and historical rating patterns.",
    limitation:
      "Results depend on the available historical user and restaurant ratings. Predictions may not generalize to new customers, new restaurants, or Japanese dining markets.",
    chartType: "confusion",
    confusion: [
      [94, 41],
      [26, 72],
    ],
    chartTitle: "Satisfaction confusion matrix",
    chartNote:
      "Rows represent actual labels; columns represent predicted labels. Class order is Not Satisfied then Satisfied.",
  },
  {
    id: "traffic",
    number: "05",
    name: "Traffic Classification",
    algorithm: "Decision Tree",
    task: "Three-class classification",
    dataset: "Recruit Restaurant Visitor Forecasting",
    region: "Japan",
    purpose:
      "Classify restaurant visitor demand as Low, Medium, or High.",
    metricName: "Test Accuracy",
    metricValue: "47.10%",
    metrics: [
      ["Accuracy", "47.10%"],
      ["Macro F1", "0.4502"],
      ["Baseline Accuracy", "40.27%"],
    ],
    explanation:
      "A Decision Tree learns branching rules from restaurant history and calendar-related features to estimate traffic categories.",
    limitation:
      "Accuracy is modest. This is a historical 2017 evaluation, not a validated present-day restaurant traffic prediction system.",
    chartType: "accuracy",
    chartData: [
      { name: "Decision Tree", accuracy: 47.1 },
      { name: "Majority Baseline", accuracy: 40.27 },
    ],
    chartTitle: "Traffic classification versus baseline",
    chartNote:
      "Higher accuracy is better. The Decision Tree improves upon the majority-class baseline but still misclassifies many examples.",
  },
  {
    id: "visitors",
    number: "06",
    name: "Visitor Forecasting",
    algorithm: "Artificial Neural Network",
    task: "Regression",
    dataset: "Recruit Restaurant Visitor Forecasting",
    region: "Japan",
    purpose: "Estimate numerical daily restaurant visitor counts.",
    metricName: "Mean Absolute Error",
    metricValue: "8.00",
    metrics: [
      ["MAE", "8.0007"],
      ["RMSE", "13.1862"],
      ["R²", "0.4853"],
    ],
    explanation:
      "An Artificial Neural Network learns nonlinear relationships between calendar features, previous visitor counts, and historical demand patterns.",
    limitation:
      "Evaluation is based on historical 2017 restaurant data. The model's inputs depend on historical observations and should not be presented as a current or future operational forecast.",
    chartType: "regression",
    chartData: [
      { name: "ANN", mae: 8.0007 },
      { name: "Linear Baseline Model", mae: 8.2416 },
    ],
    chartTitle: "Historical visitor forecasting MAE",
    chartNote:
      "Lower MAE is better. Both figures use the historical Japanese visitor prediction task.",
  },
];

const chartColors = ["#ee8a49", "#557b9a"];

function MetricChart({ model }) {
  if (model.chartType === "none") {
    return (
      <div className="insights-no-chart">
        <Database size={34} />
        <h4>Similarity-based recommendation</h4>
        <p>
          The KNN system finds attribute-nearest restaurants.
          We have not yet established a valid ranking
          evaluation score, such as Precision@K.
        </p>
      </div>
    );
  }

  if (model.chartType === "confusion") {
    const matrix = model.confusion;
    const labels =
      model.id === "sentiment"
        ? ["Negative", "Positive"]
        : ["Not Satisfied", "Satisfied"];

    return (
      <div className="insights-matrix-wrap">
        <p className="insights-axis-title">
          Columns: Predicted label
        </p>

        <div className="insights-matrix">
          <span />
          {labels.map((label) => (
            <strong key={`col-${label}`}>{label}</strong>
          ))}

          {matrix.map((row, rowIndex) => (
            <div className="insights-matrix-row" key={rowIndex}>
              <strong>{labels[rowIndex]}</strong>

              {row.map((value, colIndex) => (
                <div
                  key={colIndex}
                  className={
                    rowIndex === colIndex
                      ? "insights-matrix-cell correct"
                      : "insights-matrix-cell incorrect"
                  }
                >
                  <strong>{value}</strong>
                  <small>
                    {rowIndex === colIndex
                      ? "Correct"
                      : "Incorrect"}
                  </small>
                </div>
              ))}
            </div>
          ))}
        </div>

        <p className="insights-axis-title">
          Rows: Actual label
        </p>
      </div>
    );
  }

  const dataKey =
    model.chartType === "accuracy" ? "accuracy" : "mae";

  return (
    <div className="insights-chart">
      <ResponsiveContainer width="100%" height={290}>
        <BarChart
          data={model.chartData}
          margin={{ top: 20, right: 18, left: 10, bottom: 15 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            vertical={false}
          />

          <XAxis
            dataKey="name"
            tick={{ fontSize: 11 }}
            interval={0}
          />

          <YAxis
            tick={{ fontSize: 11 }}
            width={65}
            domain={
              model.chartType === "accuracy"
                ? [0, 100]
                : [0, "auto"]
            }
            tickFormatter={(value) =>
              value >= 1000000
                ? `${(value / 1000000).toFixed(1)}M`
                : String(Number(value.toFixed?.(1) ?? value))
            }
          />

          <Tooltip
            formatter={(value) => [
              model.chartType === "accuracy"
                ? `${Number(value).toFixed(2)}%`
                : Number(value).toLocaleString("en-US", {
                    maximumFractionDigits: 2,
                  }),
              model.chartType === "accuracy"
                ? "Accuracy"
                : "Mean Absolute Error",
            ]}
          />

          <Legend />

          <Bar
            name={
              model.chartType === "accuracy"
                ? "Accuracy (%)"
                : "MAE"
            }
            dataKey={dataKey}
            radius={[6, 6, 0, 0]}
            maxBarSize={105}
          >
            {model.chartData.map((entry, index) => (
              <Cell
                key={entry.name}
                fill={chartColors[index % chartColors.length]}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function ModelInsights({ onBack }) {
  const [selectedId, setSelectedId] = useState("sentiment");

  const selectedModel =
    modelData.find((model) => model.id === selectedId) ||
    modelData[0];

  return (
    <section className="insights-page">
      <div className="insights-container">
        <button
          type="button"
          className="insights-back"
          onClick={onBack}
        >
          <ArrowLeft size={18} />
          Back to ML Solutions
        </button>

        <div className="insights-heading">
          <span className="section-kicker">
            MODEL EVALUATION LABORATORY
          </span>

          <h1>Model Insights Dashboard</h1>

          <p>
            Explore the actual evaluation results of
            DineMind AI's six trained machine learning
            algorithms, including their datasets,
            metrics, and limitations.
          </p>
        </div>

        <div className="insights-overview">
          <div className="insights-stat">
            <BrainCircuit size={25} />
            <strong>06</strong>
            <span>Trained Models</span>
          </div>

          <div className="insights-stat">
            <Database size={25} />
            <strong>04</strong>
            <span>Dataset Families</span>
          </div>

          <div className="insights-stat">
            <BarChart3 size={25} />
            <strong>06</strong>
            <span>ML Applications</span>
          </div>
        </div>

        <div className="insights-section-title">
          <h2>Select a Model</h2>
          <p>
            Choose an algorithm to inspect its evaluation.
          </p>
        </div>

        <div className="insights-model-tabs" role="group"
          aria-label="Choose a machine learning model">
          {modelData.map((model) => (
            <button
              key={model.id}
              type="button"
              className={
                selectedId === model.id
                  ? "insights-tab active"
                  : "insights-tab"
              }
              aria-pressed={selectedId === model.id}
              onClick={() => setSelectedId(model.id)}
            >
              <span>ML-{model.number}</span>
              <strong>{model.algorithm}</strong>
            </button>
          ))}
        </div>

        <div className="insights-main-card">
          <div className="insights-model-heading">
            <div>
              <span className="section-kicker">
                ML-{selectedModel.number} · {selectedModel.task}
              </span>
              <h2>{selectedModel.name}</h2>
              <p>{selectedModel.purpose}</p>
            </div>

            <div className="insights-primary-metric">
              <span>{selectedModel.metricName}</span>
              <strong>{selectedModel.metricValue}</strong>
            </div>
          </div>

          <div className="insights-metric-grid">
            {selectedModel.metrics.map(([label, value]) => (
              <div className="insights-metric" key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>

          <div className="insights-visualization">
            <h3>{selectedModel.chartTitle}</h3>
            <MetricChart model={selectedModel} />

            {selectedModel.chartNote && (
              <p className="insights-chart-note">
                <Info size={16} />
                {selectedModel.chartNote}
              </p>
            )}
          </div>

          <div className="insights-information-grid">
            <div className="insights-info-panel">
              <div className="insights-info-title">
                <Database size={19} />
                <h3>Dataset & Source</h3>
              </div>

              <strong>{selectedModel.dataset}</strong>
              <p>Source region: {selectedModel.region}</p>
            </div>

            <div className="insights-info-panel">
              <div className="insights-info-title">
                <CheckCircle2 size={19} />
                <h3>How the Algorithm Works</h3>
              </div>

              <p>{selectedModel.explanation}</p>
            </div>
          </div>

          <div className="insights-limitation">
            <AlertTriangle size={21} />

            <div>
              <strong>Model Limitations</strong>
              <p>{selectedModel.limitation}</p>
            </div>
          </div>
        </div>

        <div className="insights-method-note">
          <Info size={20} />
          <p>
            Evaluation metrics come from different tasks and
            datasets. Accuracy and F1 apply to classification,
            while MAE, RMSE, and R² apply to regression.
            Comparing their raw numerical values across all
            six models would be misleading. The reported
            metrics reflect historical project evaluations,
            not live production monitoring.
          </p>
        </div>
      </div>
    </section>
  );
}
