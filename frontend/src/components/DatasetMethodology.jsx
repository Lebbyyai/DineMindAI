
import { useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Database,
  BookOpen,
  BrainCircuit,
  AlertTriangle,
  CheckCircle2,
  Globe2,
} from "lucide-react";

const datasets = [
  {
    id: "recruit",
    name: "Recruit Restaurant Visitor Forecasting",
    region: "Japan",
    source: "Kaggle",
    rows: "252,108 original visit records",
    use: "Decision Tree and ANN",
    description:
      "Historical restaurant visitor records, restaurant information, and calendar data used for traffic classification and visitor forecasting.",
    limitations:
      "The data cover 2016–2017. The interactive demonstrations use historical dates and cannot forecast current restaurant demand.",
    link: "https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting",
  },
  {
    id: "yelp",
    name: "Yelp Review Sentiment",
    region: "English-language restaurant reviews",
    source: "Yelp Polarity and UCI",
    rows: "20,000 sampled Yelp Polarity training reviews, plus original short reviews",
    use: "Naive Bayes",
    description:
      "Labeled customer reviews used to learn patterns associated with positive and negative restaurant experiences.",
    limitations:
      "The model was evaluated on 2,000 sampled Yelp Polarity test reviews. Its performance may differ on shorter reviews or Japanese-language text.",
    link: "https://huggingface.co/datasets/ajay232/yelp_polarity",
  },
  {
    id: "mexico",
    name: "Restaurant & Consumer Data",
    region: "Mexico",
    source: "UCI Machine Learning Repository",
    rows: "130 restaurants and 1,161 historical ratings",
    use: "KNN and SVM",
    description:
      "Restaurant attributes, cuisine information, customer preferences, and ratings used for recommendations and satisfaction classification.",
    limitations:
      "The records represent a limited historical sample in Mexico. Results cannot automatically be generalized to Japanese restaurants.",
    link: "https://archive.ics.uci.edu/dataset/232/restaurant+consumer+data",
  },
  {
    id: "tfi",
    name: "TFI Restaurant Revenue Prediction",
    region: "Turkey",
    source: "Kaggle",
    rows: "137 labeled training restaurants",
    use: "Linear Regression",
    description:
      "Restaurant opening dates, categories, city groups, and anonymized numerical attributes used to estimate revenue.",
    limitations:
      "The dataset is very small. The model underperformed a simple mean-revenue baseline and should be considered experimental.",
    link: "https://www.kaggle.com/c/restaurant-revenue-prediction",
  },
];

const models = [
  {
    number: "01",
    name: "Linear Regression",
    application: "Revenue Prediction",
    dataset: "TFI Restaurant Revenue Prediction",
    inputs: "Opening date, city group, restaurant type, P1–P5",
    output: "Estimated revenue in dataset units",
    method:
      "A linear regression pipeline learns associations between the selected restaurant features and continuous revenue values. Five-fold cross-validation was used for evaluation.",
    evaluation:
      "MAE 1,825,871.90; RMSE 2,581,812.46; R² −0.0118.",
    why:
      "Linear Regression is a useful baseline for numerical prediction and demonstrates how continuous outcomes can be modeled.",
    caution:
      "Performance was worse than the mean baseline. The prediction is not dependable for financial decisions.",
  },
  {
    number: "02",
    name: "Naive Bayes",
    application: "Customer Review Sentiment",
    dataset: "Yelp Polarity and UCI sentiment reviews",
    inputs: "English-language review text",
    output: "Positive or Negative, with estimated probabilities",
    method:
      "Text is converted into word and phrase features. Multinomial Naive Bayes uses learned feature distributions to classify sentiment. Expanded data and hyperparameter selection improved results.",
    evaluation:
      "90.95% accuracy; macro F1 0.9095 on 2,000 held-out Yelp Polarity reviews.",
    why:
      "Naive Bayes is fast and interpretable for text classification, making it suitable for demonstrating customer feedback analysis.",
    caution:
      "The scores do not establish accuracy for Japanese-language feedback or every short review.",
  },
  {
    number: "03",
    name: "K-Nearest Neighbors",
    application: "Restaurant Recommendations",
    dataset: "UCI Restaurant & Consumer Data",
    inputs: "Preferred cuisine, price, and atmosphere",
    output: "Ranked restaurants with attribute similarity",
    method:
      "Restaurant features are encoded, cuisine matches receive additional weight, and nearest-neighbor retrieval uses cosine distance.",
    evaluation:
      "Recommendations are demonstrated, but an independent Precision@K or ranking benchmark has not yet been established.",
    why:
      "KNN is intuitive for finding items with similar attributes and supports interactive preference-based recommendations.",
    caution:
      "Similarity reflects encoded attributes, not restaurant quality or guaranteed customer satisfaction.",
  },
  {
    number: "04",
    name: "Support Vector Machine",
    application: "Customer Satisfaction",
    dataset: "UCI Restaurant & Consumer Data",
    inputs: "Customer ID, restaurant ID, historical rating summaries and restaurant attributes",
    output: "Satisfied or Not Satisfied",
    method:
      "An RBF-kernel support vector classifier learns a nonlinear decision boundary from labeled historical customer ratings and selected features.",
    evaluation:
      "71.24% test accuracy, with 166 correct predictions from 233 test examples.",
    why:
      "SVM can model nonlinear boundaries between satisfaction categories in structured data.",
    caution:
      "Historical customer and restaurant information is required. Predictions may not generalize to previously unseen entities.",
  },
  {
    number: "05",
    name: "Decision Tree",
    application: "Traffic Classification",
    dataset: "Recruit Restaurant Visitor Forecasting",
    inputs: "Restaurant ID, historical date and derived visitor/calendar features",
    output: "Low, Medium, or High traffic",
    method:
      "A decision tree learns conditional splitting rules that divide observations into visitor-traffic categories. Evaluation used a historical time-based holdout.",
    evaluation:
      "47.10% accuracy; macro F1 0.4502; majority baseline accuracy 40.27%.",
    why:
      "Decision Trees provide understandable branching logic that can illustrate category prediction.",
    caution:
      "The moderate accuracy makes it unsuitable for dependable real-world planning.",
  },
  {
    number: "06",
    name: "Artificial Neural Network",
    application: "Visitor Forecasting",
    dataset: "Recruit Restaurant Visitor Forecasting",
    inputs: "Restaurant ID, historical date, calendar and lagged visitor features",
    output: "Estimated daily visitor count",
    method:
      "A multilayer neural network learns nonlinear relationships between historical attendance, restaurant characteristics and calendar information.",
    evaluation:
      "MAE 8.0007; RMSE 13.1862; R² 0.4853 on the historical holdout.",
    why:
      "ANN demonstrates a flexible nonlinear regression approach for predicting numerical demand.",
    caution:
      "The interactive tool is a retrospective 2017 demonstration and requires historical feature data.",
  },
];

export default function DatasetMethodology({
  onBack,
  onOpenInsights,
}) {
  const [selectedModel, setSelectedModel] = useState(0);
  const model = models[selectedModel];

  return (
    <section className="docs-page">
      <div className="docs-container">
        <button
          type="button"
          className="docs-back"
          onClick={onBack}
        >
          <ArrowLeft size={17} />
          Back to ML Solutions
        </button>

        <div className="docs-hero">
          <span className="section-kicker">
            RESEARCH & TRANSPARENCY
          </span>
          <h1>Datasets & Methodology</h1>
          <p>
            Explore the research datasets, machine learning
            techniques, evaluation methods, and limitations
            behind the six DineMind AI applications.
          </p>
        </div>

        <div className="docs-summary">
          <div>
            <Database size={24} />
            <strong>4</strong>
            <span>Dataset families</span>
          </div>
          <div>
            <BrainCircuit size={24} />
            <strong>6</strong>
            <span>Trained algorithms</span>
          </div>
          <div>
            <Globe2 size={24} />
            <strong>3</strong>
            <span>Dataset countries</span>
          </div>
        </div>

        <div className="docs-section-heading">
          <span className="section-kicker">
            DATA PROVENANCE
          </span>
          <h2>Our Research Datasets</h2>
          <p>
            The Japan-focused company concept uses research
            datasets from different countries. The country
            and purpose of each source are identified below.
          </p>
        </div>

        <div className="docs-dataset-grid">
          {datasets.map((dataset) => (
            <article className="docs-dataset-card" key={dataset.id}>
              <div className="docs-dataset-top">
                <Database size={23} />
                <span>{dataset.region}</span>
              </div>

              <h3>{dataset.name}</h3>

              <p>{dataset.description}</p>

              <div className="docs-dataset-meta">
                <span>Source</span>
                <strong>{dataset.source}</strong>
                <span>Scope</span>
                <strong>{dataset.rows}</strong>
                <span>Used by</span>
                <strong>{dataset.use}</strong>
              </div>

              <p className="docs-small-warning">
                <AlertTriangle size={16} />
                {dataset.limitations}
              </p>

              <a
                href={dataset.link}
                target="_blank"
                rel="noopener noreferrer"
                className="docs-source-link"
              >
                View original dataset
                <ArrowRight size={16} />
              </a>
            </article>
          ))}
        </div>

        <div className="docs-section-heading">
          <span className="section-kicker">
            ALGORITHM SELECTION
          </span>
          <h2>How Each Model Works</h2>
          <p>
            Select an algorithm to examine its purpose,
            input features, prediction output, and
            evaluation results.
          </p>
        </div>

        <div className="docs-model-selector">
          {models.map((item, index) => (
            <button
              type="button"
              key={item.number}
              className={
                selectedModel === index
                  ? "docs-model-tab active"
                  : "docs-model-tab"
              }
              aria-pressed={selectedModel === index}
              onClick={() => setSelectedModel(index)}
            >
              <span>ML-{item.number}</span>
              <strong>{item.name}</strong>
            </button>
          ))}
        </div>

        <div className="docs-model-detail">
          <span className="section-kicker">
            ML-{model.number} · {model.name}
          </span>

          <h2>{model.application}</h2>

          <div className="docs-detail-grid">
            <div className="docs-detail-item">
              <span>Dataset</span>
              <strong>{model.dataset}</strong>
            </div>
            <div className="docs-detail-item">
              <span>Inputs</span>
              <strong>{model.inputs}</strong>
            </div>
            <div className="docs-detail-item">
              <span>Output</span>
              <strong>{model.output}</strong>
            </div>
          </div>

          <div className="docs-explanation">
            <div>
              <BookOpen size={21} />
              <h3>Training Approach</h3>
            </div>
            <p>{model.method}</p>
          </div>

          <div className="docs-explanation">
            <div>
              <CheckCircle2 size={21} />
              <h3>Why This Algorithm?</h3>
            </div>
            <p>{model.why}</p>
          </div>

          <div className="docs-evaluation">
            <strong>Recorded Evaluation</strong>
            <p>{model.evaluation}</p>
          </div>

          <div className="docs-caution">
            <AlertTriangle size={20} />
            <div>
              <strong>Important Limitation</strong>
              <p>{model.caution}</p>
            </div>
          </div>
        </div>

        <div className="docs-research-standards">
          <h2>Evaluation Principles</h2>
          <div className="docs-principles">
            <article>
              <strong>01 · Training and testing</strong>
              <p>
                We use training data to fit models and
                separate validation or holdout data to
                estimate performance. Some experiments
                use cross-validation, while historical
                Japanese visitor models use a date-based
                test period.
              </p>
            </article>
            <article>
              <strong>02 · Appropriate metrics</strong>
              <p>
                Classification is evaluated with accuracy,
                F1 and confusion matrices. Numerical
                predictions use MAE, RMSE and R².
                Recommendation similarity is not itself
                an accuracy measure.
              </p>
            </article>
            <article>
              <strong>03 · Responsible interpretation</strong>
              <p>
                We report weak as well as strong model
                results. Dataset differences, time periods
                and geographic coverage must be considered
                before interpreting predictions.
              </p>
            </article>
          </div>
        </div>

        <div className="docs-insights-link">
          <div>
            <span className="section-kicker">
              EVALUATION DASHBOARD
            </span>
            <h2>Explore the Results Visually</h2>
            <p>
              Open Model Insights to review confusion
              matrices, evaluation charts and comparative
              model metrics.
            </p>
          </div>
          <button
            type="button"
            className="btn btn-primary"
            onClick={onOpenInsights}
          >
            View Model Insights
            <ArrowRight size={18} />
          </button>
        </div>

        <p className="docs-footer-note">
          DineMind AI is an academic demonstration of
          machine learning methods, not a commercially
          validated restaurant decision platform.
        </p>
      </div>
    </section>
  );
}
