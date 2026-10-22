
import { useState } from "react";
import {
  BrainCircuit,
  Menu,
  X,
  ArrowRight,
  TrendingUp,
  MessageSquare,
  UtensilsCrossed,
  HeartHandshake,
  Users,
  ChartColumnIncreasing,
  Database,
  CheckCircle2,
} from "lucide-react";

import RevenuePredictor from "./components/RevenuePredictor";
import SentimentAnalyzer from "./components/SentimentAnalyzer";
import RestaurantRecommender from "./components/RestaurantRecommender";
import SatisfactionPredictor from "./components/SatisfactionPredictor";
import TrafficClassifier from "./components/TrafficClassifier";
import VisitorForecast from "./components/VisitorForecast";
import ModelInsights from "./components/ModelInsights";
import DatasetMethodology from "./components/DatasetMethodology";

import "./App.css";

// ============================================
// DINEMIND AI 2.0
// ============================================

const modules = [
  {
    id: "revenue",
    number: "01",
    name: "Revenue Prediction",
    algorithm: "Linear Regression",
    description:
      "Explore how restaurant attributes relate to experimental revenue estimates.",
    icon: TrendingUp,
    button: "Predict Revenue",
  },
  {
    id: "sentiment",
    number: "02",
    name: "Review Sentiment",
    algorithm: "Naive Bayes",
    description:
      "Analyze English-language customer reviews to identify positive and negative feedback.",
    icon: MessageSquare,
    button: "Analyze Reviews",
  },
  {
    id: "recommendations",
    number: "03",
    name: "Restaurant Recommendations",
    algorithm: "K-Nearest Neighbors",
    description:
      "Find restaurants matching cuisine, budget, and atmosphere preferences.",
    icon: UtensilsCrossed,
    button: "Find Restaurants",
  },
  {
    id: "satisfaction",
    number: "04",
    name: "Customer Satisfaction",
    algorithm: "Support Vector Machine",
    description:
      "Explore satisfaction predictions based on historical customer ratings.",
    icon: HeartHandshake,
    button: "Predict Satisfaction",
  },
  {
    id: "traffic",
    number: "05",
    name: "Traffic Classification",
    algorithm: "Decision Tree",
    description:
      "Classify historical Japanese restaurant traffic as Low, Medium, or High.",
    icon: Users,
    button: "Classify Traffic",
  },
  {
    id: "visitors",
    number: "06",
    name: "Visitor Forecasting",
    algorithm: "Artificial Neural Network",
    description:
      "Estimate historical daily restaurant visitor counts using neural networks.",
    icon: ChartColumnIncreasing,
    button: "Forecast Visitors",
  },
];

const moduleIds = modules.map((item) => item.id);

export default function App() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [activePage, setActivePage] = useState("home");

  function openPage(page) {
    setActivePage(page);
    setMenuOpen(false);
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  function openModule(id) {
    if (moduleIds.includes(id)) {
      openPage(id);
    }
  }

  function navigateTo(sectionId) {
    setActivePage("home");
    setMenuOpen(false);

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        document.getElementById(sectionId)?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      });
    });
  }

  function closeModule() {
    navigateTo("solutions");
  }

  function renderContent() {
    switch (activePage) {
      case "revenue":
        return <RevenuePredictor onBack={closeModule} />;

      case "sentiment":
        return <SentimentAnalyzer onBack={closeModule} />;

      case "recommendations":
        return <RestaurantRecommender onBack={closeModule} />;

      case "satisfaction":
        return <SatisfactionPredictor onBack={closeModule} />;

      case "traffic":
        return <TrafficClassifier onBack={closeModule} />;

      case "visitors":
        return <VisitorForecast onBack={closeModule} />;

      case "insights":
        return <ModelInsights onBack={closeModule} />;

      case "datasets":
        return (
          <DatasetMethodology
            onBack={closeModule}
            onOpenInsights={() => openPage("insights")}
          />
        );

      default:
        return renderHomePage();
    }
  }

  function renderHomePage() {
    return (
      <>
        {/* HERO SECTION */}

        <section id="home" className="hero">
          <div className="hero-content">
            <div className="eyebrow">
              <span className="status-dot" />
              MACHINE LEARNING FOR RESTAURANTS
            </div>

            <h1>
              Smarter Decisions.
              <br />
              <span>Better Dining.</span>
            </h1>

            <p>
              Discover six machine learning applications
              exploring restaurant revenue, customer
              feedback, recommendations, satisfaction,
              and historical visitor demand.
            </p>

            <div className="hero-actions">
              <a
                href="#solutions"
                className="btn btn-primary"
                onClick={(event) => {
                  event.preventDefault();
                  navigateTo("solutions");
                }}
              >
                Explore Our Solutions
                <ArrowRight size={18} />
              </a>

              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => openPage("datasets")}
              >
                Explore the Research
              </button>
            </div>

            <div className="hero-stats">
              <div>
                <strong>06</strong>
                <span>ML Algorithms</span>
              </div>
              <div>
                <strong>06</strong>
                <span>Business Applications</span>
              </div>
              <div>
                <strong>Real</strong>
                <span>Research Datasets</span>
              </div>
            </div>
          </div>

          <div className="hero-visual">
            <div className="visual-glow" />

            <div className="visual-panel">
              <div className="panel-heading">
                <span>
                  <BrainCircuit size={18} />
                  Intelligence Overview
                </span>
                <span className="panel-label">
                  ML DEMO
                </span>
              </div>

              <div className="panel-highlight">
                <span>Six Specialized Models</span>
                <strong>Data Into Decisions</strong>
                <p>
                  Restaurant analytics powered by
                  trained machine learning models.
                </p>
              </div>

              <div className="panel-items">
                <div>
                  <TrendingUp />
                  <span>Revenue Analytics</span>
                  <CheckCircle2 />
                </div>
                <div>
                  <MessageSquare />
                  <span>Customer Sentiment</span>
                  <CheckCircle2 />
                </div>
                <div>
                  <Users />
                  <span>Visitor Demand</span>
                  <CheckCircle2 />
                </div>
              </div>

              <div className="panel-footer">
                <Database size={15} />
                Historical research data · Academic demonstration
              </div>
            </div>
          </div>
        </section>

        {/* ML SOLUTIONS */}

        <section
          id="solutions"
          className="section solutions"
        >
          <div className="section-heading">
            <span className="section-kicker">
              OUR MACHINE LEARNING LAB
            </span>
            <h2>
              Six algorithms. Six business solutions.
            </h2>
            <p>
              Explore the different ways machine
              learning can support restaurant analysis.
            </p>
          </div>

          <div className="modules-grid">
            {modules.map((module) => {
              const Icon = module.icon;

              return (
                <article
                  className="module-card"
                  key={module.id}
                >
                  <div className="module-top">
                    <span className="module-icon">
                      <Icon size={25} />
                    </span>
                    <span className="module-number">
                      {module.number}
                    </span>
                  </div>

                  <span className="algorithm-label">
                    {module.algorithm}
                  </span>

                  <h3>{module.name}</h3>
                  <p>{module.description}</p>

                  <button
                    type="button"
                    className="module-open"
                    onClick={() => openModule(module.id)}
                  >
                    {module.button}
                    <ArrowRight size={18} />
                  </button>
                </article>
              );
            })}
          </div>

          <div className="insights-promo">
            <div>
              <span className="section-kicker">
                BEYOND PREDICTIONS
              </span>

              <h2>
                Explore the science behind our models.
              </h2>

              <p>
                View evaluation metrics, confusion
                matrices, comparisons, and limitations.
              </p>
            </div>

            <button
              type="button"
              className="btn btn-primary"
              onClick={() => openPage("insights")}
            >
              View Model Insights
              <ArrowRight size={18} />
            </button>
          </div>
        </section>

        {/* ABOUT SECTION */}

        <section id="about" className="section about">
          <div>
            <span className="section-kicker">
              WHO WE ARE
            </span>

            <h2>
              Restaurant intelligence with purpose.
            </h2>

            <p>
              DineMind AI is a fictional Japan-focused
              restaurant intelligence company developed
              for an academic machine learning project.
              The website demonstrates the potential
              and limitations of data-driven
              restaurant decision support.
            </p>
          </div>

          <div className="values">
            <article>
              <h3>Our Mission</h3>
              <p>
                To make restaurant analytics accessible
                and demonstrate how machine learning
                supports informed decisions.
              </p>
            </article>

            <article>
              <h3>Our Vision</h3>
              <p>
                To encourage responsible, transparent
                and useful applications of data science
                in the restaurant industry.
              </p>
            </article>
          </div>
        </section>

        {/* METHODOLOGY PREVIEW */}

        <section
          id="methodology"
          className="section methodology"
        >
          <div className="method-card">
            <Database size={32} />

            <div>
              <span className="section-kicker">
                TRANSPARENT BY DESIGN
              </span>

              <h2>
                Real data. Real models. Clear limitations.
              </h2>

              <p>
                Our demonstrations use historical Japanese
                restaurant visits, restaurant ratings
                from Mexico, English-language Yelp reviews,
                and anonymized restaurant revenue data.
              </p>

              <small>
                These are academic demonstrations,
                not production-validated business tools.
              </small>

              <div style={{ marginTop: "22px" }}>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => openPage("datasets")}
                >
                  View Datasets & Methodology
                  <ArrowRight size={18} />
                </button>
              </div>
            </div>
          </div>
        </section>
      </>
    );
  }

  return (
    <div className="app">
      {/* NAVIGATION */}

      <header className="header">
        <a
          href="#home"
          className="brand"
          onClick={(event) => {
            event.preventDefault();
            navigateTo("home");
          }}
        >
          <div className="brand-icon">
            <UtensilsCrossed size={22} />
          </div>

          <div>
            <strong>DineMind AI</strong>
            <span>
              JAPAN · RESTAURANT INTELLIGENCE
            </span>
          </div>
        </a>

        <nav className={menuOpen ? "nav nav-open" : "nav"}>
          {[
            ["Home", "home"],
            ["ML Solutions", "solutions"],
            ["About", "about"],
          ].map(([label, id]) => (
            <a
              key={id}
              href={`#${id}`}
              onClick={(event) => {
                event.preventDefault();
                navigateTo(id);
              }}
            >
              {label}
            </a>
          ))}

          <a
            href="#datasets"
            onClick={(event) => {
              event.preventDefault();
              openPage("datasets");
            }}
          >
            Datasets & Methodology
          </a>

          <a
            href="#insights"
            onClick={(event) => {
              event.preventDefault();
              openPage("insights");
            }}
          >
            Model Insights
          </a>

          <a
            href="#solutions"
            className="nav-button"
            onClick={(event) => {
              event.preventDefault();
              navigateTo("solutions");
            }}
          >
            Explore Models
            <ArrowRight size={16} />
          </a>
        </nav>

        <button
          type="button"
          className="mobile-menu"
          aria-label="Toggle navigation"
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen(!menuOpen)}
        >
          {menuOpen ? <X /> : <Menu />}
        </button>
      </header>

      {/* ACTIVE PAGE */}

      <main>{renderContent()}</main>

      {/* FOOTER */}

      <footer className="footer">
        <div className="footer-brand">
          <UtensilsCrossed size={20} />
          <strong>DineMind AI</strong>
        </div>

        <span>
          Academic Machine Learning Project · 2026
        </span>

        <a
          href="#home"
          onClick={(event) => {
            event.preventDefault();
            navigateTo("home");
          }}
        >
          Back to top ↑
        </a>
      </footer>
    </div>
  );
}
