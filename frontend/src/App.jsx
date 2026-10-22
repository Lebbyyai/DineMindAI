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
  Sparkles,
  ShieldCheck,
  BarChart3,
  Globe2,
  BookOpen,
  Layers3,
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
import "./theme.css";
import "./premium-ui.css";
import "./brand-refinement.css";

const modules = [
  {
    id: "revenue",
    number: "01",
    name: "Revenue Prediction",
    algorithm: "Linear Regression",
    description:
      "Explore experimental revenue estimates based on restaurant characteristics.",
    icon: TrendingUp,
    button: "Predict Revenue",
  },
  {
    id: "sentiment",
    number: "02",
    name: "Review Sentiment",
    algorithm: "Naive Bayes",
    description:
      "Identify positive and negative patterns in English-language customer reviews.",
    icon: MessageSquare,
    button: "Analyze Sentiment",
  },
  {
    id: "recommendations",
    number: "03",
    name: "Restaurant Recommendations",
    algorithm: "K-Nearest Neighbors",
    description:
      "Discover restaurants matching your cuisine, price, and atmosphere preferences.",
    icon: UtensilsCrossed,
    button: "Find Restaurants",
  },
  {
    id: "satisfaction",
    number: "04",
    name: "Customer Satisfaction",
    algorithm: "Support Vector Machine",
    description:
      "Estimate satisfaction using historical customer ratings and restaurant features.",
    icon: HeartHandshake,
    button: "Predict Satisfaction",
  },
  {
    id: "traffic",
    number: "05",
    name: "Traffic Classification",
    algorithm: "Decision Tree",
    description:
      "Classify historical restaurant visitor demand as Low, Medium, or High.",
    icon: Users,
    button: "Classify Traffic",
  },
  {
    id: "visitors",
    number: "06",
    name: "Visitor Forecasting",
    algorithm: "Artificial Neural Network",
    description:
      "Estimate historical daily restaurant visitors using a neural network.",
    icon: ChartColumnIncreasing,
    button: "Forecast Visitors",
  },
];

export default function App() {
  const [activePage, setActivePage] = useState("home");
  const [menuOpen, setMenuOpen] = useState(false);

  const closeMenu = () => setMenuOpen(false);

  function openPage(page) {
    setActivePage(page);
    closeMenu();

    window.scrollTo({
      top: 0,
      behavior: "auto",
    });
  }

  function navigateTo(sectionId) {
    setActivePage("home");
    closeMenu();

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        document
          .getElementById(sectionId)
          ?.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
      });
    });
  }

  function closeModule() {
    navigateTo("solutions");
  }

  function renderHomePage() {
    return (
      <>
        {/* HERO */}

        <section id="home" className="hero">
          <div className="hero-content">
            <div className="eyebrow">
              <span className="status-dot" />
              MACHINE LEARNING FOR RESTAURANT INTELLIGENCE
            </div>

            <h1>
              Smarter Decisions.
              <br />
              <span>Better Dining.</span>
            </h1>

            <p>
              Discover the potential of intelligent
              restaurant analytics. Explore six trained
              machine learning applications that
              demonstrate predictions, classifications,
              customer insights, and recommendations.
            </p>

            <div className="hero-actions">
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => navigateTo("solutions")}
              >
                Explore AI Solutions
                <ArrowRight size={18} />
              </button>

              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => openPage("insights")}
              >
                <BarChart3 size={18} />
                Model Insights
              </button>
            </div>

            <div className="hero-stats">
              <div>
                <strong>06</strong>
                <span>Trained Models</span>
              </div>

              <div>
                <strong>06</strong>
                <span>Interactive Tools</span>
              </div>

              <div>
                <strong>04</strong>
                <span>Dataset Families</span>
              </div>
            </div>
          </div>

          <div className="hero-visual">
            <div className="visual-glow" />

            <div className="visual-panel">
              <div className="panel-heading">
                <span>
                  <BrainCircuit size={20} />
                  DineMind Intelligence
                </span>

                <span className="panel-label">
                  AI LAB
                </span>
              </div>

              <div className="panel-highlight">
                <span>
                  <Sparkles size={15} />
                  DATA-DRIVEN EXPLORATION
                </span>

                <strong>
                  Intelligence behind every insight.
                </strong>

                <p>
                  Explore six machine learning approaches
                  through interactive restaurant
                  intelligence demonstrations.
                </p>
              </div>

              <div className="panel-items">
                <div>
                  <TrendingUp size={19} />
                  <span>Revenue Analysis</span>
                  <CheckCircle2 size={17} />
                </div>

                <div>
                  <MessageSquare size={19} />
                  <span>Customer Sentiment</span>
                  <CheckCircle2 size={17} />
                </div>

                <div>
                  <Users size={19} />
                  <span>Visitor Intelligence</span>
                  <CheckCircle2 size={17} />
                </div>
              </div>

              <div className="panel-footer">
                <ShieldCheck size={16} />
                Academic research with transparent limitations
              </div>
            </div>
          </div>
        </section>

        {/* SOLUTIONS */}

        <section id="solutions" className="section solutions">
          <div className="section-heading">
            <span className="section-kicker">
              EXPLORE OUR INTELLIGENCE LAB
            </span>

            <h2>
              Six machine learning applications.
            </h2>

            <p>
              From customer experience to restaurant
              operations, explore the different ways
              machine learning can analyze and interpret
              restaurant-related data.
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
                      ML-{module.number}
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
                    onClick={() => openPage(module.id)}
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
                RESEARCH TRANSPARENCY
              </span>

              <h2>
                Understand the science behind the results.
              </h2>

              <p>
                Explore how the models were evaluated,
                review their reported metrics, and
                understand their limitations.
              </p>
            </div>

            <button
              type="button"
              className="btn btn-primary"
              onClick={() => openPage("insights")}
            >
              Explore Model Insights
              <ArrowRight size={18} />
            </button>
          </div>
        </section>

        {/* ABOUT */}

        <section id="about" className="section about">
          <div>
            <span className="section-kicker">
              THE IDEA BEHIND DINEMIND
            </span>

            <h2>
              Where dining meets intelligence.
            </h2>

            <p>
              DineMind AI is an academic restaurant
              intelligence platform that demonstrates
              how machine learning can help analyze
              restaurant-related information.
            </p>

            <p>
              The name combines "Dine", representing
              restaurants and dining experiences,
              with "Mind", representing intelligence,
              analysis, and data-driven thinking.
            </p>
          </div>

          <div className="values">
            <article>
              <h3>Our Mission</h3>
              <p>
                Make restaurant-related machine learning
                applications more understandable and
                accessible through interactive tools.
              </p>
            </article>

            <article>
              <h3>Our Vision</h3>
              <p>
                Explore how responsible, transparent,
                and data-driven technology could
                support better restaurant decisions.
              </p>
            </article>
          </div>
        </section>

        {/* RESEARCH METHODOLOGY */}

        <section
          id="methodology"
          className="section methodology dm-research-section"
        >
          <div className="dm-research-card">
            <div className="dm-research-top">
              <div className="dm-research-symbol">
                <Database size={27} />
              </div>

              <span className="dm-research-pill">
                <BookOpen size={14} />
                RESEARCH FOUNDATION
              </span>
            </div>

            <div className="dm-research-content">
              <span className="section-kicker">
                BUILT ON REAL DATA
              </span>

              <h2>
                Real datasets.
                <br />
                Real models.
                <br />
                <span>Transparent limitations.</span>
              </h2>

              <p>
                DineMind AI brings together several
                public research datasets to demonstrate
                different machine learning methods.
                We clearly identify their origins,
                evaluation metrics, and limitations.
              </p>
            </div>

            <div className="dm-research-sources">
              <div>
                <Globe2 size={20} />
                <span>Multiple dataset origins</span>
              </div>

              <div>
                <Layers3 size={20} />
                <span>Six ML approaches</span>
              </div>

              <div>
                <ShieldCheck size={20} />
                <span>Research limitations disclosed</span>
              </div>
            </div>

            <div className="dm-research-bottom">
              <div className="dm-research-notice">
                <ShieldCheck size={19} />
                <span>
                  Academic demonstration. Not a
                  production-validated business service.
                </span>
              </div>

              <button
                type="button"
                className="btn btn-primary"
                onClick={() => openPage("datasets")}
              >
                Explore Our Methodology
                <ArrowRight size={18} />
              </button>
            </div>
          </div>
        </section>
      </>
    );
  }

  function renderContent() {
    switch (activePage) {
      case "revenue":
        return <RevenuePredictor onBack={closeModule} />;

      case "sentiment":
        return <SentimentAnalyzer onBack={closeModule} />;

      case "recommendations":
        return (
          <RestaurantRecommender onBack={closeModule} />
        );

      case "satisfaction":
        return (
          <SatisfactionPredictor onBack={closeModule} />
        );

      case "traffic":
        return (
          <TrafficClassifier onBack={closeModule} />
        );

      case "visitors":
        return (
          <VisitorForecast onBack={closeModule} />
        );

      case "insights":
        return (
          <ModelInsights onBack={closeModule} />
        );

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

  return (
    <div className="app">
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
  <img
    src="/dinemind-logo.png"
    alt="DineMind AI Logo"
    className="brand-logo"
  />
</div>

          <div>
            <strong>DineMind AI</strong>
            <span>RESTAURANT INTELLIGENCE</span>
          </div>
        </a>

        <nav
          id="main-navigation"
          className={menuOpen ? "nav nav-open" : "nav"}
          aria-label="Main navigation"
        >
          {[
            ["Home", "home"],
            ["ML Solutions", "solutions"],
            ["About", "about"],
          ].map(([label, sectionId]) => (
            <a
              key={sectionId}
              href={`#${sectionId}`}
              onClick={(event) => {
                event.preventDefault();
                navigateTo(sectionId);
              }}
            >
              {label}
            </a>
          ))}

          <a
            href="#datasets"
            aria-current={
              activePage === "datasets"
                ? "page"
                : undefined
            }
            onClick={(event) => {
              event.preventDefault();
              openPage("datasets");
            }}
          >
            Methodology
          </a>

          <a
            href="#insights"
            aria-current={
              activePage === "insights"
                ? "page"
                : undefined
            }
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
          aria-label={
            menuOpen
              ? "Close navigation"
              : "Open navigation"
          }
          aria-controls="main-navigation"
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen(!menuOpen)}
        >
          {menuOpen ? (
            <X size={23} />
          ) : (
            <Menu size={23} />
          )}
        </button>
      </header>

      <main id="main-content">
        {renderContent()}
      </main>

      <footer className="footer dm-footer">
        <div className="dm-footer-main">
          <div className="dm-footer-identity">
            <div className="brand-icon">
              <BrainCircuit size={22} />
            </div>

            <div>
              <strong>DineMind AI</strong>
              <span>RESTAURANT INTELLIGENCE</span>
            </div>
          </div>

          <p>
            Exploring smarter restaurant decisions
            through machine learning.
          </p>

          <div className="dm-footer-links">
            <button
              type="button"
              onClick={() => navigateTo("solutions")}
            >
              ML Solutions
            </button>

            <button
              type="button"
              onClick={() => openPage("datasets")}
            >
              Methodology
            </button>

            <button
              type="button"
              onClick={() => openPage("insights")}
            >
              Model Insights
            </button>

            <a
              href="https://github.com/Lebbyyai/DineMindAI"
              target="_blank"
              rel="noreferrer"
              aria-label="DineMind AI GitHub repository"
            >
              <svg
  width="18"
  height="18"
  viewBox="0 0 24 24"
  fill="currentColor"
  aria-hidden="true"
>
  <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56 0-.28-.01-1.02-.02-2-3.2.69-3.88-1.36-3.88-1.36-.52-1.32-1.27-1.67-1.27-1.67-1.04-.71.08-.7.08-.7 1.15.08 1.75 1.18 1.75 1.18 1.02 1.75 2.68 1.24 3.33.95.1-.74.4-1.24.72-1.53-2.56-.29-5.26-1.28-5.26-5.7 0-1.26.45-2.29 1.18-3.1-.12-.29-.51-1.47.11-3.06 0 0 .96-.31 3.15 1.18A10.96 10.96 0 0 1 12 6.1c.98 0 1.97.13 2.89.39 2.19-1.49 3.15-1.18 3.15-1.18.62 1.59.23 2.77.11 3.06.73.81 1.18 1.84 1.18 3.1 0 4.43-2.7 5.4-5.27 5.69.41.35.77 1.02.77 2.06 0 1.49-.01 2.69-.01 3.06 0 .31.21.68.8.56A11.51 11.51 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z" />
</svg>
              GitHub
            </a>
          </div>
        </div>

        <div className="dm-footer-bottom">
          <span>
            © 2026 DineMind AI · Academic ML Project
          </span>

          <span>
            Smarter Decisions. Better Dining.
          </span>

          <button
            type="button"
            onClick={() => navigateTo("home")}
          >
            Back to top ↑
          </button>
        </div>
      </footer>
    </div>
  );
}