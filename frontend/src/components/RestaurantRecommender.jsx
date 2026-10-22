
import { useEffect, useState } from "react";
import {
  ArrowLeft,
  Search,
  UtensilsCrossed,
  Star,
  MapPin,
  AlertCircle,
  LoaderCircle,
  RotateCcw,
} from "lucide-react";

const API_URL = "http://127.0.0.1:8000";

export default function RestaurantRecommender({ onBack }) {
  const [options, setOptions] = useState(null);
  const [price, setPrice] = useState("");
  const [ambience, setAmbience] = useState("");
  const [cuisine, setCuisine] = useState("");
  const [topN, setTopN] = useState(5);

  const [recommendations, setRecommendations] = useState([]);
  const [searched, setSearched] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingOptions, setLoadingOptions] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    async function loadPreferences() {
      try {
        const response = await fetch(
          `${API_URL}/restaurants/preferences`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error("Unable to load restaurant preferences.");
        }

        const data = await response.json();

        setOptions(data);
        setPrice(
          data.prices.includes("medium")
            ? "medium"
            : data.prices[0] || ""
        );
        setAmbience(
          data.ambiences.includes("familiar")
            ? "familiar"
            : data.ambiences[0] || ""
        );
        setCuisine(
          data.cuisines.includes("Mexican")
            ? "Mexican"
            : data.cuisines[0] || ""
        );
      } catch (err) {
        if (err.name !== "AbortError") {
          setError(
            "Could not load preferences. Make sure FastAPI is running."
          );
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoadingOptions(false);
        }
      }
    }

    loadPreferences();

    return () => controller.abort();
  }, []);

  function updatePreference(setter, value) {
    setter(value);
    setRecommendations([]);
    setSearched(false);
    setError("");
  }

  async function findRestaurants(event) {
    event.preventDefault();

    if (!price || !ambience || !cuisine) {
      setError("Please choose all three restaurant preferences.");
      return;
    }

    setLoading(true);
    setError("");
    setRecommendations([]);
    setSearched(false);

    try {
      const params = new URLSearchParams({
        price,
        ambience,
        cuisine,
        top_n: String(topN),
      });

      const response = await fetch(
        `${API_URL}/predict/recommendations?${params}`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Restaurant recommendation failed."
        );
      }

      setRecommendations(data.recommendations || []);
      setSearched(true);
    } catch (err) {
      setError(
        err instanceof TypeError
          ? "Cannot connect to FastAPI. Start your backend server."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  function clearResults() {
    setRecommendations([]);
    setSearched(false);
    setError("");
  }

  return (
    <section className="recommender-page">
      <div className="recommender-container">
        <button
          type="button"
          className="recommender-back"
          onClick={onBack}
        >
          <ArrowLeft size={17} />
          Back to ML Solutions
        </button>

        <div className="recommender-heading">
          <span className="section-kicker">
            ML-03 · K-NEAREST NEIGHBORS
          </span>
          <h2>Find Your Ideal Restaurant</h2>
          <p>
            Discover restaurants matching your dining preferences
            using our trained nearest-neighbor recommendation model.
          </p>
        </div>

        <div className="recommender-layout">
          <div className="recommender-card">
            <div className="recommender-card-title">
              <UtensilsCrossed size={23} />
              <h3>Your Dining Preferences</h3>
            </div>

            {loadingOptions ? (
              <p>Loading available preferences...</p>
            ) : (
              <form onSubmit={findRestaurants}>
                <label htmlFor="rec-cuisine">
                  Preferred Cuisine
                </label>
                <select
                  id="rec-cuisine"
                  value={cuisine}
                  onChange={(e) =>
                    updatePreference(setCuisine, e.target.value)
                  }
                  disabled={!options}
                >
                  {(options?.cuisines || []).map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>

                <label htmlFor="rec-price">Price Range</label>
                <select
                  id="rec-price"
                  value={price}
                  onChange={(e) =>
                    updatePreference(setPrice, e.target.value)
                  }
                  disabled={!options}
                >
                  {(options?.prices || []).map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>

                <label htmlFor="rec-ambience">
                  Restaurant Atmosphere
                </label>
                <select
                  id="rec-ambience"
                  value={ambience}
                  onChange={(e) =>
                    updatePreference(setAmbience, e.target.value)
                  }
                  disabled={!options}
                >
                  {(options?.ambiences || []).map((item) => (
                    <option key={item} value={item}>
                      {item === "familiar"
                        ? "Family-friendly"
                        : item}
                    </option>
                  ))}
                </select>

                <label htmlFor="rec-count">
                  Number of Recommendations
                </label>
                <select
                  id="rec-count"
                  value={topN}
                  onChange={(e) => {
                    setTopN(Number(e.target.value));
                    clearResults();
                  }}
                >
                  {[3, 5, 10].map((n) => (
                    <option value={n} key={n}>
                      Top {n}
                    </option>
                  ))}
                </select>

                <div className="recommender-actions">
                  <button
                    className="btn btn-primary"
                    type="submit"
                    disabled={loading || !options}
                  >
                    {loading ? (
                      <>
                        <LoaderCircle
                          size={18}
                          className="spin"
                        />
                        Finding matches...
                      </>
                    ) : (
                      <>
                        <Search size={18} />
                        Find Restaurants
                      </>
                    )}
                  </button>

                  <button
                    type="button"
                    className="recommender-clear"
                    onClick={clearResults}
                  >
                    <RotateCcw size={16} />
                    Clear
                  </button>
                </div>
              </form>
            )}

            {error && (
              <div className="recommender-error" role="alert">
                <AlertCircle size={18} />
                {error}
              </div>
            )}
          </div>

          <div className="recommender-card">
            <div className="recommender-card-title">
              <Search size={23} />
              <h3>Recommended Restaurants</h3>
            </div>

            {!searched ? (
              <div className="recommender-empty">
                <UtensilsCrossed size={43} />
                <h4>Discover Your Matches</h4>
                <p>
                  Select your dining preferences and click
                  Find Restaurants to see recommendations.
                </p>
              </div>
            ) : recommendations.length === 0 ? (
              <p>No matching restaurants were returned.</p>
            ) : (
              <div className="recommendation-results">
                <p className="result-count" aria-live="polite">
                  {recommendations.length} restaurant matches
                </p>

                {recommendations.map((restaurant, index) => (
                  <article
                    className="recommendation-item"
                    key={restaurant.restaurant_id}
                  >
                    <div className="recommendation-item-top">
                      <span className="recommendation-rank">
                        #{index + 1}
                      </span>
                      <strong>{restaurant.name}</strong>
                    </div>

                    <p>
                      <UtensilsCrossed size={15} />
                      {restaurant.cuisines.join(", ") ||
                        "Cuisine not listed"}
                    </p>

                    <p>
                      <MapPin size={15} />
                      {restaurant.price} price ·{" "}
                      {restaurant.ambience}
                    </p>

                    <p>
                      <Star size={15} />
                      {restaurant.average_rating === null
                        ? "No ratings available"
                        : `${restaurant.average_rating.toFixed(2)}/2 from ${restaurant.rating_count} ratings`}
                    </p>

                    <div className="similarity-display">
                      Attribute similarity:{" "}
                      <strong>
                        {(restaurant.similarity * 100).toFixed(1)}%
                      </strong>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </div>
        </div>

        <p className="recommender-note">
          Educational demonstration using historical restaurant
          data from Mexico. Similarity scores indicate attribute
          matching, not guaranteed customer satisfaction.
        </p>
      </div>
    </section>
  );
}
