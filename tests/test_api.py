
import requests

BASE_URL = "http://127.0.0.1:8000"

tests = [
    {
        "name": "Backend Health",
        "method": "GET",
        "endpoint": "/health",
    },
    {
        "name": "Japanese Restaurants",
        "method": "GET",
        "endpoint": "/restaurants/japan",
    },
    {
        "name": "Restaurant Preferences",
        "method": "GET",
        "endpoint": "/restaurants/preferences",
    },
    {
        "name": "Satisfaction Options",
        "method": "GET",
        "endpoint": "/satisfaction/options",
    },
    {
        "name": "Revenue Prediction",
        "method": "POST",
        "endpoint": "/predict/revenue",
        "json": {
            "opening_year": 2010,
            "opening_month": 5,
            "city_group": "Big Cities",
            "restaurant_type": "FC",
            "P1": 4,
            "P2": 5,
            "P3": 4,
            "P4": 4,
            "P5": 2,
        },
    },
    {
        "name": "Sentiment Prediction",
        "method": "POST",
        "endpoint": "/predict/sentiment",
        "json": {
            "review": "The food was disgusting"
        },
    },
    {
        "name": "Restaurant Recommendations",
        "method": "GET",
        "endpoint": "/predict/recommendations",
        "params": {
            "price": "medium",
            "ambience": "familiar",
            "cuisine": "Mexican",
            "top_n": 5,
        },
    },
    {
        "name": "Satisfaction Prediction",
        "method": "POST",
        "endpoint": "/predict/satisfaction",
        "json": {
            "user_id": "U1001",
            "restaurant_id": 135106,
        },
    },
    {
        "name": "Traffic Classification",
        "method": "POST",
        "endpoint": "/predict/traffic",
        "json": {
            "restaurant_id": "air_00a91d42b08b08d9",
            "prediction_date": "2017-03-15",
        },
    },
    {
        "name": "Visitor Forecasting",
        "method": "POST",
        "endpoint": "/predict/visitors",
        "json": {
            "restaurant_id": "air_00a91d42b08b08d9",
            "prediction_date": "2017-03-15",
        },
    },
]


def run_tests():
    passed = 0
    failed = 0

    print("\n" + "=" * 65)
    print("DINEMIND AI 2.0 - API TEST REPORT")
    print("=" * 65)

    for test in tests:
        try:
            response = requests.request(
                method=test["method"],
                url=BASE_URL + test["endpoint"],
                json=test.get("json"),
                params=test.get("params"),
                timeout=30,
            )

            if response.status_code == 200:
                print(f"[PASS] {test['name']}")

                if test["name"] == "Sentiment Prediction":
                    data = response.json()
                    print("       Prediction:", data.get("sentiment"))

                passed += 1
            else:
                print(
                    f"[FAIL] {test['name']}: "
                    f"HTTP {response.status_code}"
                )
                print("       ", response.text[:300])
                failed += 1

        except requests.RequestException as error:
            print(f"[FAIL] {test['name']}: {error}")
            failed += 1

    print("\n" + "=" * 65)
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")
    print("=" * 65)

    if failed == 0:
        print("ALL API TESTS PASSED")
    else:
        print("SOME API TESTS REQUIRE ATTENTION")


if __name__ == "__main__":
    run_tests()
