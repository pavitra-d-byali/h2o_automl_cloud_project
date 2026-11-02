"""Smoke-test a running API. Start uvicorn in another terminal first."""

import json
from pathlib import Path
import requests

BASE_URL = "http://127.0.0.1:8000"
BASE_DIR = Path(__file__).resolve().parent


def main():
    health = requests.get(f"{BASE_URL}/health", timeout=10)
    health.raise_for_status()
    print("Health:", health.json())

    feature_response = requests.get(f"{BASE_URL}/features", timeout=10)
    feature_response.raise_for_status()
    print("Required feature count:", len(feature_response.json()["features"]))

    sample_path = BASE_DIR / "sample_request.json"
    if not sample_path.exists():
        raise FileNotFoundError("Run `python make_sample_request.py` first.")
    payload = json.loads(sample_path.read_text(encoding="utf-8"))

    response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=60)
    print("HTTP status:", response.status_code)
    print(json.dumps(response.json(), indent=2))
    response.raise_for_status()
    print("API smoke test passed.")


if __name__ == "__main__":
    main()
