"""Check real running services from within the tools container."""
import requests


def main():
    base = "http://api:8000"
    health = requests.get(f"{base}/health", timeout=10)
    health.raise_for_status()
    assert health.json()["status"] == "ready"
    texts = ["An excellent product. I absolutely love it!", "A terrible product. Completely broken and useless."]
    batch = requests.post(f"{base}/predict/batch", json={"texts": texts}, timeout=180)
    batch.raise_for_status()
    predictions = batch.json()["predictions"]
    assert [item["label"] for item in predictions] == ["positive", "negative"]
    single = requests.post(f"{base}/predict", json={"text": texts[0]}, timeout=60)
    single.raise_for_status()
    assert single.json()["label"] == predictions[0]["label"]
    invalid = requests.post(f"{base}/predict", json={"text": " "}, timeout=10)
    assert invalid.status_code == 422
    long_review = requests.post(f"{base}/predict", json={"text": "great " * 600}, timeout=60)
    long_review.raise_for_status()
    assert long_review.json()["truncated"] is True
    page = requests.get("http://dashboard:8501", timeout=10)
    page.raise_for_status()
    assert "streamlit" in page.text.lower()
    print("Real API predictions:", predictions)
    print("PASS: health, single/batch sentiment, validation, truncation, dashboard HTTP")


if __name__ == "__main__":
    main()
