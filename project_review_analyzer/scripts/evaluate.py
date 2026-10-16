"""Evaluate the same model used by the API, without HTTP overhead."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix

from app.model import SentimentModel
from app.settings import MODEL_ID, MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "data" / "evaluation.csv"
    if not source.exists():
        raise SystemExit("Download data first: python -m scripts.download_data")
    manifest = json.loads((source.parent / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["files"][source.name]["sha256"]:
        raise SystemExit("Evaluation data changed. Download the original sample again before evaluating.")
    frame = pd.read_csv(source, keep_default_na=False)
    expected = frame["label"].map({0: "negative", 1: "positive"})
    if expected.isna().any() or len(frame) == 0:
        raise SystemExit("Evaluation needs non-empty reviews with labels 0 or 1.")
    print("Loading pretrained model…", flush=True)
    model = SentimentModel()
    started = perf_counter()
    predictions = pd.DataFrame(model.predict(frame["text"].tolist()))
    elapsed = perf_counter() - started
    labels = ["negative", "positive"]
    predicted = predictions["label"]
    # Deterministic tie-breaking: negative is first when classes have equal counts.
    majority = expected.value_counts().reindex(labels, fill_value=0).idxmax()
    baseline = [majority] * len(frame)
    report = classification_report(expected, predicted, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(expected, predicted, labels=labels)
    metrics = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "data": manifest,
        "rows": len(frame),
        "accuracy": accuracy_score(expected, predicted),
        "baseline_label": majority,
        "baseline_accuracy": accuracy_score(expected, baseline),
        "classification_report": report,
        "confusion_matrix_labels": labels,
        "confusion_matrix": matrix.tolist(),
        "truncated_reviews": int(predictions["truncated"].sum()),
        "inference_seconds": elapsed,
        "note": "Inference time excludes model loading. This small balanced sample is not a production accuracy estimate.",
    }
    destination = ROOT / "reports"
    destination.mkdir(exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    output = frame.rename(columns={"label": "expected_label_id"}).assign(expected_sentiment=expected)
    output = pd.concat([output, predictions.rename(columns={"label": "predicted_sentiment"})], axis=1)
    output.to_csv(destination / "predictions.csv", index=False)
    output[expected != predicted].to_csv(destination / "errors.csv", index=False)
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(cmap="Blues", colorbar=False)
    plt.title(f"Amazon reviews · {len(frame)} examples")
    plt.tight_layout()
    plt.savefig(destination / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Accuracy: {metrics['accuracy']:.1%}; baseline: {metrics['baseline_accuracy']:.1%}")
    print(f"F1 (positive): {report['positive']['f1-score']:.3f}; inference: {elapsed:.1f}s")
    print(f"Saved metrics, predictions, errors, and confusion matrix to {destination}")


if __name__ == "__main__":
    main()
