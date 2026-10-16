"""Stream a deterministic, balanced subset without downloading all four million rows."""
import hashlib
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from app.settings import DATASET_ID, DATASET_REVISION

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def collect(split: str, per_class: int) -> pd.DataFrame:
    stream = load_dataset(
        DATASET_ID, revision=DATASET_REVISION, split=split, streaming=True
    ).shuffle(seed=SEED, buffer_size=10_000)
    counts = {0: 0, 1: 0}
    rows = []
    for item in stream:
        label = int(item["label"])
        if counts[label] < per_class:
            # Keep exactly the body, not a label-bearing title or star rating.
            rows.append({"text": item["content"], "label": label})
            counts[label] += 1
        if all(count >= per_class for count in counts.values()):
            break
    if len(rows) != 2 * per_class:
        raise RuntimeError(f"Could not collect enough rows from {split}.")
    return pd.DataFrame(rows)


def main():
    directory = ROOT / "data"
    directory.mkdir(exist_ok=True)
    manifest = {
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "seed": SEED,
        "method": "buffer shuffle (10000), take first 50/250 per class; not a uniform full-dataset sample",
        "labels": {"0": "negative", "1": "positive"},
        "files": {},
    }
    for split, per_class, filename in [("train", 50, "development.csv"), ("test", 250, "evaluation.csv")]:
        frame = collect(split, per_class)
        path = directory / filename
        frame.to_csv(path, index=False)
        manifest["files"][filename] = {
            "source_split": split,
            "rows": len(frame),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(f"Saved {len(frame)} reviews to {path}", flush=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
