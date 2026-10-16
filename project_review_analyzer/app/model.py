"""Load once, tokenize text, and run CPU inference. No model training happens here."""
from threading import Lock

from app.settings import MAX_TOKENS, MODEL_ID, MODEL_REVISION


class SentimentModel:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(2)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, use_safetensors=True
        ).eval()
        self.lock = Lock()

    def predict(self, texts: list[str]) -> list[dict]:
        import torch

        results = []
        # One request at a time; small internal batches keep CPU memory bounded.
        with self.lock, torch.inference_mode():
            for start in range(0, len(texts), 8):
                chunk = texts[start : start + 8]
                lengths = [len(ids) for ids in self.tokenizer(chunk, truncation=False)["input_ids"]]
                encoded = self.tokenizer(
                    chunk, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
                )
                probabilities = self.model(**encoded).logits.softmax(dim=-1)
                scores, labels = probabilities.max(dim=-1)
                for score, label, length in zip(scores.tolist(), labels.tolist(), lengths):
                    results.append({
                        "label": self.model.config.id2label[label].lower(),
                        "score": score,
                        "truncated": length > MAX_TOKENS,
                    })
        return results
