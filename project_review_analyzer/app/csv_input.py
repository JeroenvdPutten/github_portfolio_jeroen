"""CSV validation shared by the dashboard and tests."""
import pandas as pd

from app.settings import MAX_BATCH_SIZE, MAX_TEXT_CHARS


def validate_reviews(frame: pd.DataFrame) -> list[str]:
    if "text" not in frame.columns:
        raise ValueError("Your CSV needs a column named 'text'.")
    if not 1 <= len(frame) <= MAX_BATCH_SIZE:
        raise ValueError(f"Upload between 1 and {MAX_BATCH_SIZE} reviews.")
    if frame["text"].isna().any():
        raise ValueError("Some reviews are missing. Fill in every text cell.")
    if not frame["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("Every review must contain text.")
    texts = frame["text"].str.strip().tolist()
    if any(not text for text in texts):
        raise ValueError("Some reviews are blank. Fill in every text cell.")
    if any(len(text) > MAX_TEXT_CHARS for text in texts):
        raise ValueError(f"Each review must be at most {MAX_TEXT_CHARS:,} characters.")
    return texts
