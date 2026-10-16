"""The UI calls the API over HTTP; it never loads the model itself."""
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from app.csv_input import validate_reviews
from app.settings import MAX_TEXT_CHARS

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="Customer Review Analyzer", page_icon="💬", layout="wide")
st.title("Customer Review Analyzer")
st.write("Turn customer feedback into a clear view of positive and negative sentiment.")
st.caption("English reviews · Pretrained DistilBERT · Two sentiment classes")


def post_prediction(path, payload):
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=(5, 180))
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        st.error("Could not analyze reviews. Check that the API is running, then try again.")
        return None


single, batch = st.tabs(["One review", "Review collection"])
with single:
    with st.form("single_review"):
        text = st.text_area("Your review", placeholder="The headphones sound great and arrived quickly.", max_chars=MAX_TEXT_CHARS)
        submitted = st.form_submit_button("Analyze review", type="primary")
    if submitted:
        st.session_state.pop("single_result", None)
        if not text.strip():
            st.warning("Enter a review first.")
        else:
            with st.spinner("Reading your review…"):
                result = post_prediction("/predict", {"text": text})
            if result:
                st.session_state.single_result = (text, result)
    if "single_result" in st.session_state:
        analyzed_text, result = st.session_state.single_result
        st.caption("Result for the submitted review")
        st.write(analyzed_text)
        col1, col2 = st.columns(2)
        col1.metric("Predicted sentiment", result["label"].capitalize())
        col2.metric("Model score", f"{result['score']:.1%}")
        if result["truncated"]:
            st.warning("This review exceeded the model's limit. Only the first 512 tokens were analyzed.")

with batch:
    st.write("Upload a CSV with a **text** column. Maximum 100 reviews, 2 MB, 5,000 characters per review.")
    st.download_button("Download example CSV", Path(__file__).with_name("examples").joinpath("reviews.csv").read_bytes(), "example_reviews.csv", "text/csv")
    upload = st.file_uploader("Review CSV", type="csv")
    if upload is not None:
        # Discard old predictions when the file changes, including same-name replacements.
        content = upload.getvalue()
        if st.session_state.get("upload_content") != content:
            st.session_state.pop("batch_results", None)
            st.session_state.upload_content = content
        try:
            if len(content) > 2 * 1024 * 1024:
                raise ValueError("Upload a CSV smaller than 2 MB.")
            frame = pd.read_csv(upload, dtype={"text": str}, keep_default_na=False)
            texts = validate_reviews(frame)
            st.dataframe(frame.head(5), hide_index=True, width="stretch")
            if st.button("Analyze collection", type="primary"):
                st.session_state.pop("batch_results", None)
                with st.spinner(f"Analyzing {len(texts)} reviews…"):
                    result = post_prediction("/predict/batch", {"texts": texts})
                if result:
                    # Only export the submitted text and new predictions; avoid column collisions.
                    st.session_state.batch_results = pd.concat(
                        [pd.DataFrame({"text": texts}), pd.DataFrame(result["predictions"])], axis=1
                    )
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(str(error))
    else:
        st.session_state.pop("batch_results", None)
    if "batch_results" in st.session_state:
        results = st.session_state.batch_results
        counts = results["label"].value_counts().reindex(["positive", "negative"], fill_value=0)
        total, positive, negative = st.columns(3)
        total.metric("Reviews analyzed", len(results))
        positive.metric("Positive", f"{counts['positive'] / len(results):.0%}")
        negative.metric("Negative", f"{counts['negative'] / len(results):.0%}")
        st.bar_chart(counts.rename("Reviews"), color="#339989")
        if results["truncated"].any():
            st.warning("Some reviews were truncated to the first 512 tokens. See the truncated column.")
        st.dataframe(results, hide_index=True, width="stretch")
        st.download_button("Download predictions", results.to_csv(index=False).encode("utf-8"), "review_predictions.csv", "text/csv")

st.divider()
st.caption("A model score is not a guarantee of correctness. Neutral, mixed, sarcastic, or non-English reviews may be misclassified. Reviews are processed by your local API; the model downloads from Hugging Face on first startup.")
