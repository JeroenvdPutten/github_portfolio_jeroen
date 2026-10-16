from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests
from streamlit.testing.v1 import AppTest

from app.csv_input import validate_reviews

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("frame", [pd.DataFrame({"wrong": ["ok"]}), pd.DataFrame({"text": []}), pd.DataFrame({"text": [None]}), pd.DataFrame({"text": [" "]}), pd.DataFrame({"text": [12]}), pd.DataFrame({"text": ["x" * 5001]}), pd.DataFrame({"text": ["ok"] * 101})])
def test_invalid_csv_is_rejected(frame):
    with pytest.raises(ValueError):
        validate_reviews(frame)


def test_example_csv_is_valid():
    frame = pd.read_csv(ROOT / "examples" / "reviews.csv")
    assert len(validate_reviews(frame)) == 6


def test_dashboard_single_review_flow():
    app = AppTest.from_file(str(ROOT / "dashboard.py")).run()
    assert not app.exception
    app.text_area[0].set_value("A great product")
    response = Mock()
    response.json.return_value = {"label": "positive", "score": 0.95, "truncated": False}
    with patch("requests.post", return_value=response) as post:
        app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value == "Positive"
    post.assert_called_once_with("http://localhost:8000/predict", json={"text": "A great product"}, timeout=(5, 180))


def test_dashboard_explains_connection_error():
    app = AppTest.from_file(str(ROOT / "dashboard.py")).run()
    app.text_area[0].set_value("A great product")
    with patch("requests.post", side_effect=requests.ConnectionError):
        app.button[0].click().run()
    assert not app.exception
    assert "API is running" in app.error[0].value


def test_dashboard_csv_flow_and_replacement():
    upload = BytesIO(b'text,label\nGreat product,existing\nBroken product,existing\n')
    with patch("streamlit.file_uploader", return_value=upload):
        app = AppTest.from_file(str(ROOT / "dashboard.py")).run()
    assert not app.exception
    response = Mock()
    response.json.return_value = {"predictions": [
        {"label": "positive", "score": 0.98, "truncated": False},
        {"label": "negative", "score": 0.91, "truncated": True},
    ]}
    upload.seek(0)
    with patch("streamlit.file_uploader", return_value=upload), patch("requests.post", return_value=response):
        app.button[1].click().run()
    assert not app.exception
    assert app.metric[0].value == "2"
    assert app.metric[1].value == "50%"
    assert list(app.session_state.batch_results.columns) == ["text", "label", "score", "truncated"]
    assert "truncated" in app.warning[0].value
    # A new invalid upload must not leave results from the previous file visible.
    with patch("streamlit.file_uploader", return_value=BytesIO(b'wrong\nmissing text column\n')):
        app.run()
    assert not app.exception
    assert "text" in app.error[0].value
    assert len(app.metric) == 0
