import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple

import torch
from flask import Flask, jsonify, render_template, request
from requests import RequestException
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import requests
from bs4 import BeautifulSoup



BASE_DIR = Path(__file__).parent
ENV_FILE = BASE_DIR / ".env"
MODEL_PATH = BASE_DIR / "model" / "distilbert_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_local_env(path: Path) -> None:
    """Minimal .env loader so env vars work when running `python app.py` directly."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


load_local_env(ENV_FILE)

app = Flask(__name__)
NEWS_API_ENDPOINT = os.getenv("NEWS_API_ENDPOINT", "https://newsapi.org/v2/top-headlines")
NEWS_API_KEYS = [key.strip() for key in os.getenv("NEWS_API_KEYS", "").split(",") if key.strip()]
if not NEWS_API_KEYS:
    single_key = os.getenv("NEWS_API_KEY", "").strip()
    if single_key:
        NEWS_API_KEYS = [single_key]
NEWS_API_PARAMS = {"language": "en", "pageSize": 1}
_news_key_lock = Lock()
_news_key_index = 0


@dataclass
class PredictionResult:
    label: str
    confidence: float
    input_text: str
    error: Optional[str] = None


def load_model():
    """
    Load tokenizer and model from local directory.
    Returns (tokenizer, model) ready for inference.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        # Set human-friendly labels when not present; default to FAKE=0, REAL=1
        if not model.config.id2label or set(model.config.id2label.keys()) == {0, 1}:
            model.config.id2label = {0: "FAKE", 1: "REAL"}
            model.config.label2id = {"FAKE": 0, "REAL": 1}
        return tokenizer, model
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load model")
        raise RuntimeError(f"Unable to load model from {MODEL_PATH}: {exc}") from exc


TOKENIZER, MODEL = load_model()


def fetch_article_text(url: str, timeout: int = 8) -> str:
    """Fetch and extract readable text from a URL."""
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "AI-News-Detector/1.0"},
        )
        response.raise_for_status()
    except RequestException as exc:
        raise ValueError(f"Could not fetch the URL: {exc}") from exc

    soup = BeautifulSoup(response.text, "html.parser")
    # Collect visible paragraph text
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    text = "\n".join(p for p in paragraphs if p)
    if not text:
        raise ValueError("No readable text found at the provided URL.")
    return text


def classify_text(text: str) -> PredictionResult:
    if not text or not text.strip():
        return PredictionResult(label="", confidence=0.0, input_text="", error="Input text is empty.")

    encoded = TOKENIZER(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = MODEL(**encoded)
        logits = outputs.logits.squeeze()
        probabilities = torch.softmax(logits, dim=-1)
        confidence, predicted_class = torch.max(probabilities, dim=-1)

    label = MODEL.config.id2label.get(predicted_class.item(), f"LABEL_{predicted_class.item()}")
    return PredictionResult(label=label, confidence=confidence.item(), input_text=text)


def process_input(mode: str, text_input: str, url_input: str) -> Tuple[Optional[str], Optional[PredictionResult]]:
    if mode == "url":
        if not url_input:
            return "Please provide a URL to analyze.", None
        try:
            extracted_text = fetch_article_text(url_input)
        except ValueError as exc:
            return str(exc), None
        return None, classify_text(extracted_text)

    # default to text mode
    if not text_input:
        return "Please provide article text to analyze.", None
    return None, classify_text(text_input)


def get_next_news_api_key() -> Optional[str]:
    if not NEWS_API_KEYS:
        return None
    global _news_key_index
    with _news_key_lock:
        key = NEWS_API_KEYS[_news_key_index % len(NEWS_API_KEYS)]
        _news_key_index = (_news_key_index + 1) % len(NEWS_API_KEYS)
    return key


def fetch_latest_headline(api_key: str, timeout: int = 8) -> Tuple[str, str]:
    if not api_key:
        raise ValueError("Missing news API key.")
    response = requests.get(
        NEWS_API_ENDPOINT,
        params={**NEWS_API_PARAMS, "apiKey": api_key},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    articles = payload.get("articles") or []
    if not articles:
        raise ValueError("No articles returned from news provider.")
    article = articles[0]
    title = (article.get("title") or "").strip()
    description = (article.get("description") or "").strip()
    if not title and not description:
        raise ValueError("Article missing title/description.")
    return title, description


@app.route("/", methods=["GET", "POST"])
def index():
    error_message: Optional[str] = None
    result: Optional[PredictionResult] = None
    selected_mode = "text"
    input_value = {"text": "", "url": ""}

    if request.method == "POST":
        selected_mode = request.form.get("mode", "text")
        text_input = request.form.get("article_text", "").strip()
        url_input = request.form.get("article_url", "").strip()
        input_value = {"text": text_input, "url": url_input}

        error_message, result = process_input(selected_mode, text_input, url_input)
    return render_template(
        "index.html",
        error=error_message,
        result=result,
        mode=selected_mode,
        input_value=input_value,
    )


@app.route("/api/live-news", methods=["GET"])
def live_news():
    api_key = get_next_news_api_key()
    if not api_key:
        return jsonify(
            {
                "status": "error",
                "message": "NEWS_API_KEYS not configured.",
                "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        )
    try:
        title, description = fetch_latest_headline(api_key)
        combined_text = f"{title} {description}".strip()
        prediction = classify_text(combined_text)
        return jsonify(
            {
                "status": "ok",
                "headline": title,
                "description": description,
                "label": prediction.label,
                "confidence": prediction.confidence,
                "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Live news update failed: %s", exc)
        return jsonify(
            {
                "status": "error",
                "message": "Waiting for next update...",
                "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
