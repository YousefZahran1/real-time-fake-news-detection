# AI News Detector

End-to-end fake news detection web application built with Flask and a fine-tuned DistilBERT model.

## Features
- Local DistilBERT classifier for binary fake news detection (REAL vs FAKE).
- Two input modes: paste article text or fetch an article by URL.
- Near-instant inference with confidence scoring and colored labels.
- Dark, gradient-themed UI with interactive toggle controls (no JS frameworks).

## Project Structure
```
project/
├── app.py
├── model/
│   └── distilbert_model/        # fine-tuned model + tokenizer assets
├── templates/
│   └── index.html               # Flask template and UI markup
├── static/
│   └── style.css                # Styling for the UI
├── requirements.txt
└── README.md
```

## Prerequisites
- Python 3.9+ recommended.
- The fine-tuned model weights are already placed in `model/distilbert_model/` (using `model.safetensors` and config). Tokenizer assets from `distilbert-base-uncased` are included for fully local loading.

## Setup
1) Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
```

2) Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3) Run the Flask server:
```bash
python app.py
# App runs at http://127.0.0.1:5000
```

## Usage
1. Choose **Text** or **URL** using the toggle.
2. Paste article text or enter a URL to fetch.
3. Click **Analyze Content** to view the predicted label (REAL/FAKE) and confidence bar.

## Error Handling
- Empty input: prompts for content.
- Invalid/unreachable URL: clear error message.
- Model loading issues: surfaced in server logs and UI error banner when applicable.

## Notes
- The model runs on GPU when available; otherwise CPU is used automatically.
- If you swap in a different fine-tuned DistilBERT checkpoint, replace the files inside `model/distilbert_model/` (keep tokenizer assets alongside the weights).
