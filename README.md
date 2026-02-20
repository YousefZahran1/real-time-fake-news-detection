# Real-Time Fake News Detection using DistilBERT

An end-to-end web-based fake news detection system built with Flask and a fine-tuned DistilBERT transformer model.  
This project was developed as part of a Final Year Project (FYP) focusing on real-time NLP-based misinformation classification.

---

## Project Overview

The system classifies news content as **REAL** or **FAKE** using a fine-tuned DistilBERT model optimized for efficient inference.

The objectives of this project are:

- Achieve high classification accuracy using transformer-based NLP
- Maintain low inference latency for real-time usability
- Provide an interactive and user-friendly web interface
- Enable fully local execution without external API dependency

This implementation demonstrates how optimized transformer models can be deployed in practical web applications.

---

## Key Features

- Fine-tuned DistilBERT binary classifier (REAL vs FAKE)
- Fully local model inference (no external API calls required)
- Two input modes:
  - Direct text input
  - URL-based article fetching
- Confidence score output
- Clean, responsive UI built with Flask templates
- Automatic GPU usage if available (falls back to CPU)

---

## System Architecture

The system follows a modular pipeline:

1. User input (Text or URL)
2. Preprocessing and tokenization
3. DistilBERT inference
4. Probability scoring
5. Label output with confidence visualization

The model and tokenizer are stored locally to ensure reproducibility and offline operation.

---

## Project Structure

```
project/
├── app.py
├── model/
│   └── distilbert_model/        # Fine-tuned model + tokenizer assets
├── templates/
│   └── index.html               # Flask template and UI layout
├── static/
│   └── style.css                # UI styling
├── requirements.txt
└── README.md
```

---

## Model Information

Base Model: `distilbert-base-uncased`  
Fine-tuned for binary fake news classification.

The directory `model/distilbert_model/` contains:

- model.safetensors (trained weights)
- config.json
- tokenizer files
- vocabulary files

The model runs locally using HuggingFace Transformers.

---

## Requirements

- Python 3.9 or higher (recommended 3.10+)
- pip
- Virtual environment (recommended)

Main Libraries:

- Flask
- PyTorch
- Transformers (HuggingFace)
- scikit-learn
- numpy
- pandas

All dependencies are listed in `requirements.txt`.

---

## Installation & Setup

### 1. Clone Repository

```
git clone https://github.com/YousefZahran1/real-time-fake-news-detection.git
cd real-time-fake-news-detection
```

### 2. Create Virtual Environment

```
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application

```
python app.py
```

The application will run at:

```
http://127.0.0.1:5000
```

---

## Usage

1. Select input mode (Text or URL).
2. Paste article text or provide a news article URL.
3. Click **Analyze Content**.
4. View predicted label (**REAL** or **FAKE**) and confidence score.

---

## Error Handling

- Empty input: prompts user to provide content.
- Invalid/unreachable URL: clear error message displayed.
- Model loading issues: shown in server logs and UI error banner.

---

## Hardware Support

- Automatically uses GPU if available.
- Falls back to CPU if no GPU is detected.
- No special hardware required for basic operation.

---

## Notes for Replacing the Model

If using a different fine-tuned DistilBERT checkpoint:

- Replace files inside `model/distilbert_model/`
- Keep tokenizer files alongside model weights
- Ensure configuration files remain consistent

---

## Academic Context

This project was developed as a Final Year Project (FYP) exploring:

- Transformer-based fake news detection
- Model optimization for real-time deployment
- Practical implementation of NLP research in web systems

The work demonstrates how lightweight transformer architectures such as DistilBERT can achieve strong performance while remaining deployable in real-world environments.

---

## License

This project is developed for academic and research purposes.
