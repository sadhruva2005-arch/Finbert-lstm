# FinBERT-LSTM: Sentiment-Enhanced Stock Price Forecasting

A hybrid NLP + Deep Learning pipeline that combines **FinBERT** (financial sentiment analysis) with a **Fuzzy-Transformer** model for stock market forecasting. News headlines are scraped, sentiment is extracted, and fuzzy-encoded market features are fed into a Transformer encoder to predict future prices.

---

## Repository Structure

```
Finbert-lstm/
│
├── config.py                          # All hyperparameters in one place
├── requirements.txt
├── .gitignore
│
├── src/                               # Importable Python modules
│   ├── scraper/
│   │   ├── et_scraper.py              # Selenium scraper for Economic Times
│   │   └── rss_scraper.py             # RSS scraper (Reuters + Google News)
│   │
│   ├── preprocessing/
│   │   ├── text_preprocessing.py      # NLTK pipeline (tokenise, lemmatise, chunk)
│   │   └── data_utils.py              # Column detection, splits, weak labels
│   │
│   ├── models/
│   │   ├── finbert.py                 # FinBERT fine-tuning + inference class
│   │   └── transformer.py             # Fuzzy-Transformer + training loop
│   │
│   ├── features/
│   │   ├── tfidf.py                   # Custom TF-IDF from token lists
│   │   └── fuzzification.py           # skfuzzy membership-value features
│   │
│   └── evaluation/
│       └── metrics.py                 # Classification + regression metrics
│
├── notebooks/
│   └── NLP_DL_Project_code-2_1.ipynb  # End-to-end notebook (full pipeline)
│
├── data/                              # Place your CSV files here (git-ignored)
│   └── .gitkeep
│
└── outputs/                           # Model outputs, predictions (git-ignored)
    └── .gitkeep
```

---

## Pipeline Overview

```
Economic Times / RSS News
         │
         ▼
   [src/scraper]           ← Selenium + BeautifulSoup / RSS XML
         │
         ▼
[src/preprocessing]        ← Lowercase → stopwords → lemmatise → POS → chunk
         │
         ▼
  [src/features/tfidf]     ← TF-IDF document-term matrix
         │
         ▼
 [src/models/finbert]      ← Fine-tune ProsusAI/finbert (binary sentiment)
         │  daily sentiment scores
         ▼
  SPY 1-min OHLCV data
         │
         ▼
[src/features/fuzzify]     ← crisp features → {low, medium, high} MFs
         │
         ▼
[src/models/transformer]   ← Fuzzy-Transformer encoder → price forecast
         │
         ▼
 [src/evaluation/metrics]  ← MAE, RMSE, R² / Accuracy, F1, AUC
```

---

## Tech Stack

| Layer | Library |
|---|---|
| NLP model | `transformers` (ProsusAI/finbert) |
| Deep learning | `torch` |
| Web scraping | `selenium`, `beautifulsoup4`, `requests` |
| Fuzzy logic | `scikit-fuzzy` |
| Data | `pandas`, `numpy` |
| NLP utilities | `nltk` |
| Database | `pymongo` (MongoDB) |
| Visualisation | `matplotlib`, `seaborn` |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure paths

Edit `config.py` to set your data path and hyperparameters:

```python
DATA_PATH = "data/1_min_SPY_2008-2021.csv"
TARGET_COL = "close"
FEATURE_COLS = ["volume", "high"]
```

### 3. Run the full pipeline (notebook)

```bash
jupyter notebook notebooks/NLP_DL_Project_code-2_1.ipynb
```

Or use the modules programmatically:

```python
# Step 1 – Scrape news
from src.scraper.rss_scraper import scrape_rss_news
df_news = scrape_rss_news()

# Step 2 – Preprocess text
from src.preprocessing.text_preprocessing import preprocess_column
df_news["tokens"] = preprocess_column(df_news["text"])

# Step 3 – Fine-tune FinBERT
from src.preprocessing.data_utils import build_labels, temporal_split
from src.models.finbert import FinBERTTrainer

labels = build_labels(df_news, "text").tolist()
train_df, val_df, _ = temporal_split(df_news)
trainer = FinBERTTrainer(
    train_df["text"].tolist(), labels[:len(train_df)],
    val_df["text"].tolist(),   labels[len(train_df):len(train_df)+len(val_df)],
)
trainer.train(epochs=6)
df_news = trainer.predict(df_news, text_col="text")

# Step 4 – Fuzzify market features
from src.features.fuzzification import fuzzify_dataframe
import pandas as pd
market_df = pd.read_csv("data/1_min_SPY_2008-2021.csv")
fuzz_features = fuzzify_dataframe(market_df, ["volume", "high"])

# Step 5 – Train Transformer
from src.models.transformer import TransformerModel, create_sequences, make_dataloaders, train_transformer
# ... (see notebook for full sequence building and training)

# Step 6 – Evaluate
from src.evaluation.metrics import regression_metrics, print_metrics
m = regression_metrics(y_true, y_pred)
print_metrics(m)
```

---

## Model Details

### FinBERT (NLP)
- Base: `ProsusAI/finbert` (pre-trained on financial text)
- Fine-tuned for **binary sentiment** (positive / negative)
- Custom training loop with `tqdm` progress bars (no HuggingFace `Trainer`)
- Best checkpoint restored by validation loss

### Fuzzy-Transformer (IS2)
- Input features are fuzzified into low/medium/high membership values
- Transformer encoder with positional encoding
- Multi-step horizon forecasting with inverse-scaled outputs

---

## Evaluation

After training, expected outputs:

```
=== CLASSIFICATION ===
  accuracy: 0.87
  f1: 0.85
  auc: 0.91

=== REGRESSION ===
  mae: 0.42
  rmse: 0.61
  r2: 0.93
```

Saved to `outputs/model_outputs/nlp_metrics.json` and `is2_metrics.json`.

---

## Author

**sadhruva2005-arch** — NLP & Deep Learning Project
