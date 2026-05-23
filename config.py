"""
Central configuration for the FinBERT-LSTM project.
Edit values here; all modules import from this file.
"""

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
DATA_PATH = "data/1_min_SPY_2008-2021.csv"   # SPY 1-min OHLCV CSV
DATE_COL = "date"
TARGET_COL = "close"
FEATURE_COLS = ["volume", "high"]

# ──────────────────────────────────────────────
# Splits
# ──────────────────────────────────────────────
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# ──────────────────────────────────────────────
# Fuzzy logic
# ──────────────────────────────────────────────
FUZZY_UNIVERSE_MIN = -3.5
FUZZY_UNIVERSE_MAX = 3.5
FUZZY_UNIVERSE_STEP = 0.1

# ──────────────────────────────────────────────
# Sequence parameters
# ──────────────────────────────────────────────
LOOKBACK = 24    # input window (time steps)
HORIZON = 1      # forecast horizon (time steps ahead)

# ──────────────────────────────────────────────
# Transformer hyperparameters
# ──────────────────────────────────────────────
D_MODEL = 128
N_HEAD = 4
NUM_ENCODER_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
BATCH_SIZE = 64
N_EPOCHS = 5
LEARNING_RATE = 1e-4

# ──────────────────────────────────────────────
# FinBERT fine-tuning
# ──────────────────────────────────────────────
FINBERT_MODEL_ID = "ProsusAI/finbert"
FINBERT_MAX_LEN = 256
FINBERT_EPOCHS = 6
FINBERT_LR = 2e-5
FINBERT_BATCH_TRAIN = 4
FINBERT_BATCH_VALID = 8

# ──────────────────────────────────────────────
# Scraping
# ──────────────────────────────────────────────
ET_MAX_SCROLL_STEPS = 60
ET_SLEEP_S = 1.0
ET_HEADLESS = True
ET_PAGE_TIMEOUT = 20

# ──────────────────────────────────────────────
# MongoDB
# ──────────────────────────────────────────────
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "NLP"
MONGO_COLLECTION = "nlp"
