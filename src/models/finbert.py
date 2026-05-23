"""
FinBERT fine-tuning (binary sentiment) — Trainer-free custom loop.

Usage:
    from src.models.finbert import FinBERTTrainer
    trainer = FinBERTTrainer(train_texts, train_labels, val_texts, val_labels)
    trainer.train(epochs=6)
    df_with_scores = trainer.predict(df, text_col="text")
"""

from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

MODEL_ID = "ProsusAI/finbert"
MAX_LEN = 256
BATCH_TRAIN = 4
BATCH_VALID = 8


class _EncodedDataset(Dataset):
    def __init__(self, encodings: dict, labels: list[int] | None = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


class FinBERTTrainer:
    """Fine-tunes ProsusAI/finbert for binary sentiment classification."""

    def __init__(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str],
        val_labels: list[int],
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, num_labels=2, ignore_mismatched_sizes=True
        ).to(self.device)
        self.model.config.id2label = {0: "negative", 1: "positive"}
        self.model.config.label2id = {"negative": 0, "positive": 1}

        self._dl_train = self._make_loader(train_texts, train_labels, BATCH_TRAIN, shuffle=True)
        self._dl_valid = self._make_loader(val_texts, val_labels, BATCH_VALID, shuffle=False)

    def _encode(self, texts: list[str]) -> dict:
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        )

    def _make_loader(
        self, texts: list[str], labels: list[int], batch_size: int, shuffle: bool
    ) -> DataLoader:
        enc = self._encode(texts)
        ds = _EncodedDataset(enc, labels)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def _eval(self, criterion) -> tuple[float, float]:
        self.model.eval()
        losses, correct, total = [], 0, 0
        with torch.no_grad():
            for batch in self._dl_valid:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(self.device)
                out = self.model(**inputs)
                losses.append(criterion(out.logits, labels).item())
                correct += (out.logits.argmax(-1) == labels).sum().item()
                total += labels.numel()
        return (sum(losses) / len(losses) if losses else 0.0), (correct / total if total else 0.0)

    def train(self, epochs: int = 6, lr: float = 2e-5) -> None:
        """Fine-tune the model."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        best_val_loss, best_state = float("inf"), None

        for epoch in tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch"):
            self.model.train()
            running_loss, n_batches = 0.0, 0
            for batch in tqdm(self._dl_train, desc=f"Epoch {epoch}/{epochs}", leave=False):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                out = self.model(**inputs)
                loss = criterion(out.logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1

            val_loss, val_acc = self._eval(criterion)
            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={running_loss / max(1, n_batches):.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state:
            self.model.load_state_dict(best_state)
            print(f"Restored best model (val_loss={best_val_loss:.4f}).")

    def predict_proba(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return softmax probabilities of shape (N, 2)."""
        self.model.eval()
        all_probs = []
        for i in range(0, len(texts), batch_size):
            enc = self._encode(texts[i : i + batch_size])
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            del enc, logits
            gc.collect()
        return np.vstack(all_probs) if all_probs else np.zeros((len(texts), 2), dtype=np.float32)

    def predict(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """
        Run inference and attach sentiment columns to df.

        Adds: finbert_proba_neg, finbert_proba_pos, finbert_label, finbert_score
        """
        texts = df[text_col].astype(str).fillna("").tolist()
        probs = self.predict_proba(texts)
        df = df.copy()
        df["finbert_proba_neg"] = probs[:, 0]
        df["finbert_proba_pos"] = probs[:, 1]
        df["finbert_label"] = (probs[:, 1] >= 0.5).astype(int)
        df["finbert_score"] = probs[:, 1] - probs[:, 0]
        return df

    def daily_sentiment_index(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """Aggregate finbert_score to a daily mean index."""
        return (
            df.groupby(date_col)["finbert_score"]
            .mean()
            .rename("daily_sentiment_index")
            .reset_index()
        )
