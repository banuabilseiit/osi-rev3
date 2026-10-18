"""Классификатор статей расходов по смыслу строки (категории из vocab.py).

Обучается на синтетических формулировках, проверяется на строках из
реальных отчётов (data/real_lines.csv). Сравниваются несколько моделей.
"""
import random
import re
from pathlib import Path

import numpy as np

from .vocab import CATEGORIES, FORMS, PHRASES, VENDORS

OCR_NOISE = {"о": "0", "О": "0", "е": "e", "а": "a", "с": "c", "р": "p",
             "н": "H", "и": "n"}
PREFIXES = ["", "", "за услуги по ", "оплата за ", "расходы на ", "иные расходы ",
            "услуги по ", "оплата "]


def _noise(text, rng, rate=0.03):
    out = []
    for ch in text:
        if ch in OCR_NOISE and rng.random() < rate:
            out.append(OCR_NOISE[ch])
        elif rng.random() < rate / 3:
            continue
        else:
            out.append(ch)
    return "".join(out)


def make_training(n_per_class=250, seed=0, phrases=None):
    """Синтетические строки: фраза + префикс + контрагент + шум OCR."""
    rng = random.Random(seed)
    phrases = phrases or PHRASES
    texts, labels = [], []
    for cat in CATEGORIES:
        pool = phrases[cat]
        for _ in range(n_per_class):
            text = rng.choice(PREFIXES) + rng.choice(pool)
            roll = rng.random()
            if roll < 0.5:
                text = (f'{rng.choice(FORMS)} "{rng.choice(VENDORS)}" '
                        f'({text})')
            elif roll < 0.6:
                text = f"{text} за {rng.choice(['январь', 'март', 'июль'])} 2026 год"
            if rng.random() < 0.15:
                text = text.upper()
            texts.append(_noise(text, rng))
            labels.append(cat)
    return texts, labels


def clean(text):
    text = re.sub(r"\b(тоо|ип|ао|too)\b", " ", text.lower().replace("ё", "е"))
    text = re.sub(r'["«»()]', " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_models():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import ComplementNB
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    def char():
        return TfidfVectorizer(preprocessor=clean, analyzer="char_wb",
                               ngram_range=(2, 5), sublinear_tf=True)

    def word():
        return TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2),
                               sublinear_tf=True)

    return {
        "tfidf_char+logreg": make_pipeline(
            char(), LogisticRegression(max_iter=2000, C=10)),
        "tfidf_word+logreg": make_pipeline(
            word(), LogisticRegression(max_iter=2000, C=10)),
        "tfidf_char+svm": make_pipeline(char(), LinearSVC(C=1.0)),
        "tfidf_word+nb": make_pipeline(word(), ComplementNB(alpha=0.3)),
    }


def embedding_model():
    """Эмбеддинги мультиязычной модели + LogReg (нужен sentence-transformers
    и доступ к Hugging Face; в Colab/Space работает)."""
    from sentence_transformers import SentenceTransformer
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    class Embed(BaseEstimator, TransformerMixin):
        def __init__(self, name="paraphrase-multilingual-MiniLM-L12-v2"):
            self.name = name

        def fit(self, X, y=None):
            self.model_ = SentenceTransformer(self.name)
            return self

        def transform(self, X):
            return self.model_.encode(list(X), normalize_embeddings=True)

    return make_pipeline(Embed(), LogisticRegression(max_iter=2000, C=10))


def load_real(path="data/real_lines.csv"):
    import pandas as pd
    df = pd.read_csv(path)
    return df["text"].tolist(), df["category"].tolist()


def compare(real_path="data/real_lines.csv", seed=0, with_embeddings=False):
    """Таблица: accuracy / macro-F1 на отложенных фразах и на реальных строках."""
    from sklearn.metrics import accuracy_score, f1_score
    rng = random.Random(seed)
    train_ph, test_ph = {}, {}
    for cat, items in PHRASES.items():
        items = items[:]
        rng.shuffle(items)
        cut = max(1, int(len(items) * 0.7))
        train_ph[cat], test_ph[cat] = items[:cut], items[cut:] or items[:1]
    x_tr, y_tr = make_training(250, seed, train_ph)
    x_va, y_va = make_training(60, seed + 1, test_ph)
    x_re, y_re = load_real(real_path)
    x_full, y_full = make_training(250, seed)
    models, full_models = build_models(), build_models()
    if with_embeddings:
        models["embeddings+logreg"] = embedding_model()
        full_models["embeddings+logreg"] = embedding_model()
    rows = []
    for name, model in models.items():
        model.fit(x_tr, y_tr)
        p_va = model.predict(x_va)
        full_models[name].fit(x_full, y_full)
        p_re = full_models[name].predict(x_re)
        rows.append({
            "model": name,
            "synthetic_acc": round(accuracy_score(y_va, p_va), 3),
            "synthetic_f1": round(f1_score(y_va, p_va, average="macro"), 3),
            "real_acc": round(accuracy_score(y_re, p_re), 3),
            "real_f1": round(f1_score(y_re, p_re, average="macro"), 3),
        })
    return rows


class CategoryClassifier:
    """Обёртка: обучить на всех синтетических данных, сохранить, предсказывать."""

    def __init__(self, model=None):
        self.model = model

    def fit(self, seed=0):
        x, y = make_training(300, seed)
        self.model = build_models()["tfidf_char+logreg"].fit(x, y)
        return self

    def predict(self, texts):
        return list(self.model.predict(list(texts)))

    def predict_proba(self, texts):
        proba = self.model.predict_proba(list(texts))
        return self.model.classes_, proba

    def save(self, path="models/category.joblib"):
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path="models/category.joblib"):
        import joblib
        return cls(joblib.load(path))


def main():
    import json
    table = compare()
    for row in table:
        print(row)
    Path("docs").mkdir(exist_ok=True)
    Path("docs/metrics_classifier.json").write_text(
        json.dumps(table, indent=2, ensure_ascii=False))
    CategoryClassifier().fit().save()


if __name__ == "__main__":
    from osi.classifier import main as _main    # чтобы pickle видел пакет
    _main()
