"""Графики для EDA и результатов экспериментов (сохраняются в docs/).

Запуск: python scripts/make_figures.py [файлы отчётов ...]
Без аргументов строит графики только по синтетическим данным.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osi.analyze import classify_rows     # noqa: E402
from osi.classifier import (CategoryClassifier, build_models, load_real,  # noqa: E402
                            make_training)
from osi.report import SECTION_TITLES     # noqa: E402
from osi.vocab import CATEGORY_RU         # noqa: E402

DOCS = Path("docs")


def fig_sections(reports):
    """Структура расходов по разделам для месячных отчётов."""
    data = {}
    for rep in reports:
        if rep.kind != "monthly":
            continue
        data[rep.label] = {
            SECTION_TITLES[r.section]: r.amount or 0
            for r in rep.rows if r.role in SECTION_TITLES}
    frame = pd.DataFrame(data).fillna(0) / 1e6
    ax = frame.T.plot(kind="bar", stacked=True, figsize=(7, 4))
    ax.set_ylabel("млн ₸")
    ax.set_title("Расходы по разделам отчёта, месяцы")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(DOCS / "eda_sections.png", dpi=130)
    plt.close()


def fig_plan_fact(reports):
    """Годовой отчёт: план и факт по основным статьям (глубина 2)."""
    for rep in reports:
        if rep.kind != "annual":
            continue
        rows = [r for r in rep.rows if r.zone == "expense" and r.depth == 2
                and r.plan and r.fact]
        labels = [r.code for r in rows]
        x = range(len(rows))
        plt.figure(figsize=(7, 4))
        plt.bar([i - 0.2 for i in x], [r.plan / 1e6 for r in rows], 0.4,
                label="план")
        plt.bar([i + 0.2 for i in x], [r.fact / 1e6 for r in rows], 0.4,
                label="факт")
        plt.xticks(list(x), labels)
        plt.ylabel("млн ₸")
        plt.title("Годовой отчёт: план и факт по разделам")
        plt.legend()
        plt.tight_layout()
        plt.savefig(DOCS / "eda_plan_fact.png", dpi=130)
        plt.close()
        return


def fig_categories(reports, classifier):
    frame = classify_rows([r for r in reports if r.kind == "monthly"],
                          classifier)
    if frame.empty:
        return
    share = (frame.assign(категория=frame["категория"].map(CATEGORY_RU))
             .groupby("категория")["сумма"].sum().sort_values())
    ax = (share / 1e6).plot(kind="barh", figsize=(7, 5))
    ax.set_xlabel("млн ₸ (3 месяца)")
    ax.set_title("Расходы по смысловым категориям (классификатор)")
    plt.tight_layout()
    plt.savefig(DOCS / "eda_categories.png", dpi=130)
    plt.close()


def fig_confusion():
    from sklearn.metrics import ConfusionMatrixDisplay
    x, y = make_training(250, 0)
    model = build_models()["tfidf_char+logreg"].fit(x, y)
    x_real, y_real = load_real()
    ConfusionMatrixDisplay.from_predictions(
        y_real, model.predict(x_real), xticks_rotation=90,
        colorbar=False, ax=plt.subplots(figsize=(8, 8))[1])
    plt.title("Реальные строки отчётов: матрица ошибок")
    plt.tight_layout()
    plt.savefig(DOCS / "classifier_confusion.png", dpi=120)
    plt.close()


def fig_model_comparison():
    path = DOCS / "metrics_classifier.json"
    if not path.exists():
        return
    table = pd.DataFrame(json.loads(path.read_text()))
    ax = table.set_index("model")[["synthetic_f1", "real_f1"]].plot(
        kind="bar", figsize=(7, 4))
    ax.set_ylabel("macro-F1")
    ax.set_title("Сравнение моделей классификации статей")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(DOCS / "classifier_models.png", dpi=130)
    plt.close()


def fig_checks():
    path = DOCS / "metrics_checks.json"
    if not path.exists():
        return
    table = pd.DataFrame(json.loads(path.read_text())).T
    ax = table[["precision", "recall"]].plot(kind="bar", figsize=(8, 4))
    ax.set_ylim(0, 1.05)
    ax.set_title("Проверки на синтетических сериях с внедрёнными нарушениями")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(DOCS / "checks_metrics.png", dpi=130)
    plt.close()


def main(paths):
    DOCS.mkdir(exist_ok=True)
    reports = []
    if paths:
        from osi.analyze import load_files
        reports, _, _ = load_files(paths)
    classifier = CategoryClassifier.load()
    if reports:
        fig_sections(reports)
        fig_plan_fact(reports)
        fig_categories(reports, classifier)
    fig_confusion()
    fig_model_comparison()
    fig_checks()


if __name__ == "__main__":
    main(sys.argv[1:])
