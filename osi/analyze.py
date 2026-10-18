"""Главный сценарий: документы -> отчёты -> проверки -> вопросы -> итог."""
import re
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz

from .amounts import format_money
from .checks import (Finding, SEVERITY_ORDER, _leaf_rows, check_reports,
                     subject)
from .report import Report, build_report
from .vocab import CATEGORY_RU, CATEGORY_SECTION

REPORT_WORDS = ("отчет", "отчёт", "смета", "смет", "кондоминиум")
EVIDENCE_MIN = 20_000
AMOUNT_TOL = 1.0
WEIGHT = {"taxes": 0.3, "chair_salary": 0.3, "bank": 0.2, "accounting": 0.5}
SEVERITY_POINTS = {"high": 12, "medium": 4, "info": 0}


@dataclass
class Question:
    id: str
    text: str
    kind: str                 # receipt | document
    amount: float = None
    row_label: str = ""
    resolved_by: str = ""

    @property
    def resolved(self):
        return bool(self.resolved_by)


@dataclass
class Analysis:
    reports: list = field(default_factory=list)
    receipts: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    questions: list = field(default_factory=list)
    categories: object = None          # pandas.DataFrame
    classifier: object = None

    @property
    def open_questions(self):
        return [q for q in self.questions if not q.resolved]

    @property
    def score(self):
        pts = sum(SEVERITY_POINTS[f.severity] for f in self.findings)
        pts += 2 * len(self.open_questions)
        return min(100, pts)

    @property
    def risk(self):
        s = self.score
        return "низкий" if s < 15 else "средний" if s < 40 else "высокий"


def is_report(pages):
    from .ingest import page_header_text
    text = page_header_text(pages[0]).lower()
    return any(w in text for w in REPORT_WORDS)


def load_files(paths, reader=None):
    """Раскладывает файлы на отчёты и подтверждающие документы."""
    from .ingest import load_document
    from .receipts import read_receipt, read_statement
    reports, receipts, skipped = [], [], []
    for path in map(Path, paths):
        suffix = path.suffix.lower()
        try:
            if suffix in (".csv", ".xlsx", ".xls"):
                receipts += read_statement(path)
                continue
            pages = load_document(path, reader)
            if is_report(pages):
                reports.append(build_report(pages, path.name))
            else:
                receipts.append(read_receipt(path, reader))
        except Exception as exc:                      # noqa: BLE001
            skipped.append((path.name, str(exc)[:120]))
    return reports, receipts, skipped


def classify_rows(reports, classifier):
    """DataFrame: отчёт, строка, категория, вероятность, сумма."""
    import pandas as pd
    records = []
    for rep in reports:
        rows = list(_leaf_rows(rep))
        if not rows or classifier is None:
            continue
        texts = [subject(r.text) for r in rows]
        classes, proba = classifier.predict_proba(texts)
        for row, text, p in zip(rows, texts, proba):
            best = int(p.argmax())
            records.append({"отчёт": rep.label, "строка": text[:90],
                            "код": row.code or "", "раздел": row.section,
                            "категория": classes[best],
                            "уверенность": round(float(p[best]), 2),
                            "сумма": row.amount, "idx": row.idx})
    return pd.DataFrame(records)


def category_findings(frame):
    """Строка по смыслу относится к другому разделу отчёта."""
    out = []
    if frame is None or frame.empty:
        return out
    for rec in frame.itertuples():
        expected = CATEGORY_SECTION.get(rec.категория)
        if (rec.уверенность >= 0.6 and rec.раздел and expected
                and expected != rec.раздел and rec.категория != "other"):
            out.append(Finding(
                "category_section", "info", "Строка, возможно, не в том разделе",
                f"«{rec.строка[:60]}» похожа на «{CATEGORY_RU[rec.категория]}»,"
                f" но указана в другом разделе отчёта ({rec.отчёт}).",
                rec.отчёт, rec.idx, rec.код, rec.сумма))
    return out


def match_evidence(reports, receipts):
    """Сопоставляет строки расходов и документы по сумме (и продавцу)."""
    matches = {}
    for rep in reports:
        for row in _leaf_rows(rep):
            for doc in receipts:
                if doc.total is None or row.amount is None:
                    continue
                same = abs(doc.total - row.amount) <= AMOUNT_TOL
                near_vendor = bool(doc.vendor) and fuzz.partial_ratio(
                    re.sub(r"\W+", "", doc.vendor.lower()),
                    re.sub(r"\W+", "", row.text.lower())) >= 85
                if same or (near_vendor and
                            abs(doc.total - row.amount) <= 0.005 * row.amount):
                    matches[(rep.label, row.idx)] = doc
                    break
    return matches


def build_questions(reports, findings, frame, matches):
    questions = []
    seen = set()
    for f in findings:
        if f.severity == "info" or not f.ask or f.ask in seen:
            continue
        seen.add(f.ask)
        questions.append(Question(
            f"f{len(questions)}",
            f"{f.ask}. Причина: {f.title.lower()} ({f.report}) — "
            f"{f.detail}",
            "document", f.amount, f.report))
    cands = []
    for rep in reports:
        cats = {}
        if frame is not None and not frame.empty:
            sub = frame[frame["отчёт"] == rep.label]
            cats = dict(zip(sub["idx"], sub["категория"]))
        for row in _leaf_rows(rep):
            if row.amount is None or row.amount < EVIDENCE_MIN:
                continue
            if (rep.label, row.idx) in matches:
                continue
            weight = WEIGHT.get(cats.get(row.idx, ""), 1.0)
            cands.append((row.amount * weight, rep, row))
    cands.sort(key=lambda c: -c[0])
    for _, rep, row in cands[:6]:
        questions.append(Question(
            f"r{len(questions)}",
            f"Загрузите чек, акт или договор по расходу «{subject(row.text)[:80]}» "
            f"на {format_money(row.amount)} ₸ ({rep.label}).",
            "receipt", row.amount, rep.label))
    return questions


def analyze(reports, receipts, classifier=None, skipped=None):
    findings = check_reports(reports)
    frame = classify_rows(reports, classifier)
    findings += category_findings(frame)
    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.report))
    matches = match_evidence(reports, receipts)
    questions = build_questions(reports, findings, frame, matches)
    return Analysis(reports, receipts, skipped or [], findings, questions,
                    frame, classifier)


def apply_uploads(analysis, new_receipts):
    """Новые документы закрывают вопросы с совпавшей суммой."""
    analysis.receipts += new_receipts
    for question in analysis.open_questions:
        if question.kind != "receipt" or question.amount is None:
            continue
        for doc in new_receipts:
            if doc.total is not None and abs(doc.total - question.amount) \
                    <= max(AMOUNT_TOL, 0.005 * question.amount):
                question.resolved_by = doc.filename
                break
    return analysis


def summary_markdown(analysis):
    """Итоговый отчёт ревизора в Markdown."""
    lines = ["# Итоговый отчёт ИИ-ревизора ОСИ", ""]
    labels = ", ".join(r.label for r in analysis.reports) or "—"
    lines += [f"**Проанализировано отчётов:** {len(analysis.reports)} "
              f"({labels}); подтверждающих документов: "
              f"{len(analysis.receipts)}.",
              f"**Оценка риска:** {analysis.risk} ({analysis.score} из 100).",
              ""]
    groups = [("high", "Серьёзные нарушения"), ("medium", "Требует пояснений"),
              ("info", "К сведению")]
    for sev, title in groups:
        items = [f for f in analysis.findings if f.severity == sev]
        if not items:
            continue
        lines.append(f"## {title} ({len(items)})")
        lines += [f"- **{f.report}** — {f.title}. {f.detail}" for f in items]
        lines.append("")
    done = [q for q in analysis.questions if q.resolved]
    todo = analysis.open_questions
    lines.append(f"## Подтверждение документами: закрыто {len(done)} из "
                 f"{len(analysis.questions)}")
    lines += [f"- ✅ {q.text[:110]} → {q.resolved_by}" for q in done]
    lines += [f"- ❓ {q.text}" for q in todo]
    lines += ["", "_Отчёт подготовлен автоматически и не заменяет решение "
              "ревизионной комиссии: каждое замечание нужно проверить по "
              "первичным документам._"]
    return "\n".join(lines)
