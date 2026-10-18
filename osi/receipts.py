"""Чеки и платёжные документы: сумма, дата, продавец (OCR + правила)."""
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from .amounts import parse_amount

TOTAL_WORDS = ("итого", "итог", "всего", "к оплате", "сумма", "total",
               "оплате", "оплачено", "наличными", "карта")
_AMOUNT = re.compile(r"\d[\d ]*[.,]\s?\d{2}(?!\d)")
_DATE = re.compile(r"(\d{2})[./-](\d{2})[./-](\d{4}|\d{2})")


@dataclass
class Receipt:
    filename: str
    total: float = None
    day: date = None
    vendor: str = ""
    text: str = ""

    @property
    def ok(self):
        return self.total is not None


def _page_lines(pages):
    from .layout import text_lines
    lines = []
    for page in pages:
        if not page.words:
            continue
        tol = 0.5 * float(np.median([w.height for w in page.words]))
        lines += text_lines(page.words, tol)
    return lines


def parse_total(lines):
    """Итоговая сумма: максимум в строках с ключевыми словами, иначе общий."""
    keyed, every = [], []
    for line in lines:
        values = [parse_amount(m.group(0), allow_no_separator=True)
                  for m in _AMOUNT.finditer(line)]
        values = [v for v in values if v]
        every += values
        if values and any(k in line.lower() for k in TOTAL_WORDS):
            keyed += values
    pool = keyed or every
    return max(pool) if pool else None


def parse_day(lines):
    for line in lines:
        match = _DATE.search(line)
        if match:
            d, m, y = (int(x) for x in match.groups())
            y += 2000 if y < 100 else 0
            try:
                return date(y, m, d)
            except ValueError:
                continue
    return None


def parse_vendor(lines):
    for line in lines[:6]:
        letters = re.sub(r"[^А-Яа-яA-Za-z ]", "", line).strip()
        if len(letters) >= 4 and not re.search(r"чек|касс|кассир", line.lower()):
            return line.strip()[:60]
    return ""


def read_receipt(path, reader=None):
    from .ingest import load_document
    path = Path(path)
    pages = load_document(path, reader)
    lines = _page_lines(pages)
    return Receipt(path.name, parse_total(lines), parse_day(lines),
                   parse_vendor(lines), "\n".join(lines))


def read_statement(path):
    """CSV/XLSX выписка -> список Receipt-подобных платежей (сумма, назначение)."""
    import pandas as pd
    path = Path(path)
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") \
        else pd.read_csv(path, sep=None, engine="python")
    cols = {c: str(c).lower() for c in df.columns}
    amount_col = next((c for c, n in cols.items()
                       if re.search(r"сумм|дебет|расход|amount|debit", n)), None)
    text_col = next((c for c, n in cols.items()
                     if re.search(r"назнач|описан|контрагент|purpose", n)), None)
    date_col = next((c for c, n in cols.items()
                     if re.search(r"дата|date", n)), None)
    out = []
    if amount_col is None:
        return out
    for _, row in df.iterrows():
        value = parse_amount(str(row[amount_col]).replace("\xa0", " "))
        if value is None:
            try:
                value = round(float(str(row[amount_col]).replace(" ", "")
                                    .replace(",", ".")), 2)
            except ValueError:
                continue
        day = None
        if date_col is not None:
            parsed = pd.to_datetime(row[date_col], errors="coerce",
                                    dayfirst=True)
            day = None if pd.isna(parsed) else parsed.date()
        vendor = str(row[text_col]) if text_col is not None else ""
        out.append(Receipt(path.name, abs(value), day, vendor[:60], vendor))
    return out
