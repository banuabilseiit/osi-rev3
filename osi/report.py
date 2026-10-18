"""Модель отчёта ОСИ: строки, роли (итоги, разделы, остатки), метаданные."""
import re
from dataclasses import dataclass, field
from datetime import date

from rapidfuzz import fuzz

from .amounts import code_depth, normalize_code
from .layout import assemble_rows
from .ingest import page_header_text

ROLE_KEYS = [
    ("cur_balance", "остаток на текущем счете"),
    ("sav_balance", "остаток на сберегательном счете"),
    ("dep_balance", "остаток на депозитном счете"),
    ("income_total", "всего поступлений"),
    ("expense_total", "всего расходов"),
]
SECTION_KEYS = [
    ("mgmt", "расходы по управлению объектом"),
    ("maint", "расходы по содержанию общего"),
    ("repair", "текущий ремонт общего имущества"),
    ("utilities", "коммунальные услуги на содержание"),
    ("household", "хозяйственные расходы"),
    ("capital", "капитальный ремонт"),
]
SECTION_TITLES = {
    "mgmt": "Расходы по управлению",
    "maint": "Содержание общего имущества",
    "repair": "Текущий ремонт",
    "utilities": "Коммунальные услуги",
    "household": "Хозяйственные расходы",
    "capital": "Капитальный ремонт",
}
MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5,
    "июня": 6, "июля": 7, "августа": 8, "сентября": 9, "октября": 10,
    "ноября": 11, "декабря": 12,
}
_IGNORE_TEXT = ("расходование денег", "* расходование",
                "расходовашне", "расходованне")
_HEADER_TEXT = ("наименование", "сумма, тенге", "сумма тенге")


@dataclass
class Row:
    idx: int
    page: int
    code: str
    depth: int
    role: str
    zone: str
    section: str
    text: str
    plan: float
    fact: float

    @property
    def amount(self):
        """Основная сумма строки: факт, а если его нет — план."""
        return self.fact if self.fact is not None else self.plan


@dataclass
class Report:
    kind: str
    source: str
    org: str = ""
    address: str = ""
    start: date = None
    end: date = None
    rows: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def label(self):
        if self.start and self.end and self.kind == "monthly":
            return f"{self.start:%m.%Y}"
        if self.start and self.end:
            return f"{self.start:%d.%m.%Y}-{self.end:%d.%m.%Y}"
        return self.source

    def by_role(self, role):
        return [r for r in self.rows if r.role == role]

    def first(self, role):
        found = self.by_role(role)
        return found[0] if found else None

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame([{
            "№": r.code or "", "роль": r.role or "", "раздел": r.section,
            "наименование": r.text, "план": r.plan, "факт": r.fact,
        } for r in self.rows])


_LATIN_TO_CYR = str.maketrans({
    "B": "В", "C": "С", "c": "с", "e": "е", "E": "Е", "H": "Н", "K": "К",
    "k": "к", "M": "М", "O": "О", "o": "о", "P": "Р", "p": "р", "T": "Т",
    "X": "Х", "x": "х", "y": "у", "a": "а", "A": "А", "r": "г",
})


def _norm(text):
    """Нижний регистр, латинские «двойники» -> кириллица, без пунктуации."""
    text = text.translate(_LATIN_TO_CYR).lower().replace("ё", "е")
    return re.sub(r"[^а-я0-9 ]", "", text)


def _score(text, key):
    """Похожесть начала строки на ключевую фразу (0..100)."""
    return fuzz.ratio(key, _norm(text)[:len(key)])


def _best_key(text, keys, threshold=80):
    best, best_score = "", 0.0
    for name, key in keys:
        score = _score(text, key)
        if score > best_score:
            best, best_score = name, score
    return best if best_score >= threshold else ""


def detect_kind(header_text):
    low = header_text.lower()
    if "годов" in low:
        return "annual"
    if "ежемесяч" in low or "месяч" in low:
        return "monthly"
    if "статья затрат" in low or "смет" in low:
        return "estimate"
    return "monthly"


def parse_period(header_text):
    """-> (дата начала, дата конца) или (None, None)."""
    match = re.search(r"(\d{2})\.(\d{2})\.(\d{4}).{0,25}?"
                      r"(\d{2})\.(\d{2})\.(\d{4})", header_text, re.S)
    if match:
        d1, m1, y1, d2, m2, y2 = (int(x) for x in match.groups())
        try:
            return date(y1, m1, d1), date(y2, m2, d2)
        except ValueError:
            pass
    pattern = (r"[\"«]?(\d{1,2})[\"»]?\s*([а-я]+)\s*(\d{4}).{0,25}?"
               r"[\"«]?(\d{1,2})[\"»]?\s*([а-я]+)\s*(\d{4})")
    match = re.search(pattern, header_text.lower(), re.S)
    if match:
        d1, mo1, y1, d2, mo2, y2 = match.groups()
        if mo1 in MONTHS and mo2 in MONTHS:
            return (date(int(y1), MONTHS[mo1], int(d1)),
                    date(int(y2), MONTHS[mo2], int(d2)))
    return None, None


def parse_org(header_text):
    match = re.search(r"(?:Объединение|Обьединение|Объединенне)"
                      r"[^\"«'\n]*[\"«']([^\"»'\n]+)[\"»']", header_text)
    return match.group(1).strip() if match else ""


def parse_address(header_text):
    match = re.search(r"\(адрес\)\s*[:.]?\s*(.+)", header_text)
    return match.group(1).strip() if match else ""


def _assign_roles(rows):
    """Зоны (шапка / поступления / расходы) и роли строк."""
    zone, section = "head", ""
    for row in rows:
        role = _best_key(row.text, ROLE_KEYS)
        if role == "income_total":
            zone = "income"
        elif role == "expense_total":
            zone, section = "expense", ""
        elif zone == "expense" and not role:
            if row.depth in (0, 2):
                role = _best_key(row.text, SECTION_KEYS, threshold=78)
                if role:
                    section = role
        if zone == "expense" and role in ("cur_balance", "sav_balance",
                                          "dep_balance"):
            zone, section = "tail", ""
        row.role, row.zone, row.section = role, zone, section
    return rows


def _set_code(row, code):
    row.code = code
    row.depth = code_depth(code)


def _repair_by_roles(rows, kind):
    """Коды итоговых строк и разделов известны заранее по их роли."""
    base = 8 if kind == "monthly" else 7
    sections = {name: f"{base}.{i}" for i, (name, _) in
                enumerate(SECTION_KEYS, 1)}
    for row in rows:
        if row.role == "income_total":
            _set_code(row, "6")
        elif row.role == "expense_total":
            _set_code(row, "7" if kind == "monthly" else "7")
        elif row.role in sections and row.depth in (0, 1, 2) and (
                row.code is None or row.code == "" or row.depth <= 2) \
                and row.section == row.role:
            if not row.code or row.code.count(".") <= 1:
                _set_code(row, sections[row.role])
    if kind != "monthly":
        return
    for row in rows:
        if row.zone != "expense" or not row.section or not row.code:
            continue
        prefix = sections[row.section] + "."
        if row.code.startswith(prefix) or row.code == sections[row.section]:
            continue
        digits = row.code.replace(".", "")
        head = sections[row.section].replace(".", "")
        if digits.startswith(head) and len(digits) > len(head):
            tail = digits[len(head):]
            _set_code(row, sections[row.section] + "." + ".".join(tail))
        elif row.code.count(".") >= 2:
            _set_code(row, prefix + ".".join(row.code.split(".")[2:]))


def _repair_sequence(rows):
    """Соседние номера одного уровня идут подряд: чиним одиночные сбои."""
    last = {}
    for pos, row in enumerate(rows):
        if not row.code or "." not in row.code:
            continue
        parent, _, num = row.code.rpartition(".")
        if not num.isdigit():
            continue
        prev = last.get(parent)
        if prev is not None and int(num) != prev[0] + 1:
            nxt = next((r for r in rows[pos + 1:] if r.code
                        and r.code.rpartition(".")[0] == parent
                        and r.code.rpartition(".")[2].isdigit()), None)
            if nxt is not None and int(nxt.code.rpartition(".")[2]) \
                    == prev[0] + 2:
                _set_code(row, f"{parent}.{prev[0] + 1}")
                num = str(prev[0] + 1)
        last[parent] = (int(num), pos)


def _to_rows(raw_rows, columns):
    rows = []
    for raw in raw_rows:
        low = raw.text.lower().strip()
        if any(low.startswith(x) for x in _IGNORE_TEXT) or low in _HEADER_TEXT:
            continue
        amounts = list(raw.amounts)
        if not raw.text and not any(a is not None for a in amounts):
            continue
        plan = fact = None
        if len(amounts) >= 2:
            plan, fact = amounts[0], amounts[1]
        elif amounts:
            fact = amounts[0]
        code = normalize_code(raw.num)
        rows.append(Row(len(rows), raw.page, code,
                        code_depth(code) if code else 0, "", "head", "",
                        raw.text.strip(), plan, fact))
    return rows


def build_report(pages, name=""):
    """Страницы документа -> Report (месячный или годовой отчёт)."""
    header = page_header_text(pages[0])
    kind = detect_kind(header)
    report = Report(kind=kind, source=name or pages[0].source)
    report.org = parse_org(header)
    report.address = parse_address(header)
    report.start, report.end = parse_period(header)
    raw = []
    for page in pages:
        gap = 0.0026 * page.height
        raw.extend(assemble_rows(page, min_gap=max(gap, 2.0)))
    report.rows = _assign_roles(_to_rows(raw, kind))
    _repair_by_roles(report.rows, kind)
    _repair_sequence(report.rows)
    if not report.first("expense_total"):
        report.notes.append("Не найдена строка «Всего расходов»")
    if not report.first("income_total"):
        report.notes.append("Не найдена строка «Всего поступлений»")
    return report
