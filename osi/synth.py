"""Синтетические отчёты ОСИ: серии месяцев с внедрёнными нарушениями.

Нужны для оценки проверок (известно, что внедрено) и для демо-файлов.
"""
import calendar
import random
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .amounts import code_depth, format_money
from .report import Report, Row
from .vocab import (CATEGORY_SECTION, FORMS, ORG_NAMES, PHRASES, VENDORS)

INJECTIONS = ["sum_mismatch", "balance_savings", "balance_current", "jump",
              "legal_form", "suspicious_item", "duplicate_payment"]
SECTION_ORDER = ["mgmt", "maint", "repair", "utilities", "household",
                 "capital"]
CAPTIONS = {
    "cur_balance": "Остаток на текущем счете в банке второго уровня на "
                   "которые перечисляются взносы (платежи) собственников",
    "sav_balance": "Остаток на сберегательном счете в банке второго уровня "
                   "на который перечисляются взносы (платежи) собственников",
    "income_total": "Всего поступлений, в том числе:",
    "expense_total": "Всего расходов, в том числе:",
}
SECTION_CAPTIONS = {
    "mgmt": "Расходы по управлению объектом кондоминиума, в том числе:",
    "maint": "Расходы по содержанию общего имущества объекта "
             "кондоминиума, в том числе:",
    "repair": "Текущий ремонт общего имущества объекта кондоминиума "
              "(на основании дефектного акта)",
    "utilities": "Коммунальные услуги на содержание общего имущества "
                 "объекта кондоминиума",
    "household": "Хозяйственные расходы (приобретение инвентаря, "
                 "оборудования и другое)",
    "capital": "Капитальный ремонт",
}
INCOME_CAPTIONS = {
    "6.1": "Ежемесячные взносы (платежи) собственников квартир, нежилых "
           "помещений на управление объекта кондоминиума",
    "6.2": "Ежемесячные взносы (платежи) собственников парковочных мест, "
           "кладовок за содержание парковочного места, кладовки",
    "6.3": "Взносы собственников квартир, нежилых помещений для накопления "
           "сумм на капитальный ремонт (сберегательный счет)",
    "6.6": "Иные поступления (добровольные, частные инвестиции)",
}
# число строк и размер (тенге/мес.) по категориям
AMOUNT_RANGE = {
    "accounting": (80_000, 200_000), "legal": (50_000, 250_000),
    "chair_salary": (250_000, 450_000), "taxes": (120_000, 260_000),
    "bank": (2_000, 20_000), "office": (15_000, 60_000),
    "debt_collection": (10_000, 120_000), "engineering": (300_000, 900_000),
    "cleaning": (500_000, 1_200_000), "snow": (150_000, 700_000),
    "security": (400_000, 900_000), "fire": (30_000, 150_000),
    "gates_cameras": (30_000, 150_000), "pest": (10_000, 60_000),
    "sewer": (20_000, 80_000), "utilities": (100_000, 450_000),
    "supplies": (30_000, 300_000), "repair": (100_000, 500_000),
    "capital_repair": (500_000, 2_000_000), "other": (30_000, 150_000),
}


@dataclass
class Line:
    cat: str
    form: str
    vendor: str
    phrase: str
    base: int

    @property
    def text(self):
        return f'{self.form} "{self.vendor}" ({self.phrase})'


@dataclass
class Series:
    reports: list
    truth: list = field(default_factory=list)   # [{"kind", "month", "code"}]
    org: str = ""


def _make_lines(rng):
    """Набор постоянных статей организации."""
    cats = ["accounting", "chair_salary", "taxes", "bank", "office",
            "engineering", "cleaning", "snow", "security", "fire", "pest",
            "gates_cameras", "supplies", "utilities", "sewer"]
    rng.shuffle(cats)
    chosen = set(cats[:rng.randint(9, len(cats))]) | {
        "accounting", "chair_salary", "taxes", "engineering", "cleaning"}
    for extra in ("legal", "debt_collection", "repair", "capital_repair"):
        if rng.random() < 0.4:
            chosen.add(extra)
    lines = []
    vendors = rng.sample(VENDORS, len(VENDORS))
    forms = {v: rng.choice(FORMS) for v in VENDORS}
    for cat in sorted(chosen):
        for _ in range(rng.choice([1, 1, 2]) if cat in (
                "engineering", "cleaning", "supplies", "office") else 1):
            low, high = AMOUNT_RANGE[cat]
            vendor = vendors.pop() if vendors else rng.choice(VENDORS)
            lines.append(Line(cat, forms[vendor], vendor,
                              rng.choice(PHRASES[cat]),
                              int(rng.uniform(low, high) // 1000 * 1000)))
    return lines


def _month_amount(rng, base):
    return int(round(base * rng.uniform(0.92, 1.08)))


def _row(rows, code, text, fact, role="", zone="head", section=""):
    rows.append(Row(len(rows), 0, code, code_depth(code) if code else 0,
                    role, zone, section, text, None, fact))
    return rows[-1]


def _build_month(rng, lines, month, org, state, flags):
    """Один месячный отчёт; state — остатки на конец прошлого месяца."""
    last = calendar.monthrange(month.year, month.month)[1]
    rows = []
    amounts = []
    for line in lines:
        mult = flags.get(("jump", line.text), 1)
        amounts.append(int(_month_amount(rng, line.base) * mult))
    by_section = {s: [] for s in SECTION_ORDER}
    for line, amount in zip(lines, amounts):
        by_section[CATEGORY_SECTION[line.cat]].append((line, amount))
    for extra in flags.get("extra_lines", []):
        by_section[extra[2]].append((extra[0], extra[1]))
    section_total = {s: sum(a for _, a in v) for s, v in by_section.items()}
    expense = sum(section_total.values())
    fees = int(sum(amounts) * rng.uniform(0.8, 1.1))
    inc61 = int(fees * 0.75 + rng.randint(-50_000, 50_000))
    inc62 = int(fees * 0.2)
    inc63 = int(rng.uniform(300_000, 900_000) // 100 * 100)
    inc66 = rng.choice([0, 0, 15_000])
    income = inc61 + inc62 + inc63 + inc66
    cur = round(state["cur"] + income - inc63 - expense, 2)
    sav = round(state["sav"] + inc63, 2)
    _row(rows, "1", CAPTIONS["cur_balance"], cur, "cur_balance")
    _row(rows, "2", CAPTIONS["sav_balance"], sav, "sav_balance")
    _row(rows, "6", CAPTIONS["income_total"], income, "income_total", "income")
    for code, value in (("6.1", inc61), ("6.2", inc62), ("6.3", inc63),
                        ("6.6", inc66 or None)):
        _row(rows, code, INCOME_CAPTIONS[code], value, "", "income")
    _row(rows, "7", CAPTIONS["expense_total"], expense, "expense_total",
         "expense")
    for number, section in enumerate(SECTION_ORDER, 1):
        items = by_section[section]
        code = f"8.{number}"
        total = section_total[section] or None
        _row(rows, code, SECTION_CAPTIONS[section], total,
             section, "expense", section)
        if not items:
            continue
        if section == "mgmt":
            for k, (line, amount) in enumerate(items, 1):
                _row(rows, f"{code}.{k}", line.phrase.capitalize(), amount,
                     "", "expense", section)
                _row(rows, "", line.text, amount, "", "expense", section)
                rows[-1].code = None
                rows[-1].depth = 0
        elif section == "maint":
            _row(rows, f"{code}.1", "Оплата услуг по договорам субъектам "
                 "сервисной деятельности, в том числе:", total, "",
                 "expense", section)
            for line, amount in items:
                _row(rows, "", line.text, amount, "", "expense", section)
                rows[-1].code = None
                rows[-1].depth = 0
        else:
            for line, amount in items:
                _row(rows, "", line.text, amount, "", "expense", section)
                rows[-1].code = None
                rows[-1].depth = 0
    state["cur"], state["sav"] = cur, sav
    report = Report("monthly", f"{org}_{month:%Y_%m}.pdf", org=org,
                    address="г. Астана, ул. Тестовая, дом 1",
                    start=month.replace(day=1),
                    end=month.replace(day=last), rows=rows)
    for i, row in enumerate(rows):
        row.idx = i
    return report


def _swap_form(text):
    for old, new in (("ТОО", "ИП"), ("ИП", "ТОО")):
        if text.startswith(old):
            return new + text[len(old):]
    return text


def make_series(seed, months=3, injections=None):
    """Серия отчётов за несколько месяцев + список внедрённых нарушений."""
    rng = random.Random(seed)
    org = rng.choice(ORG_NAMES)
    lines = _make_lines(rng)
    start_month = rng.randint(1, 12 - months + 1)
    dates = [date(2026, start_month + i, 1) for i in range(months)]
    injections = list(injections or [])
    plan = {}
    for kind in injections:
        plan[kind] = rng.randint(1, months - 1)
    big = [ln for ln in lines if ln.base >= 250_000]
    flags = [dict() for _ in range(months)]
    truth = []
    if "jump" in plan and big:
        line = rng.choice(big)
        flags[plan["jump"]][("jump", line.text)] = rng.choice([3, 4])
        truth.append({"kind": "jump", "month": plan["jump"]})
    if "suspicious_item" in plan:
        sus = Line("other", "АО", "Standard Life", "страхование жизни", 0)
        flags[plan["suspicious_item"]].setdefault("extra_lines", []).append(
            (sus, rng.choice([60_000, 85_000, 120_000]), "mgmt"))
        truth.append({"kind": "suspicious_item",
                      "month": plan["suspicious_item"]})
    if "duplicate_payment" in plan:
        cats = ("engineering", "supplies", "cleaning", "security")
        smalls = [ln for ln in lines if ln.cat in cats]
        if smalls:
            dup = rng.choice(smalls)
            amount = int(dup.base * 0.9)
            section = CATEGORY_SECTION[dup.cat]
            flags[plan["duplicate_payment"]].setdefault(
                "extra_lines", []).extend([(dup, amount, section)] * 2)
            truth.append({"kind": "duplicate_payment",
                          "month": plan["duplicate_payment"]})
    swapped = None
    if "legal_form" in plan:
        swapped = rng.choice([ln for ln in lines if ln.form in ("ТОО", "ИП")])
        truth.append({"kind": "legal_form", "month": plan["legal_form"]})
    state = {"cur": rng.randint(300_000, 3_000_000),
             "sav": rng.randint(1_000_000, 8_000_000)}
    reports = []
    for i, month in enumerate(dates):
        month_lines = lines
        if swapped is not None and i >= plan["legal_form"]:
            month_lines = [Line(ln.cat, "ИП" if ln.form == "ТОО" else "ТОО",
                                ln.vendor, ln.phrase, ln.base)
                           if ln is swapped else ln for ln in lines]
        report = _build_month(rng, month_lines, month, org, state, flags[i])
        reports.append(report)
    for kind, month_idx in plan.items():
        report = reports[month_idx]
        if kind == "sum_mismatch":
            kids = [r for r in report.rows if r.zone == "expense"
                    and r.code and r.depth == 3 and r.fact]
            if kids:
                victim = rng.choice(kids)
                victim.fact = round(victim.fact + rng.choice([5_000, 25_000, 100_000]), 2)
                truth.append({"kind": "sum_mismatch", "month": month_idx})
        elif kind == "balance_savings":
            row = report.first("sav_balance")
            row.fact = round(row.fact + rng.choice([-250_000, 120_000, 500_000]), 2)
            truth.append({"kind": "balance_savings", "month": month_idx})
        elif kind == "balance_current":
            row = report.first("cur_balance")
            row.fact = round(row.fact + rng.choice([-300_000, 180_000, 400_000]), 2)
            truth.append({"kind": "balance_current", "month": month_idx})
    return Series(reports, truth, org)


# ---------- рендер в PDF ----------
def _wrap(text, width_chars):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width_chars and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + ([cur] if cur else [])


def render_pdf(report, path):
    """Рисует отчёт как PDF с текстовым слоем и линиями таблицы."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    assets = Path(__file__).parent / "assets"
    pdfmetrics.registerFont(TTFont("DV", str(assets / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DVB", str(assets / "DejaVuSans-Bold.ttf")))
    width, height = A4
    x = [40, 78, 430, 555]
    pdf = canvas.Canvas(str(path), pagesize=A4)
    size, lead = 8, 10

    def header():
        y = height - 40
        pdf.setFont("DVB", 10)
        pdf.drawCentredString(width / 2, y, "Ежемесячный отчет")
        pdf.setFont("DV", 8)
        pdf.drawCentredString(width / 2, y - 12, "по управлению объектом "
                              "кондоминиума и содержанию общего имущества")
        pdf.drawCentredString(
            width / 2, y - 24,
            f"за период с {report.start:%d.%m.%Y} года по "
            f"{report.end:%d.%m.%Y} года")
        pdf.drawString(x[0], y - 44, f"Местонахождение дома (адрес): "
                       f"{report.address}")
        pdf.drawString(x[0], y - 56, "Форма управления объектом кондоминиума:"
                       f' Объединение собственников имущества "{report.org}"')
        top = y - 70
        pdf.line(x[0], top, x[3], top)
        pdf.drawString(x[0] + 8, top - 12, "№")
        pdf.drawString(x[1] + 140, top - 12, "Наименование")
        pdf.drawString(x[2] + 22, top - 8, "Сумма,")
        pdf.drawString(x[2] + 22, top - 18, "тенге")
        bottom = top - 24
        pdf.line(x[0], bottom, x[3], bottom)
        return top, bottom

    top, y = header()
    seg_top = top
    pdf.setFont("DV", size)

    def verticals(y_from, y_to):
        for xv in x:
            pdf.line(xv, y_from, xv, y_to)

    for row in report.rows:
        text = row.text
        lines = _wrap(text, 78) or [""]
        h = lead * len(lines) + 4
        if y - h < 50:
            verticals(seg_top, y)
            pdf.showPage()
            pdf.setFont("DV", size)
            y = height - 40
            seg_top = y
            pdf.line(x[0], y, x[3], y)
        for i, line in enumerate(lines):
            pdf.drawString(x[1] + 4 + (12 if not row.code else 0),
                           y - 10 - i * lead, line)
        if row.code:
            pdf.drawString(x[0] + 4, y - 10, row.code)
        if row.fact is not None:
            pdf.drawRightString(x[3] - 4, y - 10, format_money(row.fact))
        y -= h
        pdf.line(x[0], y, x[3], y)
    verticals(seg_top, y)
    pdf.save()
    return path
