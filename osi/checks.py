"""Проверки ревизора: арифметика отчёта, остатки, план/факт, динамика, контрагенты."""
import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .amounts import format_money, parent_code

TOL = 0.011
SEVERITY_ORDER = {"high": 0, "medium": 1, "info": 2}
MAX_SUBSET = 16
JUMP_RATIO = 2.0
JUMP_MIN = 300_000
PLAN_DEV_RATIO = 0.25
PLAN_DEV_MIN = 100_000
SUSPICIOUS = ("страхование жизни", "страхованию жизни", "подар", "штраф", "премия", "матпомощь",
              "материальная помощь", "банкет", "корпоратив")
LEGAL_FORMS = ("тоо", "ип", "ао", "тоо.", "оо")


@dataclass
class Finding:
    kind: str
    severity: str
    title: str
    detail: str
    report: str = ""
    row_idx: int = -1
    code: str = ""
    amount: float = None
    expected: float = None
    ask: str = ""            # какой документ запросить у пользователя
    tags: list = field(default_factory=list)


def _cents(value):
    return int(round(value * 100))


def subset_sums(values):
    """Все суммы подмножеств (в копейках); для длинных списков — префиксы."""
    sums = {0}
    if len(values) > MAX_SUBSET:
        total = 0
        for v in values:
            total += v
            sums.add(total)
        return sums
    for v in values:
        sums |= {s + v for s in sums}
    return sums


def _val(row, name):
    return getattr(row, name)


class _Group:
    def __init__(self, row):
        self.row = row
        self.details = []
        self.children = []


def build_groups(report):
    """Нумерованные строки + «хвосты» без номера, которые к ним относятся."""
    groups, cur = [], None
    for row in report.rows:
        if row.zone not in ("income", "expense"):
            cur = None
            continue
        if row.code:
            cur = _Group(row)
            groups.append(cur)
        elif cur is not None and row.zone == cur.row.zone:
            cur.details.append(row)
    by_code = {g.row.code: g for g in groups}
    for g in groups:
        parent = parent_code(g.row.code)
        if parent in by_code:
            by_code[parent].children.append(g)
    return groups


def find_subset(values, target):
    """Индексы элементов, дающих target (копейки), или None."""
    n = len(values)
    if n > MAX_SUBSET:
        total = 0
        for i, v in enumerate(values):
            total += v
            if total == target:
                return list(range(i + 1))
        return None
    for mask in range(1, 1 << n):
        if sum(v for i, v in enumerate(values) if mask >> i & 1) == target:
            return [i for i in range(n) if mask >> i & 1]
    return None


def _effective(group, field_name, report, out):
    """-> (сумма строки, «лишние» строки без номера в копейках)."""
    row = group.row
    own = _val(row, field_name)
    if group.children:
        total, extras = 0.0, []
        for child in group.children:
            value, left = _effective(child, field_name, report, out)
            total += value or 0.0
            extras += left
        total = round(total, 2)
        if own is None:
            return (total or None), extras
        diff = _cents(own) - _cents(total)
        if abs(diff) <= 1:
            return own, extras
        picked = find_subset(extras, diff) if diff > 0 else None
        if picked is not None:
            used = [extras[i] for i in picked]
            extras = [e for i, e in enumerate(extras) if i not in picked]
            out.append(Finding(
                "ambiguous", "info", "Строки без номера учтены как отдельные расходы",
                f"В строке {row.code} сумма сходится, только если строки "
                f"без номера на {format_money(sum(used) / 100)} считать "
                "самостоятельными расходами, а не расшифровкой.",
                report.label, row.idx, row.code, own, total))
            return own, extras
        out.append(Finding(
            "sum_mismatch", "high", "Сумма строки не равна сумме её частей",
            f"Строка {row.code} «{row.text[:60]}»: указано "
            f"{format_money(own)}, а сумма подпунктов — "
            f"{format_money(total)} (разница {format_money(own - total)}).",
            report.label, row.idx, row.code, own, total,
            ask=f"Уточните расшифровку строки {row.code}"))
        return own, extras
    parts = [_cents(_val(d, field_name)) for d in group.details
             if _val(d, field_name) is not None]
    if own is None:
        return (round(sum(parts) / 100, 2) if parts else None), []
    if not parts:
        return own, []
    picked = find_subset(parts, _cents(own))
    if picked is None:
        return own, parts
    return own, [p for i, p in enumerate(parts) if i not in picked]


def check_sums(report):
    """Иерархия сумм: итоги поступлений/расходов, разделы, подпункты."""
    out = []
    groups = build_groups(report)
    fields = ["fact"] + (["plan"] if report.kind == "annual" else [])
    for name in fields:
        for zone, role in (("income", "income_total"),
                           ("expense", "expense_total")):
            roots = [g for g in groups if g.row.role == role]
            if not roots:
                continue
            root = roots[0]
            root.children = [g for g in groups if g.row.zone == zone
                             and g.row.depth == 2 and g is not root]
            _effective(root, name, report, out)
            # вложенные разделы уже проверены рекурсивно
    seen, unique = set(), []
    for f in out:
        key = (f.kind, f.code, f.amount)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _first(report, role):
    return report.first(role)


def check_balances(reports):
    """Остатки сходятся между месяцами и внутри года."""
    out = []
    monthly = sorted((r for r in reports if r.kind == "monthly" and r.start),
                     key=lambda r: r.start)
    for prev, cur in zip(monthly, monthly[1:]):
        out += _check_pair(prev, cur)
    for rep in reports:
        if rep.kind == "annual":
            out += _check_annual(rep)
    return out


def _amount(report, role, idx=0):
    rows = report.by_role(role)
    return rows[idx].amount if len(rows) > idx else None


def _sum_code(report, code):
    for row in report.rows:
        if row.code == code and row.zone == "income":
            return row.amount
    return None


def _check_pair(prev, cur):
    out = []
    p_sav, c_sav = _amount(prev, "sav_balance"), _amount(cur, "sav_balance")
    contrib = _sum_code(cur, "6.3")
    if None not in (p_sav, c_sav):
        expected = round(p_sav + (contrib or 0), 2)
        if abs(expected - c_sav) > TOL:
            out.append(Finding(
                "balance_savings", "high",
                "Остаток накопительного счёта не сходится с предыдущим месяцем",
                f"{prev.label}: {format_money(p_sav)} + взносы на капремонт "
                f"{format_money(contrib or 0)} = {format_money(expected)}, "
                f"а в отчёте {cur.label} указано {format_money(c_sav)}.",
                cur.label, amount=c_sav, expected=expected,
                ask="Выписка по сберегательному счёту за месяц"))
    p_cur, c_cur = _amount(prev, "cur_balance"), _amount(cur, "cur_balance")
    inc, exp = _amount(cur, "income_total"), _amount(cur, "expense_total")
    if None not in (p_cur, c_cur, inc, exp):
        expected = round(p_cur + inc - (contrib or 0) - exp, 2)
        if abs(expected - c_cur) > TOL:
            out.append(Finding(
                "balance_current", "high",
                "Остаток текущего счёта не сходится с движением денег",
                f"{prev.label}: {format_money(p_cur)} + поступления "
                f"{format_money(inc)} − на накопительный "
                f"{format_money(contrib or 0)} − расходы {format_money(exp)} "
                f"= {format_money(expected)}, в отчёте {cur.label}: "
                f"{format_money(c_cur)}.",
                cur.label, amount=c_cur, expected=expected,
                ask="Выписка по текущему счёту за месяц"))
    return out


def _check_annual(rep):
    out = []
    o_cur, o_sav = _amount(rep, "cur_balance", 0), _amount(rep, "sav_balance", 0)
    c_cur, c_sav = _amount(rep, "cur_balance", 1), _amount(rep, "sav_balance", 1)
    dep = _amount(rep, "dep_balance")
    inc, exp = _amount(rep, "income_total"), _amount(rep, "expense_total")
    values = (o_cur, o_sav, c_cur, c_sav, inc, exp)
    if None in values:
        return out
    expected = round(o_cur + o_sav + inc - exp, 2)
    actual = round(c_cur + c_sav + (dep or 0), 2)
    if abs(expected - actual) > TOL:
        out.append(Finding(
            "balance_annual", "high", "Годовой баланс денег не сходится",
            f"Остаток на начало ({format_money(o_cur + o_sav)}) + "
            f"поступления − расходы = {format_money(expected)}, а остатки на "
            f"конец (счета + депозит) — {format_money(actual)}.",
            rep.label, amount=actual, expected=expected,
            ask="Годовые выписки по всем счетам"))
    if dep:
        out.append(Finding(
            "deposit", "info", "Есть средства на депозите",
            f"На депозитном счёте {format_money(dep)}. Рекомендуем "
            "проверить решение собрания и условия размещения.",
            rep.label, amount=dep, ask="Решение собрания о размещении депозита"))
    return out


def check_plan_fact(report):
    """Годовой отчёт: сильные отклонения факта от плана по строкам."""
    out = []
    if report.kind != "annual":
        return out
    for row in report.rows:
        if row.zone != "expense" or row.plan is None or row.fact is None:
            continue
        if row.depth > 3 or not row.code:
            continue
        diff = row.fact - row.plan
        base = max(abs(row.plan), 1.0)
        if abs(diff) >= PLAN_DEV_MIN and abs(diff) / base >= PLAN_DEV_RATIO:
            word = "превышение" if diff > 0 else "недорасход"
            out.append(Finding(
                "plan_deviation", "medium" if diff > 0 else "info",
                f"Отклонение от плана: {word}",
                f"Строка {row.code} «{row.text[:55]}»: план "
                f"{format_money(row.plan)}, факт {format_money(row.fact)} "
                f"({diff / base:+.0%}).",
                report.label, row.idx, row.code, row.fact, row.plan,
                ask=f"Обоснование отклонения по строке {row.code}"))
    return out


def _leaf_rows(report):
    """Строки «нижнего уровня»: расшифровки и пункты без расшифровки."""
    parents = {parent_code(r.code) for r in report.rows if r.code}
    groups = build_groups(report)
    with_details = {g.row.idx for g in groups
                    if any(d.amount for d in g.details)}
    for row in report.rows:
        if row.zone != "expense" or row.amount is None or not row.text:
            continue
        if row.role or row.depth == 0 and not row.code:
            if not row.code:
                yield row
            continue
        if row.code and row.code not in parents and row.idx not in with_details:
            yield row


def subject(text):
    """Суть строки: у типовой «Наименование физ. или юр. лица ... за услуги»
    берём то, что идёт после слов «сервисной деятельности»."""
    low = text.lower()
    if low.startswith(("наименован", "нанменован", "наименованне")):
        pos = low.find("деятельности")
        if pos >= 0:
            tail = text[pos + len("деятельности"):].strip(" :;.-")
            if len(tail) > 3:
                return re.sub(r"^за\s+", "", tail)
    return text


def _line_key(text):
    clean = subject(text).lower().replace("ё", "е")
    return re.sub(r"[^а-яa-z ]", "", clean)[:70]


def check_dynamics(reports):
    """Скачки сумм по одной и той же статье в соседних месяцах."""
    out = []
    monthly = sorted((r for r in reports if r.kind == "monthly" and r.start),
                     key=lambda r: r.start)
    for prev, cur in zip(monthly, monthly[1:]):
        before = {_line_key(r.text): r for r in _leaf_rows(prev)}
        for row in _leaf_rows(cur):
            key = _line_key(row.text)
            match = before.get(key)
            if match is None:
                best = max(before, key=lambda k: fuzz.ratio(k, key),
                           default=None)
                if best is not None and fuzz.ratio(best, key) >= 92:
                    match = before[best]
            if match is None:
                continue
            a, b = match.amount or 0.0, row.amount
            if max(a, b) - min(a, b) < JUMP_MIN:
                continue
            if min(a, b) == 0 or max(a, b) / min(a, b) >= JUMP_RATIO:
                out.append(Finding(
                    "jump", "info" if min(a, b) == 0 else "medium",
                    "Резкое изменение суммы статьи",
                    f"«{subject(row.text)[:70]}»: {prev.label} — {format_money(a)}, "
                    f"{cur.label} — {format_money(b)}.",
                    cur.label, row.idx, row.code or "", b, a,
                    ask="Документы, объясняющие рост по этой статье"))
    return out


_LOOK = str.maketrans({"T": "Т", "O": "О", "A": "А", "P": "Р", "H": "Н",
                       "M": "М", "K": "К", "C": "С", "X": "Х", "E": "Е",
                       "B": "В"})
_MENTION = re.compile(
    r"(?<![а-я])(тоо|ип|ао)\s*[\"«'“]?\s*([а-яa-z][^\"»'”()\d,;:]{1,38}?)"
    r"(?=\s*[\"»'”(),;:\d]|\s+(?:за|договор|оплата|от)\b|$)")


def vendor_mentions(text):
    """[(форма, название)] из строки: «ТОО ТКМ-Сервис за январь» -> тоо, ткм-сервис."""
    low = text.translate(_LOOK).lower().replace("ё", "е")
    return [(m.group(1), re.sub(r"\s+", " ", m.group(2)).strip())
            for m in _MENTION.finditer(low)]


def vendor_name(text):
    found = vendor_mentions(text)
    return found[0][1] if found else ""


def legal_form(text):
    found = vendor_mentions(text)
    return found[0][0] if found else ""


def _vendor_rows(report):
    for row in report.rows:
        if row.zone == "expense" and not row.code and row.amount \
                and len(row.text) > 4:
            yield row


def check_vendors(reports):
    """Один контрагент под разными формами (ТОО/ИП/АО) в отчётах."""
    out = []
    known = {}               # название -> {форма: (отчёт, строка)}
    reported = set()
    ordered = sorted(reports, key=lambda r: (r.start is None, r.start or 0))
    for rep in ordered:
        for row in rep.rows:
            if row.zone != "expense" or row.role or len(row.text) < 6:
                continue
            for form, name in vendor_mentions(row.text):
                if len(name) < 4:
                    continue
                key = next((k for k in known
                            if fuzz.ratio(name, k) >= 88), name)
                forms = known.setdefault(key, {})
                for other, (o_rep, _) in list(forms.items()):
                    pair = (key, frozenset((form, other)))
                    if other != form and pair not in reported:
                        reported.add(pair)
                        out.append(Finding(
                            "legal_form", "medium",
                            "Контрагент сменил организационно-правовую форму",
                            f"«{key}» указан как {other.upper()} "
                            f"({o_rep.label}) и как {form.upper()} "
                            f"({rep.label}).",
                            rep.label, row.idx, row.code or "", row.amount,
                            ask="Договор и реквизиты контрагента"))
                forms[form] = (rep, row)
    return out


def check_duplicates(report):
    """Две одинаковые строки (контрагент + назначение + сумма) в одном месяце."""
    out, seen = [], {}
    for row in _vendor_rows(report):
        key = (re.sub(r"\W+", "", row.text.lower()), row.amount)
        if key in seen:
            out.append(Finding(
                "duplicate_payment", "medium", "Возможный двойной платёж",
                f"Строка «{row.text[:70]}» на {format_money(row.amount)} "
                f"встречается дважды в отчёте {report.label}.",
                report.label, row.idx, row.code, row.amount,
                ask="Платёжные документы по обоим платежам"))
        seen[key] = row.idx
    return out


def check_suspicious(reports):
    out = []
    for rep in reports:
        taken = set()
        for row in sorted(rep.rows, key=lambda r: -len(r.text)):
            low = row.text.lower()
            if row.zone == "expense" and any(k in low for k in SUSPICIOUS) \
                    and (row.amount, rep.label) not in taken:
                taken.add((row.amount, rep.label))
                out.append(Finding(
                    "suspicious_item", "medium",
                    "Необычная статья расходов",
                    f"«{row.text[:80]}» — {format_money(row.amount)}. "
                    "Такие расходы обычно требуют решения собрания.",
                    rep.label, row.idx, row.code, row.amount,
                    ask="Решение собрания и договор по этой статье"))
    return out


def check_reports(reports):
    """Все проверки -> список Finding, самые важные сверху."""
    findings = []
    for rep in reports:
        findings += check_sums(rep)
        findings += check_plan_fact(rep)
        findings += check_duplicates(rep)
    findings += check_balances(reports)
    findings += check_dynamics(reports)
    findings += check_vendors(reports)
    findings += check_suspicious(reports)
    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.report))
    return findings
