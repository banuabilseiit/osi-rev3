"""Сборка строк таблицы из слов с координатами (общая для сканов и PDF).

Вход: слова страницы, горизонтальные линии таблицы и границы колонок.
Выход: строки «номер | текст | суммы по колонкам».
"""
import re
import statistics
from dataclasses import dataclass, field

from .amounts import clean_amount_text, parse_amount


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def xc(self):
        return (self.x0 + self.x1) / 2

    @property
    def yc(self):
        return (self.y0 + self.y1) / 2

    @property
    def height(self):
        return self.y1 - self.y0


@dataclass
class Page:
    index: int
    width: float
    height: float
    words: list
    hrules: list = field(default_factory=list)
    cols: list = field(default_factory=list)
    source: str = ""
    is_scan: bool = False


@dataclass
class RawRow:
    page: int
    y0: float
    y1: float
    num: str
    text: str
    amounts: list
    confidence: float = 1.0


def _cluster(values, tol):
    """Группирует отсортированные числа, расстояние между соседями <= tol."""
    groups = []
    for value in sorted(values):
        if groups and value - groups[-1][-1] <= tol:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def merge_rules(rules, min_gap):
    """Сливает близкие линии (двойные границы Excel, толстые линии скана)."""
    return [sum(g) / len(g) for g in _cluster(rules, min_gap)]


def header_bottom(page):
    """Нижняя граница шапки таблицы: по словам заголовка колонок."""
    marks = ("Сумма,", "тенге", "план", "факт", "Сумма")
    top = [w for w in page.words if w.y0 < 0.45 * page.height]
    found = [w for w in top if w.text in marks]
    if not found:
        return 0.0
    ys = [w.y1 for w in found] + [w.y1 for w in top
                                  if w.text == "Наименование"
                                  and abs(w.y1 - found[0].y1) < 80]
    return max(ys)


def column_of(word, cols):
    """Индекс колонки по центру слова или None, если вне таблицы."""
    for i in range(len(cols) - 1):
        if cols[i] <= word.xc < cols[i + 1]:
            return i
    return None


def _join_line(words, gap):
    """Склеивает слова одной строки; близкие фрагменты без пробела."""
    out = ""
    prev = None
    for word in sorted(words, key=lambda w: w.x0):
        if prev is not None:
            out += "" if word.x0 - prev.x1 < gap else " "
        out += word.text
        prev = word
    return out


def text_lines(words, tol):
    """Слова колонки текста -> строки сверху вниз."""
    lines = []
    for word in sorted(words, key=lambda w: w.yc):
        if lines and abs(word.yc - lines[-1]["yc"]) <= tol:
            lines[-1]["words"].append(word)
            n = len(lines[-1]["words"])
            lines[-1]["yc"] += (word.yc - lines[-1]["yc"]) / n
        else:
            lines.append({"yc": word.yc, "words": [word]})
    return [" ".join(w.text for w in sorted(line["words"],
                                            key=lambda w: w.x0))
            for line in lines]


def _split_points(words, cols, tol):
    """Где резать интервал без линий: перед каждым номером строки."""
    numbers = [w for w in words if column_of(w, cols) == 0
               and re.search(r"\d", w.text)]
    ys = [sum(g) / len(g) for g in _cluster([w.yc for w in numbers],
                                            3 * tol)]
    return [y - 1.2 * tol for y in ys[1:]]


_NUMERIC = re.compile(r"[\d\s.,OoОо|lI-]*\d[\d\s.,OoОо|lI-]*|[.,]\d{1,2}")


def _amount_split_points(words, cols, tol):
    """Где резать интервал, если в колонке сумм стоят несколько чисел."""
    ys = [w.yc for w in words if (column_of(w, cols) or 0) >= 2
          and _NUMERIC.fullmatch(w.text)]
    groups = _cluster(ys, 1.5 * tol)
    centers = [sum(g) / len(g) for g in groups]
    return [(a + b) / 2 for a, b in zip(centers, centers[1:])]


def assemble_rows(page, min_gap, row_min=0.0):
    """Строки таблицы страницы. Линии берём из page.hrules."""
    words = [w for w in page.words if column_of(w, page.cols) is not None]
    if not words:
        return []
    heights = [w.height for w in words if w.height > 0]
    tol = 0.5 * statistics.median(heights) if heights else 5.0
    top = header_bottom(page)
    rules = merge_rules(page.hrules, min_gap)
    rules = [y for y in rules if y >= top - min_gap]
    if top == 0.0 or not rules:             # страница-продолжение: без шапки
        rules.insert(0, 0.0)
    bottom = max(w.y1 for w in words) + min_gap
    if rules[-1] < bottom - 3 * min_gap:
        rules.append(bottom)
    body = [w for w in words if w.yc > rules[0]]
    cuts = []
    for lo, hi in zip(rules, rules[1:]):
        inside = [w for w in body if lo <= w.yc < hi]
        if not inside:
            continue
        points = sorted(_split_points(inside, page.cols, tol)
                        + _amount_split_points(inside, page.cols, tol))
        merged = []
        for y in points:
            if not merged or y - merged[-1] > 2 * tol:
                merged.append(y)
        points = merged
        edges = [lo] + points + [hi]
        for a, b in zip(edges, edges[1:]):
            cuts.append((a, b))
    rows = []
    amount_cols = list(range(2, len(page.cols) - 1))
    for lo, hi in cuts:
        inside = [w for w in body if lo <= w.yc < hi]
        if not inside:
            continue
        by_col = {}
        for word in inside:
            by_col.setdefault(column_of(word, page.cols), []).append(word)
        num = " ".join(w.text for w in sorted(
            by_col.get(0, []), key=lambda w: (w.yc, w.x0)))
        lines = text_lines(by_col.get(1, []), tol)
        amounts = []
        for col in amount_cols:
            numeric = [w for w in by_col.get(col, [])
                       if re.fullmatch(r"[\d\s.,OoОо|lI-]*\d[\d\s.,OoОо|lI-]*"
                                       r"|[.,]\d{1,2}", w.text)]
            raw = _join_line(numeric, gap=0.8 * tol * 2)
            amounts.append(parse_amount(raw, allow_no_separator=page.is_scan)
                           if clean_amount_text(raw) else None)
        rows.append(RawRow(page.index, lo, hi, num, " ".join(lines), amounts))
    return rows
