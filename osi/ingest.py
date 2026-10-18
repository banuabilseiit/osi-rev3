"""Чтение документов: PDF с текстом (pdfplumber) и сканы (OCR EasyOCR).

Результат — список Page со словами, линиями таблицы и границами колонок.
"""
import os
import re
from pathlib import Path

import numpy as np

from .layout import Page, Word, _cluster

_TEXT_PDF_MIN_WORDS = 8


# ---------- PDF с текстом ----------
def _pdf_page(page, index, source):
    words = [Word(w["text"], w["x0"], w["top"], w["x1"], w["bottom"])
             for w in page.extract_words(x_tolerance=1.5, y_tolerance=2)]
    hrules = [e["top"] for e in page.edges
              if e["orientation"] == "h" and e["width"] > 0.3 * page.width]
    vrules = [e["x0"] for e in page.edges
              if e["orientation"] == "v" and e["height"] > 0.05 * page.height]
    cols = [sum(g) / len(g) for g in _cluster(vrules, 3.0)]
    return Page(index, page.width, page.height, words, hrules, cols, source)


def _render(page, target_width=1700):
    """Рендер страницы шириной ~1700 px (достаточно для OCR таблиц)."""
    dpi = 72 * target_width / page.width
    image = page.to_image(resolution=dpi).original.convert("L")
    return np.array(image)


def load_pdf(path, reader=None):
    """Страницы PDF. Если текста нет (скан внутри PDF) — OCR."""
    import pdfplumber
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        has_text = any(len(pg.extract_words()) >= _TEXT_PDF_MIN_WORDS
                       for pg in pdf.pages)
        for i, page in enumerate(pdf.pages):
            if has_text:
                pages.append(_pdf_page(page, i, Path(path).name))
            else:
                pages.append(read_scan(_render(page), i, Path(path).name,
                                       reader))
    return pages


# ---------- сканы ----------
_READER = None


def get_ocr_reader():
    """EasyOCR (ru+en). Создаётся один раз: загрузка занимает секунды."""
    global _READER
    if _READER is None:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "OCR недоступен в этой версии приложения: сканы и фото "
                "не читаются, загрузите PDF с текстом") from exc
        try:
            import torch
            gpu = torch.cuda.is_available()
        except ImportError:
            gpu = False
        _READER = easyocr.Reader(
            ["ru", "en"], gpu=gpu, verbose=False,
            model_storage_directory=os.environ.get(
                "EASYOCR_DIR", str(Path.home() / ".EasyOCR" / "model")))
    return _READER


def skew_angle(gray, max_angle=3.0, step=0.1):
    """Угол наклона страницы: максимум дисперсии горизонтальной проекции."""
    import cv2
    scale = 900 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    binary = cv2.adaptiveThreshold(
        small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        25, 12)
    h, w = binary.shape
    best, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + step, step):
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle), 1.0)
        rotated = cv2.warpAffine(binary, matrix, (w, h),
                                 flags=cv2.INTER_NEAREST)
        score = float(np.var(rotated.sum(axis=1)))
        if score > best_score:
            best, best_score = float(angle), score
    return best


def deskew(gray):
    import cv2
    angle = skew_angle(gray)
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(gray, matrix, (w, h),
                         borderMode=cv2.BORDER_REPLICATE)
    return out, angle


def _binary(gray):
    import cv2
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 21, 12)


def find_hrules(gray, x_from=0.15, x_to=0.86, cover=0.55):
    """Y-координаты горизонтальных линий таблицы."""
    import cv2
    h, w = gray.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    mask = cv2.morphologyEx(_binary(gray), cv2.MORPH_OPEN, kernel)
    profile = (mask[:, int(x_from * w):int(x_to * w)] > 0).mean(axis=1)
    ys = np.where(profile > cover)[0]
    return [float(np.mean(g)) for g in _cluster(list(ys), 4)]


def find_vrules(gray, y_from, y_to, cover=0.35):
    """X-координаты вертикальных линий (границы колонок)."""
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
    mask = cv2.morphologyEx(_binary(gray), cv2.MORPH_OPEN, kernel)
    band = mask[int(y_from):int(y_to)] > 0
    if band.size == 0:
        return []
    xs = np.where(band.mean(axis=0) > cover)[0]
    return [float(np.mean(g)) for g in _cluster(list(xs), 6)]


def read_scan(gray, index=0, source="", reader=None):
    """Скан страницы -> Page (выпрямление, линии, OCR)."""
    gray, _ = deskew(gray)
    h, w = gray.shape
    hrules = find_hrules(gray)
    cols = []
    if len(hrules) >= 3:
        cols = find_vrules(gray, hrules[0], hrules[-1])
    if len(cols) < 3:                       # запасной вариант: доли ширины
        cols = [0.10 * w, 0.19 * w, 0.835 * w, 0.93 * w]
    reader = reader or get_ocr_reader()
    words = []
    for box, text, conf in reader.readtext(gray):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        words.append(Word(text, float(min(xs)), float(min(ys)),
                          float(max(xs)), float(max(ys))))
    return Page(index, float(w), float(h), words, hrules, cols, source,
                is_scan=True)


def load_image(path, reader=None):
    import cv2
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Не удалось открыть изображение: {path}")
    return [read_scan(gray, 0, Path(path).name, reader)]


def load_document(path, reader=None):
    """Любой поддерживаемый файл -> список страниц."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path, reader)
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        return load_image(path, reader)
    raise ValueError(f"Формат {suffix} пока не поддерживается")


def page_header_text(page, limit_y=None):
    """Текст над таблицей (название, период, адрес) по строкам."""
    from .layout import header_bottom, text_lines
    limit = limit_y if limit_y is not None else header_bottom(page)
    if not limit and page.hrules:           # нет слов шапки: берём верх таблицы
        limit = min(page.hrules)
    top = [w for w in page.words if w.y1 <= limit + 2]
    if not top:
        return ""
    tol = 0.5 * float(np.median([w.height for w in top]))
    lines = text_lines(top, tol)
    return re.sub(r"[ \t]+", " ", "\n".join(lines))
