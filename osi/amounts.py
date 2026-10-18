"""Разбор денежных сумм и номеров строк (с поправками на ошибки OCR)."""
import re

_OCR_DIGITS = str.maketrans({
    "O": "0", "o": "0", "О": "0", "о": "0", "l": "1", "I": "1",
    "|": "1", "З": "3", "S": "5", "Б": "6",
})
_AMOUNT = re.compile(r"-?\d[\d ]*[.,]\d{2}(?!\d)")
_NO_SEPARATOR = re.compile(r"^\d{1,3}(?: \d{3})* \d{2}$")
_CODE = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,3}$")


def clean_amount_text(text):
    """Склеивает обрывки вроде '4470604,' + ',39' и чинит цифры."""
    text = text.strip().translate(_OCR_DIGITS)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.,]\s*[.,]", ",", text)
    text = re.sub(r"(\d)\s+([.,])", r"\1\2", text)
    text = re.sub(r"([.,])\s+(\d)", r"\1\2", text)
    text = re.sub(r"([.,]\d)\s+(\d)(?!\d)", r"\1\2", text)
    return text


def parse_amount(text, allow_no_separator=False):
    """'1 720 000,00' -> 1720000.0; не сумма -> None."""
    text = clean_amount_text(text)
    match = _AMOUNT.fullmatch(text)
    if match:
        return round(float(text.replace(" ", "").replace(",", ".")), 2)
    if allow_no_separator and _NO_SEPARATOR.match(text):
        head, tail = text.rsplit(" ", 1)
        return round(float(head.replace(" ", "") + "." + tail), 2)
    return None


def format_money(value):
    """12345.5 -> '12 345,50' (для отчётов)."""
    if value is None:
        return "-"
    return f"{value:,.2f}".replace(",", " ").replace(".", ",")


def normalize_code(text):
    """'8,1.2 .' -> '8.1.2'; мусор -> None."""
    if not text:
        return None
    text = text.replace(",", ".").replace(" ", "")
    text = re.sub(r"[^0-9.]", "", text)
    text = re.sub(r"\.+", ".", text).strip(".")
    return text if _CODE.match(text) else None


def code_depth(code):
    return code.count(".") + 1


def parent_code(code):
    """'7.1.4' -> '7.1'; у корня ('7') родителя нет."""
    if "." not in code:
        return None
    return code.rsplit(".", 1)[0]
