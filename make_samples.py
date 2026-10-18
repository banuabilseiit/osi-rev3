"""Генерация демо-файлов: отчёты за 3 месяца с нарушениями + чеки.

Запуск: python make_samples.py  ->  data/samples/*.pdf
Все данные вымышленные.
"""
from pathlib import Path

from osi.amounts import format_money
from osi.analyze import analyze
from osi.synth import make_series, render_pdf

OUT = Path("data/samples")


def render_receipt(path, vendor, day, total, items):
    from reportlab.lib.pagesizes import A6
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    font = Path(__file__).parent / "osi" / "assets" / "DejaVuSans.ttf"
    pdfmetrics.registerFont(TTFont("DV", str(font)))
    pdf = canvas.Canvas(str(path), pagesize=A6)
    width, height = A6
    pdf.setFont("DV", 9)
    y = height - 28
    for line in (vendor, "Кассовый чек", f"Дата: {day}", "-" * 34):
        pdf.drawCentredString(width / 2, y, line)
        y -= 13
    for name, price in items:
        pdf.drawString(14, y, name[:26])
        pdf.drawRightString(width - 14, y, format_money(price))
        y -= 12
    pdf.drawString(14, y - 6, "ИТОГО К ОПЛАТЕ:")
    pdf.drawRightString(width - 14, y - 6, format_money(total))
    pdf.save()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    series = make_series(
        7, 3, ["sum_mismatch", "balance_savings", "jump", "legal_form",
               "suspicious_item", "duplicate_payment"])
    for i, report in enumerate(series.reports, 1):
        render_pdf(report, OUT / f"demo_otchet_{i}.pdf")
    questions = [q for q in analyze(series.reports, []).questions
                 if q.kind == "receipt"]
    for made, q in enumerate(questions[:2], 1):
        render_receipt(OUT / f"demo_chek_{made}.pdf", 'ТОО "Строй Маркет"',
                       "15.05.2026", q.amount, [("Товары по счёту", q.amount)])
    print("truth:", series.truth)


if __name__ == "__main__":
    main()
