from osi.analyze import analyze, apply_uploads, load_files
from osi.classifier import CategoryClassifier
from osi.receipts import Receipt, parse_day, parse_total
from osi.synth import make_series, render_pdf


def test_pdf_roundtrip_matches_structure(tmp_path):
    series = make_series(11, 2, [])
    path = render_pdf(series.reports[0], tmp_path / "r.pdf")
    reports, receipts, skipped = load_files([path])
    assert len(reports) == 1 and not receipts and not skipped
    original = series.reports[0]
    parsed = reports[0]
    assert parsed.first("income_total").fact == original.first(
        "income_total").fact
    assert parsed.first("expense_total").fact == original.first(
        "expense_total").fact
    assert parsed.start == original.start


def test_receipt_total_and_date():
    lines = ["ТОО Магазин", "Дата: 15.05.2026", "Хлеб 120,00",
             "ИТОГО К ОПЛАТЕ: 12 500,00"]
    assert parse_total(lines) == 12500.0
    assert str(parse_day(lines)) == "2026-05-15"


def test_evidence_closes_question():
    series = make_series(2, 3, [])
    clf = CategoryClassifier().fit()
    result = analyze(series.reports, [], clf)
    question = next(q for q in result.questions if q.kind == "receipt")
    apply_uploads(result, [Receipt("c.pdf", total=question.amount)])
    assert question.resolved
