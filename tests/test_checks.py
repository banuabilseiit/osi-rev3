from osi.checks import check_reports, find_subset, vendor_mentions
from osi.synth import make_series


def kinds(series):
    return {f.kind for f in check_reports(series.reports)}


def test_clean_series_has_no_violations():
    for seed in range(10):
        found = kinds(make_series(seed, 3, []))
        assert not found & {"sum_mismatch", "balance_savings",
                            "balance_current", "legal_form",
                            "duplicate_payment"}


def test_injected_violations_are_found():
    for kind in ("sum_mismatch", "balance_savings", "balance_current",
                 "legal_form", "duplicate_payment", "suspicious_item"):
        series = make_series(3, 3, [kind])
        assert kind in kinds(series), kind


def test_find_subset():
    assert find_subset([100, 250, 300], 350) == [0, 1]
    assert find_subset([100, 250], 999) is None


def test_vendor_mentions_handles_latin_lookalikes():
    found = vendor_mentions('TOO ТКМ-Сервис за январь 2026 год')
    assert found and found[0][0] == "тоо" and "ткм" in found[0][1]
