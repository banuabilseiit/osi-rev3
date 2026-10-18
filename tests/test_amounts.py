from osi.amounts import (code_depth, format_money, normalize_code,
                         parent_code, parse_amount)


def test_parse_amount_variants():
    assert parse_amount("1 720 000,00") == 1720000.0
    assert parse_amount("2165000,00") == 2165000.0
    assert parse_amount("3701949,3 1") == 3701949.31      # разрыв цифры OCR
    assert parse_amount("4470604, 39") == 4470604.39
    assert parse_amount("abc") is None


def test_parse_amount_scan_without_separator():
    assert parse_amount("26 411 00", allow_no_separator=True) == 26411.0
    assert parse_amount("26 411 00") is None


def test_codes():
    assert normalize_code("8,1.2 .") == "8.1.2"
    assert normalize_code("мусор") is None
    assert code_depth("7.1.3") == 3
    assert parent_code("7.1.3") == "7.1"
    assert parent_code("7") is None


def test_format_money():
    assert format_money(1234567.5) == "1 234 567,50"
    assert format_money(None) == "-"
