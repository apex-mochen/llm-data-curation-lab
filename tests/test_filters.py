from curation_lab.filters import FilterConfig, check, normalize_text


def test_normalize_text_nfkc_and_trim():
    assert normalize_text("  ＡＢＣ  ") == "ABC"


def test_good_text_passes():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    assert check(text, FilterConfig(min_chars=20)) is None


def test_short_text_rejected():
    assert check("short", FilterConfig(min_chars=20)) == "too_short"


def test_low_alpha_rejected():
    assert check("12345 67890 12345 67890", FilterConfig(min_chars=5)) == "low_alpha"


def test_high_symbol_rejected():
    text = "a" + "!" * 50
    assert check(text, FilterConfig(min_chars=5, min_alpha_ratio=0.0)) == "high_symbol"


def test_repeated_lines_rejected():
    text = "\n".join(["the same line"] * 20)
    assert check(text, FilterConfig(min_chars=20, min_alpha_ratio=0.0)) == "repeated_lines"
