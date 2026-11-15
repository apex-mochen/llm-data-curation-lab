from curation_lab.decontam import NGramDecontaminator


def test_exact_reference_copy_is_contaminated():
    reference = "the quick brown fox jumps over the lazy dog near the river bank"
    decontam = NGramDecontaminator([reference], n=3, threshold=0.5)
    assert decontam.match_ratio(reference) == 1.0
    assert decontam.is_contaminated(reference)


def test_unrelated_text_is_clean():
    reference = "the quick brown fox jumps over the lazy dog near the river bank"
    decontam = NGramDecontaminator([reference], n=3, threshold=0.5)
    other = "completely unrelated content about databases and distributed systems"
    assert decontam.match_ratio(other) == 0.0
    assert not decontam.is_contaminated(other)


def test_short_text_has_zero_ratio():
    reference = "the quick brown fox jumps over the lazy dog"
    decontam = NGramDecontaminator([reference], n=5, threshold=0.5)
    assert decontam.match_ratio("fox") == 0.0


def test_empty_reference_raises():
    try:
        NGramDecontaminator(["too short"], n=5)
    except ValueError as exc:
        assert "no n-grams" in str(exc)
    else:
        raise AssertionError("expected ValueError")
