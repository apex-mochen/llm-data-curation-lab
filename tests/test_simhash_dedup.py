from curation_lab.simhash_dedup import SimHashLSHDedup


def test_identical_documents_are_duplicates():
    dedup = SimHashLSHDedup(ngram=5, max_hamming=3)
    text = " ".join(f"w{i}" for i in range(100))
    assert dedup.is_duplicate(text) is False
    assert dedup.is_duplicate(text) is True
    assert dedup.kept == 1


def test_one_word_change_is_caught_with_loose_hamming():
    dedup = SimHashLSHDedup(ngram=5, max_hamming=8)
    base = " ".join(f"w{i}" for i in range(100))
    changed = " ".join(["other"] + [f"w{i}" for i in range(1, 100)])
    assert dedup.is_duplicate(base) is False
    assert dedup.is_duplicate(changed) is True


def test_unrelated_documents_are_kept():
    dedup = SimHashLSHDedup(ngram=5, max_hamming=3)
    a = " ".join(f"alpha{i}" for i in range(100))
    b = " ".join(f"beta{i}" for i in range(100))
    assert dedup.is_duplicate(a) is False
    assert dedup.is_duplicate(b) is False
    assert dedup.kept == 2


def test_invalid_hamming_raises():
    try:
        SimHashLSHDedup(max_hamming=64)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
