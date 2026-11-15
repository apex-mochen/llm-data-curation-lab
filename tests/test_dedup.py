from curation_lab.dedup import ExactDedup, MinHashLSHDedup


def test_exact_dedup_keeps_first():
    dedup = ExactDedup()
    assert dedup.check_and_add("hello world") is False
    assert dedup.check_and_add("hello world") is True
    assert dedup.check_and_add("hello there") is False
    assert dedup.size == 2


def test_lsh_detects_verbatim_copy():
    dedup = MinHashLSHDedup(num_perm=128, bands=16, rows=8)
    text = " ".join(f"token{i}" for i in range(100))
    assert dedup.check_and_add(text) is False
    assert dedup.check_and_add(text) is True
    assert dedup.num_docs == 1


def test_lsh_detects_one_word_change():
    dedup = MinHashLSHDedup(num_perm=128, bands=16, rows=8)
    base = " ".join(f"token{i}" for i in range(100))
    changed = " ".join(["other"] + [f"token{i}" for i in range(1, 100)])
    assert dedup.check_and_add(base) is False
    assert dedup.check_and_add(changed) is True


def test_lsh_keeps_unrelated_text():
    dedup = MinHashLSHDedup(num_perm=128, bands=16, rows=8)
    a = " ".join(f"alpha{i}" for i in range(100))
    b = " ".join(f"beta{i}" for i in range(100))
    assert dedup.check_and_add(a) is False
    assert dedup.check_and_add(b) is False
    assert dedup.num_docs == 2


def test_lsh_rejects_invalid_band_params():
    try:
        MinHashLSHDedup(num_perm=32, bands=8, rows=8)
    except ValueError as exc:
        assert "bands * rows" in str(exc)
    else:
        raise AssertionError("expected ValueError")
