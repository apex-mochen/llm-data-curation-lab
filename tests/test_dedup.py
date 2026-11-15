import numpy as np

from curation_lab.dedup import ExactDedup, MinHashLSHDedup


def _signature(bands, filler):
    sig = np.zeros(128, dtype=np.uint64)
    for band in range(16):
        sig[band * 8 : (band + 1) * 8] = filler * 100 + band
    for band, value in bands.items():
        sig[band * 8 : (band + 1) * 8] = value
    return sig


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


def test_transitive_clustering_drops_chain_member():
    dedup = MinHashLSHDedup(num_perm=128, bands=16, rows=8, transitive=True)
    signatures = {
        "A": _signature({0: 1}, 10),
        "B": _signature({0: 1, 1: 2}, 20),
        "C": _signature({1: 2}, 30),
    }
    dedup.signature = lambda text: signatures[text]
    assert dedup.check_and_add("A") is False
    assert dedup.check_and_add("B") is True
    assert dedup.check_and_add("C") is True
    assert dedup.num_docs == 1
    assert dedup.num_seen == 3


def test_first_match_keeps_chain_member():
    dedup = MinHashLSHDedup(num_perm=128, bands=16, rows=8, transitive=False)
    signatures = {
        "A": _signature({0: 1}, 10),
        "B": _signature({0: 1, 1: 2}, 20),
        "C": _signature({1: 2}, 30),
    }
    dedup.signature = lambda text: signatures[text]
    assert dedup.check_and_add("A") is False
    assert dedup.check_and_add("B") is True
    assert dedup.check_and_add("C") is False
    assert dedup.num_docs == 2
