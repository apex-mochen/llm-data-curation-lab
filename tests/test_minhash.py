import numpy as np

from curation_lab.hashing import shingle_hashes
from curation_lab.minhash import MinHash, jaccard_estimate


def true_jaccard(a, b, n=5, mode="word"):
    sa = set(shingle_hashes(a, n, mode))
    sb = set(shingle_hashes(b, n, mode))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def test_signature_estimate_tracks_true_jaccard():
    base = " ".join(f"w{i}" for i in range(200))
    changed = " ".join(["other"] + [f"w{i}" for i in range(1, 200)])
    different = " ".join(f"x{i}" for i in range(200))

    mh = MinHash(num_perm=256, seed=7)
    sig_base = mh.signature(base)
    sig_changed = mh.signature(changed)
    sig_different = mh.signature(different)

    assert abs(jaccard_estimate(sig_base, sig_changed) - true_jaccard(base, changed)) < 0.08
    assert jaccard_estimate(sig_base, sig_different) < 0.05


def test_signature_is_deterministic_for_same_seed():
    text = " ".join(f"token{i}" for i in range(100))
    a = MinHash(num_perm=64, seed=11).signature(text)
    b = MinHash(num_perm=64, seed=11).signature(text)
    assert np.array_equal(a, b)


def test_block_boundary_does_not_change_signature(monkeypatch):
    text = " ".join(f"w{i}" for i in range(5000))
    normal = MinHash(num_perm=64, seed=3).signature(text)

    import curation_lab.minhash as minhash_module

    monkeypatch.setattr(minhash_module, "_BLOCK", 100000)
    single_block = MinHash(num_perm=64, seed=3).signature(text)
    assert np.array_equal(normal, single_block)


def test_character_mode_counts_char_shingles():
    mh = MinHash(num_perm=64, seed=5, ngram=3, mode="char")
    a = mh.signature("abcdef")
    b = mh.signature("abcdef")
    assert np.array_equal(a, b)


def test_text_shorter_than_ngram_is_all_max():
    mh = MinHash(num_perm=32, seed=5, ngram=10)
    sig = mh.signature("only three words")
    assert np.all(sig == np.uint64((1 << 64) - 1))
