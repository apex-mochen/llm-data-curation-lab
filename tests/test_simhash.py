from curation_lab.simhash import SimHash


def test_identical_text_has_zero_hamming():
    hasher = SimHash(ngram=5)
    text = " ".join(f"w{i}" for i in range(100))
    fp = hasher.fingerprint(text)
    assert hasher.hamming(fp, fp) == 0
    assert hasher.similarity(fp, fp) == 1.0


def test_one_word_change_is_close():
    hasher = SimHash(ngram=5)
    base = " ".join(f"w{i}" for i in range(100))
    changed = " ".join(["other"] + [f"w{i}" for i in range(1, 100)])
    distance = hasher.hamming(hasher.fingerprint(base), hasher.fingerprint(changed))
    assert distance <= 12


def test_unrelated_text_is_far():
    hasher = SimHash(ngram=5)
    a = " ".join(f"alpha{i}" for i in range(100))
    b = " ".join(f"beta{i}" for i in range(100))
    distance = hasher.hamming(hasher.fingerprint(a), hasher.fingerprint(b))
    assert distance >= 16


def test_character_mode_is_deterministic():
    hasher = SimHash(ngram=3, mode="char")
    text = "注意力机制是深度学习中的一个重要组成部分"
    assert hasher.fingerprint(text) == hasher.fingerprint(text)
