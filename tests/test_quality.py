from curation_lab.quality import QualityScorer, quality_score, quality_stats


def test_empty_text_is_zero():
    stats = quality_stats("")
    assert stats.tokens == 0
    assert quality_score(stats) == 0.0


def test_repetitive_text_scores_lower_than_diverse_text():
    clean = "the quick brown fox jumps over the lazy dog while birds sing"
    repeated = "the the the the the the the the the the the the the the the"
    assert quality_stats(clean).unique_ratio > quality_stats(repeated).unique_ratio
    assert quality_score(quality_stats(clean)) > quality_score(quality_stats(repeated))


def test_repeated_four_grams_are_detected():
    stats = quality_stats("a b c d a b c d a b c d a b c d")
    assert stats.repeat_4gram_ratio > 0.5


def test_scorer_threshold():
    scorer = QualityScorer(min_score=0.6)
    assert scorer.keep("alpha beta gamma delta epsilon zeta eta theta")
    assert not scorer.keep("word word word word word word word word")
