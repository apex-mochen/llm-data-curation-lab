from curation_lab.lsh_params import collision_probability, optimal_lsh_params


def test_collision_probability_boundaries():
    assert collision_probability(0.0, 16, 8) == 0.0
    assert collision_probability(1.0, 16, 8) == 1.0
    assert 0.0 < collision_probability(0.7, 16, 8) < 1.0


def test_optimal_params_respect_budget_and_threshold():
    result = optimal_lsh_params(0.7, num_perm=128)
    assert result["bands"] * result["rows"] <= 128
    assert abs(result["s_curve_threshold"] - 0.7) < 0.15
    assert 0.0 <= result["score"] <= 1.0


def test_optimal_params_change_with_threshold():
    low = optimal_lsh_params(0.5, num_perm=128)
    high = optimal_lsh_params(0.9, num_perm=128)
    assert (low["bands"], low["rows"]) != (high["bands"], high["rows"])
    assert high["s_curve_threshold"] > low["s_curve_threshold"]
