import pandas as pd

from data.generate import generate


def test_generation_is_reproducible():
    a, b = generate(n_users=3_000, seed=1), generate(n_users=3_000, seed=1)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_different_seed_gives_different_data():
    a, b = generate(n_users=3_000, seed=1), generate(n_users=3_000, seed=2)
    assert not a["outcomes"].equals(b["outcomes"])


def test_holdout_is_roughly_half():
    share = generate(n_users=5_000)["assignments"]["treated"].mean()
    assert 0.45 < share < 0.55


def test_model_tables_hide_ground_truth():
    tables = generate(n_users=2_000)
    for name in ["users", "assignments", "outcomes"]:
        assert not {"segment", "true_uplift", "p0", "p1"} & set(tables[name].columns)
