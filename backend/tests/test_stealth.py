from ml import stealth


def test_style_features_cover_all_names():
    text = (
        "Honestly, I think this tool works well. It's fast and easy!\n\n"
        "- One point here\n- Another point here"
    )
    features = stealth.style_features(text)
    assert set(stealth.STYLE_FEATURE_NAMES) <= set(features)
    assert features["first_person_frac"] > 0
    assert features["style_bullet_frac"] > 0
    assert features["paragraph_count"] == 2


def test_merge_features_requires_lm_features():
    assert stealth.merge_features("Some text.", None) is None
    assert stealth.merge_features("Some text.", {}) is None
    merged = stealth.merge_features("Some text.", {"words": 3.0})
    assert merged["words"] == 3.0
    assert "avg_word_len" in merged


def test_score_from_features_returns_ensemble():
    features = {name: 0.0 for name in stealth.STYLE_FEATURE_NAMES}
    score = stealth.score_from_features(features)
    assert score is not None
    assert 0.0 <= score["risk"] <= 100.0
    assert 0.0 <= score["worst"] <= 100.0
    for key in ("zerogpt", "gptzero", "pangram", "humanized", "tropa3"):
        assert key in score["components"]
