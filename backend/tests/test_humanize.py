import re

from app.services import humanizer
from ml import humanize as ml_humanize


def fake_humanize(text, tone, strength, target="external"):
    return {
        "original": text,
        "rewritten": "rewritten text",
        "tone": tone,
        "strength": strength,
        "target": target,
        "before": {"ai_percent": 96.0, "verdict": "ai"},
        "after": {"ai_percent": 4.0, "verdict": "human"},
        "zerogpt": {"before": 71.0, "after": 28.0},
        "stealth": {
            "before": {"risk": 60.0, "components": {"zerogpt": 71.0}},
            "after": {"risk": 22.0, "components": {"zerogpt": 28.0}},
        },
        "changes": [{"source": "leverage", "target": "use", "count": 2}],
        "warnings": ["Scores are tuned to this detector."],
        "elapsed_ms": 1.0,
    }


def test_humanize_defaults(client, monkeypatch):
    monkeypatch.setattr(humanizer, "run", fake_humanize)
    response = client.post("/api/humanize", json={"text": "AI text here."})
    assert response.status_code == 200
    body = response.json()
    assert body["tone"] == "casual"
    assert body["strength"] == "medium"
    assert body["target"] == "external"
    assert body["before"]["ai_percent"] == 96.0
    assert body["after"]["verdict"] == "human"
    assert body["zerogpt"]["before"] == 71.0
    assert body["stealth"]["after"]["risk"] == 22.0
    assert body["changes"][0]["target"] == "use"
    assert body["rewritten"] == "rewritten text"


def test_humanizer_rewrites_when_classifier_says_human(monkeypatch):
    def fake_score(text):
        return {
            "ai_percent": 6.0,
            "verdict": "human",
            "class_percent": {},
        }

    def fake_zg(text):
        pct = 62.0 if "leverage" in text else 18.0
        return {"pct": pct, "prob": pct / 100.0}

    monkeypatch.setattr(ml_humanize, "score_only", fake_score)
    monkeypatch.setattr(ml_humanize, "score_many", lambda texts: [fake_score(t) for t in texts])
    monkeypatch.setattr(ml_humanize, "zerogpt_style_score", fake_zg)
    monkeypatch.setattr(
        ml_humanize, "zerogpt_style_scores", lambda texts: [fake_zg(t) for t in texts]
    )
    result = ml_humanize.humanize(
        "We leverage robust solutions to deliver value. Moreover, we utilize "
        "cutting-edge tools.",
        tone="casual",
        strength="light",
        target="local",
    )
    assert result["rewritten"] != result["original"]
    assert result["zerogpt"]["before"] == 62.0
    assert any("ZeroGPT-style" in w for w in result["warnings"])
    assert not any("no rewrite was needed" in w for w in result["warnings"])


def test_humanizer_reports_zerogpt_after(monkeypatch):
    def fake_score(text):
        return {"ai_percent": 90.0, "verdict": "ai", "class_percent": {}}

    def fake_zg(text):
        return {"pct": 80.0 if "leverage" in text else 25.0, "prob": 0.5}

    monkeypatch.setattr(ml_humanize, "score_only", fake_score)
    monkeypatch.setattr(ml_humanize, "score_many", lambda texts: [fake_score(t) for t in texts])
    monkeypatch.setattr(ml_humanize, "zerogpt_style_score", fake_zg)
    monkeypatch.setattr(
        ml_humanize, "zerogpt_style_scores", lambda texts: [fake_zg(t) for t in texts]
    )
    result = ml_humanize.humanize(
        "We leverage synergies and utilize best practices.",
        tone="casual",
        target="local",
    )
    assert result["zerogpt"]["before"] == 80.0
    assert result["zerogpt"]["after"] == 25.0


def test_humanizer_zg_mode_ranks_by_zerogpt(monkeypatch):
    def fake_score(text):
        return {"ai_percent": 2.0, "verdict": "human", "class_percent": {}}

    def fake_zg(text):
        return {"pct": 35.0 if text[:1].islower() else 60.0, "prob": 0.5}

    monkeypatch.setattr(ml_humanize, "score_only", fake_score)
    monkeypatch.setattr(ml_humanize, "score_many", lambda texts: [fake_score(t) for t in texts])
    monkeypatch.setattr(ml_humanize, "zerogpt_style_score", fake_zg)
    monkeypatch.setattr(
        ml_humanize, "zerogpt_style_scores", lambda texts: [fake_zg(t) for t in texts]
    )
    result = ml_humanize.humanize(
        "We leverage robust solutions to deliver value.",
        tone="casual",
        strength="light",
        target="local",
    )
    assert result["rewritten"] != result["original"]
    assert result["rewritten"][0].islower()
    assert result["zerogpt"] == {"before": 60.0, "after": 35.0}
    assert any("stricter signal" in w for w in result["warnings"])
    assert any("moved from 60.0% to 35.0%" in w for w in result["warnings"])


def test_humanizer_zg_mode_runs_refinement(monkeypatch):
    def fake_score(text):
        return {"ai_percent": 5.0, "verdict": "human", "class_percent": {}}

    def fake_zg(text):
        pct = 41.0 if "thinking about this a lot" in text.lower() else 45.0
        return {"pct": pct, "prob": pct / 100.0}

    monkeypatch.setattr(ml_humanize, "score_only", fake_score)
    monkeypatch.setattr(ml_humanize, "score_many", lambda texts: [fake_score(t) for t in texts])
    monkeypatch.setattr(ml_humanize, "zerogpt_style_score", fake_zg)
    monkeypatch.setattr(
        ml_humanize, "zerogpt_style_scores", lambda texts: [fake_zg(t) for t in texts]
    )
    result = ml_humanize.humanize(
        "We leverage robust solutions to deliver value. Moreover, we utilize "
        "cutting-edge tools.",
        tone="casual",
        target="local",
    )
    assert "thinking about this a lot" in result["rewritten"].lower()
    assert result["zerogpt"] == {"before": 45.0, "after": 41.0}


def test_humanizer_zg_mode_rewrites_when_zg_does_not_improve(monkeypatch):
    def fake_score(text):
        return {"ai_percent": 3.0, "verdict": "human", "class_percent": {}}

    def fake_zg(text):
        if "leverage" in text.lower():
            pct = 48.0
        elif "thinking about this a lot" in text.lower():
            pct = 52.0
        else:
            pct = 50.0
        return {"pct": pct, "prob": pct / 100.0}

    monkeypatch.setattr(ml_humanize, "score_only", fake_score)
    monkeypatch.setattr(ml_humanize, "score_many", lambda texts: [fake_score(t) for t in texts])
    monkeypatch.setattr(ml_humanize, "zerogpt_style_score", fake_zg)
    monkeypatch.setattr(
        ml_humanize, "zerogpt_style_scores", lambda texts: [fake_zg(t) for t in texts]
    )
    result = ml_humanize.humanize(
        "We leverage robust solutions to deliver value.",
        tone="casual",
        strength="light",
        target="local",
    )
    assert result["rewritten"] != result["original"]
    assert result["zerogpt"] == {"before": 48.0, "after": 50.0}
    assert any("did not improve" in w for w in result["warnings"])


def test_humanizer_warns_when_zerogpt_unavailable(monkeypatch):
    def fake_score(text):
        return {"ai_percent": 10.0, "verdict": "human", "class_percent": {}}

    monkeypatch.setattr(ml_humanize, "score_only", fake_score)
    monkeypatch.setattr(ml_humanize, "score_many", lambda texts: [fake_score(t) for t in texts])
    monkeypatch.setattr(ml_humanize, "zerogpt_style_score", lambda text: None)
    monkeypatch.setattr(
        ml_humanize, "zerogpt_style_scores", lambda texts: [None] * len(texts)
    )
    result = ml_humanize.humanize(
        "We leverage robust solutions to deliver value.",
        tone="casual",
        target="local",
    )
    assert result["rewritten"] != result["original"]
    assert result["zerogpt"] == {"before": None, "after": None}
    assert any("ZeroGPT-style estimate unavailable" in w for w in result["warnings"])


def test_humanizer_external_preserves_layout_and_ranks(monkeypatch):
    def fake_score(text):
        ai = 80.0 if "leverage" in text else 20.0
        return {
            "ai_percent": ai,
            "verdict": "ai" if ai >= 60 else "human",
            "class_percent": {},
        }

    def fake_zg_features(texts):
        return [
            {
                "pct": 70.0 if "leverage" in text else 20.0,
                "prob": 0.5,
                "features": {
                    "words": 10.0,
                    "marker": 65.0 if "leverage" in text else 15.0,
                },
            }
            for text in texts
        ]

    monkeypatch.setattr(ml_humanize, "score_only", fake_score)
    monkeypatch.setattr(
        ml_humanize, "score_many", lambda texts: [fake_score(t) for t in texts]
    )
    monkeypatch.setattr(
        ml_humanize, "zerogpt_style_features_scores", fake_zg_features
    )
    monkeypatch.setattr(
        ml_humanize.stealth,
        "score_from_features",
        lambda features: {
            "risk": features.get("marker", 0.0),
            "components": {"zerogpt": features.get("marker", 0.0)},
        },
    )
    text = (
        "# Guide\n\nWe leverage robust solutions.\n\n"
        "- Leverage the tools\n- Use best practices"
    )
    result = ml_humanize.humanize(
        text, tone="formal", strength="medium", target="external"
    )
    assert result["target"] == "external"
    assert result["stealth"]["before"]["risk"] == 65.0
    assert result["stealth"]["after"]["risk"] == 15.0
    assert "# Guide" in result["rewritten"]
    assert "\n\n- " in result["rewritten"]
    assert "leverage" not in result["rewritten"].lower()
    assert result["zerogpt"] == {"before": 70.0, "after": 20.0}
    assert any("moved from 65.0% to 15.0%" in w for w in result["warnings"])


def test_humanizer_local_preserves_paragraphs(monkeypatch):
    def fake_score(text):
        ai = 80.0 if "leverage" in text else 20.0
        return {
            "ai_percent": ai,
            "verdict": "ai" if ai >= 60 else "human",
            "class_percent": {},
        }

    def fake_zg(text):
        pct = 70.0 if "leverage" in text else 20.0
        return {"pct": pct, "prob": pct / 100.0}

    monkeypatch.setattr(ml_humanize, "score_only", fake_score)
    monkeypatch.setattr(
        ml_humanize, "score_many", lambda texts: [fake_score(t) for t in texts]
    )
    monkeypatch.setattr(ml_humanize, "zerogpt_style_score", fake_zg)
    monkeypatch.setattr(
        ml_humanize, "zerogpt_style_scores", lambda texts: [fake_zg(t) for t in texts]
    )
    text = (
        "# My Guide\n\n"
        "We leverage robust solutions to deliver value.\n\n"
        "- Leverage the tools\n- Use best practices\n\n"
        "Moreover, we utilize cutting-edge tools daily."
    )
    result = ml_humanize.humanize(
        text, tone="casual", strength="medium", target="local"
    )
    out = result["rewritten"]
    assert result["target"] == "local"
    # Paragraph breaks and structure survive the rewrite.
    assert out.count("\n\n") >= 3
    lines = out.split("\n")
    assert any(line.strip() == "# My Guide" for line in lines)
    assert sum(1 for line in lines if line.startswith("- ")) == 2
    # The AI-ish phrasing was rewritten.
    assert "leverage robust solutions" not in out.lower()


def test_humanizer_local_preserves_headings_tables_and_code(monkeypatch):
    def fake_score(text):
        return {"ai_percent": 80.0, "verdict": "ai", "class_percent": {}}

    def fake_zg(text):
        return {"pct": 70.0, "prob": 0.7}

    monkeypatch.setattr(ml_humanize, "score_only", fake_score)
    monkeypatch.setattr(
        ml_humanize, "score_many", lambda texts: [fake_score(t) for t in texts]
    )
    monkeypatch.setattr(ml_humanize, "zerogpt_style_score", fake_zg)
    monkeypatch.setattr(
        ml_humanize, "zerogpt_style_scores", lambda texts: [fake_zg(t) for t in texts]
    )
    text = (
        "# My Title Stays\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "We leverage robust solutions to deliver value here.\n\n"
        "```\ncode line\n```"
    )
    result = ml_humanize.humanize(
        text, tone="casual", strength="strong", target="local"
    )
    out = result["rewritten"]
    lines = out.split("\n")
    # Heading keeps its title case (lower_style must not touch structure).
    assert any(line.strip() == "# My Title Stays" for line in lines)
    # Table rows and separator survive.
    assert "|---|---|" in out
    assert "| 1 | 2 |" in out
    # Code fence stays verbatim -- no phrase swap or capitalization.
    assert "```\ncode line\n```" in out
    # Prose was still rewritten.
    assert "leverage robust solutions" not in out.lower()


def test_humanizer_external_keeps_original_when_worse(monkeypatch):
    def fake_score(text):
        return {"ai_percent": 20.0, "verdict": "human", "class_percent": {}}

    def fake_zg_features(texts):
        return [
            {
                "pct": 20.0 if "leverage" in text else 60.0,
                "prob": 0.2,
                "features": {
                    "words": 5.0,
                    "marker": 20.0 if "leverage" in text else 60.0,
                },
            }
            for text in texts
        ]

    monkeypatch.setattr(ml_humanize, "score_only", fake_score)
    monkeypatch.setattr(
        ml_humanize, "score_many", lambda texts: [fake_score(t) for t in texts]
    )
    monkeypatch.setattr(
        ml_humanize, "zerogpt_style_features_scores", fake_zg_features
    )
    monkeypatch.setattr(
        ml_humanize.stealth,
        "score_from_features",
        lambda features: {
            "risk": features.get("marker", 0.0),
            "components": {"zerogpt": features.get("marker", 0.0)},
        },
    )
    result = ml_humanize.humanize(
        "We leverage robust solutions.", tone="casual", target="external"
    )
    assert result["rewritten"] == result["original"]
    assert result["stealth"]["after"]["risk"] == 20.0
    assert any("original is unchanged" in w for w in result["warnings"])


def test_humanizer_external_falls_back_to_local(monkeypatch):
    def fake_score(text):
        return {"ai_percent": 90.0, "verdict": "ai", "class_percent": {}}

    def fake_zg(text):
        pct = 80.0 if "leverage" in text else 30.0
        return {"pct": pct, "prob": pct / 100.0}

    monkeypatch.setattr(ml_humanize, "score_only", fake_score)
    monkeypatch.setattr(
        ml_humanize, "score_many", lambda texts: [fake_score(t) for t in texts]
    )
    monkeypatch.setattr(ml_humanize, "zerogpt_style_score", fake_zg)
    monkeypatch.setattr(
        ml_humanize, "zerogpt_style_scores", lambda texts: [fake_zg(t) for t in texts]
    )
    monkeypatch.setattr(ml_humanize.stealth, "available", lambda: False)
    result = ml_humanize.humanize(
        "We leverage synergies.", tone="casual", target="external"
    )
    assert result["target"] == "local"
    assert any("External proxy mode is unavailable" in w for w in result["warnings"])


def test_humanize_tone_and_strength(client, monkeypatch):
    monkeypatch.setattr(humanizer, "run", fake_humanize)
    response = client.post(
        "/api/humanize",
        json={"text": "Some text.", "tone": "formal", "strength": "strong"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["tone"] == "formal"
    assert body["strength"] == "strong"


def test_humanize_invalid_options(client):
    assert client.post("/api/humanize", json={"text": "x", "tone": "sarcastic"}).status_code == 422
    assert client.post("/api/humanize", json={"text": "x", "strength": "extreme"}).status_code == 422


def test_humanize_empty_rejected(client):
    assert client.post("/api/humanize", json={"text": ""}).status_code == 422
    assert client.post("/api/humanize", json={"text": "   \n  "}).status_code == 400


def test_humanize_rewrites_full_long_input(client, monkeypatch):
    captured = {}

    def spy(text, tone, strength, target="external"):
        captured["text"] = text
        return fake_humanize(text, tone, strength, target)

    monkeypatch.setattr(humanizer, "run", spy)
    text = "word here. " * 3000
    response = client.post("/api/humanize", json={"text": text})
    assert response.status_code == 200
    body = response.json()
    # The humanizer receives and returns the full input — no silent truncation.
    assert len(captured["text"]) == len(text)
    assert len(body["original"]) == len(text)
    assert not any("limited to the first" in w for w in body["warnings"])


def test_humanize_value_error_is_400(client, monkeypatch):
    def boom(text, tone, strength, target="external"):
        raise ValueError("Text is empty after cleaning.")

    monkeypatch.setattr(humanizer, "run", boom)
    response = client.post("/api/humanize", json={"text": "x"})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_humanize_not_saved_to_history(client, monkeypatch):
    monkeypatch.setattr(humanizer, "run", fake_humanize)
    client.post("/api/humanize", json={"text": "Some AI text."})
    assert client.get("/api/history").json()["total"] == 0


def test_lower_style_lowercase_prose_but_not_acronyms():
    text = "AI matters here. NASA leads the work. Companies ship daily."
    out = ml_humanize._lower_style(text, 1)
    # Casual lowercase sentence starts still apply to ordinary prose.
    assert "companies ship" in out
    assert "the work" in out
    # Acronyms and initialisms are never mangled into "aI" / "nASA".
    assert "AI matters" in out
    assert "NASA leads" in out


def test_lower_style_preserves_structural_lines():
    text = "# My Title\n- Item one\n| 1 | 2 |\n```\ncode line\n```"
    out = ml_humanize._lower_style(text, 2)
    assert "# My Title" in out
    assert "- Item one" in out
    assert "| 1 | 2 |" in out
    assert "code line" in out


def test_score_view_is_bounded_and_spans_document():
    section = (
        "## Heading {i}\n\nParagraph {i} explains how the operations team "
        "measured throughput during the quarter under review.\n\n"
    )
    doc = "# Title\n\n" + "".join(
        section.format(i=i) for i in range(1, 301)
    ) + "TAIL_SENTENCE: the last line of a very long document."
    view = ml_humanize._score_view(doc)
    # Long inputs collapse to roughly one scoring chunk...
    assert len(view) < 2800
    # ...but still cover the head, the body and the tail.
    assert "# Title" in view
    assert "TAIL_SENTENCE" in view
    headings = [
        int(m.group(1)) for m in re.finditer(r"## Heading (\d+)", view)
    ]
    assert max(headings) > 100, f"sample not spread over the body: {headings[:5]}"


def test_score_view_leaves_short_input_untouched():
    text = "A short sentence that is comfortably under the scoring limit."
    assert ml_humanize._score_view(text) == text
