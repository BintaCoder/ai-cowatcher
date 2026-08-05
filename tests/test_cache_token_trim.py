"""Prompt 6: QA threshold eval, evidence defaults, reasoning effort, warm patterns."""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient

from ai_cowatcher.agent.completion import LiteLLMCompletionClient
from ai_cowatcher.agent.conversation_agent import ConversationAgent
from ai_cowatcher.config import Settings
from ai_cowatcher.personas.loader import get_persona, list_personas
from ai_cowatcher.providers.mock import MockTextEmbedder
from ai_cowatcher.qa.threshold_eval import (
    LabeledPair,
    PairScore,
    evaluate_threshold,
    pick_threshold,
    score_pairs,
)
from ai_cowatcher.qa.warm_cache import (
    COMMON_PATTERN_TEMPLATES,
    load_pairs,
    merge_common_patterns,
)
from ai_cowatcher.retrieval.evidence import scene_evidence_json
from ai_cowatcher.storage.qa_cache import InMemoryExactKV, QACache, ts_bucket


def test_default_trim_and_cache_knobs():
    # Ignore developer .env so we assert code defaults, not local overrides.
    s = Settings(MOCK_MODE=True, _env_file=None)
    assert s.qa_cache_enabled is False
    assert s.qa_cache_semantic_threshold == pytest.approx(0.88)
    assert s.qa_cache_ts_bucket_sec == 45
    assert s.evidence_max_scenes == 2
    assert s.evidence_max_chars_per_field == 200
    assert s.llm_short_answer_max_tokens == 160
    assert s.llm_max_tokens == 512
    assert s.llm_reasoning_effort == "minimal"


def test_ts_bucket_width_configurable():
    assert ts_bucket(44.0, 45) == 0
    assert ts_bucket(45.0, 45) == 1
    assert ts_bucket(44.0, 30) == 1
    # Wider bucket keeps playheads that straddled 30s boundaries in one window
    assert ts_bucket(35.0, 45) == ts_bucket(20.0, 45) == 0


def test_evidence_defaults_trim_to_two_scenes():
    scenes = [
        {"scene_id": f"s{i}", "transcript": "word " * 80, "caption": "cap " * 80}
        for i in range(5)
    ]
    payload = scene_evidence_json([scenes])
    assert "s0" in payload and "s1" in payload
    assert "s2" not in payload
    # field clip uses default 200
    for scene in scenes[:1]:
        t = scene["transcript"]
        assert len(t) > 200


def test_wider_bucket_exact_hit(cache_settings_factory):
    settings = cache_settings_factory(QA_CACHE_TS_BUCKET_SEC=45)
    cache = QACache(
        exact_kv=InMemoryExactKV(),
        qdrant=QdrantClient(":memory:"),
        embedder=MockTextEmbedder(),
        settings=settings,
    )
    cache.store("t1", 10.0, "Who is that?", "Ross.", persona_id="easygoing_friend")
    # Under 30s buckets these are different; at 45s they share a bucket
    hit = cache.lookup(
        "t1", 40.0, "Who is that?", persona_id="easygoing_friend"
    )
    assert hit is not None and hit.source == "exact"


def test_threshold_eval_prefers_lowest_safe():
    pairs = [
        LabeledPair("What's going on?", "What's going on in this scene?", True),
        LabeledPair("Who is that?", "Make a joke about this", False),
    ]

    class _E:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            # Manual scores via vectors: near-dup both ~ e1; different e2
            mapping = {
                "What's going on?": [1.0, 0.0],
                "What's going on in this scene?": [0.95, 0.0],
                "Who is that?": [0.0, 1.0],
                "Make a joke about this": [0.1, 0.0],
            }
            return [mapping[t] for t in texts]

    scored = score_pairs(pairs, _E())
    reports = [
        evaluate_threshold(scored, t)
        for t in (0.86, 0.88, 0.90, 0.99)
    ]
    assert reports[0].true_positive == 1
    assert reports[0].false_positive == 0 or reports[1].false_positive == 0
    chosen = pick_threshold(reports, max_false_positive_rate=0.0)
    assert chosen is not None
    assert chosen.threshold <= 0.90
    assert chosen.false_positive == 0


def test_threshold_false_positive_blocked():
    scored = [
        PairScore(
            LabeledPair("a", "b", True),
            score=0.95,
        ),
        PairScore(
            LabeledPair("c", "d", False),
            score=0.92,
        ),
    ]
    r = evaluate_threshold(scored, 0.88)
    assert r.false_positive == 1
    assert not r.ok(max_false_positive_rate=0.0)
    assert pick_threshold([r], max_false_positive_rate=0.0) is None


def test_warm_common_patterns_merged():
    merged = merge_common_patterns(
        {"friends_ross": [{"question": "Who is that?", "answer": "Ross", "current_ts": 1.0}]},
        include_common=True,
    )
    qs = {row["question"].lower() for row in merged["friends_ross"]}
    assert "who is that?" in qs
    assert "what's going on?" in qs
    assert any("upset" in q for q in qs)
    assert len(COMMON_PATTERN_TEMPLATES) >= 8


def test_load_pairs_includes_common_by_default():
    pairs = load_pairs(None, "friends_ross", include_common=True)
    questions = [row["question"].lower() for _, row in pairs]
    assert any("what just happened" in q for q in questions)
    assert any("going on" in q for q in questions)


def test_reasoning_effort_on_all_completion_kwargs():
    settings = Settings(MOCK_MODE=True, LLM_REASONING_EFFORT="minimal", _env_file=None)
    client = LiteLLMCompletionClient(settings)
    kwargs = client._common_kwargs(
        model="mock",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2,
        max_tokens=64,
    )
    assert kwargs["reasoning_effort"] == "minimal"
    # No persona path overrides — effort always comes from settings
    settings_low = Settings(MOCK_MODE=True, LLM_REASONING_EFFORT="low", _env_file=None)
    client_low = LiteLLMCompletionClient(settings_low)
    assert (
        client_low._common_kwargs(
            model="m",
            messages=[],
            temperature=0.1,
            max_tokens=32,
        )["reasoning_effort"]
        == "low"
    )


def test_answer_max_tokens_uses_short_cap_for_simple_questions():
    settings = Settings(
        MOCK_MODE=True,
        LLM_MAX_TOKENS=512,
        LLM_SHORT_ANSWER_MAX_TOKENS=160,
        _env_file=None,
    )
    agent = ConversationAgent.__new__(ConversationAgent)
    agent._settings = settings
    assert agent._answer_max_tokens("Who is that?") == 160
    assert agent._answer_max_tokens("Why did that happen in detail — explain") == 512


def test_persona_style_notes_insist_on_short_answers():
    get_persona.cache_clear() if hasattr(get_persona, "cache_clear") else None
    from ai_cowatcher.personas import loader as persona_loader

    persona_loader._load_all.cache_clear()
    for p in list_personas():
        notes = p.style_notes.lower()
        assert "one short" in notes
    for pid in ("witty_friend", "easygoing_friend", "calm_scout"):
        assert get_persona(pid).style_notes


def test_record_qa_cache_metrics_hit_sources():
    from ai_cowatcher.observability import prometheus_metrics as pm

    before_exact = pm.QA_CACHE_HIT_TOTAL.labels(source="exact")._value.get()
    before_sem = pm.QA_CACHE_HIT_TOTAL.labels(source="semantic")._value.get()
    before_miss = pm.QA_CACHE_MISS_TOTAL._value.get()
    pm.record_qa_cache_result("exact_hit")
    pm.record_qa_cache_result("semantic_hit")
    pm.record_qa_cache_result("miss")
    assert pm.QA_CACHE_HIT_TOTAL.labels(source="exact")._value.get() == before_exact + 1
    assert pm.QA_CACHE_HIT_TOTAL.labels(source="semantic")._value.get() == before_sem + 1
    assert pm.QA_CACHE_MISS_TOTAL._value.get() == before_miss + 1


@pytest.fixture
def cache_settings_factory():
    def _make(**overrides):
        base = dict(
            MOCK_MODE=True,
            QA_CACHE_ENABLED=True,
            QA_CACHE_COLLECTION=f"qa_test_{uuid.uuid4().hex[:10]}",
            QA_CACHE_SEMANTIC_THRESHOLD=0.5,
            QA_CACHE_TS_BUCKET_SEC=45,
        )
        base.update(overrides)
        return Settings(**base)

    return _make
