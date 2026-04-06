import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import dspy
from prometheus_client import REGISTRY

from src.conversation.decoder import GenericClassifier
from src.conversation.encoder import TemplateEngine
from src.engine.cognitive_engine import CognitiveEngine
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry


def _sample_value(metric_name: str, labels: dict[str, str]) -> float:
    value = REGISTRY.get_sample_value(metric_name, labels)
    return 0.0 if value is None else float(value)


class FakeModel:
    @classmethod
    def model_json_schema(cls):
        return {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
            },
        }

    def __init__(self, **kwargs):
        self._data = kwargs

    def model_dump(self):
        return self._data


class StubDecoder:
    def classify(self, user_input: str, utterance_type: str, slots_to_extract: list, slot_values: dict = None) -> dict:
        normalized = user_input.strip().lower()
        if utterance_type == "routing":
            if normalized == "yes":
                return {"affirmation": "confirmed"}
            return {"conversation-category": "task_related"}

        if utterance_type == "task_related":
            result = {}
            if "lighting" in normalized:
                result["intent"] = "configure-lighting"
            if "bedroom" in normalized:
                result["lighting-params.room"] = "bedroom"
            if "low" in normalized:
                result["lighting-params.budget"] = "low"
            if "scheduled" in normalized:
                result["lighting-params.automation-level"] = "scheduled"
            return result

        return {}


def _build_engine() -> CognitiveEngine:
    repo_root = Path(__file__).resolve().parents[1]
    from generators.csv_parser import CSVParser
    parser = CSVParser(str(repo_root / "agent_config"))
    parser.parse_all()
    registry = ToolRegistry()
    registry.discover_and_register("src.tools.plugins")
    return CognitiveEngine(
        parser=parser,
        decoder=StubDecoder(),
        encoder=TemplateEngine(parser),
        tool_executor=ToolExecutor(registry),
    )


def _advance_until_message(engine: CognitiveEngine, max_cycles: int = 20):
    for _ in range(max_cycles):
        msg = engine.run_cycle()
        if msg is not None:
            return msg
    raise AssertionError("Engine did not produce a message within the cycle limit")


def test_decoder_token_usage_and_cost_metrics(monkeypatch):
    parser = MagicMock()
    parser.agent_config = {
        "llm_model": "test-model",
        "llm_cost_per_million_prompt_tokens": "1.0",
        "llm_cost_per_million_completion_tokens": "2.0",
    }
    parser.decoder_slots = []
    parser.dspy_examples = []
    parser.unknown_sentinel = "unknown"
    models_module = SimpleNamespace(MODELS={"test_type": FakeModel})

    monkeypatch.setattr(dspy, "LM", lambda *args, **kwargs: object())
    monkeypatch.setattr(dspy.settings, "configure", lambda **kwargs: None)
    monkeypatch.setattr(dspy, "Predict", lambda *args, **kwargs: object())

    classifier = GenericClassifier(parser, models_module)

    prompt_before = _sample_value(
        "llm_token_usage_total",
        {"model": "test-model", "utterance_type": "test_type", "token_type": "prompt"},
    )
    cost_before = _sample_value("llm_estimated_cost_usd_total", {"model": "test-model"})
    tps_before = _sample_value(
        "llm_tokens_per_second_count",
        {"model": "test-model", "utterance_type": "test_type", "token_scope": "total"},
    )

    classifier._record_token_usage("test_type", 100, 50, latency_seconds=0.5)

    prompt_after = _sample_value(
        "llm_token_usage_total",
        {"model": "test-model", "utterance_type": "test_type", "token_type": "prompt"},
    )
    completion_after = _sample_value(
        "llm_token_usage_total",
        {"model": "test-model", "utterance_type": "test_type", "token_type": "completion"},
    )
    cost_after = _sample_value("llm_estimated_cost_usd_total", {"model": "test-model"})
    tps_after = _sample_value(
        "llm_tokens_per_second_count",
        {"model": "test-model", "utterance_type": "test_type", "token_scope": "total"},
    )

    assert prompt_after - prompt_before == 100
    assert completion_after == 50
    assert cost_after - cost_before == 0.0002
    assert tps_after - tps_before == 1
    assert classifier.usage_snapshot()["estimated_cost_usd"] == 0.0002


def test_engine_records_constraint_churn_and_causal_trace_metrics():
    engine = _build_engine()

    overwrite_before = _sample_value(
        "agent_constraint_violation_total",
        {"violation_type": "single_slot_overwrite", "slot_name": "lighting-params.room"},
    )
    churn_before = _sample_value(
        "agent_working_memory_churn_total",
        {
            "slot_name": "lighting-params.room",
            "mutation_type": "overwrite",
            "source": "test",
        },
    )
    engine._apply_slot_update("lighting-params.room", "bedroom", "test")
    engine._apply_slot_update("lighting-params.room", "kitchen", "test")

    overwrite_after = _sample_value(
        "agent_constraint_violation_total",
        {"violation_type": "single_slot_overwrite", "slot_name": "lighting-params.room"},
    )
    churn_after = _sample_value(
        "agent_working_memory_churn_total",
        {
            "slot_name": "lighting-params.room",
            "mutation_type": "overwrite",
            "source": "test",
        },
    )
    assert overwrite_after - overwrite_before == 1
    assert churn_after - churn_before == 1

    engine = _build_engine()
    _advance_until_message(engine)
    engine.process_input("I want lighting in the bedroom with low budget and scheduled automation")
    _advance_until_message(engine)
    engine.process_input("yes")
    _advance_until_message(engine)

    trace_count = _sample_value(
        "agent_decision_trace_total",
        {"intent": "configure-lighting", "status": "traceable"},
    )
    assert trace_count >= 1
    assert engine.last_causal_trace is not None
    assert engine.last_causal_trace["metadata"]["matched_rule_id"]
    assert _sample_value(
        "rule_retrieval_total",
        {"intent": "configure-lighting", "status": "exact_match"},
    ) >= 1


def test_engine_progress_metrics_move_with_cycles():
    engine = _build_engine()

    _advance_until_message(engine)
    assert _sample_value("agent_cycles_since_last_tool_execution", {}) >= 1

    engine.process_input("I want lighting")
    _advance_until_message(engine)
    engine.process_input("yes")
    _advance_until_message(engine)
    engine.process_input("bedroom")
    _advance_until_message(engine)
    engine.process_input("low")
    _advance_until_message(engine)
    engine.process_input("scheduled")
    _advance_until_message(engine)

    assert _sample_value("agent_cycles_since_last_tool_execution", {}) <= 2
    assert _sample_value("agent_cycles_since_last_progress", {}) <= 2
    assert _sample_value("agent_working_memory_peak", {"state": "known"}) >= 1


def test_encoder_latency_metric_is_recorded():
    parser = MagicMock()
    parser.encoder_templates = [
        SimpleNamespace(utterance_template_id="test_template", template_text="Hello {name}")
    ]
    encoder = TemplateEngine(parser)

    before = _sample_value("agent_nlg_generation_seconds_count", {"template_id": "test_template"})
    encoder.generate("test_template", {"name": "World"})
    after = _sample_value("agent_nlg_generation_seconds_count", {"template_id": "test_template"})

    assert after - before == 1


def test_dashboard_smoke_includes_new_metrics():
    repo_root = Path(__file__).resolve().parents[1]
    dashboard_path = repo_root / "grafana" / "dashboards" / "llm_observability.json"
    dashboard = json.loads(dashboard_path.read_text())
    dashboard_text = json.dumps(dashboard)

    assert "agent_constraint_violation_total" in dashboard_text
    assert "agent_cycles_since_last_tool_execution" in dashboard_text
    assert "agent_session_estimated_cost_usd" in dashboard_text
    assert "llm_tokens_per_second" in dashboard_text
    assert "rule_retrieval_total" in dashboard_text
