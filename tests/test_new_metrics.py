import pytest
from unittest.mock import MagicMock, patch
from src.conversation.decoder import GenericClassifier
from src.engine.cognitive_engine import CognitiveEngine
from src.conversation.encoder import TemplateEngine
from prometheus_client import REGISTRY

def test_token_counting_metrics():
    # Mock parser and models
    parser = MagicMock()
    parser.agent_config = {"llm_model": "test-model", "unknown_sentinel": "unknown"}
    parser.decoder_slots = []
    parser.dspy_examples = []
    models_module = MagicMock()
    models_module.MODELS = {"test_type": MagicMock()}
    
    classifier = GenericClassifier(parser, models_module)
    
    # Manually trigger log_token_usage
    classifier._log_token_usage("test_type", 100, 50)
    
    # Check Prometheus metrics
    token_usage = REGISTRY.get_sample_value('llm_token_usage_total', {'model': 'test-model', 'utterance_type': 'test_type', 'token_type': 'prompt'})
    assert token_usage == 100
    
    cost = REGISTRY.get_sample_value('llm_estimated_cost_usd_total', {'model': 'test-model'})
    # (100 * 0.05 / 1,000,000) + (50 * 0.15 / 1,000,000) = 0.000005 + 0.0000075 = 0.0000125
    assert cost == pytest.approx(0.0000125)

def test_cvr_and_faithfulness_metrics():
    parser = MagicMock()
    parser.slots = []
    parser.unknown_sentinel = "unknown"
    
    engine = CognitiveEngine(parser, MagicMock(), MagicMock(), MagicMock())
    engine.slots = {"test-slot": "unknown"}
    
    # 1. Test Faithful update
    engine._apply_slot_update("test-slot", "value1", "tool:test_tool")
    faithfulness = REGISTRY.get_sample_value('agent_faithfulness_total', {'source': 'tool:test_tool'})
    assert faithfulness == 1
    
    # 2. Test CVR (Unauthorized Overwrite)
    # Update from value1 to value2
    engine._apply_slot_update("test-slot", "value2", "tool:test_tool")
    cvr = REGISTRY.get_sample_value('agent_constraint_violation_total', {'violation_type': 'unauthorized_overwrite', 'slot_name': 'test-slot'})
    assert cvr == 1

def test_nlg_latency_metric():
    parser = MagicMock()
    template = MagicMock()
    template.utterance_template_id = "test_template"
    template.template_text = "Hello {name}"
    parser.encoder_templates = [template]
    
    encoder = TemplateEngine(parser)
    encoder.generate("test_template", {"name": "World"})
    
    latency = REGISTRY.get_sample_value('agent_nlg_generation_seconds_count', {'template_id': 'test_template'})
    assert latency == 1
