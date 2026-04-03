import importlib.util
from pathlib import Path

from generators.csv_parser import CSVParser
from generators.generate_models import generate_conversation_models


def load_generated_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("test_generated_models", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generate_conversation_models_builds_expected_models(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    parser = CSVParser(str(repo_root / "agent_config"))
    parser.parse_all()

    output_path = tmp_path / "conversation_models.py"
    generate_conversation_models(parser, str(output_path))

    module = load_generated_module(output_path)

    assert "routing" in module.MODELS
    assert "task_related" in module.MODELS
    assert "climate_related" in module.MODELS

    task_related_fields = module.TaskRelatedModel.model_fields
    assert "intent" in task_related_fields
    assert "budget" in task_related_fields
    assert task_related_fields["intent"].default == "unknown"


def test_generate_conversation_models_escapes_csv_strings(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    parser = CSVParser(str(repo_root / "agent_config"))
    parser.parse_all()

    parser.decoder_slots[0].description = 'He said "hello" and it\'s fine'
    parser.decoder_slots[0].accepted_values = 'task_related|say"hi"|unknown'

    output_path = tmp_path / "conversation_models.py"
    generate_conversation_models(parser, str(output_path))

    module = load_generated_module(output_path)

    routing_field = module.RoutingModel.model_fields["conversation_category"]
    assert routing_field.description == 'He said "hello" and it\'s fine'
