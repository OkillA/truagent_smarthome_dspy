from pathlib import Path

from generators.csv_parser import CSVParser


def test_dspy_examples_are_loaded_from_spec():
    repo_root = Path(__file__).resolve().parents[1]
    parser = CSVParser(str(repo_root / "agent_config"))
    parser.parse_all()

    assert len(parser.dspy_examples) >= 3
    assert {example.utterance_type for example in parser.dspy_examples} == {"routing"}
    assert any("set up some lights" in example.user_input for example in parser.dspy_examples)


def test_dspy_eval_cases_are_loaded_from_spec():
    repo_root = Path(__file__).resolve().parents[1]
    parser = CSVParser(str(repo_root / "agent_config"))
    parser.parse_all()

    assert len(parser.dspy_eval_cases) >= 4
    assert any(case.case_id == "task-related-reactive" for case in parser.dspy_eval_cases)
