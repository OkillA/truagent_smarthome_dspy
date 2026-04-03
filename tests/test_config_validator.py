from pathlib import Path

from generators.config_validator import ConfigValidationError, validate_or_raise, validate_parser_config
from generators.csv_parser import CSVParser, Operator


def build_parser() -> CSVParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = CSVParser(str(repo_root / "agent_config"))
    parser.parse_all()
    return parser


def test_current_project_config_validates_cleanly():
    parser = build_parser()

    issues = validate_parser_config(parser)

    assert issues == []


def test_validator_flags_unknown_operator_slot_reference():
    parser = build_parser()
    parser.operators.append(
        Operator(
            operator_id="broken-op",
            operator_type="orchestration",
            phase="test",
            priority=1,
            conditions="missing-slot==unknown",
            utterance_template_id="",
            classifier_tool="",
            write_method="",
            tool_name="",
            requires_slot="",
            affected_slot="dialogue-phase",
            expected_value="init",
        )
    )

    issues = validate_parser_config(parser)

    assert any("missing-slot" in issue.message for issue in issues)


def test_validate_or_raise_raises_on_invalid_config():
    parser = build_parser()
    parser.operators.append(
        Operator(
            operator_id="broken-tool",
            operator_type="action",
            phase="test",
            priority=1,
            conditions="",
            utterance_template_id="",
            classifier_tool="",
            write_method="single",
            tool_name="does_not_exist",
            requires_slot="intent",
            affected_slot="task-complete",
            expected_value="yes",
        )
    )

    try:
        validate_or_raise(parser)
    except ConfigValidationError as exc:
        assert "does_not_exist" in str(exc)
    else:
        raise AssertionError("Expected ConfigValidationError to be raised")
