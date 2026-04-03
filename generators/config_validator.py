from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from generators.csv_parser import CSVParser
from src.tools.registry import ToolRegistry


VALID_OPERATOR_TYPES = {"orchestration", "action", "nlu"}


@dataclass
class ValidationIssue:
    message: str


class ConfigValidationError(Exception):
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        lines = "\n".join(f"- {issue.message}" for issue in issues)
        super().__init__(f"Configuration validation failed:\n{lines}")


def _split_csv_values(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_pipe_values(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def _intent_slot_prefix(intent: str) -> str:
    if intent.startswith("configure-"):
        return f"{intent.split('configure-', 1)[1]}-params"
    return ""


def _condition_slot_names(condition: str) -> Iterable[str]:
    if not condition:
        return []

    slot_names = []
    for part in condition.split(" AND "):
        part = part.strip()
        if "==" in part:
            slot_name, _ = part.split("==", 1)
            slot_names.append(slot_name.strip())
        elif "!=" in part:
            slot_name, _ = part.split("!=", 1)
            slot_names.append(slot_name.strip())
    return slot_names


def validate_parser_config(parser: CSVParser, plugins_package: str = "src.tools.plugins") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    slot_names = {slot.slot_name for slot in parser.slots}
    template_ids = {template.utterance_template_id for template in parser.encoder_templates}
    utterance_types = {slot.utterance_type for slot in parser.decoder_slots}
    decoder_full_slot_names = {
        f"{slot.parent_slot}.{slot.slot_name}" if slot.parent_slot else slot.slot_name
        for slot in parser.decoder_slots
    }

    registry = ToolRegistry()
    registry.discover_and_register(plugins_package)
    tool_names = set(registry._tools.keys())

    if not parser.agent_config.get("chaining_classifier"):
        issues.append(ValidationIssue("agent_config is missing 'chaining_classifier'."))
    elif parser.agent_config["chaining_classifier"] not in utterance_types:
        issues.append(
            ValidationIssue(
                f"agent_config chaining_classifier='{parser.agent_config['chaining_classifier']}' "
                "does not match any utterance_type in 04_utterance_decoder.csv."
            )
        )

    if not parser.agent_config.get("chaining_trigger_slot"):
        issues.append(ValidationIssue("agent_config is missing 'chaining_trigger_slot'."))
    elif parser.agent_config["chaining_trigger_slot"] not in slot_names:
        issues.append(
            ValidationIssue(
                f"agent_config chaining_trigger_slot='{parser.agent_config['chaining_trigger_slot']}' "
                "does not match any slot in 01_intent_tree.csv."
            )
        )

    if not parser.agent_config.get("affirmation_field"):
        issues.append(ValidationIssue("agent_config is missing 'affirmation_field'."))
    elif parser.agent_config["affirmation_field"] not in slot_names:
        issues.append(
            ValidationIssue(
                f"agent_config affirmation_field='{parser.agent_config['affirmation_field']}' "
                "does not match any slot in 01_intent_tree.csv."
            )
        )

    for category in _split_pipe_values(parser.agent_config.get("routing_interrupt_categories", "")):
        if category not in {"off_topic", "help_request", "exit"}:
            issues.append(
                ValidationIssue(
                    f"agent_config routing_interrupt_categories contains unsupported category '{category}'."
                )
            )

    for template_id in _split_pipe_values(parser.agent_config.get("clarification_metric_templates", "")):
        if template_id not in template_ids:
            issues.append(
                ValidationIssue(
                    f"agent_config clarification_metric_templates references missing template '{template_id}'."
                )
            )

    for decoder_slot in parser.decoder_slots:
        full_slot_name = (
            f"{decoder_slot.parent_slot}.{decoder_slot.slot_name}" if decoder_slot.parent_slot else decoder_slot.slot_name
        )
        if full_slot_name not in slot_names:
            issues.append(
                ValidationIssue(
                    f"decoder slot '{full_slot_name}' is not defined in 01_intent_tree.csv."
                )
            )

    for eval_case in parser.dspy_eval_cases:
        if eval_case.utterance_type not in utterance_types:
            issues.append(
                ValidationIssue(
                    f"dspy eval case '{eval_case.case_id}' references unknown utterance_type "
                    f"'{eval_case.utterance_type}'."
                )
            )
        for slot_name in _split_csv_values(eval_case.slots_to_extract):
            if slot_name not in slot_names:
                issues.append(
                    ValidationIssue(
                        f"dspy eval case '{eval_case.case_id}' extracts unknown slot '{slot_name}'."
                    )
                )
        try:
            expected_output = json.loads(eval_case.expected_output_json) if eval_case.expected_output_json else {}
        except Exception:
            issues.append(
                ValidationIssue(
                    f"dspy eval case '{eval_case.case_id}' has invalid expected_output_json."
                )
            )
        else:
            for slot_name in expected_output:
                if slot_name not in slot_names:
                    issues.append(
                        ValidationIssue(
                            f"dspy eval case '{eval_case.case_id}' expects unknown slot '{slot_name}'."
                        )
                    )
                if eval_case.slots_to_extract and slot_name not in _split_csv_values(eval_case.slots_to_extract):
                    issues.append(
                        ValidationIssue(
                            f"dspy eval case '{eval_case.case_id}' expects slot '{slot_name}' "
                            "that is not listed in slots_to_extract."
                        )
                    )

    rule_subjects: dict[str, dict[str, str]] = {}
    for triple in parser.tribal_knowledge:
        rule_subjects.setdefault(triple.subject, {})[triple.predicate] = triple.object

    for subject, rule in rule_subjects.items():
        task_type = rule.get("task-type")
        metric = rule.get("metric")
        if task_type and metric:
            slot_prefix = _intent_slot_prefix(task_type)
            metric_slot = f"{slot_prefix}.{metric}" if slot_prefix else metric
            if metric_slot not in slot_names:
                issues.append(
                    ValidationIssue(
                        f"tribal knowledge rule '{subject}' metric '{metric}' does not map to a known slot."
                    )
                )

        secondary_metric = rule.get("also-require-metric")
        if task_type and secondary_metric:
            slot_prefix = _intent_slot_prefix(task_type)
            metric_slot = f"{slot_prefix}.{secondary_metric}" if slot_prefix else secondary_metric
            if metric_slot not in slot_names:
                issues.append(
                    ValidationIssue(
                        f"tribal knowledge rule '{subject}' secondary metric '{secondary_metric}' "
                        "does not map to a known slot."
                    )
                )

    for operator in parser.operators:
        if operator.operator_type not in VALID_OPERATOR_TYPES:
            issues.append(
                ValidationIssue(
                    f"operator '{operator.operator_id}' has unsupported operator_type '{operator.operator_type}'."
                )
            )

        for slot_name in _condition_slot_names(operator.conditions):
            if slot_name not in slot_names:
                issues.append(
                    ValidationIssue(
                        f"operator '{operator.operator_id}' condition references unknown slot '{slot_name}'."
                    )
                )

        for slot_name in _split_csv_values(operator.requires_slot):
            if slot_name not in slot_names:
                issues.append(
                    ValidationIssue(
                        f"operator '{operator.operator_id}' requires unknown slot '{slot_name}'."
                    )
                )

        for slot_name in _split_csv_values(operator.affected_slot):
            if slot_name not in slot_names:
                issues.append(
                    ValidationIssue(
                        f"operator '{operator.operator_id}' affects unknown slot '{slot_name}'."
                    )
                )

        if operator.utterance_template_id and operator.utterance_template_id not in template_ids:
            issues.append(
                ValidationIssue(
                    f"operator '{operator.operator_id}' references missing template "
                    f"'{operator.utterance_template_id}'."
                )
            )

        if operator.operator_type == "nlu":
            if not operator.classifier_tool:
                issues.append(
                    ValidationIssue(
                        f"operator '{operator.operator_id}' is an nlu operator but has no classifier_tool."
                    )
                )
            elif operator.classifier_tool != parser.agent_config.get("verbatim_classifier", "target_input") and (
                operator.classifier_tool not in utterance_types
            ):
                issues.append(
                    ValidationIssue(
                        f"operator '{operator.operator_id}' references unknown classifier_tool "
                        f"'{operator.classifier_tool}'."
                    )
                )

        if operator.operator_type == "action":
            if not operator.tool_name:
                issues.append(
                    ValidationIssue(
                        f"operator '{operator.operator_id}' is an action operator but has no tool_name."
                    )
                )
            elif operator.tool_name not in tool_names:
                issues.append(
                    ValidationIssue(
                        f"operator '{operator.operator_id}' references unknown tool '{operator.tool_name}'."
                    )
                )

    for full_slot_name in decoder_full_slot_names:
        if full_slot_name not in slot_names:
            issues.append(
                ValidationIssue(
                    f"decoder schema references unknown slot '{full_slot_name}'."
                )
            )

    return issues


def validate_or_raise(parser: CSVParser, plugins_package: str = "src.tools.plugins") -> None:
    issues = validate_parser_config(parser, plugins_package=plugins_package)
    if issues:
        raise ConfigValidationError(issues)


def main() -> None:
    import os

    root_dir = os.path.dirname(os.path.dirname(__file__))
    config_dir = os.path.join(root_dir, "agent_config")

    parser = CSVParser(config_dir)
    parser.parse_all()
    validate_or_raise(parser)
    print("Configuration validation passed.")


if __name__ == "__main__":
    main()
