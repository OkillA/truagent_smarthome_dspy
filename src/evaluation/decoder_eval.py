import json
import os
from dataclasses import dataclass

from generators.csv_parser import CSVParser, DSPyEvalCase
from generators.config_validator import validate_or_raise
from src.conversation.decoder import GenericClassifier
from src.generated import conversation_models


@dataclass
class EvalResult:
    case_id: str
    utterance_type: str
    passed: bool
    expected: dict[str, str]
    predicted: dict[str, str]
    missing_keys: list[str]
    wrong_values: dict[str, dict[str, str]]
    extra_keys: list[str]


def parse_json_object(raw: str) -> dict[str, str]:
    if not raw:
        return {}
    return json.loads(raw)


def parse_slots_to_extract(raw: str) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def evaluate_case(classifier: GenericClassifier, case: DSPyEvalCase) -> EvalResult:
    state = parse_json_object(case.state_json)
    expected = parse_json_object(case.expected_output_json)
    predicted = classifier.classify(
        user_input=case.user_input,
        utterance_type=case.utterance_type,
        slots_to_extract=parse_slots_to_extract(case.slots_to_extract),
        slot_values=state,
    )

    missing_keys = [key for key in expected if key not in predicted]
    wrong_values = {
        key: {"expected": str(expected[key]), "predicted": str(predicted[key])}
        for key in expected
        if key in predicted and str(predicted[key]) != str(expected[key])
    }
    extra_keys = [key for key in predicted if key not in expected]

    return EvalResult(
        case_id=case.case_id,
        utterance_type=case.utterance_type,
        passed=not missing_keys and not wrong_values and not extra_keys,
        expected=expected,
        predicted=predicted,
        missing_keys=missing_keys,
        wrong_values=wrong_values,
        extra_keys=extra_keys,
    )


def format_summary(results: list[EvalResult]) -> str:
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    lines = [f"Decoder eval: {passed}/{total} cases passed."]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"[{status}] {result.case_id} ({result.utterance_type})")
        if result.passed:
            continue
        if result.missing_keys:
            lines.append(f"  missing: {', '.join(result.missing_keys)}")
        if result.extra_keys:
            lines.append(f"  extra: {', '.join(result.extra_keys)}")
        for key, values in result.wrong_values.items():
            lines.append(
                f"  wrong {key}: expected={values['expected']} predicted={values['predicted']}"
            )
    return "\n".join(lines)


def main() -> None:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_dir = os.path.join(root_dir, "agent_config")

    parser = CSVParser(config_dir)
    parser.parse_all()
    validate_or_raise(parser)

    classifier = GenericClassifier(parser, conversation_models)
    results = [evaluate_case(classifier, case) for case in parser.dspy_eval_cases]
    summary = format_summary(results)
    print(summary)

    if any(not result.passed for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
