from src.evaluation.decoder_eval import EvalResult, format_summary


def test_format_summary_reports_failures_cleanly():
    results = [
        EvalResult(
            case_id="ok-case",
            utterance_type="routing",
            passed=True,
            expected={"conversation-category": "task_related"},
            predicted={"conversation-category": "task_related"},
            missing_keys=[],
            wrong_values={},
            extra_keys=[],
        ),
        EvalResult(
            case_id="bad-case",
            utterance_type="task_related",
            passed=False,
            expected={"intent": "configure-lighting"},
            predicted={"intent": "configure-climate", "extra-slot": "x"},
            missing_keys=[],
            wrong_values={
                "intent": {"expected": "configure-lighting", "predicted": "configure-climate"}
            },
            extra_keys=["extra-slot"],
        ),
    ]

    summary = format_summary(results)

    assert "Decoder eval: 1/2 cases passed." in summary
    assert "[PASS] ok-case" in summary
    assert "[FAIL] bad-case" in summary
    assert "extra: extra-slot" in summary
    assert "wrong intent: expected=configure-lighting predicted=configure-climate" in summary
