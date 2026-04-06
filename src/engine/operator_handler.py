from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class OperatorExecutionResult:
    message: Optional[str] = None
    tool_executed: bool = False
    pending_operator: Any | None = None
    interrupted_operator: Any | None = None
    resumed_operator: Any | None = None
    causal_result: Any | None = None
    slot_updates: list[tuple[str, str, str]] = field(default_factory=list)


class OperatorHandler:
    """Generic operator executor for orchestration, action, and NLU operators."""

    def __init__(self, parser, encoder, tool_executor):
        self.parser = parser
        self.encoder = encoder
        self.tool_executor = tool_executor

    @staticmethod
    def _split_csv_values(raw: str) -> list[str]:
        if not raw:
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]

    def _orchestration_updates(self, operator_spec) -> list[tuple[str, str, str]]:
        affected_slots = self._split_csv_values(operator_spec.affected_slot)
        expected_values = self._split_csv_values(operator_spec.expected_value)
        if not affected_slots:
            return []
        if not expected_values:
            expected_values = [""] * len(affected_slots)
        if len(expected_values) == 1 and len(affected_slots) > 1:
            expected_values = expected_values * len(affected_slots)
        updates: list[tuple[str, str, str]] = []
        for index, slot_name in enumerate(affected_slots):
            value = expected_values[index] if index < len(expected_values) else expected_values[-1]
            updates.append((slot_name, value, "orchestration"))
        return updates

    def execute(
        self,
        operator_spec,
        slot_values: dict[str, str],
        interrupted_operator,
        *,
        metadata: dict,
        on_seed_followup: Callable[[str], list[str]],
        on_nlu_pending: Callable[[Any], None],
    ) -> OperatorExecutionResult:
        result = OperatorExecutionResult()

        if operator_spec.operator_type == "orchestration":
            result.slot_updates.extend(self._orchestration_updates(operator_spec))
            recovery_prompts = on_seed_followup(operator_spec.operator_id)
            if operator_spec.utterance_template_id:
                message = self.encoder.generate(operator_spec.utterance_template_id, slot_values)
                if recovery_prompts:
                    message = "\n".join([message, *recovery_prompts])
                result.message = message
            return result

        if operator_spec.operator_type == "action":
            tool_result = self.tool_executor.execute_from_operator_spec(
                operator_spec=operator_spec,
                slot_values=slot_values,
                metadata=metadata,
            )
            result.tool_executed = True
            result.causal_result = tool_result
            for key, value in tool_result.data.items():
                result.slot_updates.append((key, str(value), f"tool:{operator_spec.tool_name}"))
            return result

        if operator_spec.operator_type == "nlu":
            result.pending_operator = operator_spec
            on_nlu_pending(operator_spec)
            if operator_spec.utterance_template_id:
                result.message = self.encoder.generate(operator_spec.utterance_template_id, slot_values)
            return result

        return result
