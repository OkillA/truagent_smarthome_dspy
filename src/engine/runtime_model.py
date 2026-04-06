from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeModel:
    phase_order: dict[str, int]
    phase_goals: dict[str, str]
    intent_utterance_types: dict[str, str]

    @classmethod
    def from_parser(cls, parser) -> "RuntimeModel":
        phase_order = cls._build_phase_order(parser)
        phase_goals = cls._build_phase_goals(phase_order)
        intent_utterance_types = cls._build_intent_utterance_types(parser)
        return cls(
            phase_order=phase_order,
            phase_goals=phase_goals,
            intent_utterance_types=intent_utterance_types,
        )

    @staticmethod
    def _build_phase_order(parser) -> dict[str, int]:
        dialogue_phase = next((slot for slot in parser.slots if slot.slot_name == "dialogue-phase"), None)
        if not dialogue_phase or not dialogue_phase.accepted_values:
            return {}
        phases = [value.strip() for value in dialogue_phase.accepted_values.split("|") if value.strip()]
        return {phase: index for index, phase in enumerate(phases)}

    @staticmethod
    def _goal_label_for_phase(phase: str) -> str:
        return phase.replace("-", " ")

    @classmethod
    def _build_phase_goals(cls, phase_order: dict[str, int]) -> dict[str, str]:
        return {phase: cls._goal_label_for_phase(phase) for phase in phase_order}

    @staticmethod
    def _build_intent_utterance_types(parser) -> dict[str, str]:
        intent_map: dict[str, str] = {}
        for operator in parser.operators:
            if operator.operator_type != "nlu":
                continue
            if not operator.classifier_tool or operator.classifier_tool in {"routing", "target_input"}:
                continue
            for condition in operator.conditions.split(" AND "):
                condition = condition.strip()
                if not condition.startswith("intent=="):
                    continue
                intent = condition.split("==", 1)[1].strip()
                intent_map.setdefault(intent, operator.classifier_tool)
        return intent_map

