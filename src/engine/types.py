from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class WME:
    identifier: str
    attribute: str
    value: str
    slot_name: str
    parent_identifier: Optional[str] = None
    provenance: str = "seed"


@dataclass(frozen=True)
class Condition:
    slot_name: str
    comparator: str
    expected_value: str


@dataclass
class Preference:
    kind: str
    weight: int
    source: str


@dataclass
class OperatorProposal:
    operator_id: str
    operator_type: str
    phase: str
    operator_spec: Any
    preferences: list[Preference] = field(default_factory=list)
    matched_conditions: list[Condition] = field(default_factory=list)
    supporting_productions: list[str] = field(default_factory=list)

    @property
    def score(self) -> int:
        score = 0
        for preference in self.preferences:
            if preference.kind == "reject":
                score -= 10_000 + abs(preference.weight)
            elif preference.kind == "require":
                score += 10_000 + abs(preference.weight)
            else:
                score += preference.weight
        return score

    def preference_summary(self) -> list[dict[str, object]]:
        return [
            {
                "kind": preference.kind,
                "weight": preference.weight,
                "source": preference.source,
            }
            for preference in self.preferences
        ]


@dataclass
class Production:
    production_id: str
    conditions: list[Condition]
    operator_spec: Any
    preferences: list[Preference]
    production_type: str = "operator"


@dataclass
class Impasse:
    impasse_id: str
    kind: str
    goal: str
    context: dict[str, str]
    cycle_index: int
    resolved: bool = False


@dataclass
class Substate:
    substate_id: str
    parent_goal: str
    impasse_id: str
    depth: int
    context: dict[str, str]


@dataclass
class RecoveryDirective:
    next_operator_id: str | None = None
    slot_updates: dict[str, str] = field(default_factory=dict)
    suppress_impasse_increment: bool = False
    resume_interrupted: bool = False
    prompt_template_ids: list[str] = field(default_factory=list)


@dataclass
class SelectionOutcome:
    status: str
    selected_proposal: Optional[OperatorProposal] = None
    competing_proposals: list[OperatorProposal] = field(default_factory=list)
    reason: str = ""


@dataclass
class TraceEvent:
    event_type: str
    cycle_index: int
    payload: dict[str, Any]
