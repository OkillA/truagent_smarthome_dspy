from __future__ import annotations

from .types import Impasse, RecoveryDirective, Substate


class ImpasseManager:
    def __init__(self):
        self.impasses: list[Impasse] = []
        self.substates: list[Substate] = []

    def open_impasse(self, kind: str, goal: str, context: dict[str, str], cycle_index: int) -> Impasse:
        impasse = Impasse(
            impasse_id=f"impasse-{len(self.impasses) + 1}",
            kind=kind,
            goal=goal,
            context=dict(context),
            cycle_index=cycle_index,
        )
        self.impasses.append(impasse)
        self.substates.append(
            Substate(
                substate_id=f"substate-{len(self.substates) + 1}",
                parent_goal=goal,
                impasse_id=impasse.impasse_id,
                depth=len(self.substates) + 1,
                context=dict(context),
            )
        )
        return impasse

    def resolve_latest(self) -> None:
        if self.impasses:
            self.impasses[-1].resolved = True
        if self.substates:
            self.substates.pop()

    def current_depth(self) -> int:
        return len(self.substates)

    def handle_interrupt(
        self,
        routing_category: str,
        goal: str,
        context: dict[str, str],
        cycle_index: int,
        has_interrupted_operator: bool,
    ) -> tuple[Impasse, RecoveryDirective]:
        impasse_kind = "interpretation-failure" if routing_category == "help_request" else "missing-required-parameter"
        impasse = self.open_impasse(
            kind=impasse_kind,
            goal=goal,
            context=context,
            cycle_index=cycle_index,
        )
        directive = RecoveryDirective()
        if has_interrupted_operator:
            directive.resume_interrupted = True
            return impasse, directive

        if routing_category == "help_request":
            directive.slot_updates = {
                "greeting-issued": "true",
                "dialogue-phase": "intent-collection",
            }
            directive.next_operator_id = "classify-intent"
        elif routing_category == "off_topic":
            directive.slot_updates = {
                "greeting-issued": "true",
                "dialogue-phase": "intent-collection",
            }
            directive.next_operator_id = "classify-intent"
            directive.prompt_template_ids = ["classify-intent"]
        return impasse, directive

    def handle_ambiguous_confirmation(
        self,
        goal: str,
        context: dict[str, str],
        cycle_index: int,
    ) -> tuple[Impasse, RecoveryDirective]:
        impasse = self.open_impasse(
            kind="insufficient-preference",
            goal=goal,
            context=context,
            cycle_index=cycle_index,
        )
        return impasse, RecoveryDirective(suppress_impasse_increment=True)

    def handle_no_operator(
        self,
        goal: str,
        context: dict[str, str],
        cycle_index: int,
    ) -> tuple[Impasse, RecoveryDirective]:
        impasse = self.open_impasse(
            kind="no-operator",
            goal=goal,
            context=context,
            cycle_index=cycle_index,
        )
        return impasse, RecoveryDirective(next_operator_id="impasse-recovery")

    def handle_tie(
        self,
        goal: str,
        context: dict[str, str],
        cycle_index: int,
    ) -> tuple[Impasse, RecoveryDirective]:
        impasse = self.open_impasse(
            kind="tie",
            goal=goal,
            context=context,
            cycle_index=cycle_index,
        )
        return impasse, RecoveryDirective(next_operator_id="impasse-recovery")

    def handle_interpretation_failure(
        self,
        goal: str,
        context: dict[str, str],
        cycle_index: int,
    ) -> tuple[Impasse, RecoveryDirective]:
        impasse = self.open_impasse(
            kind="interpretation-failure",
            goal=goal,
            context=context,
            cycle_index=cycle_index,
        )
        return impasse, RecoveryDirective(next_operator_id="impasse-recovery")
