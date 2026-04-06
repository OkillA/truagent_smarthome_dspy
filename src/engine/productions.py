from __future__ import annotations

from collections import OrderedDict

from .types import Condition, OperatorProposal, Preference, Production, SelectionOutcome


class ProductionCompiler:
    def compile_action_space(self, operators: list) -> list[Production]:
        productions: list[Production] = []
        for operator in operators:
            production_id = f"operator:{operator.operator_id}"
            conditions = self.parse_conditions(operator.conditions)
            preferences = [
                Preference(
                    kind="acceptable",
                    weight=operator.priority,
                    source=production_id,
                )
            ]
            productions.append(
                Production(
                    production_id=production_id,
                    conditions=conditions,
                    operator_spec=operator,
                    preferences=preferences,
                    production_type="operator",
                )
            )
        return productions

    def compile_tribal_knowledge(self, triples: list) -> dict[str, dict[str, str]]:
        grouped: dict[str, dict[str, str]] = OrderedDict()
        for triple in triples:
            grouped.setdefault(triple.subject, {})
            grouped[triple.subject][triple.predicate] = triple.object
        return grouped

    def parse_conditions(self, conditions_str: str) -> list[Condition]:
        if not conditions_str:
            return []

        parsed_conditions: list[Condition] = []
        for part in conditions_str.split(" AND "):
            part = part.strip()
            if "==" in part:
                slot_name, expected_value = [token.strip() for token in part.split("==", 1)]
                parsed_conditions.append(Condition(slot_name, "==", expected_value))
            elif "!=" in part:
                slot_name, expected_value = [token.strip() for token in part.split("!=", 1)]
                parsed_conditions.append(Condition(slot_name, "!=", expected_value))
        return parsed_conditions


class ProductionMatcher:
    def __init__(self, unknown_sentinel: str):
        self.unknown_sentinel = unknown_sentinel

    def _condition_matches(self, memory, condition: Condition) -> bool:
        actual_value = memory.get_slot(condition.slot_name, self.unknown_sentinel)
        if condition.comparator == "==":
            return actual_value == condition.expected_value
        if condition.comparator == "!=":
            return actual_value != condition.expected_value
        return False

    def propose(self, productions: list[Production], memory) -> list[OperatorProposal]:
        proposals: list[OperatorProposal] = []
        for production in productions:
            operator = production.operator_spec
            if operator.operator_type == "orchestration" and operator.affected_slot:
                current_val = memory.get_slot(operator.affected_slot, "")
                if current_val == operator.expected_value:
                    continue

            if all(self._condition_matches(memory, condition) for condition in production.conditions):
                proposals.append(
                    OperatorProposal(
                        operator_id=operator.operator_id,
                        operator_type=operator.operator_type,
                        phase=operator.phase,
                        operator_spec=operator,
                        preferences=list(production.preferences),
                        matched_conditions=list(production.conditions),
                        supporting_productions=[production.production_id],
                    )
                )
        return proposals


class PreferenceResolver:
    def select(self, proposals: list[OperatorProposal]) -> SelectionOutcome:
        if not proposals:
            return SelectionOutcome(status="none", reason="no proposals matched current working memory")
        proposals = sorted(
            proposals,
            key=lambda proposal: (proposal.score, proposal.operator_id),
            reverse=True,
        )
        best_score = proposals[0].score
        top_proposals = [proposal for proposal in proposals if proposal.score == best_score]
        if len(top_proposals) > 1:
            return SelectionOutcome(
                status="tie",
                competing_proposals=top_proposals,
                reason=f"multiple operators share top score {best_score}",
            )
        return SelectionOutcome(
            status="selected",
            selected_proposal=proposals[0],
            reason=f"selected highest scoring operator with score {best_score}",
        )
