from __future__ import annotations


class GoalState:
    """Projects current goals and subgoals into working memory."""

    def __init__(self, memory, unknown_sentinel: str, phase_goals: dict[str, str] | None = None):
        self.memory = memory
        self.unknown_sentinel = unknown_sentinel
        self.phase_goals = phase_goals or {}

    def sync(self, pending_operator, impasse_manager) -> None:
        phase = self.memory.get_slot("dialogue-phase", self.unknown_sentinel)
        current_goal = self.phase_goals.get(phase, phase or self.unknown_sentinel)
        current_subgoal = getattr(pending_operator, "operator_id", self.unknown_sentinel)

        current_impasse_kind = self.unknown_sentinel
        if impasse_manager.impasses:
            current_impasse_kind = impasse_manager.impasses[-1].kind

        self.memory.set_slot("control.current-goal", current_goal, provenance="goal_state", parent_slot="control")
        self.memory.set_slot(
            "control.current-subgoal",
            current_subgoal,
            provenance="goal_state",
            parent_slot="control",
        )
        self.memory.set_slot(
            "control.substate-depth",
            str(impasse_manager.current_depth()),
            provenance="goal_state",
            parent_slot="control",
        )
        self.memory.set_slot(
            "control.current-impasse-kind",
            current_impasse_kind,
            provenance="goal_state",
            parent_slot="control",
        )
