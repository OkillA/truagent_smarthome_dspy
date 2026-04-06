from __future__ import annotations


class ControlState:
    """Projects execution control state into working memory for symbolic inspection."""

    def __init__(self, memory, unknown_sentinel: str):
        self.memory = memory
        self.unknown_sentinel = unknown_sentinel
        self._pending_operator = None
        self._interrupted_operator = None
        self._sync()

    def _sync(self) -> None:
        pending_id = getattr(self._pending_operator, "operator_id", self.unknown_sentinel)
        interrupted_id = getattr(self._interrupted_operator, "operator_id", self.unknown_sentinel)
        self.memory.set_slot("control.pending-operator-id", pending_id, provenance="control", parent_slot="control")
        self.memory.set_slot(
            "control.interrupted-operator-id",
            interrupted_id,
            provenance="control",
            parent_slot="control",
        )

    @property
    def pending_operator(self):
        return self._pending_operator

    @pending_operator.setter
    def pending_operator(self, operator) -> None:
        self._pending_operator = operator
        self._sync()

    @property
    def interrupted_operator(self):
        return self._interrupted_operator

    @interrupted_operator.setter
    def interrupted_operator(self, operator) -> None:
        self._interrupted_operator = operator
        self._sync()

    def clear_pending(self) -> None:
        self.pending_operator = None

    def clear_interrupted(self) -> None:
        self.interrupted_operator = None

