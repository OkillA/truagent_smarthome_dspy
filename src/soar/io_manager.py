from __future__ import annotations


class IOManager:
    """Compatibility I/O boundary around the current cognitive engine."""

    def __init__(self, engine, max_cycles: int = 100):
        self.engine = engine
        self.max_cycles = max_cycles

    def write_user_input(self, user_input: str) -> None:
        self.engine.process_input(user_input)

    def advance_until_output(self) -> str | None:
        for _ in range(self.max_cycles):
            message = self.engine.run_cycle()
            if message is not None:
                return self._compose_output(message)
        return "[HALT]"

    def _compose_output(self, message: str) -> str:
        action_message = self.engine.slots.get("agent-message")
        if action_message:
            self.engine.slots.pop("agent-message", None)
            if message != "[HALT]":
                return f"{action_message}\n{message}"
        return message
