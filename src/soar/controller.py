from __future__ import annotations


class SoarController:
    """Cycle coordinator that treats the engine as a Soar-like decision system."""

    def __init__(self, engine, io_manager):
        self.engine = engine
        self.io = io_manager

    def boot(self) -> str | None:
        return self.io.advance_until_output()

    def handle_user_turn(self, user_input: str) -> str | None:
        self.io.write_user_input(user_input)
        return self.io.advance_until_output()

    def is_complete(self) -> bool:
        return self.engine.slots.get("task-complete") == "yes"

    def current_phase(self) -> str:
        return self.engine.slots.get("dialogue-phase", "unknown")
