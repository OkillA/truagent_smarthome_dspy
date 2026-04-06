from pathlib import Path

from generators.csv_parser import CSVParser
from src.conversation.encoder import TemplateEngine
from src.engine.cognitive_engine import CognitiveEngine
from src.soar.controller import SoarController
from src.soar.io_manager import IOManager
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry


class StubDecoder:
    def classify(self, user_input: str, utterance_type: str, slots_to_extract: list, slot_values: dict = None) -> dict:
        normalized = user_input.strip().lower()
        if utterance_type == "routing":
            if normalized in {"yes", "y"}:
                return {"affirmation": "confirmed"}
            return {"conversation-category": "task_related"}

        if utterance_type == "task_related":
            result = {}
            if "lighting" in normalized:
                result["intent"] = "configure-lighting"
            return result

        return {}


def _build_engine() -> CognitiveEngine:
    repo_root = Path(__file__).resolve().parents[1]
    parser = CSVParser(str(repo_root / "agent_config"))
    parser.parse_all()
    registry = ToolRegistry()
    registry.discover_and_register("src.tools.plugins")
    return CognitiveEngine(
        parser=parser,
        decoder=StubDecoder(),
        encoder=TemplateEngine(parser),
        tool_executor=ToolExecutor(registry),
    )


def test_io_manager_composes_action_message_with_halt():
    class _Engine:
        def __init__(self):
            self.slots = {"agent-message": "Task completed successfully. Goodbye!"}

        def run_cycle(self):
            return "[HALT]"

    io_manager = IOManager(_Engine())
    assert io_manager.advance_until_output() == "[HALT]"


def test_soar_controller_boots_and_handles_turns_through_io_manager():
    engine = _build_engine()
    controller = SoarController(engine, IOManager(engine))

    greeting = controller.boot()
    assert greeting is not None
    assert "Smart Home Configuration Agent" in greeting

    confirmation = controller.handle_user_turn("I want lighting")
    assert confirmation is not None
    assert "configure-lighting" in confirmation
