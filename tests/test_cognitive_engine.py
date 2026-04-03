from pathlib import Path

from generators.csv_parser import CSVParser
from src.conversation.encoder import TemplateEngine
from src.engine.cognitive_engine import CognitiveEngine
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry


class StubDecoder:
    def classify(self, user_input: str, utterance_type: str, slots_to_extract: list, slot_values: dict = None) -> dict:
        if utterance_type == "routing":
            normalized = user_input.strip().lower()
            if "baseball" in normalized:
                return {"conversation-category": "off_topic"}
            if normalized in {"help", "what can you do?"}:
                return {"conversation-category": "help_request"}
            if normalized in {"exit", "quit"}:
                return {"conversation-category": "exit"}
            if normalized in {"yes", "y"}:
                return {"affirmation": "confirmed"}
            if normalized in {"no", "n"}:
                return {"affirmation": "declined"}
            return {"conversation-category": "task_related"}

        if utterance_type == "task_related":
            normalized = user_input.strip().lower()
            result = {}
            if "lighting" in normalized:
                result["intent"] = "configure-lighting"
            if "bedroom" in normalized:
                result["lighting-params.room"] = "bedroom"
            if "low" in normalized or "low cost" in normalized or "cheap" in normalized:
                result["lighting-params.budget"] = "low"
            if "scheduled" in normalized:
                result["lighting-params.automation-level"] = "scheduled"
            if "sensor based" in normalized or "sensor-based" in normalized or "motion" in normalized:
                result["lighting-params.automation-level"] = "reactive"
            return result

        return {}


def advance_until_message(engine: CognitiveEngine, max_cycles: int = 20):
    for _ in range(max_cycles):
        msg = engine.run_cycle()
        if msg is not None:
            return msg
    raise AssertionError("Engine did not produce a message within the cycle limit")


def build_engine() -> CognitiveEngine:
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


def test_engine_reaches_recommendation_and_completion_for_lighting_flow():
    engine = build_engine()

    greeting = advance_until_message(engine)
    assert "Smart Home Configuration Agent" in greeting

    engine.process_input("I want lighting")
    confirmation = advance_until_message(engine)
    assert "configure-lighting" in confirmation

    engine.process_input("yes")
    room_prompt = advance_until_message(engine)
    assert "Which room" in room_prompt

    engine.process_input("bedroom")
    budget_prompt = advance_until_message(engine)
    assert "budget" in budget_prompt

    engine.process_input("low")
    automation_prompt = advance_until_message(engine)
    assert "automation" in automation_prompt

    engine.process_input("scheduled")
    recommendation = advance_until_message(engine)
    assert "wifi-smart-plugs" in recommendation
    assert engine.slots["lighting-params.recommended-method"] == "wifi-smart-plugs"

    engine.process_input("yes")
    halt_message = advance_until_message(engine)
    assert halt_message == "[HALT]"
    assert engine.slots["task-complete"] == "yes"
    assert engine.slots["agent-message"] == "Task completed successfully. Goodbye!"


class OvereagerConfirmationDecoder(StubDecoder):
    def classify(self, user_input: str, utterance_type: str, slots_to_extract: list, slot_values: dict = None) -> dict:
        if utterance_type == "routing" and user_input.strip().lower() == "yes":
            return {"conversation-category": "task_related", "affirmation": "confirmed"}

        if utterance_type == "task_related" and user_input.strip().lower() == "yes":
            return {
                "intent": "configure-lighting",
                "lighting-params.room": "living-room",
                "lighting-params.automation-level": "reactive",
            }

        return super().classify(user_input, utterance_type, slots_to_extract, slot_values)


def test_confirmation_turn_does_not_chain_into_task_extraction():
    repo_root = Path(__file__).resolve().parents[1]
    parser = CSVParser(str(repo_root / "agent_config"))
    parser.parse_all()

    registry = ToolRegistry()
    registry.discover_and_register("src.tools.plugins")

    engine = CognitiveEngine(
        parser=parser,
        decoder=OvereagerConfirmationDecoder(),
        encoder=TemplateEngine(parser),
        tool_executor=ToolExecutor(registry),
    )

    advance_until_message(engine)
    engine.process_input("I want lighting")
    advance_until_message(engine)

    engine.process_input("yes")

    assert engine.slots["confirmation-status"] == "confirmed"
    assert engine.slots["lighting-params.room"] == "unknown"
    assert engine.slots["lighting-params.automation-level"] == "unknown"


def test_rich_initial_utterance_skips_repeated_questions():
    engine = build_engine()

    advance_until_message(engine)
    engine.process_input("I want lighting in the bedroom with a low budget and scheduled automation")
    confirmation = advance_until_message(engine)

    assert "configure-lighting" in confirmation
    assert engine.slots["lighting-params.room"] == "bedroom"
    assert engine.slots["lighting-params.budget"] == "low"
    assert engine.slots["lighting-params.automation-level"] == "scheduled"

    engine.process_input("yes")
    recommendation = advance_until_message(engine)

    assert "wifi-smart-plugs" in recommendation
    assert "Which room" not in recommendation
    assert "What is your budget" not in recommendation
    assert "How much automation" not in recommendation


def test_sensor_based_low_cost_request_gets_reactive_rule_directly():
    engine = build_engine()

    advance_until_message(engine)
    engine.process_input("I want to set up lighting in my bedroom at a low cost that is sensor based")
    confirmation = advance_until_message(engine)

    assert "configure-lighting" in confirmation
    assert engine.slots["lighting-params.room"] == "bedroom"
    assert engine.slots["lighting-params.budget"] == "low"
    assert engine.slots["lighting-params.automation-level"] == "reactive"

    engine.process_input("yes")
    recommendation = advance_until_message(engine)

    assert "budget-motion-sensor-bulbs" in recommendation


def test_off_topic_input_during_parameter_collection_redirects_cleanly():
    engine = build_engine()

    advance_until_message(engine)
    engine.process_input("I want lighting")
    advance_until_message(engine)
    engine.process_input("yes")
    room_prompt = advance_until_message(engine)
    assert "Which room" in room_prompt

    engine.process_input("baseball is better actually")
    off_topic_message = advance_until_message(engine)

    assert "focused on smart home configuration" in off_topic_message
    assert engine.slots["lighting-params.room"] == "unknown"
    assert "Which room" in off_topic_message
    assert engine.pending_operator is not None


def test_exit_input_returns_farewell_and_marks_task_complete():
    engine = build_engine()

    advance_until_message(engine)
    engine.process_input("exit")
    goodbye = advance_until_message(engine)

    assert "Ending the session now" in goodbye
    assert engine.slots["task-complete"] == "yes"
