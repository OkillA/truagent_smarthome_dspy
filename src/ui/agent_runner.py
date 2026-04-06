import logging
import os
import time
import uuid

from prometheus_client import REGISTRY, Counter, Gauge, Histogram, Summary, push_to_gateway, start_http_server

from generators.config_validator import validate_or_raise
from generators.csv_parser import CSVParser
from src.conversation.decoder import GenericClassifier
from src.conversation.encoder import TemplateEngine
from src.engine.cognitive_engine import CognitiveEngine
from src.generated import conversation_models
from src.soar.controller import SoarController
from src.soar.io_manager import IOManager
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry

AGENT_SESSION_TOTAL = Counter(
    "agent_session_total",
    "Total number of agent sessions",
    ["status"],
)
AGENT_SESSION_DURATION = Histogram(
    "agent_session_duration_seconds",
    "Duration of agent sessions in seconds",
)
AGENT_TURNS_PER_SESSION = Summary(
    "agent_turns_per_session",
    "Number of turns per conversation",
)
AGENT_TASK_SUCCESS_TOTAL = Counter(
    "agent_task_success_total",
    "Total number of successfully completed tasks",
)
AGENT_ONE_SHOT_SUCCESS_TOTAL = Counter(
    "agent_one_shot_success_total",
    "Total number of successful sessions completed in a single user turn.",
)
AGENT_USER_TURN_LATENCY_SECONDS = Histogram(
    "agent_user_turn_latency_seconds",
    "Latency from receiving a user turn to producing the next agent response.",
    ["phase"],
)
AGENT_METRICS_SERVER_STATUS = Gauge(
    "agent_metrics_server_status",
    "Whether the Prometheus metrics HTTP server is currently running (1=yes, 0=no).",
)
AGENT_SESSION_LLM_TOKENS = Gauge(
    "agent_session_llm_tokens",
    "Per-session LLM token usage broken down by token type.",
    ["token_type"],
)
AGENT_SESSION_ESTIMATED_COST_USD = Gauge(
    "agent_session_estimated_cost_usd",
    "Estimated LLM cost for the current session in USD.",
)

DEFAULT_METRICS_PORT = 8000
DEFAULT_PUSHGATEWAY_URL = os.getenv("PROMETHEUS_PUSHGATEWAY_URL", "http://localhost:9091")


class AgentRunner:
    def __init__(self, config_dir: str):
        self.parser = CSVParser(config_dir)
        self.parser.parse_all()
        validate_or_raise(self.parser)

        self.registry = ToolRegistry()
        self.registry.discover_and_register("src.tools.plugins")
        self.executor = ToolExecutor(self.registry)
        self.decoder = GenericClassifier(self.parser, conversation_models)
        self.encoder = TemplateEngine(self.parser)
        self.engine = CognitiveEngine(
            parser=self.parser,
            decoder=self.decoder,
            encoder=self.encoder,
            tool_executor=self.executor,
        )
        self.io_manager = IOManager(self.engine)
        self.controller = SoarController(self.engine, self.io_manager)

        self.session_id = uuid.uuid4().hex
        self.start_time = time.time()
        self.turn_count = 0
        AGENT_SESSION_TOTAL.labels(status="started").inc()

    def start_metrics_server(self, port: int = DEFAULT_METRICS_PORT) -> None:
        try:
            start_http_server(port)
            AGENT_METRICS_SERVER_STATUS.set(1)
        except Exception as exc:
            AGENT_METRICS_SERVER_STATUS.set(0)
            logging.error(f"Failed to start Prometheus server: {exc}")

    def start_session(self) -> str | None:
        message = self.controller.boot()
        self._push_metrics_snapshot()
        return message

    def handle_turn(self, user_input: str) -> str | None:
        self.turn_count += 1
        turn_start = time.time()
        message = self.controller.handle_user_turn(user_input)
        AGENT_USER_TURN_LATENCY_SECONDS.labels(
            phase=self.controller.current_phase()
        ).observe(time.time() - turn_start)
        self._push_metrics_snapshot()
        return message

    def finalize_session(self) -> None:
        duration = time.time() - self.start_time
        AGENT_SESSION_DURATION.observe(duration)
        AGENT_TURNS_PER_SESSION.observe(self.turn_count)
        if self.engine.slots.get("task-complete") == "yes":
            AGENT_TASK_SUCCESS_TOTAL.inc()
            if self.turn_count == 1:
                AGENT_ONE_SHOT_SUCCESS_TOTAL.inc()
        usage = self.decoder.usage_snapshot()
        AGENT_SESSION_LLM_TOKENS.labels(token_type="prompt").set(usage["prompt_tokens"])
        AGENT_SESSION_LLM_TOKENS.labels(token_type="completion").set(usage["completion_tokens"])
        AGENT_SESSION_ESTIMATED_COST_USD.set(usage["estimated_cost_usd"])
        self._push_metrics_snapshot()

    def run_interactive(self) -> None:
        print("Initializing Smart Home Configuration Agent...\n")
        print(f"Prometheus metrics available at http://localhost:{DEFAULT_METRICS_PORT}/metrics\n")
        self.start_metrics_server(DEFAULT_METRICS_PORT)

        message = self.start_session()
        if message and message != "[HALT]":
            print(f"Agent: {message}")

        try:
            while True:
                if self.controller.is_complete():
                    print("Task marked as complete. Halting.")
                    AGENT_SESSION_TOTAL.labels(status="completed").inc()
                    break

                user_input = input("You: ")
                message = self.handle_turn(user_input)
                if message == "[HALT]":
                    AGENT_SESSION_TOTAL.labels(status="completed").inc()
                    break
                if message:
                    print(f"Agent: {message}")
        finally:
            self.finalize_session()

    def _push_metrics_snapshot(self) -> None:
        try:
            push_to_gateway(
                DEFAULT_PUSHGATEWAY_URL,
                job="smart_home_agent_session",
                registry=REGISTRY,
                grouping_key={"session_id": self.session_id},
            )
        except Exception as exc:
            logging.error(f"Failed to push metrics to Pushgateway: {exc}")


def get_default_config_dir() -> str:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(root_dir, "agent_config")
