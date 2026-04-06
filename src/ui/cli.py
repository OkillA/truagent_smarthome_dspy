import os
import logging
import uuid
import time
import pynvml

from generators.csv_parser import CSVParser
from generators.config_validator import validate_or_raise
from src.generated import conversation_models
from src.conversation.decoder import GenericClassifier
from src.conversation.encoder import TemplateEngine
from src.tools.registry import ToolRegistry
from src.tools.executor import ToolExecutor
from src.engine.cognitive_engine import CognitiveEngine
from prometheus_client import REGISTRY, start_http_server, Counter, Histogram, Summary, Gauge, push_to_gateway

# Hardware Metrics
HARDWARE_GPU_VRAM_BYTES = Gauge(
    "hardware_gpu_vram_used_bytes",
    "Amount of GPU VRAM currently in use.",
    ["device_id"]
)

# Macro-Level Metrics
AGENT_SESSION_TOTAL = Counter(
    'agent_session_total', 
    'Total number of agent sessions', 
    ['status']
)
AGENT_SESSION_DURATION = Histogram(
    'agent_session_duration_seconds', 
    'Duration of agent sessions in seconds'
)
AGENT_TURNS_PER_SESSION = Summary(
    'agent_turns_per_session', 
    'Number of turns per conversation'
)
AGENT_TASK_SUCCESS_TOTAL = Counter(
    'agent_task_success_total', 
    'Total number of successfully completed tasks'
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

logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

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
            tool_executor=self.executor
        )
        
        # Session State
        self.session_id = uuid.uuid4().hex
        self.start_time = time.time()
        self.turn_count = 0
        AGENT_SESSION_TOTAL.labels(status='started').inc()

        # Init Hardware Monitoring
        self.pynvml_available = False
        try:
            pynvml.nvmlInit()
            self.pynvml_available = True
        except Exception:
            logging.debug("NVIDIA GPU monitoring not available.")

    def _refresh_hardware_metrics(self):
        if not self.pynvml_available:
            return
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                HARDWARE_GPU_VRAM_BYTES.labels(device_id=str(i)).set(info.used)
        except Exception as e:
            logging.debug(f"Failed to refresh GPU metrics: {e}")

    def run(self):
        print("Initializing Smart Home Configuration Agent...\n")
        print(f"Prometheus metrics available at http://localhost:{DEFAULT_METRICS_PORT}/metrics\n")
        
        # Start Prometheus metrics server
        try:
            start_http_server(DEFAULT_METRICS_PORT)
            AGENT_METRICS_SERVER_STATUS.set(1)
        except Exception as e:
            AGENT_METRICS_SERVER_STATUS.set(0)
            logging.error(f"Failed to start Prometheus server: {e}")
        
        msg = self._advance_to_message()
        if msg and msg != "[HALT]":
            print(f"Agent: {msg}")

        try:
            while True:
                self._refresh_hardware_metrics()
                if self.engine.slots.get('task-complete') == 'yes':
                    print("Task marked as complete. Halting.")
                    AGENT_TASK_SUCCESS_TOTAL.inc()
                    AGENT_SESSION_TOTAL.labels(status='completed').inc()
                    break

                user_input = input("You: ")
                self.turn_count += 1
                turn_start = time.time()

                self.engine.process_input(user_input)

                msg = self._advance_to_message()
                AGENT_USER_TURN_LATENCY_SECONDS.labels(
                    phase=self.engine.slots.get("dialogue-phase", "unknown")
                ).observe(time.time() - turn_start)
                if msg == "[HALT]":
                    if self.engine.slots.get('task-complete') == 'yes':
                        AGENT_TASK_SUCCESS_TOTAL.inc()
                    AGENT_SESSION_TOTAL.labels(status='completed').inc()
                    break
                elif msg:
                    print(f"Agent: {msg}")
        finally:
            # Final Session Metrics
            duration = time.time() - self.start_time
            AGENT_SESSION_DURATION.observe(duration)
            AGENT_TURNS_PER_SESSION.observe(self.turn_count)
            if self.engine.slots.get('task-complete') == 'yes' and self.turn_count == 1:
                AGENT_ONE_SHOT_SUCCESS_TOTAL.inc()
            self._push_metrics_snapshot()


    def _advance_to_message(self):
        max_cycles = 100
        for _ in range(max_cycles):
            msg = self.engine.run_cycle()
            if msg is not None:
                if 'agent-message' in self.engine.slots:
                    action_msg = self.engine.slots.pop('agent-message')
                    if msg != "[HALT]":
                        msg = action_msg + "\n" + msg
                    else:
                        print(f"Agent: {action_msg}")
                return msg
        print("Agent stalled (exceeded max_cycles without NLU interaction).")
        return "[HALT]"

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
    return os.path.join(root_dir, 'agent_config')


def main() -> None:
    config_dir = get_default_config_dir()
    runner = AgentRunner(config_dir)
    runner.run()


if __name__ == "__main__":
    main()
