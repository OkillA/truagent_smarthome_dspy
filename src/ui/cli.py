import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from generators.csv_parser import CSVParser
from src.generated import conversation_models
from src.conversation.decoder import GenericClassifier
from src.conversation.encoder import TemplateEngine
from src.tools.registry import ToolRegistry
from src.tools.executor import ToolExecutor
from src.engine.cognitive_engine import CognitiveEngine
from prometheus_client import start_http_server, Counter, Histogram, Summary
import time

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

logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

class AgentRunner:
    def __init__(self, config_dir: str):
        self.parser = CSVParser(config_dir)
        self.parser.parse_all()

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
        self.start_time = time.time()
        self.turn_count = 0
        AGENT_SESSION_TOTAL.labels(status='started').inc()

    def run(self):
        print("Initializing Smart Home Configuration Agent...\n")
        print("Prometheus metrics available at http://localhost:8000/metrics\n")
        
        # Start Prometheus metrics server
        try:
            start_http_server(8000)
        except Exception as e:
            logging.error(f"Failed to start Prometheus server: {e}")
        
        msg = self._advance_to_message()
        if msg and msg != "[HALT]":
            print(f"Agent: {msg}")

        while True:
            if self.engine.slots.get('task-complete') == 'yes':
                print("Task marked as complete. Halting.")
                AGENT_TASK_SUCCESS_TOTAL.inc()
                AGENT_SESSION_TOTAL.labels(status='completed').inc()
                break

            user_input = input("You: ")
            self.turn_count += 1
            
            if user_input.lower() in ['exit', 'quit']:
                AGENT_SESSION_TOTAL.labels(status='abandoned').inc()
                break

            self.engine.process_input(user_input)

            msg = self._advance_to_message()
            if msg == "[HALT]":
                AGENT_SESSION_TOTAL.labels(status='completed').inc()
                break
            elif msg:
                print(f"Agent: {msg}")

        # Final Session Metrics
        duration = time.time() - self.start_time
        AGENT_SESSION_DURATION.observe(duration)
        AGENT_TURNS_PER_SESSION.observe(self.turn_count)


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

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_dir = os.path.join(root_dir, 'agent_config')
    runner = AgentRunner(config_dir)
    runner.run()
