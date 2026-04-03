import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from prometheus_client import Counter, Histogram

TOOL_EXECUTIONS_TOTAL = Counter(
    "tool_executions_total",
    "Total number of tool executions.",
    ["tool_name", "status"],
)
TOOL_EXECUTION_SECONDS = Histogram(
    "tool_execution_seconds",
    "Tool execution latency in seconds.",
    ["tool_name"],
)

class ToolStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PARTIAL = "PARTIAL"
    SKIPPED = "SKIPPED"

@dataclass
class ToolResult:
    status: ToolStatus
    data: Dict[str, Any]
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0

@dataclass
class ToolContext:
    operator_id: str
    phase: str
    inputs: Dict[str, Any]
    write_method: str
    affected_slots: List[str]
    metadata: Dict[str, Any]

    def get_input(self, key: str, default: Any = None) -> Any:
        return self.inputs.get(key, default)

class BaseTool:
    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def description(self) -> str:
        raise NotImplementedError

    def execute(self, context: ToolContext) -> ToolResult:
        start_time = time.time()
        try:
            result = self._execute_impl(context)
            result.execution_time = time.time() - start_time
            TOOL_EXECUTIONS_TOTAL.labels(tool_name=self.name, status=result.status.value).inc()
            TOOL_EXECUTION_SECONDS.labels(tool_name=self.name).observe(result.execution_time)
            return result
        except Exception as e:
            logging.exception(f"Error in tool {self.name}: {e}")
            result = self._create_failure(str(e))
            result.execution_time = time.time() - start_time
            TOOL_EXECUTIONS_TOTAL.labels(tool_name=self.name, status=result.status.value).inc()
            TOOL_EXECUTION_SECONDS.labels(tool_name=self.name).observe(result.execution_time)
            return result

    def _execute_impl(self, context: ToolContext) -> ToolResult:
        raise NotImplementedError

    def _create_success(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> ToolResult:
        return ToolResult(ToolStatus.SUCCESS, data=data, metadata=metadata)

    def _create_failure(self, error: str, metadata: Optional[Dict[str, Any]] = None) -> ToolResult:
        return ToolResult(ToolStatus.FAILURE, data={}, error=error, metadata=metadata)
