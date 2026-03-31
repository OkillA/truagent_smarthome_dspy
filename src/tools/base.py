import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

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
            return result
        except Exception as e:
            logging.exception(f"Error in tool {self.name}: {e}")
            return self._create_failure(str(e))

    def _execute_impl(self, context: ToolContext) -> ToolResult:
        raise NotImplementedError

    def _create_success(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> ToolResult:
        return ToolResult(ToolStatus.SUCCESS, data=data, metadata=metadata)

    def _create_failure(self, error: str, metadata: Optional[Dict[str, Any]] = None) -> ToolResult:
        return ToolResult(ToolStatus.FAILURE, data={}, error=error, metadata=metadata)
