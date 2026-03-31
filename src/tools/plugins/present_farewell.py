from typing import Dict, Any
from ..base import BaseTool, ToolContext, ToolResult

class PresentFarewellTool(BaseTool):
    @property
    def name(self) -> str:
        return "present_farewell"

    @property
    def description(self) -> str:
        return "Presents the summary."

    def _execute_impl(self, context: ToolContext) -> ToolResult:
        return self._create_success(data={
            "agent-message": "Task completed successfully. Goodbye!",
            "task-complete": "yes"
        })
