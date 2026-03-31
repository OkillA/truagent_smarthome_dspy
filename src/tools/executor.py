import logging
from typing import Dict, Any

from .registry import ToolRegistry
from .base import ToolContext, ToolResult

class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute_from_operator_spec(self, operator_spec: Any, slot_values: Dict[str, Any], metadata: Dict[str, Any]) -> ToolResult:
        tool_name = operator_spec.tool_name
        if not tool_name:
            raise ValueError("Operator spec missing tool_name")

        tool = self.registry.get(tool_name)
        
        inputs = {}
        if operator_spec.requires_slot:
            for slot_req in operator_spec.requires_slot.split(','):
                slot_req = slot_req.strip()
                if slot_req in slot_values:
                    inputs[slot_req] = slot_values[slot_req]

        affected_slots = []
        if operator_spec.affected_slot:
            affected_slots = [s.strip() for s in operator_spec.affected_slot.split(',')]

        context = ToolContext(
            operator_id=operator_spec.operator_id,
            phase=operator_spec.phase,
            inputs=inputs,
            write_method=operator_spec.write_method,
            affected_slots=affected_slots,
            metadata=metadata
        )

        logging.info(f"Executing tool {tool_name} for operator {operator_spec.operator_id}")
        return tool.execute(context)
