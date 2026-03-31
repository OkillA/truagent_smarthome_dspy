import logging
from typing import Dict, Optional

class CognitiveEngine:
    def __init__(self, parser, decoder, encoder, tool_executor):
        self.parser = parser
        self.decoder = decoder
        self.encoder = encoder
        self.tool_executor = tool_executor
        
        # Working Memory (Flat dict of slots)
        self.slots: Dict[str, str] = {s.slot_name: s.default_value for s in parser.slots}
        
        self.pending_operator = None

    def _eval_conditions(self, conditions_str: str) -> bool:
        if not conditions_str:
            return True
            
        parts = conditions_str.split(' AND ')
        for part in parts:
            part = part.strip()
            if '==' in part:
                k, v = part.split('==')
                k = k.strip()
                v = v.strip()
                if self.slots.get(k, 'unknown') != v:
                    return False
            elif '!=' in part:
                k, v = part.split('!=')
                k = k.strip()
                v = v.strip()
                if self.slots.get(k, 'unknown') == v:
                    return False
        return True

    def _find_best_operator(self):
        valid_ops = []
        for op in self.parser.operators:
            
            # Idempotency guard for orchestration
            if op.operator_type == 'orchestration' and op.affected_slot:
                current_val = self.slots.get(op.affected_slot, '')
                if current_val == op.expected_value:
                    continue
                    
            if self._eval_conditions(op.conditions):
                valid_ops.append(op)
                
        if not valid_ops:
            return None
            
        valid_ops.sort(key=lambda x: x.priority, reverse=True)
        return valid_ops[0]

    def process_input(self, user_input: str):
        if not self.pending_operator:
            return

        op = self.pending_operator
        
        # Capture pre-processing state to detect No-Change impasse
        old_slots = self.slots.copy()

        if op.classifier_tool == "target_input":
            if op.affected_slot:
                self.slots[op.affected_slot] = user_input
        else:
            extracted = self.decoder.classify(
                user_input=user_input,
                utterance_type=op.classifier_tool,
                slots_to_extract=[],
                slot_values=self.slots
            )
            
            for k, v in extracted.items():
                if v != "unknown":
                    self.slots[k] = v

            trigger_slot = self.parser.agent_config.get("chaining_trigger_slot", "conversation-category")
            trigger_val = self.parser.agent_config.get("chaining_trigger_value", "task_related")

            if self.slots.get(trigger_slot) == trigger_val:
                chained = self.decoder.classify(
                    user_input=user_input,
                    utterance_type=trigger_val,
                    slots_to_extract=[],
                    slot_values=self.slots
                )
                for k, v in chained.items():
                    if v != "unknown":
                        self.slots[k] = v

            aff_field = self.parser.agent_config.get("affirmation_field", "affirmation")
            aff_skip = self.parser.agent_config.get("affirmation_skip_slot", "")
            aff_val = self.slots.get(aff_field, "unknown")
            if aff_val != "unknown" and op.affected_slot and op.affected_slot != aff_skip:
                self.slots[op.affected_slot] = aff_val
                self.slots[aff_field] = "unknown"

        # Impasse Detection: Did we actually change anything?
        # Only check against slots that are NOT impasse-count
        state_changed = False
        for k in self.slots:
            if k == 'impasse-count':
                continue
            if self.slots[k] != old_slots.get(k):
                state_changed = True
                break
        
        if not state_changed:
            current_impasse = int(self.slots.get('impasse-count', 0))
            self.slots['impasse-count'] = str(current_impasse + 1)
        else:
            self.slots['impasse-count'] = '0'

        self.pending_operator = None

    def run_cycle(self) -> Optional[str]:
        if self.pending_operator:
            return None

        if self.slots.get('task-complete') == 'yes':
            return "[HALT]"

        best_op = self._find_best_operator()
        if not best_op:
            logging.debug("No operator proposed.")
            return "[HALT]"

        logging.info(f"Cycler Selected: {best_op.operator_id} (Type: {best_op.operator_type})")

        if best_op.operator_type == "orchestration":
            if best_op.affected_slot:
                self.slots[best_op.affected_slot] = best_op.expected_value
            return None 

        elif best_op.operator_type == "action":
            result = self.tool_executor.execute_from_operator_spec(
                operator_spec=best_op, 
                slot_values=self.slots,
                metadata={'parser': self.parser}
            )
            for k, v in result.data.items():
                self.slots[k] = str(v)
            return None

        elif best_op.operator_type == "nlu":
            self.pending_operator = best_op
            if best_op.utterance_template_id:
                msg = self.encoder.generate(best_op.utterance_template_id, self.slots)
                return msg
            return None
            
        return None
