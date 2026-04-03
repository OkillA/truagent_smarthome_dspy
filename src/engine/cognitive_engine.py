import json
import logging
import time
from typing import Dict, Optional
from prometheus_client import Counter, Gauge, Histogram

# Impasse Metrics
AGENT_IMPASSE_TOTAL = Counter(
    'agent_impasse_total', 
    'Total number of impasses (no state change)', 
    ['utterance_type']
)
AGENT_IMPASSE_LEVEL = Gauge(
    'agent_impasse_level', 
    'Current impasse count for the active session'
)
AGENT_OPERATOR_SELECTED_TOTAL = Counter(
    "agent_operator_selected_total",
    "Total number of selected operators.",
    ["operator_id", "operator_type", "phase"],
)
AGENT_OPERATOR_EXECUTION_SECONDS = Histogram(
    "agent_operator_execution_seconds",
    "Operator execution latency in seconds.",
    ["operator_id", "operator_type"],
)
AGENT_SLOT_UPDATES_TOTAL = Counter(
    "agent_slot_updates_total",
    "Total number of slot updates applied by the engine.",
    ["source", "slot_name"],
)
AGENT_PENDING_NLU_TOTAL = Counter(
    "agent_pending_nlu_total",
    "Total number of times the engine prompted for NLU input.",
    ["operator_id", "utterance_type"],
)
AGENT_ROUTING_SIGNAL_TOTAL = Counter(
    "agent_routing_signal_total",
    "Total number of routing signals detected.",
    ["conversation_category"],
)
AGENT_CLARIFICATION_REQUEST_TOTAL = Counter(
    "agent_clarification_request_total",
    "Total number of clarification prompts issued to users.",
    ["utterance_template_id"],
)
AGENT_DECISION_CYCLE_SECONDS = Histogram(
    "agent_decision_cycle_seconds",
    "Total time spent selecting and applying one cognitive cycle.",
    ["operator_type", "phase"],
)
AGENT_NO_CHANGE_TOTAL = Counter(
    "agent_no_change_total",
    "Total number of user turns that caused no meaningful state change.",
    ["operator_id", "utterance_type"],
)
AGENT_CANDIDATE_OPERATORS = Histogram(
    "agent_candidate_operator_count",
    "Number of operators that matched the current working state in a cycle.",
)
AGENT_WORKING_MEMORY_SLOTS = Gauge(
    "agent_working_memory_slots",
    "Current working-memory slot counts by state.",
    ["state"],
)
AGENT_WORKING_MEMORY_BYTES = Gauge(
    "agent_working_memory_payload_bytes",
    "Approximate size of serialized working memory payload in bytes.",
)
AGENT_SLOT_OVERWRITE_TOTAL = Counter(
    "agent_slot_overwrite_total",
    "Total number of times a populated slot was overwritten with a new value.",
    ["slot_name", "source"],
)

class CognitiveEngine:
    def __init__(self, parser, decoder, encoder, tool_executor):
        self.parser = parser
        self.decoder = decoder
        self.encoder = encoder
        self.tool_executor = tool_executor
        self.unknown_sentinel = parser.unknown_sentinel
        
        # Working Memory (Flat dict of slots)
        self.slots: Dict[str, str] = {s.slot_name: s.default_value for s in parser.slots}
        
        self.pending_operator = None
        self.interrupted_operator = None
        self._refresh_memory_metrics()

    def _agent_config_list(self, key: str, delimiter: str = "|") -> list[str]:
        raw = self.parser.agent_config.get(key, "")
        if not raw:
            return []
        return [item.strip() for item in raw.split(delimiter) if item.strip()]

    def _refresh_memory_metrics(self) -> None:
        total_slots = len(self.slots)
        known_slots = sum(1 for value in self.slots.values() if value != self.unknown_sentinel)
        unknown_slots = total_slots - known_slots

        AGENT_WORKING_MEMORY_SLOTS.labels(state="total").set(total_slots)
        AGENT_WORKING_MEMORY_SLOTS.labels(state="known").set(known_slots)
        AGENT_WORKING_MEMORY_SLOTS.labels(state="unknown").set(unknown_slots)
        AGENT_WORKING_MEMORY_BYTES.set(len(json.dumps(self.slots, sort_keys=True)))

    def _apply_slot_update(self, slot_name: str, value: str, source: str) -> None:
        previous_value = self.slots.get(slot_name, self.unknown_sentinel)
        if (
            previous_value != self.unknown_sentinel
            and value != self.unknown_sentinel
            and previous_value != value
        ):
            AGENT_SLOT_OVERWRITE_TOTAL.labels(slot_name=slot_name, source=source).inc()

        self.slots[slot_name] = value
        AGENT_SLOT_UPDATES_TOTAL.labels(source=source, slot_name=slot_name).inc()
        self._refresh_memory_metrics()

    def _should_chain_followup_classification(self, op, extracted: Dict[str, str]) -> bool:
        trigger_slot = self.parser.agent_config.get("chaining_trigger_slot", "conversation-category")
        trigger_val = self.parser.agent_config.get("chaining_trigger_value", "task_related")
        aff_field = self.parser.agent_config.get("affirmation_field", "affirmation")

        if self.slots.get(trigger_slot) != trigger_val:
            return False

        # Pure confirmation/decline turns should not also trigger domain extraction.
        if self.slots.get(aff_field, self.unknown_sentinel) != self.unknown_sentinel:
            return False

        # Only chain when the current step is performing routing/classification,
        # not when we're already inside a domain-specific extractor.
        return op.classifier_tool == self.parser.agent_config.get("chaining_classifier", "routing")

    def _slots_for_utterance_type(self, utterance_type: str) -> list[str]:
        slots = []
        for slot in self.parser.decoder_slots:
            if slot.utterance_type != utterance_type:
                continue
            full_slot_name = f"{slot.parent_slot}.{slot.slot_name}" if slot.parent_slot else slot.slot_name
            if self.slots.get(full_slot_name, self.unknown_sentinel) == self.unknown_sentinel:
                slots.append(full_slot_name)
        return slots

    def _all_slots_for_utterance_type(self, utterance_type: str) -> list[str]:
        slots = []
        for slot in self.parser.decoder_slots:
            if slot.utterance_type != utterance_type:
                continue
            full_slot_name = f"{slot.parent_slot}.{slot.slot_name}" if slot.parent_slot else slot.slot_name
            slots.append(full_slot_name)
        return slots

    def _precheck_routing_signal(self, user_input: str) -> str | None:
        routing_type = self.parser.agent_config.get("chaining_classifier", "routing")
        extracted = self.decoder.classify(
            user_input=user_input,
            utterance_type=routing_type,
            slots_to_extract=self._all_slots_for_utterance_type(routing_type),
            slot_values=self.slots,
        )

        conversation_category = extracted.get("conversation-category", self.unknown_sentinel)
        if conversation_category in set(self._agent_config_list("routing_interrupt_categories")):
            AGENT_ROUTING_SIGNAL_TOTAL.labels(conversation_category=conversation_category).inc()
            self._apply_slot_update("conversation-category", conversation_category, "routing_precheck")
            if "affirmation" in extracted and extracted["affirmation"] != self.unknown_sentinel:
                self._apply_slot_update("affirmation", extracted["affirmation"], "routing_precheck")
            return conversation_category

        return None

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
                if self.slots.get(k, self.unknown_sentinel) != v:
                    return False
            elif '!=' in part:
                k, v = part.split('!=')
                k = k.strip()
                v = v.strip()
                if self.slots.get(k, self.unknown_sentinel) == v:
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

        AGENT_CANDIDATE_OPERATORS.observe(len(valid_ops))
                
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
                self._apply_slot_update(op.affected_slot, user_input, "target_input")
        else:
            if op.classifier_tool != self.parser.agent_config.get("chaining_classifier", "routing"):
                routing_category = self._precheck_routing_signal(user_input)
                if routing_category:
                    if routing_category in {"off_topic", "help_request"}:
                        self.interrupted_operator = op
                    self.pending_operator = None
                    return

            slots_to_extract = self._slots_for_utterance_type(op.classifier_tool)
            extracted = self.decoder.classify(
                user_input=user_input,
                utterance_type=op.classifier_tool,
                slots_to_extract=slots_to_extract,
                slot_values=self.slots
            )
            
            for k, v in extracted.items():
                if v != self.unknown_sentinel:
                    self._apply_slot_update(k, v, op.classifier_tool)

            if self._should_chain_followup_classification(op, extracted):
                trigger_val = self.parser.agent_config.get("chaining_trigger_value", "task_related")
                chained = self.decoder.classify(
                    user_input=user_input,
                    utterance_type=trigger_val,
                    slots_to_extract=self._slots_for_utterance_type(trigger_val),
                    slot_values=self.slots
                )
                for k, v in chained.items():
                    if v != self.unknown_sentinel:
                        self._apply_slot_update(k, v, f"chained:{trigger_val}")

            aff_field = self.parser.agent_config.get("affirmation_field", "affirmation")
            aff_skip = self.parser.agent_config.get("affirmation_skip_slot", "")
            aff_val = self.slots.get(aff_field, self.unknown_sentinel)
            
            if aff_val != self.unknown_sentinel:
                logging.debug(f"Handling affirmation: {aff_val}")
                if op.affected_slot and op.affected_slot != aff_skip:
                    self._apply_slot_update(op.affected_slot, aff_val, "affirmation_map")
                    self._apply_slot_update(aff_field, self.unknown_sentinel, "affirmation_map")
                    logging.debug(f"Mapped affirmation to slot: {op.affected_slot}")

        # Impasse Detection: Did we actually change anything?
        # Only check against slots that are NOT impasse-count
        state_changed = False
        for k in self.slots:
            if k == 'impasse-count':
                continue
            if self.slots[k] != old_slots.get(k):
                logging.debug(f"State Changed: Slot '{k}' from '{old_slots.get(k)}' to '{self.slots[k]}'")
                state_changed = True
                break
        
        if not state_changed:
            logging.debug("No state change detected, incrementing impasse.")
            AGENT_NO_CHANGE_TOTAL.labels(
                operator_id=op.operator_id,
                utterance_type=op.classifier_tool,
            ).inc()
            try:
                current_impasse = int(self.slots.get('impasse-count', 0))
            except (ValueError, TypeError):
                current_impasse = 0
            
            new_impasse = current_impasse + 1
            self._apply_slot_update('impasse-count', str(new_impasse), "impasse")
            
            # Record Prometheus Metrics
            AGENT_IMPASSE_LEVEL.set(new_impasse)
            AGENT_IMPASSE_TOTAL.labels(utterance_type=op.classifier_tool).inc()
        else:
            self._apply_slot_update('impasse-count', '0', "impasse_reset")
            AGENT_IMPASSE_LEVEL.set(0)

        self.pending_operator = None

    def run_cycle(self) -> Optional[str]:
        cycle_start = time.time()
        if self.pending_operator:
            return None

        if self.slots.get('task-complete') == 'yes':
            return "[HALT]"

        best_op = self._find_best_operator()
        if not best_op:
            logging.debug("No operator proposed.")
            return "[HALT]"

        logging.info(f"Cycler Selected: {best_op.operator_id} (Type: {best_op.operator_type})")
        AGENT_OPERATOR_SELECTED_TOTAL.labels(
            operator_id=best_op.operator_id,
            operator_type=best_op.operator_type,
            phase=best_op.phase,
        ).inc()
        op_start = time.time()

        if best_op.operator_type == "orchestration":
            if best_op.affected_slot:
                self._apply_slot_update(best_op.affected_slot, best_op.expected_value, "orchestration")
            if best_op.utterance_template_id:
                message = self.encoder.generate(best_op.utterance_template_id, self.slots)
                if self.interrupted_operator and self.slots.get("conversation-category", self.unknown_sentinel) == self.unknown_sentinel:
                    resumed_operator = self.interrupted_operator
                    self.pending_operator = resumed_operator
                    self.interrupted_operator = None
                    if resumed_operator.utterance_template_id:
                        resume_prompt = self.encoder.generate(resumed_operator.utterance_template_id, self.slots)
                        message = f"{message}\n{resume_prompt}"
                AGENT_OPERATOR_EXECUTION_SECONDS.labels(
                    operator_id=best_op.operator_id, operator_type=best_op.operator_type
                ).observe(time.time() - op_start)
                AGENT_DECISION_CYCLE_SECONDS.labels(
                    operator_type=best_op.operator_type, phase=best_op.phase
                ).observe(time.time() - cycle_start)
                return message
            AGENT_OPERATOR_EXECUTION_SECONDS.labels(
                operator_id=best_op.operator_id, operator_type=best_op.operator_type
            ).observe(time.time() - op_start)
            AGENT_DECISION_CYCLE_SECONDS.labels(
                operator_type=best_op.operator_type, phase=best_op.phase
            ).observe(time.time() - cycle_start)
            return None 

        elif best_op.operator_type == "action":
            result = self.tool_executor.execute_from_operator_spec(
                operator_spec=best_op, 
                slot_values=self.slots,
                metadata={'parser': self.parser}
            )
            for k, v in result.data.items():
                self._apply_slot_update(k, str(v), f"tool:{best_op.tool_name}")
            AGENT_OPERATOR_EXECUTION_SECONDS.labels(
                operator_id=best_op.operator_id, operator_type=best_op.operator_type
            ).observe(time.time() - op_start)
            AGENT_DECISION_CYCLE_SECONDS.labels(
                operator_type=best_op.operator_type, phase=best_op.phase
            ).observe(time.time() - cycle_start)
            return None

        elif best_op.operator_type == "nlu":
            self.pending_operator = best_op
            AGENT_PENDING_NLU_TOTAL.labels(
                operator_id=best_op.operator_id, utterance_type=best_op.classifier_tool
            ).inc()
            clarification_templates = set(self._agent_config_list("clarification_metric_templates"))
            clarification_prefixes = self._agent_config_list("clarification_metric_prefixes")
            if best_op.utterance_template_id and (
                best_op.utterance_template_id in clarification_templates
                or any(best_op.utterance_template_id.startswith(prefix) for prefix in clarification_prefixes)
            ):
                AGENT_CLARIFICATION_REQUEST_TOTAL.labels(
                    utterance_template_id=best_op.utterance_template_id
                ).inc()
            AGENT_OPERATOR_EXECUTION_SECONDS.labels(
                operator_id=best_op.operator_id, operator_type=best_op.operator_type
            ).observe(time.time() - op_start)
            AGENT_DECISION_CYCLE_SECONDS.labels(
                operator_type=best_op.operator_type, phase=best_op.phase
            ).observe(time.time() - cycle_start)
            if best_op.utterance_template_id:
                msg = self.encoder.generate(best_op.utterance_template_id, self.slots)
                return msg
            return None
            
        return None
