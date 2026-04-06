import json
import logging
import time
from typing import Any, Dict, Optional
from prometheus_client import Counter, Gauge, Histogram
from .control import ControlState
from .goals import GoalState
from .impasse import ImpasseManager
from .operator_handler import OperatorHandler
from .policy import CognitivePolicy
from .productions import PreferenceResolver, ProductionCompiler, ProductionMatcher
from .runtime_model import RuntimeModel
from .tracing import TraceRecorder
from .working_memory import WorkingMemory

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
AGENT_IMPASSE_PEAK = Gauge(
    "agent_impasse_peak",
    "Peak impasse count observed during the active session.",
)
AGENT_IMPASSE_EVENT_TOTAL = Counter(
    "agent_impasse_event_total",
    "Total number of impasse and recovery events opened by the engine.",
    ["kind", "utterance_type"],
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
AGENT_WORKING_MEMORY_CHURN_TOTAL = Counter(
    "agent_working_memory_churn_total",
    "Total number of working-memory slot mutations by mutation type.",
    ["slot_name", "mutation_type", "source"],
)
AGENT_WORKING_MEMORY_PEAK = Gauge(
    "agent_working_memory_peak",
    "Peak working-memory size observed during the session.",
    ["state"],
)
AGENT_CYCLE_INDEX = Gauge(
    "agent_cycle_index",
    "Current cognitive cycle index for the active session.",
)
AGENT_CYCLES_SINCE_LAST_TOOL_EXECUTION = Gauge(
    "agent_cycles_since_last_tool_execution",
    "Number of cognitive cycles since the last tool execution.",
)
AGENT_CYCLES_SINCE_LAST_PROGRESS = Gauge(
    "agent_cycles_since_last_progress",
    "Number of cognitive cycles since the last meaningful state change.",
)
AGENT_CONSTRAINT_VIOLATION_TOTAL = Counter(
    "agent_constraint_violation_total",
    "Total number of symbolic constraint violations detected by the engine.",
    ["violation_type", "slot_name"],
)
AGENT_CAUSAL_TRACE_TOTAL = Counter(
    "agent_causal_trace_total",
    "Total number of structured causal traces emitted by operator executions.",
    ["trace_type", "operator_id", "status"],
)
AGENT_DECISION_TRACE_TOTAL = Counter(
    "agent_decision_trace_total",
    "Total number of recommendation decisions classified by traceability.",
    ["intent", "status"],
)
AGENT_SUBSTATE_DEPTH = Gauge(
    "agent_substate_depth",
    "Current explicit substate depth created by impasse handling.",
)
AGENT_SELECTION_IMPASSE_TOTAL = Counter(
    "agent_selection_impasse_total",
    "Total number of selection-time impasses by kind.",
    ["kind"],
)

class CognitiveEngine:
    def __init__(self, parser, decoder, encoder, tool_executor):
        self.parser = parser
        self.decoder = decoder
        self.encoder = encoder
        self.tool_executor = tool_executor
        self.unknown_sentinel = parser.unknown_sentinel

        self.memory = WorkingMemory.from_slots(parser.slots, unknown_sentinel=self.unknown_sentinel)
        self.control = ControlState(self.memory, unknown_sentinel=self.unknown_sentinel)
        self.slots: Dict[str, str] = self.memory.slot_view
        self.runtime_model = RuntimeModel.from_parser(parser)

        self.pending_operator = None
        self.interrupted_operator = None
        self.slot_definitions = {slot.slot_name: slot for slot in parser.slots}
        self.phase_order = self.runtime_model.phase_order
        self.peak_known_slots = 0
        self.peak_payload_bytes = 0
        self.cycle_index = 0
        self.last_tool_cycle = 0
        self.last_progress_cycle = 0
        self.peak_impasse_level = 0
        self._state_changed_this_cycle = False
        self._tool_executed_this_cycle = False
        self._queued_message: Optional[str] = None
        self.last_causal_trace: Optional[dict[str, Any]] = None
        self.production_compiler = ProductionCompiler()
        self.operator_productions = self.production_compiler.compile_action_space(parser.operators)
        self.matcher = ProductionMatcher(unknown_sentinel=self.unknown_sentinel)
        self.preference_resolver = PreferenceResolver()
        self.impasse_manager = ImpasseManager()
        self.operator_handler = OperatorHandler(parser=parser, encoder=encoder, tool_executor=tool_executor)
        self.goal_state = GoalState(
            self.memory,
            unknown_sentinel=self.unknown_sentinel,
            phase_goals=self.runtime_model.phase_goals,
        )
        self.policy = CognitivePolicy(
            parser=parser,
            unknown_sentinel=self.unknown_sentinel,
            runtime_model=self.runtime_model,
        )
        self.trace_recorder = TraceRecorder()
        self._refresh_memory_metrics()

    @property
    def pending_operator(self):
        return self.control.pending_operator

    @pending_operator.setter
    def pending_operator(self, operator) -> None:
        self.control.pending_operator = operator
        if hasattr(self, "impasse_manager"):
            self._refresh_memory_metrics()

    @property
    def interrupted_operator(self):
        return self.control.interrupted_operator

    @interrupted_operator.setter
    def interrupted_operator(self, operator) -> None:
        self.control.interrupted_operator = operator
        if hasattr(self, "impasse_manager"):
            self._refresh_memory_metrics()

    def _find_operator_by_id(self, operator_id: str):
        for operator in self.parser.operators:
            if operator.operator_id == operator_id:
                return operator
        return None

    def _apply_recovery_directive(self, directive) -> None:
        for slot_name, value in directive.slot_updates.items():
            self._apply_slot_update(slot_name, value, "recovery")

    def _render_recovery_prompts(self, directive) -> list[str]:
        messages: list[str] = []
        for template_id in directive.prompt_template_ids:
            messages.append(self.encoder.generate(template_id, self.slots))
        return messages

    def _activate_recovery_operator(self, operator_id: str) -> Optional[str]:
        operator = self._find_operator_by_id(operator_id)
        if operator is None:
            return None
        self.pending_operator = operator
        if operator.utterance_template_id:
            AGENT_PENDING_NLU_TOTAL.labels(
                operator_id=operator.operator_id,
                utterance_type=operator.classifier_tool,
            ).inc()
            return self.encoder.generate(operator.utterance_template_id, self.slots)
        return None

    def _agent_config_list(self, key: str, delimiter: str = "|") -> list[str]:
        raw = self.parser.agent_config.get(key, "")
        if not raw:
            return []
        return [item.strip() for item in raw.split(delimiter) if item.strip()]

    def _refresh_memory_metrics(self) -> None:
        if hasattr(self, "goal_state") and hasattr(self, "impasse_manager"):
            self.goal_state.sync(self.pending_operator, self.impasse_manager)
        slot_snapshot = self.memory.snapshot_slots()
        self.slots = self.memory.slot_view
        total_slots = len(slot_snapshot)
        known_slots = sum(1 for value in slot_snapshot.values() if value != self.unknown_sentinel)
        unknown_slots = total_slots - known_slots

        AGENT_WORKING_MEMORY_SLOTS.labels(state="total").set(total_slots)
        AGENT_WORKING_MEMORY_SLOTS.labels(state="known").set(known_slots)
        AGENT_WORKING_MEMORY_SLOTS.labels(state="unknown").set(unknown_slots)
        payload_bytes = len(json.dumps(slot_snapshot, sort_keys=True))
        AGENT_WORKING_MEMORY_BYTES.set(payload_bytes)
        self.peak_known_slots = max(self.peak_known_slots, known_slots)
        self.peak_payload_bytes = max(self.peak_payload_bytes, payload_bytes)
        AGENT_WORKING_MEMORY_PEAK.labels(state="known").set(self.peak_known_slots)
        AGENT_WORKING_MEMORY_PEAK.labels(state="payload_bytes").set(self.peak_payload_bytes)
        current_impasse_level = max(self._current_impasse_level(), self.impasse_manager.current_depth())
        self.peak_impasse_level = max(self.peak_impasse_level, current_impasse_level)
        AGENT_IMPASSE_LEVEL.set(current_impasse_level)
        AGENT_IMPASSE_PEAK.set(self.peak_impasse_level)
        AGENT_SUBSTATE_DEPTH.set(self.impasse_manager.current_depth())

    def _current_impasse_level(self) -> int:
        try:
            return int(self.slots.get("impasse-count", 0))
        except (ValueError, TypeError):
            return 0

    def _record_impasse_event(self, kind: str, utterance_type: str) -> None:
        AGENT_IMPASSE_EVENT_TOTAL.labels(
            kind=kind,
            utterance_type=utterance_type or "unknown",
        ).inc()

    def _allowed_values_for_slot(self, slot_name: str) -> set[str]:
        slot = self.slot_definitions.get(slot_name)
        if not slot or not slot.accepted_values:
            return set()
        return {value.strip() for value in slot.accepted_values.split("|") if value.strip()}

    def _operator_allowed_values(self, operator) -> set[str]:
        if not getattr(operator, "affected_slot", ""):
            return set()
        slot_names = [slot_name.strip() for slot_name in operator.affected_slot.split(",") if slot_name.strip()]
        allowed: set[str] = set()
        for slot_name in slot_names:
            allowed.update(self._allowed_values_for_slot(slot_name))
        return allowed

    def _should_suppress_off_topic_interrupt(self, operator, user_input: str, routing_category: str) -> bool:
        if routing_category != "off_topic":
            return False
        allowed_values = self._operator_allowed_values(operator)
        if not allowed_values:
            return False
        word_count = len([token for token in user_input.strip().split() if token])
        return word_count <= 2

    def _parameter_retry_message(self, operator) -> Optional[str]:
        if not operator or not operator.utterance_template_id:
            return None
        allowed_values = self._operator_allowed_values(operator)
        if not allowed_values:
            return None
        prompt = self.encoder.generate(operator.utterance_template_id, self.slots)
        return f"{prompt}\nPlease answer using one of the listed options."

    def _record_constraint_violations(self, slot_name: str, value: str, source: str) -> None:
        slot = self.slot_definitions.get(slot_name)
        previous_value = self.slots.get(slot_name, self.unknown_sentinel)
        if slot is None:
            AGENT_CONSTRAINT_VIOLATION_TOTAL.labels(
                violation_type="unknown_slot",
                slot_name=slot_name,
            ).inc()
            return

        allowed_values = self._allowed_values_for_slot(slot_name)
        if allowed_values and value not in allowed_values:
            AGENT_CONSTRAINT_VIOLATION_TOTAL.labels(
                violation_type="invalid_value",
                slot_name=slot_name,
            ).inc()

        if (
            slot.value_type == "single"
            and previous_value != self.unknown_sentinel
            and value != self.unknown_sentinel
            and previous_value != value
            and source not in {"impasse_reset"}
        ):
            AGENT_CONSTRAINT_VIOLATION_TOTAL.labels(
                violation_type="single_slot_overwrite",
                slot_name=slot_name,
            ).inc()

        if slot_name == "dialogue-phase":
            old_rank = self.phase_order.get(previous_value)
            new_rank = self.phase_order.get(value)
            if (
                old_rank is not None
                and new_rank is not None
                and new_rank < old_rank
            ):
                AGENT_CONSTRAINT_VIOLATION_TOTAL.labels(
                    violation_type="phase_regression",
                    slot_name=slot_name,
                ).inc()

    def _record_working_memory_churn(self, slot_name: str, previous_value: str, value: str, source: str) -> None:
        if previous_value == value:
            mutation_type = "reassert"
        elif previous_value == self.unknown_sentinel and value != self.unknown_sentinel:
            mutation_type = "populate"
        elif previous_value != self.unknown_sentinel and value == self.unknown_sentinel:
            mutation_type = "clear"
        elif previous_value != self.unknown_sentinel and value != self.unknown_sentinel:
            mutation_type = "overwrite"
        else:
            mutation_type = "unknown"

        AGENT_WORKING_MEMORY_CHURN_TOTAL.labels(
            slot_name=slot_name,
            mutation_type=mutation_type,
            source=source,
        ).inc()

    def _finalize_cycle_metrics(self) -> None:
        if self._state_changed_this_cycle:
            self.last_progress_cycle = self.cycle_index
        if self._tool_executed_this_cycle:
            self.last_tool_cycle = self.cycle_index

        AGENT_CYCLE_INDEX.set(self.cycle_index)
        AGENT_CYCLES_SINCE_LAST_TOOL_EXECUTION.set(self.cycle_index - self.last_tool_cycle)
        AGENT_CYCLES_SINCE_LAST_PROGRESS.set(self.cycle_index - self.last_progress_cycle)

    def _record_causal_trace(self, best_op, result: Any) -> None:
        metadata = getattr(result, "metadata", None) or {}
        if not metadata:
            return

        trace = {
            "cycle_index": self.cycle_index,
            "operator_id": best_op.operator_id,
            "operator_type": best_op.operator_type,
            "tool_name": best_op.tool_name,
            "phase": best_op.phase,
            "metadata": metadata,
        }
        self.last_causal_trace = trace
        self.trace_recorder.record(
            "operator_execution",
            self.cycle_index,
            operator_id=best_op.operator_id,
            operator_type=best_op.operator_type,
            tool_name=best_op.tool_name,
            phase=best_op.phase,
            metadata=metadata,
        )

        trace_status = "traceable" if metadata.get("matched_rule_id") or metadata.get("rationale") else "untraceable"
        AGENT_CAUSAL_TRACE_TOTAL.labels(
            trace_type="operator_execution",
            operator_id=best_op.operator_id,
            status=trace_status,
        ).inc()

        if best_op.tool_name == "rule_evaluator":
            intent = self.slots.get("intent", self.unknown_sentinel)
            AGENT_DECISION_TRACE_TOTAL.labels(intent=intent, status=trace_status).inc()

    def _apply_slot_update(self, slot_name: str, value: str, source: str) -> None:
        previous_value = self.memory.get_slot(slot_name, self.unknown_sentinel)
        self._record_constraint_violations(slot_name, value, source)
        self._record_working_memory_churn(slot_name, previous_value, value, source)
        if (
            previous_value != self.unknown_sentinel
            and value != self.unknown_sentinel
            and previous_value != value
        ):
            AGENT_SLOT_OVERWRITE_TOTAL.labels(slot_name=slot_name, source=source).inc()

        parent_slot = self.slot_definitions.get(slot_name).parent if slot_name in self.slot_definitions else None
        self.memory.set_slot(slot_name, value, provenance=source, parent_slot=parent_slot)
        AGENT_SLOT_UPDATES_TOTAL.labels(source=source, slot_name=slot_name).inc()
        if previous_value != value:
            self._state_changed_this_cycle = True
        self._refresh_memory_metrics()

    def _slots_for_utterance_type(self, utterance_type: str) -> list[str]:
        slots = []
        for slot in self.parser.decoder_slots:
            if slot.utterance_type != utterance_type:
                continue
            full_slot_name = f"{slot.parent_slot}.{slot.slot_name}" if slot.parent_slot else slot.slot_name
            if self.memory.get_slot(full_slot_name, self.unknown_sentinel) == self.unknown_sentinel:
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

    def _opportunistic_domain_extraction(self, user_input: str, extracted: Dict[str, str]) -> None:
        intent = extracted.get("intent", self.memory.get_slot("intent", self.unknown_sentinel))
        if not intent or intent == self.unknown_sentinel:
            return

        intent_domain_eval = self.policy.evaluate_intent_domain(intent)
        self._record_policy_evaluation(
            "intent_domain",
            intent_domain_eval,
            intent=intent,
        )
        utterance_type = self.policy.intent_utterance_type(intent)
        if not utterance_type:
            return

        if utterance_type == "task_related":
            return

        chained = self.decoder.classify(
            user_input=user_input,
            utterance_type=utterance_type,
            slots_to_extract=self._slots_for_utterance_type(utterance_type),
            slot_values=self.slots,
        )
        for key, value in chained.items():
            if value != self.unknown_sentinel:
                self._apply_slot_update(key, value, f"chained:{utterance_type}")

    def _precheck_routing_signal(self, user_input: str) -> tuple[str | None, dict[str, str]]:
        routing_type = self.parser.agent_config.get("chaining_classifier", "routing")
        extracted = self.decoder.classify(
            user_input=user_input,
            utterance_type=routing_type,
            slots_to_extract=self._all_slots_for_utterance_type(routing_type),
            slot_values=self.slots,
        )

        conversation_category = extracted.get("conversation-category", self.unknown_sentinel)
        routing_eval = self.policy.evaluate_routing_interrupt(conversation_category)
        self._record_policy_evaluation(
            "routing_interrupt",
            routing_eval,
            routing_category=conversation_category,
        )
        if routing_eval.decision:
            return conversation_category, extracted

        return None, extracted

    def _record_proposals(self, proposals) -> None:
        for proposal in proposals:
            self.trace_recorder.record(
                "operator_proposed",
                self.cycle_index,
                operator_id=proposal.operator_id,
                supporting_productions=proposal.supporting_productions,
                score=proposal.score,
                preferences=proposal.preference_summary(),
                matched_conditions=[
                    {
                        "slot_name": condition.slot_name,
                        "comparator": condition.comparator,
                        "expected_value": condition.expected_value,
                    }
                    for condition in proposal.matched_conditions
                ],
            )

    def _record_policy_evaluation(self, policy_type: str, evaluation, **context) -> None:
        if not evaluation.matched_rule_ids:
            return
        self.trace_recorder.record(
            "policy_evaluated",
            self.cycle_index,
            policy_type=policy_type,
            decision=evaluation.decision,
            matched_rule_ids=list(evaluation.matched_rule_ids),
            payloads=list(evaluation.payloads),
            **context,
        )

    def _find_best_operator(self):
        proposals = self.matcher.propose(self.operator_productions, self.memory)
        self._record_proposals(proposals)
        AGENT_CANDIDATE_OPERATORS.observe(len(proposals))
        if not proposals:
            return {"status": "none", "operator": None, "proposals": []}
        selection = self.preference_resolver.select(proposals)
        if selection.status != "selected" or selection.selected_proposal is None:
            return {
                "status": selection.status,
                "operator": None,
                "proposals": selection.competing_proposals,
                "reason": selection.reason,
            }
        best_proposal = selection.selected_proposal
        self.trace_recorder.record(
            "operator_selected",
            self.cycle_index,
            operator_id=best_proposal.operator_id,
            supporting_productions=best_proposal.supporting_productions,
            score=best_proposal.score,
            reason=selection.reason,
            preferences=best_proposal.preference_summary(),
        )
        return {
            "status": "selected",
            "operator": best_proposal.operator_spec,
            "proposals": proposals,
            "reason": selection.reason,
        }

    def process_input(self, user_input: str):
        if not self.pending_operator:
            return

        op = self.pending_operator
        
        # Capture pre-processing state to detect No-Change impasse
        old_slots = self.memory.snapshot_slots()

        if op.classifier_tool == "target_input":
            if op.affected_slot:
                self._apply_slot_update(op.affected_slot, user_input, "target_input")
        else:
            if op.classifier_tool != self.parser.agent_config.get("chaining_classifier", "routing"):
                routing_category, routing_extracted = self._precheck_routing_signal(user_input)
                if routing_category:
                    if self._should_suppress_off_topic_interrupt(op, user_input, routing_category):
                        routing_category = None
                if routing_category:
                    AGENT_ROUTING_SIGNAL_TOTAL.labels(conversation_category=routing_category).inc()
                    self._apply_slot_update("conversation-category", routing_category, "routing_precheck")
                    if "affirmation" in routing_extracted and routing_extracted["affirmation"] != self.unknown_sentinel:
                        self._apply_slot_update("affirmation", routing_extracted["affirmation"], "routing_precheck")
                    capture_eval = self.policy.evaluate_interrupt_capture(routing_category)
                    self._record_policy_evaluation(
                        "interrupt_capture",
                        capture_eval,
                        routing_category=routing_category,
                        interrupted_operator_id=op.operator_id,
                    )
                    if capture_eval.decision:
                        self.interrupted_operator = op
                    self.pending_operator = None
                    _, directive = self.impasse_manager.handle_interrupt(
                        routing_category=routing_category,
                        goal=op.operator_id,
                        context=self.memory.snapshot_slots(),
                        cycle_index=self.cycle_index,
                        has_interrupted_operator=self.interrupted_operator is not None,
                    )
                    self._record_impasse_event("interrupt", routing_category)
                    self._apply_recovery_directive(directive)
                    self._refresh_memory_metrics()
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

            self._opportunistic_domain_extraction(user_input, extracted)

            chaining_eval = self.policy.evaluate_chaining(self.memory, op, extracted)
            self._record_policy_evaluation(
                "chaining",
                chaining_eval,
                operator_id=op.operator_id,
                classifier_tool=op.classifier_tool,
            )
            if chaining_eval.decision:
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
                self._opportunistic_domain_extraction(user_input, chained)

            affirmation_eval = self.policy.evaluate_affirmation_mapping(self.memory, op)
            self._record_policy_evaluation(
                "affirmation_mapping",
                affirmation_eval,
                operator_id=op.operator_id,
                affected_slot=op.affected_slot,
            )
            affirmation_updates = self.policy.affirmation_updates(self.memory, op)
            for slot_name, value, source in affirmation_updates:
                self._apply_slot_update(slot_name, value, source)

        # Impasse Detection: Did we actually change anything?
        # Only check against slots that are NOT impasse-count
        state_changed = False
        current_slots = self.memory.snapshot_slots()
        for k in current_slots:
            if k == 'impasse-count':
                continue
            if current_slots[k] != old_slots.get(k):
                logging.debug(f"State Changed: Slot '{k}' from '{old_slots.get(k)}' to '{current_slots[k]}'")
                state_changed = True
                break

        ambiguous_eval = self.policy.evaluate_ambiguous_confirmation(self.memory, op, state_changed)
        self._record_policy_evaluation(
            "ambiguous_confirmation",
            ambiguous_eval,
            operator_id=op.operator_id,
            state_changed=state_changed,
        )
        ambiguous_confirmation = ambiguous_eval.decision

        if ambiguous_confirmation:
            logging.debug("Ambiguous confirmation detected; re-asking without impasse escalation.")
            self.pending_operator = op
            _, directive = self.impasse_manager.handle_ambiguous_confirmation(
                goal=op.operator_id,
                context=current_slots,
                cycle_index=self.cycle_index,
            )
            self._record_impasse_event("ambiguous_confirmation", op.classifier_tool)
            self._apply_recovery_directive(directive)
            self._refresh_memory_metrics()
            if op.utterance_template_id:
                message = self.encoder.generate(op.utterance_template_id, self.slots)
                self._queued_message = f"{message}\nPlease answer yes or no."
            return

        if not state_changed:
            parameter_retry = self._parameter_retry_message(op)
            if parameter_retry:
                logging.debug("No state change on constrained parameter; re-asking current prompt.")
                self.pending_operator = op
                self._refresh_memory_metrics()
                self._queued_message = parameter_retry
                return
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
            _, directive = self.impasse_manager.handle_interpretation_failure(
                goal=op.operator_id,
                context=current_slots,
                cycle_index=self.cycle_index,
            )
            self._record_impasse_event("no_change", op.classifier_tool)
            self._apply_recovery_directive(directive)
            
            # Record Prometheus Metrics
            AGENT_IMPASSE_TOTAL.labels(utterance_type=op.classifier_tool).inc()
        else:
            self._apply_slot_update('impasse-count', '0', "impasse_reset")
            AGENT_IMPASSE_LEVEL.set(0)
            self.impasse_manager.resolve_latest()
            self._refresh_memory_metrics()

        self.pending_operator = None

    def _seed_followup_after_interrupt_response(self, operator_id: str) -> list[str]:
        reseed_eval = self.policy.evaluate_interrupt_reseed(
            self.memory,
            operator_id,
            has_interrupted_operator=self.interrupted_operator is not None,
        )
        self._record_policy_evaluation(
            "interrupt_reseed",
            reseed_eval,
            operator_id=operator_id,
            has_interrupted_operator=self.interrupted_operator is not None,
        )
        routing_category = self.policy.interrupt_reseed_category(
            self.memory,
            operator_id,
            has_interrupted_operator=self.interrupted_operator is not None,
        )
        if routing_category:
            _, directive = self.impasse_manager.handle_interrupt(
                routing_category=routing_category,
                goal=operator_id,
                context=self.memory.snapshot_slots(),
                cycle_index=self.cycle_index,
                has_interrupted_operator=False,
            )
            self._record_impasse_event("interrupt", routing_category)
            self._apply_recovery_directive(directive)
            if directive.next_operator_id:
                self.pending_operator = self._find_operator_by_id(directive.next_operator_id)
            return self._render_recovery_prompts(directive)
        return []

    def _interrupt_resume_prompt(self, operator) -> tuple[bool, Optional[str]]:
        resume_eval = self.policy.evaluate_interrupt_resume(
            self.memory,
            has_interrupted_operator=self.interrupted_operator is not None,
        )
        self._record_policy_evaluation(
            "interrupt_resume",
            resume_eval,
            operator_id=operator.operator_id,
            has_interrupted_operator=self.interrupted_operator is not None,
        )
        if not resume_eval.decision or not self.interrupted_operator:
            return False, None

        resumed_operator = self.interrupted_operator
        resume_prompt = None
        if resumed_operator.utterance_template_id:
            resume_prompt = self.encoder.generate(resumed_operator.utterance_template_id, self.slots)
        return True, resume_prompt

    def _mark_nlu_pending(self, operator) -> None:
        AGENT_PENDING_NLU_TOTAL.labels(
            operator_id=operator.operator_id, utterance_type=operator.classifier_tool
        ).inc()
        clarification_templates = set(self._agent_config_list("clarification_metric_templates"))
        clarification_prefixes = self._agent_config_list("clarification_metric_prefixes")
        if operator.utterance_template_id and (
            operator.utterance_template_id in clarification_templates
            or any(operator.utterance_template_id.startswith(prefix) for prefix in clarification_prefixes)
        ):
            AGENT_CLARIFICATION_REQUEST_TOTAL.labels(
                utterance_template_id=operator.utterance_template_id
            ).inc()
    def run_cycle(self) -> Optional[str]:
        cycle_start = time.time()
        if self._queued_message is not None:
            message = self._queued_message
            self._queued_message = None
            return message
        if self.pending_operator:
            return None

        if self.memory.get_slot('task-complete', self.unknown_sentinel) == 'yes':
            return "[HALT]"

        selection = self._find_best_operator()
        best_op = selection["operator"]
        if selection["status"] == "tie":
            AGENT_SELECTION_IMPASSE_TOTAL.labels(kind="tie").inc()
            self.trace_recorder.record(
                "selection_impasse",
                self.cycle_index,
                kind="tie",
                competing_operator_ids=[proposal.operator_id for proposal in selection["proposals"]],
                reason=selection.get("reason", ""),
            )
            _, directive = self.impasse_manager.handle_tie(
                goal=self.memory.get_slot("dialogue-phase", self.unknown_sentinel) or "unknown",
                context=self.memory.snapshot_slots(),
                cycle_index=self.cycle_index,
            )
            self._record_impasse_event("selection_tie", "selection")
            self._apply_recovery_directive(directive)
            self._refresh_memory_metrics()
            if directive.next_operator_id:
                message = self._activate_recovery_operator(directive.next_operator_id)
                if message is not None:
                    return message
            return "[HALT]"

        if selection["status"] == "none" or not best_op:
            logging.debug("No operator proposed.")
            AGENT_SELECTION_IMPASSE_TOTAL.labels(kind="none").inc()
            self.trace_recorder.record(
                "selection_impasse",
                self.cycle_index,
                kind="none",
                competing_operator_ids=[],
                reason=selection.get("reason", ""),
            )
            _, directive = self.impasse_manager.handle_no_operator(
                goal=self.memory.get_slot("dialogue-phase", self.unknown_sentinel) or "unknown",
                context=self.memory.snapshot_slots(),
                cycle_index=self.cycle_index,
            )
            self._record_impasse_event("no_operator", "selection")
            self._apply_recovery_directive(directive)
            self._refresh_memory_metrics()
            if directive.next_operator_id:
                message = self._activate_recovery_operator(directive.next_operator_id)
                if message is not None:
                    extra_prompts = self._render_recovery_prompts(directive)
                    if extra_prompts:
                        return "\n".join([message, *extra_prompts])
                    return message
            return "[HALT]"

        self.cycle_index += 1
        self._state_changed_this_cycle = False
        self._tool_executed_this_cycle = False

        logging.info(f"Cycler Selected: {best_op.operator_id} (Type: {best_op.operator_type})")
        AGENT_OPERATOR_SELECTED_TOTAL.labels(
            operator_id=best_op.operator_id,
            operator_type=best_op.operator_type,
            phase=best_op.phase,
        ).inc()
        op_start = time.time()
        execution = self.operator_handler.execute(
            operator_spec=best_op,
            slot_values=self.slots,
            interrupted_operator=self.interrupted_operator,
            metadata={"parser": self.parser},
            on_seed_followup=self._seed_followup_after_interrupt_response,
            on_nlu_pending=self._mark_nlu_pending,
        )

        for slot_name, value, source in execution.slot_updates:
            self._apply_slot_update(slot_name, value, source)

        if execution.causal_result is not None:
            self._tool_executed_this_cycle = execution.tool_executed
            self._record_causal_trace(best_op, execution.causal_result)

        if execution.pending_operator is not None:
            self.pending_operator = execution.pending_operator

        if best_op.operator_type == "orchestration":
            should_resume, resume_prompt = self._interrupt_resume_prompt(best_op)
            if should_resume and self.interrupted_operator is not None:
                resumed_operator = self.interrupted_operator
                self.pending_operator = resumed_operator
                self.interrupted_operator = None
                execution.resumed_operator = resumed_operator
                execution.message = (
                    f"{execution.message}\n{resume_prompt}" if execution.message and resume_prompt else resume_prompt or execution.message
                )

        AGENT_OPERATOR_EXECUTION_SECONDS.labels(
            operator_id=best_op.operator_id, operator_type=best_op.operator_type
        ).observe(time.time() - op_start)
        AGENT_DECISION_CYCLE_SECONDS.labels(
            operator_type=best_op.operator_type, phase=best_op.phase
        ).observe(time.time() - cycle_start)
        self._finalize_cycle_metrics()
        return execution.message
