from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyEvaluation:
    decision: bool
    matched_rule_ids: tuple[str, ...]
    payloads: tuple[dict[str, object], ...]

    @property
    def first_payload(self) -> dict[str, object] | None:
        return self.payloads[0] if self.payloads else None


@dataclass(frozen=True)
class PolicyCondition:
    source: str
    key: str
    comparator: str
    expected: object


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    rule_type: str
    conditions: tuple[PolicyCondition, ...]
    payload: dict[str, object]


class CognitivePolicy:
    def __init__(self, parser, unknown_sentinel: str, runtime_model=None):
        self.parser = parser
        self.unknown_sentinel = unknown_sentinel
        self.runtime_model = runtime_model
        self.rules = self._build_rules()

    def _build_rules(self) -> list[PolicyRule]:
        chaining_classifier = self.parser.agent_config.get("chaining_classifier", "routing")
        trigger_slot = self.parser.agent_config.get("chaining_trigger_slot", "conversation-category")
        trigger_val = self.parser.agent_config.get("chaining_trigger_value", "task_related")
        affirmation_field = self.parser.agent_config.get("affirmation_field", "affirmation")
        affirmation_skip = self.parser.agent_config.get("affirmation_skip_slot", "conversation-category")
        interrupt_categories = [
            item.strip()
            for item in self.parser.agent_config.get("routing_interrupt_categories", "").split("|")
            if item.strip()
        ]

        rules: list[PolicyRule] = [
            PolicyRule(
                rule_id="chain-followup-routing",
                rule_type="chaining",
                conditions=(
                    PolicyCondition("memory", trigger_slot, "==", trigger_val),
                    PolicyCondition("memory", affirmation_field, "==", self.unknown_sentinel),
                    PolicyCondition("operator", "classifier_tool", "==", chaining_classifier),
                ),
                payload={"enabled": True},
            ),
            PolicyRule(
                rule_id="ambiguous-confirmation-routing",
                rule_type="ambiguous_confirmation",
                conditions=(
                    PolicyCondition("runtime", "state_changed", "==", False),
                    PolicyCondition("operator", "classifier_tool", "==", chaining_classifier),
                    PolicyCondition("operator", "affected_slot", "in", ("confirmation-status", "method-confirmation-result")),
                ),
                payload={"enabled": True},
            ),
            PolicyRule(
                rule_id="resume-interrupted-operator",
                rule_type="interrupt_resume",
                conditions=(
                    PolicyCondition("runtime", "has_interrupted_operator", "==", True),
                    PolicyCondition("memory", "conversation-category", "==", self.unknown_sentinel),
                ),
                payload={"resume": True},
            ),
            PolicyRule(
                rule_id="interrupt-reseed-help",
                rule_type="interrupt_reseed",
                conditions=(
                    PolicyCondition("value", "operator_id", "==", "respond-help"),
                    PolicyCondition("memory", "dialogue-phase", "==", "init"),
                    PolicyCondition("runtime", "has_interrupted_operator", "==", False),
                ),
                payload={"routing_category": "help_request"},
            ),
            PolicyRule(
                rule_id="interrupt-reseed-off-topic",
                rule_type="interrupt_reseed",
                conditions=(
                    PolicyCondition("value", "operator_id", "==", "respond-off-topic"),
                    PolicyCondition("memory", "dialogue-phase", "==", "init"),
                    PolicyCondition("runtime", "has_interrupted_operator", "==", False),
                ),
                payload={"routing_category": "off_topic"},
            ),
        ]

        for category in interrupt_categories:
            rules.append(
                PolicyRule(
                    rule_id=f"interrupt-category-{category}",
                    rule_type="interrupt_category",
                    conditions=(PolicyCondition("value", "routing_category", "==", category),),
                    payload={"enabled": True},
                )
            )

        for category in ("off_topic", "help_request"):
            rules.append(
                PolicyRule(
                    rule_id=f"capture-interrupted-{category}",
                    rule_type="interrupt_capture",
                    conditions=(PolicyCondition("value", "routing_category", "==", category),),
                    payload={"capture": True},
                )
            )

        rules.append(
            PolicyRule(
                rule_id="affirmation-map",
                rule_type="affirmation_mapping",
                conditions=(
                    PolicyCondition("memory", affirmation_field, "!=", self.unknown_sentinel),
                    PolicyCondition("operator", "affected_slot", "!=", ""),
                    PolicyCondition("operator", "affected_slot", "!=", affirmation_skip),
                ),
                payload={"affirmation_field": affirmation_field},
            )
        )

        intent_utterance_types = getattr(self.runtime_model, "intent_utterance_types", {}) or {}
        for intent, utterance_type in intent_utterance_types.items():
            rules.append(
                PolicyRule(
                    rule_id=f"intent-domain-{intent}",
                    rule_type="intent_domain",
                    conditions=(PolicyCondition("value", "intent", "==", intent),),
                    payload={"utterance_type": utterance_type},
                )
            )

        return rules

    def _resolve(self, source: str, key: str, memory, operator, values: dict[str, object]) -> object:
        if source == "memory":
            return memory.get_slot(key, self.unknown_sentinel)
        if source == "operator":
            return getattr(operator, key, "")
        if source == "runtime":
            return values.get(key)
        if source == "value":
            return values.get(key)
        return None

    def _matches(self, condition: PolicyCondition, memory, operator, values: dict[str, object]) -> bool:
        actual = self._resolve(condition.source, condition.key, memory, operator, values)
        if condition.comparator == "==":
            return actual == condition.expected
        if condition.comparator == "!=":
            return actual != condition.expected
        if condition.comparator == "in":
            return actual in condition.expected
        return False

    def _matching_rules(self, rule_type: str, memory, operator=None, **values) -> list[PolicyRule]:
        matched: list[PolicyRule] = []
        for rule in self.rules:
            if rule.rule_type != rule_type:
                continue
            if all(self._matches(condition, memory, operator, values) for condition in rule.conditions):
                matched.append(rule)
        return matched

    def _evaluate(self, rule_type: str, memory, operator=None, **values) -> PolicyEvaluation:
        matched = self._matching_rules(rule_type, memory, operator, **values)
        return PolicyEvaluation(
            decision=bool(matched),
            matched_rule_ids=tuple(rule.rule_id for rule in matched),
            payloads=tuple(rule.payload for rule in matched),
        )

    def evaluate_chaining(self, memory, operator, extracted: dict[str, str]) -> PolicyEvaluation:
        return self._evaluate("chaining", memory, operator, extracted=extracted)

    def evaluate_routing_interrupt(self, routing_category: str) -> PolicyEvaluation:
        return self._evaluate("interrupt_category", memory=None, routing_category=routing_category)

    def evaluate_interrupt_capture(self, routing_category: str) -> PolicyEvaluation:
        return self._evaluate("interrupt_capture", memory=None, routing_category=routing_category)

    def evaluate_ambiguous_confirmation(self, memory, operator, state_changed: bool) -> PolicyEvaluation:
        return self._evaluate(
            "ambiguous_confirmation",
            memory,
            operator,
            state_changed=state_changed,
        )

    def affirmation_updates(self, memory, operator) -> list[tuple[str, str, str]]:
        evaluation = self.evaluate_affirmation_mapping(memory, operator)
        if not evaluation.decision:
            return []
        affirmation_field = evaluation.payloads[0]["affirmation_field"]
        affirmation_value = memory.get_slot(affirmation_field, self.unknown_sentinel)
        return [
            (operator.affected_slot, affirmation_value, "affirmation_map"),
            (affirmation_field, self.unknown_sentinel, "affirmation_map"),
        ]

    def evaluate_affirmation_mapping(self, memory, operator) -> PolicyEvaluation:
        return self._evaluate("affirmation_mapping", memory, operator)

    def evaluate_intent_domain(self, intent: str) -> PolicyEvaluation:
        return self._evaluate("intent_domain", memory=None, intent=intent)

    def intent_utterance_type(self, intent: str) -> str | None:
        evaluation = self.evaluate_intent_domain(intent)
        payload = evaluation.first_payload
        if not payload:
            return None
        utterance_type = payload.get("utterance_type")
        return str(utterance_type) if utterance_type else None

    def evaluate_interrupt_reseed(self, memory, operator_id: str, has_interrupted_operator: bool) -> PolicyEvaluation:
        return self._evaluate(
            "interrupt_reseed",
            memory,
            operator=None,
            operator_id=operator_id,
            has_interrupted_operator=has_interrupted_operator,
        )

    def interrupt_reseed_category(self, memory, operator_id: str, has_interrupted_operator: bool) -> str | None:
        evaluation = self.evaluate_interrupt_reseed(memory, operator_id, has_interrupted_operator)
        payload = evaluation.first_payload
        if not payload:
            return None
        routing_category = payload.get("routing_category")
        return str(routing_category) if routing_category else None

    def evaluate_interrupt_resume(self, memory, has_interrupted_operator: bool) -> PolicyEvaluation:
        return self._evaluate(
            "interrupt_resume",
            memory,
            operator=None,
            has_interrupted_operator=has_interrupted_operator,
        )
