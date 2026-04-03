from typing import Dict, Any
from prometheus_client import Counter
from ..base import BaseTool, ToolContext, ToolResult

RULE_FALLBACK_TOTAL = Counter(
    "rule_fallback_total",
    "Total number of times recommendation fallback rules are used.",
    ["intent", "recommended_method"],
)
RULE_MATCH_TOTAL = Counter(
    "rule_match_total",
    "Total number of successfully matched recommendation rules.",
    ["intent", "recommended_method"],
)
RULE_NO_MATCH_TOTAL = Counter(
    "rule_no_match_total",
    "Total number of recommendation evaluations that found no matching or fallback rule.",
    ["intent"],
)
RULE_TRACEABILITY_TOTAL = Counter(
    "rule_traceability_total",
    "Total number of recommendation decisions with or without a usable rationale trace.",
    ["intent", "status"],
)

class RuleEvaluatorTool(BaseTool):
    @property
    def name(self) -> str:
        return "rule_evaluator"

    @property
    def description(self) -> str:
        return "Evaluates tribal knowledge against current slot values."

    def _group_rules(self, parser) -> dict[str, dict[str, str]]:
        rules = {}
        for triple in parser.tribal_knowledge:
            if triple.subject not in rules:
                rules[triple.subject] = {}
            rules[triple.subject][triple.predicate] = triple.object
        return rules

    def _resolve_slot_prefix(self, intent: str, rule: dict[str, str]) -> str:
        slot_prefix = rule.get("slot-prefix", "").strip()
        if slot_prefix:
            return slot_prefix

        if "-" in intent:
            return f"{intent.split('-')[1]}-params"

        return "unknown"

    def _metric_condition_matches(
        self,
        context: ToolContext,
        slot_prefix: str,
        metric_name: str,
        comparator: str,
        threshold: str,
        unknown_sentinel: str,
    ) -> bool:
        metric_val = context.get_input(f"{slot_prefix}.{metric_name}", unknown_sentinel)
        if comparator == "==" and metric_val == threshold:
            return True
        if comparator == "!=" and metric_val != threshold:
            return True
        return False

    def _matches_rule(
        self,
        context: ToolContext,
        slot_prefix: str,
        rule: dict[str, str],
        unknown_sentinel: str,
    ) -> bool:
        metric_name = rule.get("metric", "")
        if metric_name:
            comparator = rule.get("comparator", "==")
            threshold = rule.get("threshold", unknown_sentinel)
            if not self._metric_condition_matches(
                context, slot_prefix, metric_name, comparator, threshold, unknown_sentinel
            ):
                return False

        secondary_metric = rule.get("also-require-metric")
        if not secondary_metric:
            return bool(metric_name) or not metric_name

        secondary_comparator = rule.get("also-require-comparator", "==")
        secondary_threshold = rule.get("also-require-threshold", unknown_sentinel)
        if not self._metric_condition_matches(
            context,
            slot_prefix,
            secondary_metric,
            secondary_comparator,
            secondary_threshold,
            unknown_sentinel,
        ):
            return False

        return True

    def _fallback_for_task(
        self,
        context: ToolContext,
        rules: dict[str, dict[str, str]],
        intent: str,
        unknown_sentinel: str,
    ) -> dict[str, str] | None:
        for rule in rules.values():
            if rule.get("task-type") != intent:
                continue
            if rule.get("fallback", "").lower() == "true":
                slot_prefix = self._resolve_slot_prefix(intent, rule)
                if self._matches_rule(
                    context=context,
                    slot_prefix=slot_prefix,
                    rule=rule,
                    unknown_sentinel=unknown_sentinel,
                ):
                    return rule
        return None

    def _execute_impl(self, context: ToolContext) -> ToolResult:
        parser = context.metadata.get('parser')
        unknown_sentinel = parser.unknown_sentinel
        intent = context.get_input('intent', unknown_sentinel)

        rules = self._group_rules(parser)

        recommended_method = unknown_sentinel
        rationale = unknown_sentinel

        for r_id, r in rules.items():
            if r.get('task-type') != intent:
                continue
            if r.get("fallback", "").lower() == "true":
                continue

            slot_prefix = self._resolve_slot_prefix(intent, r)
            if self._matches_rule(context, slot_prefix, r, unknown_sentinel):
                recommended_method = r.get('recommended-method', unknown_sentinel)
                rationale = r.get('rationale', '')
                target_slot_prefix = slot_prefix
                RULE_MATCH_TOTAL.labels(intent=intent, recommended_method=recommended_method).inc()
                break

        if recommended_method == unknown_sentinel:
            fallback_rule = self._fallback_for_task(context, rules, intent, unknown_sentinel)
            if fallback_rule:
                recommended_method = fallback_rule.get("recommended-method", unknown_sentinel)
                rationale = fallback_rule.get("rationale", unknown_sentinel)
                target_slot_prefix = self._resolve_slot_prefix(intent, fallback_rule)
                RULE_FALLBACK_TOTAL.labels(intent=intent, recommended_method=recommended_method).inc()
            else:
                target_slot_prefix = self._resolve_slot_prefix(intent, {})
                recommended_method = unknown_sentinel
                rationale = unknown_sentinel
                RULE_NO_MATCH_TOTAL.labels(intent=intent).inc()
        else:
            target_slot_prefix = locals().get("target_slot_prefix", self._resolve_slot_prefix(intent, {}))

        traceability_status = "traceable" if rationale and rationale != unknown_sentinel else "untraceable"
        RULE_TRACEABILITY_TOTAL.labels(intent=intent, status=traceability_status).inc()

        return self._create_success(data={
            f"{target_slot_prefix}.recommended-method": recommended_method,
            f"{target_slot_prefix}.rule-rationale": rationale,
            "rules-evaluated": "true"
        })
