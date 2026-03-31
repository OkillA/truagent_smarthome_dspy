from typing import Dict, Any
from ..base import BaseTool, ToolContext, ToolResult

class RuleEvaluatorTool(BaseTool):
    @property
    def name(self) -> str:
        return "rule_evaluator"

    @property
    def description(self) -> str:
        return "Evaluates tribal knowledge against current slot values."

    def _execute_impl(self, context: ToolContext) -> ToolResult:
        parser = context.metadata.get('parser')
        intent = context.get_input('intent', 'unknown')
        
        # Group triples by subject
        rules = {}
        for t in parser.tribal_knowledge:
            if t.subject not in rules:
                rules[t.subject] = {}
            rules[t.subject][t.predicate] = t.object
            
        recommended_method = "unknown"
        rationale = "unknown"
        
        prefix = "unknown"
        if '-' in intent:
            prefix = intent.split('-')[1] # lighting from configure-lighting

        for r_id, r in rules.items():
            if r.get('task-type') != intent:
                continue
                
            metric_name = r.get('metric', '')
            metric_val = context.get_input(f"{prefix}-params.{metric_name}", "unknown")
            cmpr = r.get('comparator', '==')
            thresh = r.get('threshold', 'unknown')
            
            match_primary = False
            if cmpr == '==' and metric_val == thresh:
                match_primary = True
            elif cmpr == '!=' and metric_val != thresh:
                match_primary = True
                
            match_secondary = True
            sec_metric = r.get('also-require-metric')
            if sec_metric:
                sec_val = context.get_input(f"{prefix}-params.{sec_metric}", "unknown")
                sec_cmpr = r.get('also-require-comparator', '==')
                sec_thresh = r.get('also-require-threshold', 'unknown')
                
                if sec_cmpr == '==' and sec_val != sec_thresh:
                    match_secondary = False
                elif sec_cmpr == '!=' and sec_val == sec_thresh:
                    match_secondary = False
                    
            if match_primary and match_secondary:
                recommended_method = r.get('recommended-method', 'unknown')
                rationale = r.get('rationale', '')
                break

        if recommended_method == "unknown":
            if prefix == "climate":
                recommended_method = "generic-smart-thermostat"
                rationale = "We couldn't find a specialized climate rule for your exact combination, but standard Generic Smart Thermostats are a great reliable fallback!"
            elif prefix == "security":
                type_val = context.get_input("security-params.type", "unknown")
                if type_val == "cameras":
                    recommended_method = "generic-wifi-camera"
                    rationale = "We couldn't find a specialized camera rule for your budget, but a standard generic WiFi camera is a great entry-level choice."
                elif type_val == "locks":
                    recommended_method = "generic-smart-lock"
                    rationale = "We couldn't find a specialized lock rule for your budget, but a standard generic smart lock will work with most doors."
                else:
                    recommended_method = "generic-security-hub"
                    rationale = "We couldn't find a specialized rule for your security setup, but a generic smart home security hub is a great way to start."
            else:
                recommended_method = "universal-smart-bulbs"
                rationale = "We couldn't find a highly specialized rule for your exact combination, but standard Universal Smart Bulbs are a great reliable fallback!"

        return self._create_success(data={
            f"{prefix}-params.recommended-method": recommended_method,
            f"{prefix}-params.rule-rationale": rationale,
            "rules-evaluated": "true"
        })
