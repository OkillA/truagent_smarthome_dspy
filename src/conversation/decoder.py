import os
import json
import logging
import dspy
import tiktoken
from pydantic import ValidationError
from dotenv import load_dotenv
from langfuse import observe
from prometheus_client import Counter, Histogram, Counter as PromCounter
import time
from .classifier_pipeline import (
    ClassifierOutputParser,
    ClassifierValidator,
    SchemaScopeBuilder,
)

load_dotenv()

# Prometheus Metrics
LLM_CALLS_TOTAL = Counter(
    'llm_calls_total', 
    'Total number of LLM calls', 
    ['model', 'utterance_type', 'status']
)
LLM_LATENCY_SECONDS = Histogram(
    'llm_latency_seconds', 
    'Latency of LLM calls in seconds', 
    ['model', 'utterance_type']
)
LLM_TOKEN_USAGE_TOTAL = Counter(
    "llm_token_usage_total",
    "Total number of tokens consumed by the LLM.",
    ["model", "utterance_type", "token_type"],
)
LLM_ESTIMATED_COST_USD = Counter(
    "llm_estimated_cost_usd",
    "Estimated cost of LLM calls in USD.",
    ["model"],
)
LLM_EXTRACTED_FIELDS_TOTAL = PromCounter(
    "llm_extracted_fields_total",
    "Total number of non-unknown fields extracted by the classifier.",
    ["utterance_type", "field_name"],
)
LLM_SCOPED_REQUESTS_TOTAL = PromCounter(
    "llm_scoped_requests_total",
    "Total number of classifier calls by extraction scope size.",
    ["utterance_type", "scope"],
)
LLM_HANDSHAKE_FAILURE_TOTAL = PromCounter(
    "llm_handshake_failure_total",
    "Total number of extracted values that do not map cleanly into the allowed schema or requested slots.",
    ["utterance_type", "reason"],
)
LLM_CONTEXT_SLOT_COUNT = Histogram(
    "llm_context_slot_count",
    "Number of currently populated working-memory slots included in the classifier state.",
    ["utterance_type"],
)
LLM_CONTEXT_PAYLOAD_BYTES = Histogram(
    "llm_context_payload_bytes",
    "Approximate serialized size of the classifier state payload in bytes.",
    ["utterance_type"],
)
LLM_TOKENS_PER_SECOND = Histogram(
    "llm_tokens_per_second",
    "Approximate token throughput computed from prompt+completion tokens over end-to-end classifier latency.",
    ["model", "utterance_type", "token_scope"],
)

# Define the DSPy Signature with 'state' for short-term memory
class ExtractJSON(dspy.Signature):
    """
    You are a precise data extraction assistant for a Smart Home AI.
    Given a user message, the current conversation state, and a target JSON schema, 
    you must extract the fields accurately while maintaining consistency with the current state.
    Respond ONLY with the raw JSON object.
    """
    context = dspy.InputField(desc="The system personality.")
    state = dspy.InputField(desc="Current filled slots/working memory.")
    instruction = dspy.InputField(desc="Specific extraction instructions.")
    target_schema = dspy.InputField(desc="The required JSON schema.")
    user_input = dspy.InputField(desc="The user message to process.")
    
    output = dspy.OutputField(desc="A valid JSON object matching the schema.")

class GenericClassifier:
    def __init__(self, parser, models_module):
        self.parser = parser
        self.models_module = models_module
        
        self.instructions = {}
        for slot in parser.decoder_slots:
            if slot.instructions and slot.utterance_type not in self.instructions:
                self.instructions[slot.utterance_type] = slot.instructions

        self.api_key = os.getenv("NVIDIA_API_KEY", "")
        self.model_name = self.parser.agent_config.get("llm_model", "microsoft/phi-3.5-mini-instruct")
        
        # Configure DSPy LM
        self.lm = dspy.LM(
            model=f"openai/{self.model_name}",
            api_base="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key,
            temperature=0.0,
            max_tokens=2048,
            cache=False
        )
        dspy.settings.configure(lm=self.lm)
        
        # The Predictor
        self.predictor = dspy.Predict(ExtractJSON)
        self.examples_by_utterance_type = self._load_examples()
        self.debug_extractions = (
            self.parser.agent_config.get("debug_extractions", "false").strip().lower() == "true"
        )
        self.unknown_sentinel = self.parser.unknown_sentinel
        self.prompt_cost_per_million = float(
            self.parser.agent_config.get("llm_cost_per_million_prompt_tokens", "0.05")
        )
        self.completion_cost_per_million = float(
            self.parser.agent_config.get("llm_cost_per_million_completion_tokens", "0.15")
        )
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_estimated_cost_usd = 0.0
        self.schema_builder = SchemaScopeBuilder()
        self.output_parser = ClassifierOutputParser()
        self.validator = ClassifierValidator()
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _load_examples(self) -> dict[str, list[dspy.Example]]:
        examples_by_type: dict[str, list[dspy.Example]] = {}

        for example in self.parser.dspy_examples:
            examples_by_type.setdefault(example.utterance_type, []).append(
                dspy.Example(
                    context=example.context,
                    state=example.state,
                    instruction=example.instruction,
                    target_schema="{}",
                    user_input=example.user_input,
                    output=example.output,
                ).with_inputs("context", "state", "instruction", "target_schema", "user_input")
            )

        return examples_by_type

    def _slot_to_field_name(self, slot_name: str) -> str:
        return self.schema_builder.slot_to_field_name(slot_name)

    def _build_schema_for_slots(self, model_class, slots_to_extract: list[str]) -> str:
        return self.schema_builder.build_schema_for_slots(model_class, slots_to_extract)

    def _allowed_values_for_utterance_type(self, utterance_type: str) -> dict[str, set[str]]:
        return self.validator.allowed_values_for_utterance_type(self.parser, utterance_type)

    def _extract_json_object(self, raw_output: str) -> dict:
        return self.output_parser.extract_json_object(raw_output)

    def _estimate_token_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.prompt_cost_per_million / 1_000_000
            + completion_tokens * self.completion_cost_per_million / 1_000_000
        )

    def _record_token_usage(
        self,
        utterance_type: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_seconds: float | None = None,
    ) -> None:
        LLM_TOKEN_USAGE_TOTAL.labels(
            model=self.model_name,
            utterance_type=utterance_type,
            token_type="prompt",
        ).inc(prompt_tokens)
        LLM_TOKEN_USAGE_TOTAL.labels(
            model=self.model_name,
            utterance_type=utterance_type,
            token_type="completion",
        ).inc(completion_tokens)
        estimated_cost = self._estimate_token_cost(prompt_tokens, completion_tokens)
        LLM_ESTIMATED_COST_USD.labels(model=self.model_name).inc(estimated_cost)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_estimated_cost_usd += estimated_cost
        if latency_seconds and latency_seconds > 0:
            total_tokens = prompt_tokens + completion_tokens
            LLM_TOKENS_PER_SECOND.labels(
                model=self.model_name,
                utterance_type=utterance_type,
                token_scope="total",
            ).observe(total_tokens / latency_seconds)
            if completion_tokens > 0:
                LLM_TOKENS_PER_SECOND.labels(
                    model=self.model_name,
                    utterance_type=utterance_type,
                    token_scope="completion",
                ).observe(completion_tokens / latency_seconds)

    def usage_snapshot(self) -> dict[str, float]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "estimated_cost_usd": self.total_estimated_cost_usd,
        }

    @observe(name="decoder_classify", as_type="generation", capture_input=False, capture_output=False)
    def classify(self, user_input: str, utterance_type: str, slots_to_extract: list, slot_values: dict = None) -> dict:
        if utterance_type not in self.models_module.MODELS:
            raise ValueError(f"No generated model for: {utterance_type}")

        ModelClass = self.models_module.MODELS[utterance_type]
        system_instruction = self.instructions.get(utterance_type, "Extract the requested properties accurately.")
        agent_context = self.parser.agent_config.get('llm_prompt_context', 'You are a smart home assistant.')
        schema_json = self._build_schema_for_slots(ModelClass, slots_to_extract)

        scope_hint = ""
        if slots_to_extract:
            scope_hint = self.schema_builder.scope_hint(slots_to_extract)

        strictness_hint = (
            " Extract only information that is directly and explicitly stated in the user's message."
            " Do not infer, guess, or fill plausible defaults for missing fields."
            f" If a field is not clearly stated, return '{self.unknown_sentinel}'."
        )
        scoped_instruction = f"{system_instruction} {scope_hint}{strictness_hint}".strip()
        
        # Format the state to be readable for a small model
        current_state = {k: v for k, v in (slot_values or {}).items() if v != self.unknown_sentinel}
        state_str = json.dumps(current_state, sort_keys=True)
        LLM_CONTEXT_SLOT_COUNT.labels(utterance_type=utterance_type).observe(len(current_state))
        LLM_CONTEXT_PAYLOAD_BYTES.labels(utterance_type=utterance_type).observe(len(state_str.encode("utf-8")))
        self.predictor.demos = self.examples_by_utterance_type.get(utterance_type, [])
        LLM_SCOPED_REQUESTS_TOTAL.labels(
            utterance_type=utterance_type,
            scope="scoped" if slots_to_extract else "full",
        ).inc()

        start_time = time.time()
        try:
            # Predict with State Injection
            with dspy.settings.context(lm=self.lm):
                result = self.predictor(
                    context=agent_context,
                    state=state_str,
                    instruction=scoped_instruction,
                    target_schema=schema_json,
                    user_input=user_input
                )
            
            latency = time.time() - start_time
            LLM_LATENCY_SECONDS.labels(model=self.model_name, utterance_type=utterance_type).observe(latency)
            LLM_CALLS_TOTAL.labels(model=self.model_name, utterance_type=utterance_type, status='success').inc()
            prompt_str = f"{agent_context} {state_str} {scoped_instruction} {schema_json} {user_input}"
            prompt_tokens = len(self.tokenizer.encode(prompt_str))
            completion_tokens = len(self.tokenizer.encode(result.output))
            self._record_token_usage(
                utterance_type,
                prompt_tokens,
                completion_tokens,
                latency_seconds=latency,
            )
            
            dict_out = self._extract_json_object(result.output)
            
            # Validate
            validated = self.validator.validate(ModelClass, dict_out)
            final_dict = validated.model_dump()
            
            if self.debug_extractions:
                logging.info("DSPy extracted for %s: %s", utterance_type, final_dict)
            
            final_result = {}
            for slot in self.parser.decoder_slots:
                if slot.utterance_type == utterance_type:
                    pydantic_field = slot.slot_name.split('.')[-1].replace('-', '_')
                    full_slot_name = f"{slot.parent_slot}.{slot.slot_name}" if slot.parent_slot else slot.slot_name
                    if slots_to_extract and full_slot_name not in slots_to_extract:
                        continue
                    if pydantic_field in final_dict:
                        val = final_dict[pydantic_field]
                        if val and val != self.unknown_sentinel:
                            final_result[full_slot_name] = str(val)
                            LLM_EXTRACTED_FIELDS_TOTAL.labels(
                                utterance_type=utterance_type,
                                field_name=full_slot_name,
                            ).inc()

            if slots_to_extract:
                unexpected_slots = [slot for slot in final_result if slot not in slots_to_extract]
                if unexpected_slots:
                    for _slot in unexpected_slots:
                        LLM_HANDSHAKE_FAILURE_TOTAL.labels(
                            utterance_type=utterance_type,
                            reason="unexpected_slot",
                        ).inc()

            return final_result

        except ValidationError as e:
            for reason in self.validator.translate_validation_error(self.parser, utterance_type, e, dict_out):
                LLM_HANDSHAKE_FAILURE_TOTAL.labels(
                    utterance_type=utterance_type,
                    reason=reason,
                ).inc()

            LLM_CALLS_TOTAL.labels(model=self.model_name, utterance_type=utterance_type, status='error').inc()
            logging.error(f"DSPy Validation failed: {e}")
            return {}
        except Exception as e:
            LLM_CALLS_TOTAL.labels(model=self.model_name, utterance_type=utterance_type, status='error').inc()
            LLM_HANDSHAKE_FAILURE_TOTAL.labels(
                utterance_type=utterance_type,
                reason="exception",
            ).inc()
            logging.error(f"DSPy Extraction failed: {e}")
            return {}
