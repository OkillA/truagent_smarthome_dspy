import os
import json
import logging
import dspy
import re
from typing import Dict, List, Optional, Type
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langfuse import observe
from prometheus_client import Counter, Histogram
import time

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
    schema = dspy.InputField(desc="The required JSON schema.")
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
        
        # Bootstrap with examples that include 'state'
        self._add_routing_examples()

    def _add_routing_examples(self):
        """Add few-shot examples showing how to use the 'state' to avoid drift."""
        routing_schema = "{\"properties\": {\"conversation_category\": {\"enum\": [\"task_related\", \"greeting\", \"help_request\", \"off_topic\", \"unknown\"], \"type\": \"string\"}, \"affirmation\": {\"enum\": [\"confirmed\", \"declined\", \"unknown\"], \"type\": \"string\"}}, \"type\": \"object\"}"
        
        examples = [
            dspy.Example(
                context="Smart home assistant",
                state="{}",
                instruction="Classify the message.",
                schema=routing_schema,
                user_input="I want to set up some lights",
                output='{"conversation_category": "task_related", "affirmation": "unknown"}'
            ).with_inputs("context", "state", "instruction", "schema", "user_input"),
            dspy.Example(
                context="Smart home assistant",
                state="{'intent': 'configure-lighting'}",
                instruction="Classify the message.",
                schema=routing_schema,
                user_input="Yes, that is correct",
                output='{"conversation_category": "unknown", "affirmation": "confirmed"}'
            ).with_inputs("context", "state", "instruction", "schema", "user_input"),
             dspy.Example(
                context="Smart home assistant",
                state="{'intent': 'configure-lighting', 'room': 'living-room'}",
                instruction="Classify the message.",
                schema=routing_schema,
                user_input="I'd like it to be reactive",
                output='{"conversation_category": "task_related", "affirmation": "unknown"}'
            ).with_inputs("context", "state", "instruction", "schema", "user_input")
        ]
        
        self.predictor.demos = examples

    @observe()
    def classify(self, user_input: str, utterance_type: str, slots_to_extract: list, slot_values: dict = None) -> dict:
        if utterance_type not in self.models_module.MODELS:
            raise ValueError(f"No generated model for: {utterance_type}")

        ModelClass = self.models_module.MODELS[utterance_type]
        system_instruction = self.instructions.get(utterance_type, "Extract the requested properties accurately.")
        agent_context = self.parser.agent_config.get('llm_prompt_context', 'You are a smart home assistant.')
        schema_json = json.dumps(ModelClass.model_json_schema())
        
        # Format the state to be readable for a small model
        current_state = {k: v for k, v in (slot_values or {}).items() if v != 'unknown'}
        state_str = str(current_state)

        start_time = time.time()
        try:
            # Predict with State Injection
            with dspy.settings.context(lm=self.lm):
                result = self.predictor(
                    context=agent_context,
                    state=state_str,
                    instruction=system_instruction,
                    schema=schema_json,
                    user_input=user_input
                )
            
            latency = time.time() - start_time
            LLM_LATENCY_SECONDS.labels(model=self.model_name, utterance_type=utterance_type).observe(latency)
            LLM_CALLS_TOTAL.labels(model=self.model_name, utterance_type=utterance_type, status='success').inc()
            
            raw_output = result.output
            
            # Clean JSON
            json_str = raw_output.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            else:
                match = re.search(r'\{.*\}', json_str, re.DOTALL)
                if match:
                    json_str = match.group(0)

            dict_out = json.loads(json_str)
            
            # Validate
            validated = ModelClass(**dict_out)
            final_dict = validated.model_dump()
            
            print(f"\n[DSPy DEBUG] Extracted for {utterance_type}:\n{final_dict}\n")
            
            final_result = {}
            for slot in self.parser.decoder_slots:
                if slot.utterance_type == utterance_type:
                    pydantic_field = slot.slot_name.split('.')[-1].replace('-', '_')
                    if pydantic_field in final_dict:
                        val = final_dict[pydantic_field]
                        if val and val != 'unknown':
                            full_slot_name = f"{slot.parent_slot}.{slot.slot_name}" if slot.parent_slot else slot.slot_name
                            final_result[full_slot_name] = str(val)
                            
            return final_result

        except Exception as e:
            LLM_CALLS_TOTAL.labels(model=self.model_name, utterance_type=utterance_type, status='error').inc()
            logging.error(f"DSPy Extraction failed: {e}")
            return {}
