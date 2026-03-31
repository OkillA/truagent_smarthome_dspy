import os
import json
import logging
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

class GenericClassifier:
    def __init__(self, parser, models_module):
        self.parser = parser
        self.models_module = models_module
        
        self.instructions = {}
        for slot in parser.decoder_slots:
            if slot.instructions and slot.utterance_type not in self.instructions:
                self.instructions[slot.utterance_type] = slot.instructions

        self.api_key = os.getenv("NVIDIA_API_KEY", "")
        if not self.api_key:
            logging.warning("NVIDIA_API_KEY is not set.")

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key
        )
        self.model = self.parser.agent_config.get("llm_model", "nvidia/glm-4-9b-chat")

    def classify(self, user_input: str, utterance_type: str, slots_to_extract: list, slot_values: dict = None) -> dict:
        if utterance_type not in self.models_module.MODELS:
            raise ValueError(f"No generated model for: {utterance_type}")

        ModelClass = self.models_module.MODELS[utterance_type]
        system_instruction = self.instructions.get(utterance_type, "Extract the properties.")
        agent_context = self.parser.agent_config.get('llm_prompt_context', '')

        schema = ModelClass.model_json_schema()
        
        prompt = f"""
{agent_context}

{system_instruction}

User input: "{user_input}"

Respond ONLY with valid JSON exactly matching the properties in this schema. Do not include markdown codeblocks. Just the raw JSON object. Use "unknown" if a piece of information is missing.
Schema:
{json.dumps(schema, indent=2)}
"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = completion.choices[0].message.content
            if response_text is None:
                logging.error(f"Response text is None! Full completion: {completion}")
                response_text = "{}"
            
            # sometimes models wrap json in markdown
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json\n", "").replace("\n```", "")
                
            print(f"\n[DEBUG] Raw model output:\n{response_text}\n")
            
            parsed_data = json.loads(response_text)
            
            validated = ModelClass(**parsed_data)
            dict_out = validated.model_dump()
            
            final_result = {}
            for slot in self.parser.decoder_slots:
                if slot.utterance_type == utterance_type:
                    pydantic_field = slot.slot_name.split('.')[-1].replace('-', '_')
                    if pydantic_field in dict_out:
                        val = dict_out[pydantic_field]
                        if val != 'unknown':
                            full_slot_name = f"{slot.parent_slot}.{slot.slot_name}" if slot.parent_slot else slot.slot_name
                            final_result[full_slot_name] = str(val)
                            
            return final_result

        except Exception as e:
            logging.error(f"Classification failed: {e}")
            return {}
