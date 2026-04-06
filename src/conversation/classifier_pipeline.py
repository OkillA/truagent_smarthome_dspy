from __future__ import annotations

import json
import re

from pydantic import ValidationError


class SchemaScopeBuilder:
    @staticmethod
    def slot_to_field_name(slot_name: str) -> str:
        return slot_name.split(".")[-1].replace("-", "_")

    def build_schema_for_slots(self, model_class, slots_to_extract: list[str]) -> str:
        full_schema = model_class.model_json_schema()
        properties = full_schema.get("properties", {})

        if not slots_to_extract:
            return json.dumps(full_schema)

        target_fields: list[str] = []
        for slot_name in slots_to_extract:
            field_name = self.slot_to_field_name(slot_name)
            if field_name in properties and field_name not in target_fields:
                target_fields.append(field_name)

        if not target_fields:
            return json.dumps(full_schema)

        scoped_schema = {
            "type": "object",
            "properties": {field_name: properties[field_name] for field_name in target_fields},
        }
        return json.dumps(scoped_schema)

    def scope_hint(self, slots_to_extract: list[str]) -> str:
        if not slots_to_extract:
            return ""

        field_names: list[str] = []
        for slot_name in slots_to_extract:
            field_name = self.slot_to_field_name(slot_name)
            if field_name not in field_names:
                field_names.append(field_name)
        if not field_names:
            return ""
        return (
            "Only extract values for these fields if they are explicitly stated in the user input: "
            + ", ".join(field_names)
            + "."
        )


class ClassifierOutputParser:
    def extract_json_object(self, raw_output: str) -> dict:
        json_str = raw_output.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            match = re.search(r"\{.*\}", json_str, re.DOTALL)
            if match:
                json_str = match.group(0)
        return json.loads(json_str)


class ClassifierValidator:
    def allowed_values_for_utterance_type(self, parser, utterance_type: str) -> dict[str, set[str]]:
        allowed_values: dict[str, set[str]] = {}
        for slot in parser.decoder_slots:
            if slot.utterance_type != utterance_type:
                continue
            field_name = SchemaScopeBuilder.slot_to_field_name(slot.slot_name)
            values = {value.strip() for value in slot.accepted_values.split("|") if value.strip()}
            if values:
                allowed_values[field_name] = values
        return allowed_values

    def validate(self, model_class, dict_out: dict):
        return model_class(**dict_out)

    def translate_validation_error(self, parser, utterance_type: str, error: ValidationError, dict_out: dict) -> list[str]:
        allowed_values = self.allowed_values_for_utterance_type(parser, utterance_type)
        reasons: list[str] = []
        for item in error.errors():
            field_name = item.get("loc", ["unknown"])[0]
            bad_value = dict_out.get(field_name) if isinstance(dict_out, dict) else None
            if field_name in allowed_values and bad_value not in allowed_values[field_name]:
                reasons.append("invalid_enum")
            else:
                reasons.append("validation_error")
        return reasons or ["validation_error"]
