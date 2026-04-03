import json
from collections import defaultdict

from generators.csv_parser import CSVParser


def _python_string_literal(value: str) -> str:
    return repr(value)


def _field_description_literal(description: str) -> str:
    return json.dumps(description or "")

def generate_conversation_models(parser: CSVParser, output_path: str):
    unknown_sentinel = parser.unknown_sentinel
    code = [
        "from typing import Optional, Literal",
        "from pydantic import BaseModel, Field",
        "",
        "MODELS = {}"
    ]

    slots_by_type = defaultdict(list)
    for slot in parser.decoder_slots:
        slots_by_type[slot.utterance_type].append(slot)

    for ut_type, slots in slots_by_type.items():
        class_name = "".join([word.capitalize() for word in ut_type.split('_')]) + "Model"
        
        code.append(f"\nclass {class_name}(BaseModel):")
        for slot in slots:
            field_name = slot.slot_name.replace('-', '_').replace('.', '_')
            description_literal = _field_description_literal(slot.description)
            
            if slot.accepted_values:
                literals = [_python_string_literal(val) for val in slot.accepted_values.split('|')]
                lit_str = ", ".join(literals)
                unknown_literal = _python_string_literal(unknown_sentinel)
                if unknown_literal not in lit_str:
                    lit_str += f", {unknown_literal}"
                code.append(
                    f'    {field_name}: Optional[Literal[{lit_str}]] = '
                    f'Field(default={unknown_literal}, description={description_literal})'
                )
            else:
                code.append(
                    f'    {field_name}: Optional[str] = '
                    f'Field(default={_python_string_literal(unknown_sentinel)}, description={description_literal})'
                )

        code.append(f"MODELS['{ut_type}'] = {class_name}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(code))
    
    print(f"Generated {output_path}")
