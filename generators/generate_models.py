import sys
import os
from csv_parser import CSVParser

def generate_conversation_models(parser: CSVParser, output_path: str):
    code = [
        "from typing import Optional, Literal",
        "from pydantic import BaseModel, Field",
        "import typing",
        "",
        "MODELS = {}"
    ]

    from collections import defaultdict
    slots_by_type = defaultdict(list)
    for slot in parser.decoder_slots:
        slots_by_type[slot.utterance_type].append(slot)

    for ut_type, slots in slots_by_type.items():
        class_name = "".join([word.capitalize() for word in ut_type.split('_')]) + "Model"
        
        code.append(f"\nclass {class_name}(BaseModel):")
        for slot in slots:
            field_name = slot.slot_name.replace('-', '_').replace('.', '_')
            
            if slot.accepted_values:
                # Need to escape strings
                literals = []
                for val in slot.accepted_values.split('|'):
                    literals.append(f"'{val}'")
                lit_str = ", ".join(literals)
                if "'unknown'" not in lit_str:
                    lit_str += ", 'unknown'"
                # Use Literal type
                code.append(f'    {field_name}: Optional[Literal[{lit_str}]] = Field(default="unknown", description="{slot.description}")')
            else:
                code.append(f'    {field_name}: Optional[str] = Field(default="unknown", description="{slot.description}")')

        code.append(f"MODELS['{ut_type}'] = {class_name}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(code))
    
    print(f"Generated {output_path}")
