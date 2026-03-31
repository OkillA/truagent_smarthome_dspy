import csv
from dataclasses import dataclass
from typing import Dict, List
import os

@dataclass
class Slot:
    slot_name: str
    default_value: str
    level: str
    parent: str
    value_type: str
    accepted_values: str
    description: str

@dataclass
class Operator:
    operator_id: str
    operator_type: str
    phase: str
    priority: int
    conditions: str
    utterance_template_id: str
    classifier_tool: str
    write_method: str
    tool_name: str
    requires_slot: str
    affected_slot: str
    expected_value: str

@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    category: str
    description: str

@dataclass
class DecoderSlot:
    slot_name: str
    parent_slot: str
    utterance_type: str
    value_type: str
    accepted_values: str
    description: str
    instructions: str
    skip_initial: str

@dataclass
class UtteranceTemplate:
    utterance_template_id: str
    template_text: str

class CSVParser:
    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        self.agent_config: Dict[str, str] = {}
        self.slots: List[Slot] = []
        self.operators: List[Operator] = []
        self.tribal_knowledge: List[Triple] = []
        self.decoder_slots: List[DecoderSlot] = []
        self.encoder_templates: List[UtteranceTemplate] = []

    def parse_all(self):
        self._parse_00()
        self._parse_01()
        self._parse_02()
        self._parse_03()
        self._parse_04()
        self._parse_05()

    def _parse_00(self):
        path = os.path.join(self.config_dir, "00_agent_config.csv")
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row.pop(None, None)
                self.agent_config[row['key']] = row['value']

    def _parse_01(self):
        path = os.path.join(self.config_dir, "01_intent_tree.csv")
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row.pop(None, None)
                self.slots.append(Slot(**row))

    def _parse_02(self):
        path = os.path.join(self.config_dir, "02_action_space.csv")
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row.pop(None, None)
                row['priority'] = int(row['priority'])
                self.operators.append(Operator(**row))

    def _parse_03(self):
        path = os.path.join(self.config_dir, "03_tribal_knowledge.csv")
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row.pop(None, None)
                self.tribal_knowledge.append(Triple(**row))

    def _parse_04(self):
        path = os.path.join(self.config_dir, "04_utterance_decoder.csv")
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row.pop(None, None)
                self.decoder_slots.append(DecoderSlot(**row))

    def _parse_05(self):
        path = os.path.join(self.config_dir, "05_utterance_encoder.csv")
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row.pop(None, None)
                self.encoder_templates.append(UtteranceTemplate(**row))
