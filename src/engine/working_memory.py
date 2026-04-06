from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from .types import WME


class WorkingMemory:
    def __init__(self, unknown_sentinel: str):
        self.unknown_sentinel = unknown_sentinel
        self._wmes: "OrderedDict[str, WME]" = OrderedDict()
        self.slot_view: dict[str, str] = {}

    @classmethod
    def from_slots(cls, slots: Iterable, unknown_sentinel: str) -> "WorkingMemory":
        slot_list = list(slots)
        memory = cls(unknown_sentinel=unknown_sentinel)
        for slot in slot_list:
            memory.set_slot(
                slot_name=slot.slot_name,
                value=slot.default_value,
                provenance="seed",
                parent_slot=slot.parent or None,
            )
        return memory

    def _identifier_for_slot(self, slot_name: str) -> str:
        return f"wme:{slot_name}"

    def _parent_identifier(self, slot_name: str, parent_slot: str | None) -> str | None:
        if parent_slot:
            return self._identifier_for_slot(parent_slot)
        if "." in slot_name:
            return self._identifier_for_slot(slot_name.rsplit(".", 1)[0])
        return "wme:state"

    def set_slot(
        self,
        slot_name: str,
        value: str,
        provenance: str,
        parent_slot: str | None = None,
    ) -> WME:
        identifier = self._identifier_for_slot(slot_name)
        attribute = slot_name.split(".")[-1]
        wme = WME(
            identifier=identifier,
            attribute=attribute,
            value=value,
            slot_name=slot_name,
            parent_identifier=self._parent_identifier(slot_name, parent_slot),
            provenance=provenance,
        )
        self._wmes[identifier] = wme
        self.slot_view[slot_name] = value
        return wme

    def get_slot(self, slot_name: str, default: str | None = None) -> str | None:
        return self.slot_view.get(slot_name, default)

    def get_wme(self, slot_name: str) -> WME | None:
        return self._wmes.get(self._identifier_for_slot(slot_name))

    def snapshot_slots(self) -> dict[str, str]:
        return dict(self.slot_view)

    def all_wmes(self) -> list[WME]:
        return list(self._wmes.values())
