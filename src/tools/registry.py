import importlib
import inspect
import logging
import pkgutil
from typing import Dict

from .base import BaseTool

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def discover_and_register(self, plugins_package: str) -> int:
        count = 0
        pkg = importlib.import_module(plugins_package)
        for _, module_name, _ in pkgutil.iter_modules(pkg.__path__):
            full_module_name = f"{plugins_package}.{module_name}"
            module = importlib.import_module(full_module_name)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BaseTool) and obj is not BaseTool:
                    if not inspect.isabstract(obj):
                        try:
                            tool_instance = obj()
                            self.register(tool_instance)
                            count += 1
                        except Exception as e:
                            logging.error(f"Failed to register tool {obj.__name__}: {e}")
        return count

    def register(self, tool: BaseTool):
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name} already registered")
        self._tools[tool.name] = tool
        logging.debug(f"Registered tool: {tool.name}")

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Tool {name} not found in registry")
        return self._tools[name]
