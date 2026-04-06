import re
import time
from prometheus_client import Histogram

AGENT_NLG_GENERATION_SECONDS = Histogram(
    "agent_nlg_generation_seconds",
    "Time taken to generate a response from a template.",
    ["template_id"],
)

class TemplateEngine:
    def __init__(self, parser):
        self.parser = parser
        self.templates = {t.utterance_template_id: t.template_text for t in parser.encoder_templates}
        
    def generate(self, template_id: str, slot_values: dict) -> str:
        start_time = time.time()
        template = self.templates.get(template_id)
        if not template:
            return f"[Missing template: {template_id}]"

        placeholders = re.findall(r'\{([^{}]+)\}', template)
        
        result = template
        for ph in placeholders:
            val = slot_values.get(ph, f"[{ph} missing]")
            result = result.replace(f"{{{ph}}}", str(val))

        AGENT_NLG_GENERATION_SECONDS.labels(template_id=template_id).observe(time.time() - start_time)
        return result
