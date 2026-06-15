import os
from typing import Optional
from jinja2 import Environment, FileSystemLoader


class PromptRegistry:
    """Loads and renders Jinja2 prompt templates.

    Templates live in src/llm/templates/ as .jinja2 files.
    They are version-controlled and editable without touching Python code.
    """

    def __init__(self, templates_dir: Optional[str] = None):
        if templates_dir is None:
            templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        self._env = Environment(loader=FileSystemLoader(templates_dir))

    def render(self, template_name: str, **kwargs) -> tuple[str, str]:
        """Render a prompt template.

        Every template renders to two parts separated by a marker:
        ---SYSTEM--- and ---USER---, so the caller gets both.

        Returns:
            (system_prompt, user_prompt): A tuple of the two parts.
        """
        template = self._env.get_template(template_name)
        rendered = template.render(**kwargs)

        # Split on the convention marker
        if "---USER---" in rendered:
            parts = rendered.split("---USER---", 1)
            system_part = parts[0].replace("---SYSTEM---", "").strip()
            user_part = parts[1].strip()
            return system_part, user_part

        # If no markers, treat entire output as user prompt
        return "", rendered.strip()
