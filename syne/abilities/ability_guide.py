"""Ability Guide builder (CORE).

Assembles the ``# Abilities`` section for the system prompt.
Iterates ALL registered abilities (bundled + user-created) and calls
their ``get_guide(enabled, config)`` method.

User-created abilities define their own ``get_guide()`` in their
ability class file — registered via DB, not via files in this repo.
"""

import json
import logging

logger = logging.getLogger("syne.abilities.guide")

_CREATION_GUIDE = """\
## Creating a New Ability

When the owner asks for a capability you don't have, create an ability.

### Steps
1. Write the file to `syne/abilities/custom/<name>.py` using `file_write`
2. Register: `update_ability(action='create', name='<name>', description='...')`
3. The ability is immediately available — no restart needed

### Template (mandatory structure)
```python
from syne.abilities.base import Ability

class MyAbility(Ability):
    name = "my_ability"
    description = "What this ability does"
    version = "1.0"

    def get_guide(self, enabled: bool, config: dict) -> str:
        # Return status + usage for system prompt
        if enabled:
            return "- Status: **ready**\\n- Use: `my_ability(param='...')`"
        return "- Status: **not ready**\\n- Setup: configure via update_ability"

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "my_ability",
                "description": "What this ability does",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param": {"type": "string", "description": "..."},
                    },
                    "required": ["param"],
                },
            },
        }

    async def execute(self, params: dict, context: dict) -> dict:
        # context["config"] has ability config from DB
        return {"success": True, "result": "done"}

    def get_required_config(self) -> list[str]:
        return []  # or ["api_key"] if needed
```

### External Dependencies
If the ability needs an external binary, package, or service to work,
override `ensure_dependencies()`. It is called **automatically** when the
user enables the ability — no manual install step needed.

```python
async def ensure_dependencies(self) -> tuple[bool, str]:
    import shutil
    if shutil.which("mytool"):
        return True, ""
    # Try to install...
    ok = await self._install_mytool()
    if ok:
        return True, "mytool installed"
    return False, "mytool not found. Install: apt install mytool"
```

- Return `(True, "")` — deps satisfied, ability will be enabled
- Return `(True, "message")` — deps installed, message shown to user
- Return `(False, "error")` — install failed, ability stays disabled
- Also called on bot restart for already-enabled abilities (re-checks deps)
- Do NOT assume deps are pre-installed — always check and auto-install if possible

### Rules
- File goes in `syne/abilities/custom/` — the ONLY writable path under `syne/`
- Class MUST extend `Ability` and implement: `execute`, `get_schema`, `get_guide`
- Validator checks syntax, structure, and schema before registration
- Config (API keys etc.) stored in DB via `update_ability(action='config')`
- Never hardcode secrets — read from `context["config"]`
"""


async def build() -> str:
    """Build the full ``# Abilities`` section for the system prompt.

    Flow:
    1. Load all ability records from DB (name, enabled, config, module_path)
    2. Bundled abilities — instantiate from loader, call get_guide()
    3. Dynamic abilities — load from module_path, call get_guide()

    Returns:
        Multi-line string ready to append to the system prompt.
    """
    try:
        from ..db.connection import get_connection
        from .loader import get_bundled_ability_classes, load_dynamic_ability_safe

        async with get_connection() as conn:
            rows = await conn.fetch(
                "SELECT name, enabled, config, module_path FROM abilities ORDER BY name"
            )
        db_info = {r["name"]: r for r in rows}

        parts = ["# Abilities"]

        # ── Bundled abilities ──
        bundled_names = set()
        for cls in get_bundled_ability_classes():
            try:
                instance = cls()
                bundled_names.add(instance.name)
                row = db_info.get(instance.name)
                enabled = row["enabled"] if row else False
                config = json.loads(row["config"]) if row and row["config"] else {}
                parts.append(f"## {instance.name}")
                parts.append(instance.get_guide(enabled, config))
                parts.append("")
            except Exception as e:
                logger.error(f"Failed to load bundled ability {cls.__name__}: {e}")

        # ── Dynamic abilities (user-created / installed) ──
        broken = []
        for row in rows:
            if row["name"] in bundled_names:
                continue
            try:
                instance, load_err = load_dynamic_ability_safe(row["module_path"])
                if instance and hasattr(instance, "get_guide"):
                    enabled = row["enabled"]
                    config = json.loads(row["config"]) if row["config"] else {}
                    parts.append(f"## {instance.name}")
                    parts.append(instance.get_guide(enabled, config))
                    parts.append("")
                elif instance is None:
                    broken.append((row["name"], row["module_path"],
                                   load_err or "Unknown error"))
            except Exception as e:
                broken.append((row["name"], row["module_path"], str(e)))
                logger.debug(f"Skipping guide for '{row['name']}': {e}")

        # ── Broken abilities — surface to bot for self-healing ──
        if broken:
            parts.append("## Broken Abilities (need repair)")
            parts.append("The following user-created abilities failed to load.")
            parts.append("**Fix them proactively** using the Self-Healing steps.")
            parts.append("")
            for name, mod_path, err in broken:
                parts.append(f"- **{name}** (`{mod_path}`): {err}")
            parts.append("")

        # ── Creation guide ──
        parts.append(_CREATION_GUIDE)

        return "\n".join(parts)
    except Exception as e:
        return f"# Abilities\n(Error loading: {e})\n"
