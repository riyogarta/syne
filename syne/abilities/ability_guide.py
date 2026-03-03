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
    permission = 0o700  # see Permission section below

    def get_guide(self, enabled: bool, config: dict) -> str:
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

### Permission System

Every ability declares a `permission` class attribute — a 3-digit octal number
that controls who can use it. Format: `0oOFP` (Owner / Family / Public).

**If omitted, defaults to `0o700` (owner-only) — always safe.**

#### The 3 digits
| Position | Who | Example |
|----------|-----|---------|
| 1st digit | **owner** — the system administrator | `7xx` |
| 2nd digit | **family** — trusted users (household, close friends) | `x7x` |
| 3rd digit | **public** — anyone else who can message the bot | `xx7` |

#### What each bit means
| Bit | Value | Meaning |
|-----|-------|---------|
| r | 4 | **read** — query, fetch, view data |
| w | 2 | **write** — store, update, modify data |
| x | 1 | **execute** — perform action, send, generate |

Combine bits by adding: `r+w+x = 7`, `r+x = 5`, `r = 4`, none = `0`.

#### Common permission patterns
| Permission | Meaning | Use when |
|------------|---------|----------|
| `0o700` | owner-only | Sensitive ops: system config, admin, private data |
| `0o770` | owner + family | Family features: messaging, scheduling, personal tools |
| `0o750` | owner full + family read/exec | Family can use but not modify |
| `0o550` | owner + family read/exec | Read-only tools for trusted users |
| `0o555` | everyone read/exec | Safe public tools: search, lookup, info |
| `0o777` | everyone full access | Safe stateless tools with NO side effects (see warning below) |

**WARNING about `0o777`**: This grants access to ALL users including anonymous/public
users — anyone who can message the bot. Only use for abilities that are:
- Completely stateless (no data modification)
- Free or negligible cost per call
- Cannot be abused for spam or resource exhaustion
- Do not expose private information
Example: image generation (creative, no side effects, output goes only to requester)

#### How to choose the right permission
Ask these questions **in order** — stop at the first YES:

1. **Does it access private/sensitive data?** (health, finances, personal notes, credentials)
   → `0o700` — owner-only, non-negotiable
2. **Can it modify system state?** (write files, change config, manage users, delete data)
   → `0o700` — owner-only
3. **Does it send messages to external services?** (email, WhatsApp, SMS, webhooks)
   → max `0o770` — never give public send access
4. **Does it cost significant money per call?** (paid APIs, token-heavy operations)
   → `0o700` unless owner explicitly requests otherwise
5. **Is it useful for family members?** (scheduling, personal tools, family-shared features)
   → `0o770` (read+write) or `0o550` (read-only)
6. **Is it safe for literally anyone?** (lookup, search, display public info, creative generation)
   → `0o555` (read/exec) or `0o777` (full access if truly harmless)

#### Security rules — MANDATORY
These rules are non-negotiable. Always follow them, even if the owner asks otherwise.
If the owner insists on a risky permission, **warn them clearly** before applying it.

1. **Default to `0o700`** when unsure — it is always safe to be restrictive first.
   The owner can relax permissions later; damage from overly permissive defaults cannot be undone.
2. **Never give public (xx7) write/exec access** to abilities that modify system state,
   send external messages, or access private data — regardless of what is requested.
3. **Always warn the owner** when setting permission higher than `0o700`:
   - For `0o770`: "This allows family members to use this ability."
   - For `0o555`: "This allows ALL users including public/anonymous to use this ability."
   - For `0o777`: "This allows ALL users full access. Confirm this ability has no side effects and no significant cost."
4. **The `blocked` access level is always denied** regardless of permission value — no exceptions.
5. **Validate permission makes sense** for the ability's behavior:
   - An ability that runs shell commands should NEVER be > `0o700`
   - An ability that reads files from disk should NEVER be > `0o700`
   - An ability that calls paid APIs without rate limiting should NEVER be > `0o770`
6. **When in doubt, ask the owner** — it is better to ask "should public users have access?"
   than to silently grant it.

### External Dependencies
If the ability needs an external binary, package, or service to work,
override `ensure_dependencies()`. It is called **automatically** when the
user enables the ability — no manual install step needed.

```python
async def ensure_dependencies(self) -> tuple[bool, str]:
    import shutil
    if shutil.which("mytool"):
        return True, ""
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
- Always set `permission` explicitly — don't rely on the default
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
