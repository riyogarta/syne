# Security

Syne is a self-hosted AI assistant that runs an LLM with tool access on
your infrastructure, optionally exposed via internet-facing channels
(Telegram). This document describes the threat model, current mitigations,
and configuration choices that materially affect the attack surface.

If you find a security issue not covered here, please open a private
report via GitHub (Security tab → Report a vulnerability) instead of
filing a public issue.

---

## Threat model

### In scope

- **Prompt injection from external content** — web pages fetched via
  `web_fetch`, file contents read into context, memories retrieved during
  recall. Any of these can attempt to coerce the LLM into actions it
  shouldn't take.
- **Multi-user channels** — Telegram bot can receive messages from
  `family` and `public` (per policy). Owner-only tools must remain
  inaccessible to lower tiers regardless of conversational coercion.
- **Self-modification path** — `update_ability(action='create')` lets the
  LLM write Python files into `syne/abilities/custom/` and load them.
  This is intentional for owner-driven extension, but is a
  prompt-injection-to-RCE vector if the flag is on for public-facing
  deployments.

### Out of scope

- **Database / disk-level compromise.** If the attacker has root or
  database access, all bets are off. Encrypt your disk, manage secrets,
  use OS-level isolation.
- **OAuth provider attacks.** If Google/Anthropic/OpenAI is compromised,
  that's their problem. We pass tokens straight through.
- **Network MITM.** Use HTTPS for all upstream calls (default).
- **Trusted owner.** The owner is assumed trusted. Owner-level access
  *is* full code execution by design — that's the assistant's purpose.

---

## Permission model

Linux-style 3-digit octal per tool / ability:
`0o{owner}{family}{public}`. Each digit is `r=4 w=2 x=1`.

| Tier | Who | Examples |
|---|---|---|
| **owner** | First user who messages Syne; verified by platform ID | Full access |
| **family** | Trusted users — household, close friends | Most work tools, no config changes |
| **public** | Anyone else who manages to message the bot | Read-only safe tools; memory access only via Rule 765 exception |
| **blocked** | Denied users | Nothing |

Tools register their permission at code level. `filter_tools_for_group()`
strips owner-only tools from the schema sent to the LLM in group chats,
so the LLM physically cannot see them and cannot be coerced into using them.

`exec` is `0o700` (owner-only) and never exposed to family or public
under any circumstances. Same for `db_query` (read-only, owner-only),
`file_write`, `update_*`, `manage_*`.

### Rule 760 / 765 — memory privacy

All memories are private by default (Rule 760). Public users get zero
results from `memory_search` and zero file retrievals from
`memory_get_file` / `memory_analyze_file` **unless** the memory's category
is explicitly listed in `memory.public_categories` (Rule 765).

This is enforced server-side after the search, not via tool permissions —
the LLM can call the tool but gets filtered results. The category gate
is the actual access boundary.

---

## Self-modification — OFF by default

`abilities.self_modification_enabled` controls whether the LLM is allowed
to **create new abilities** via `update_ability(action='create', ...)`.

**Default: `false`.**

### Why this matters

A self-modification path means an attacker who achieves prompt injection
(e.g., a malicious web page fetched via `web_fetch`, or a poisoned memory
written by a previous compromised session) can ask the LLM to "create an
ability that exec's this code" and gain arbitrary code execution.

With the flag off, that path is dead at the handler level. The LLM gets
a clear refusal message and the owner stays in control of which custom
abilities exist.

### What this flag does NOT do

- **It does not disable existing custom abilities.** Files already in
  `syne/abilities/custom/` and registered in the DB continue to run.
  To deactivate one, use:
  ```
  update_ability(action='disable', name='X')
  ```
  Or remove the file and restart.
- **It does not sandbox running code.** A custom ability that was
  created when the flag was on still has full Python access. Treat
  every custom ability as fully trusted code by the owner.

### Defense in depth

Sub-agents are already blocked from `update_ability` entirely via
`SUBAGENT_BLOCKED_TOOLS` in `security.py`, regardless of this flag.
The handler-level check is a second layer.

### Enabling self-modification

If you want the LLM to extend itself, enable explicitly:
```
update_config(key='abilities.self_modification_enabled', value='true')
```

Recommended only when:
- The bot is in a controlled environment (DM only with owner, or
  trusted family group)
- You actively review and audit every newly-created ability
- You understand that any prompt injection vector becomes an RCE vector

---

## Prompt injection — what we do and don't mitigate

| Vector | Mitigation |
|---|---|
| External web page via `web_fetch` | None at content level. Result is plain text in context. Don't `web_fetch` URLs from untrusted message senders in owner sessions. |
| Memory poisoning (attacker writes a memory, later retrieved) | Rule 760/765 limits who can write memory. Family/owner only by default. Don't enable auto-capture on bot exposed to untrusted public. |
| Filename / metadata injection | Filenames go through `_safe_filename()` (regex strip). MIME types from user-supplied uploads are accepted as-is — don't trust them as security boundary. |
| Tool description injection | Tool/ability descriptions are static in code. They're not user-modifiable. |

The LLM provider's own safety layer (Claude, Gemini, etc.) is the first
line of defense against many of these. Syne does not add a separate
content scanner.

---

## Hardening checklist for production

If you're running Syne for more than your own personal use:

- [ ] **`telegram.dm_policy = "approval"`** (default) — new users need
      owner approval before they can message
- [ ] **`telegram.group_policy = "allowlist"`** (default) — bot ignores
      groups it wasn't explicitly added to
- [ ] **`telegram.require_mention = true`** (default) — in groups, bot
      only responds when explicitly mentioned
- [ ] **`abilities.self_modification_enabled = false`** (default) — keep
      off unless you actively want the LLM to extend itself
- [ ] **`memory.auto_capture = false`** (default) — only store what user
      explicitly asks to remember; prevents memory poisoning
- [ ] **`memory.public_categories = []`** (default) — opens nothing to
      public users unless you explicitly list categories
- [ ] **Audit custom abilities periodically** —
      `ls syne/abilities/custom/` and review each file. Anything you
      didn't write yourself is a trust delegation.
- [ ] **Disable `web_fetch` from public-tier contexts** if your threat
      model includes public users sending URLs. Currently `web_fetch` is
      `0o555` (everyone can use it) — consider downgrading to `0o550`
      in your fork if that fits your model.
- [ ] **Run behind a process supervisor** (systemd, default) so crashes
      don't leave the bot in a half-state.
- [ ] **Keep `~/.log-syne/` log directory readable only by the syne
      user** — logs may contain user content.

---

## Reporting

Found something not in this document, or that contradicts it? Open a
GitHub security report (Security → Report a vulnerability). For
non-security correctness bugs, use regular Issues.

---

## Out-of-tree (roadmap)

Not yet implemented but tracked:

- **Sandbox for self-created abilities** — subprocess isolation + import
  allowlist, so even with the flag enabled, a malicious ability cannot
  escape into Syne's process. This is engineering work, separate from
  the off-by-default flag above.
- **Content scanning for `web_fetch` results** — flag suspicious patterns
  before they enter the LLM context.
- **Per-ability maturity labels** (`stable` / `beta` / `experimental`)
  so fork users understand which abilities are stress-tested vs
  community-maintained.
