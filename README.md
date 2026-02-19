# Syne 🧠

**AI Agent Framework with Unlimited Memory**

*"I remember, therefore I am"*

Named after [Mnemosyne](https://en.wikipedia.org/wiki/Mnemosyne), the Greek goddess of memory and mother of the Muses.

Syne is a standalone, open-source AI agent framework built in Python. It features **PostgreSQL-native memory** with semantic search, an **ability-based architecture** for extensibility, and **self-evolution** capabilities where the agent can create new abilities for itself.

---

## Why Syne?

Most AI assistants forget everything between sessions. They have no persistent memory, no learning, no growth. Syne is different:

- **Unlimited memory** — Semantic search over millions of memories using pgvector
- **Anti-hallucination** — 3-layer defense ensures only user-confirmed facts are stored
- **Self-evolving** — Syne can create new abilities for itself (with your permission)
- **No config files** — Everything lives in PostgreSQL. Change behavior through conversation, not YAML
- **Near-zero cost** — Chat via Google Gemini OAuth (free). Embedding via Together AI (~$0.008/1M tokens)
- **Interactive CLI** — Code like Claude Code, but with persistent memory and tools

---

## Cost

The table below shows the **minimum cost** setup using free OAuth providers. During `syne init`, you choose your own chat LLM and embedding provider — costs vary depending on your choice.

**Minimum cost setup (Google Gemini + Together AI):**

| Component | Model | Cost | Notes |
|-----------|-------|------|-------|
| Chat LLM | Gemini 2.5 Pro (Google OAuth) | **$0** | Free, rate-limited |
| Embedding | bge-base-en-v1.5 (Together AI) | **~$0.008/1M tokens** | $5 free credit on signup |
| Image Gen | FLUX.1-schnell (Together AI) | **~$0.003/image** | Optional ability |
| PostgreSQL | Self-hosted (Docker) | **$0** | |
| Telegram Bot | Telegram Bot API | **$0** | |
| **Typical monthly** | | **< $1** | |

**Other provider options available during install:**

| Type | Providers |
|------|-----------|
| Chat (OAuth, free) | Google Gemini, ChatGPT, Claude |
| Chat (API key, paid) | OpenAI, Anthropic, Together AI, Groq |
| Embedding (paid) | Together AI, OpenAI |

> Costs depend entirely on which providers you choose. The free OAuth + Together AI combo above is the cheapest path.
>
> **Coming soon:** Ollama support for local embedding — bringing the total cost to **$0**. Trade-off: requires additional CPU and RAM on your server.

---

## Minimum Requirements

| Requirement | Details |
|-------------|---------|
| **CPU** | 1 vCPU minimum (2+ recommended) |
| **OS** | Linux (Ubuntu 22.04+, Debian 12+) |
| **Python** | 3.11+ |
| **RAM** | 1 GB minimum with 1 GB swap (2 GB recommended) |
| **Storage** | 500 MB (excluding Docker images) |
| **Docker** | Required — PostgreSQL 16 + pgvector runs in Docker |
| **Network** | Access to: Together AI (embedding), Google OAuth (chat), Telegram API (bot), Brave Search (optional) |

---

## Quick Start

### Installation

```bash
git clone https://github.com/riyogarta/syne.git
cd syne
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
syne init
```

### What `syne init` Does

`syne init` is fully automated — no manual steps mid-install:

1. **Choose AI provider** — OAuth (free) or API key (paid)
2. **Choose embedding provider**
3. **Enter Telegram bot token** — from @BotFather
4. **Start PostgreSQL** — Docker container with pgvector, auto-install Docker if needed
5. **Initialize database** — Schema, identity, credentials saved to DB
6. **Setup systemd service** — Auto-start on boot, linger enabled

When init finishes, Syne is running.

### Verify Installation

```bash
syne repair     # Diagnose: DB, OAuth, Telegram, abilities
syne status     # Quick status check
```

---

## No Config Files — Just Talk

Most agent frameworks require editing `SOUL.md`, `AGENTS.md`, or `CONFIG.yaml`. Syne has none of that. Everything lives in PostgreSQL, and you change it through conversation:

```
You:  Be more casual and use humor.
Syne: Updated my personality. ✅

You:  Add a rule: never share my location.
Syne: Added as a hard rule. ✅

You:  Switch to GPT-5.2 for chat.
Syne: Model switched. ✅
```

| Traditional File | Syne Equivalent | How to change |
|------------------|-----------------|---------------|
| `SOUL.md` | `soul` table | *"Be more casual and witty"* |
| `AGENTS.md` | `rules` table | *"Add a rule: never share my location"* |
| `IDENTITY.md` | `identity` table | *"Change your name to Atlas"* |
| `CONFIG.yaml` | `config` table | *"Switch to GPT for chat"* |

Fresh install comes with sensible defaults. Override anything through conversation — no SQL, no files.

---

## Memory System

### Three-Layer Anti-Hallucination

Only user-confirmed facts are stored. Never assistant suggestions.

```
User message
    │
    ├─ Layer 1: Quick Filter (no LLM call)
    │   └─ Skip: greetings, short messages, questions-only
    │
    ├─ Layer 2: LLM Evaluation
    │   └─ Is this worth remembering?
    │
    └─ Layer 3: Similarity Dedup
        └─ Does this already exist?
```

### Conflict Resolution

When storing a new memory, similarity to existing memories determines the action:

| Similarity | Action | Example |
|------------|--------|---------|
| < 0.70 | **Insert** new memory | "I have a dog" (no prior pet info) |
| 0.70–0.84 | **Update** existing | "I moved to Bandung" updates "lives in Jakarta" |
| ≥ 0.85 | **Skip** duplicate | "I live in Jakarta" (already stored) |

### Auto Capture vs Manual

| Mode | Trigger | Cost impact |
|------|---------|-------------|
| `auto_capture = false` (default) | Only when user says "remember this" | No extra LLM calls |
| `auto_capture = true` | Every message evaluated automatically | +1 LLM call + 1 embedding per message |

> ⚠️ `auto_capture = true` adds extra LLM + embedding calls per message. On free-tier OAuth this exhausts rate limits faster.

### Managing Your Memories

```
You:  What do you remember about my family?
Syne: [recalls relevant memories via semantic search]

You:  Remember: I'm allergic to shellfish.
Syne: Stored. ✅

You:  Forget that I like sushi.
Syne: Removed from memory. ✅
```

Via CLI: `syne memory stats`, `syne memory search "query"`, `syne memory add "info"`

---

## Ability System

### Core Tools (13 — Always Available)

| Tool | Description |
|------|-------------|
| `exec` | Execute shell commands |
| `memory_search` | Semantic search over memories |
| `memory_store` | Store new memories |
| `spawn_subagent` | Spawn background agents |
| `subagent_status` | Check sub-agent status |
| `update_config` | Change runtime configuration |
| `update_ability` | Enable/disable/create abilities |
| `update_soul` | Modify behavioral rules |
| `manage_group` | Manage group chat settings |
| `manage_user` | Manage user access levels |
| `web_search` | Search the web (Brave Search API) |
| `web_fetch` | Fetch and extract content from URLs |
| `read_source` | Read Syne's own source code (for self-healing) |

### Bundled Abilities

| Ability | Description | Provider |
|---------|-------------|----------|
| `image_gen` | Generate images from text | Together AI (FLUX.1-schnell) |
| `image_analysis` | Analyze and describe images | Google Gemini vision |
| `maps` | Places, directions, geocoding | Google Maps/Places |

### Ability Interface

```python
class Ability:
    name: str
    description: str
    version: str

    async def execute(self, params: dict, context: dict) -> dict: ...
    def get_schema(self) -> dict: ...
```

### Managing Abilities

```
You:  Enable image generation
Syne: Done — image_gen enabled. ✅

You:  What abilities do I have?
Syne: ✅ image_gen, ✅ image_analysis, ❌ maps (disabled)
```

---

## Self-Modification

Syne can create new abilities at runtime — no restart required:

### Flow

```
User: "I wish you could check Bitcoin prices"
    │
    ├─ Syne writes syne/abilities/crypto_price.py
    ├─ Registers via update_ability (source='self_created')
    ├─ Ability is immediately available
    │
    └─ "Created 'crypto_price' ability. Try: what's BTC now?"
```

### Safety Rules

| Rule | Description |
|------|-------------|
| ✅ CAN | Create/edit files in `syne/abilities/` |
| ❌ CANNOT | Modify core code (`syne/` engine, tools, channels, db, llm, security) |
| ❌ CANNOT | Modify `syne/db/schema.sql` |
| 📝 INSTEAD | Core bugs → draft GitHub issue for owner to post |

### ⚠️ Security Warning (exec)

The `exec` tool gives Syne shell access on the host system. This is powerful but dangerous:

- **Owner-only** — Only users with `owner` access level can trigger exec
- **Timeout** — Configurable per-session via `session.max_tool_rounds` (default: 100)
- **Sub-agents** — Inherit exec access but run in isolated sessions
- **Your responsibility** — Review what Syne executes, especially on production systems

---

## Sub-agents

Syne can spawn isolated background agents for parallel tasks:

```
User: "Write full documentation for the project"
    │
    ├─ Syne spawns sub-agent (background)
    ├─ Main session continues chatting
    └─ Sub-agent completes → results delivered back
```

| Setting | Default | Description |
|---------|---------|-------------|
| `subagents.enabled` | `true` | Master ON/OFF switch |
| `subagents.max_concurrent` | `2` | Max simultaneous sub-agents |
| `subagents.timeout_seconds` | `300` | Sub-agent timeout (5 min) |

Sub-agents inherit abilities and memory access but run in isolated sessions. They cannot spawn other sub-agents.

---

## Multi-User Access

Syne supports multiple users with different access levels:

| Level | Permissions |
|-------|------------|
| `owner` | Full access — exec, config, abilities, memory, all tools |
| `admin` | Most tools except system-level config |
| `family` | Memory access, conversation, limited tools |
| `friend` | Conversation, basic tools |
| `public` | Conversation only |

The first user to message Syne automatically becomes `owner`.

Manage via conversation: *"Make @alice an admin"*, *"Remove @bob's access"*

---

## Configuration Reference

All configuration lives in the `config` table. Change via conversation or SQL.

### Provider Settings

| Key | Default | Description |
|-----|---------|-------------|
| `provider.primary` | `google` | LLM provider |
| `provider.chat_model` | `gemini-2.5-pro` | Chat model |
| `provider.embedding_model` | `BAAI/bge-base-en-v1.5` | Embedding model (Together AI) |
| `provider.embedding_dimensions` | `768` | Vector dimensions |

### Memory Settings

| Key | Default | Description |
|-----|---------|-------------|
| `memory.auto_capture` | `false` | Auto-evaluate messages for storage |
| `memory.recall_limit` | `10` | Max memories per query |

### Session Settings

| Key | Default | Description |
|-----|---------|-------------|
| `session.compaction_threshold` | `80000` | Tokens before compaction |
| `session.max_messages` | `100` | Messages before compaction |
| `session.max_tool_rounds` | `100` | Max tool call rounds per turn |
| `session.thinking_budget` | `null` | Thinking: `0`=off, `1024`=low, `4096`=medium, `8192`=high, `24576`=max |
| `session.reasoning_visible` | `false` | Show thinking in responses |

### Sub-agent Settings

| Key | Default | Description |
|-----|---------|-------------|
| `subagents.enabled` | `true` | Enable sub-agents |
| `subagents.max_concurrent` | `2` | Max concurrent sub-agents |
| `subagents.timeout_seconds` | `300` | Sub-agent timeout |

---

## CLI Commands

```bash
# Setup & Running
syne init                  # Interactive setup (fully automated)
syne start                 # Start Telegram agent
syne start --debug         # Start with debug logging
syne cli                   # Interactive terminal chat
syne cli --debug           # CLI with debug logging
syne status                # Show status
syne repair                # Diagnose and repair
syne restart               # Restart agent
syne stop                  # Stop agent

# Database
syne db init               # Initialize schema
syne db reset              # Reset database (destructive!)

# Identity
syne identity              # View identity
syne identity name "Syne"  # Set identity value
syne prompt                # Show system prompt

# Memory
syne memory stats          # Memory statistics
syne memory search "query" # Semantic search
syne memory add "info"     # Manually add memory
```

### Interactive CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show commands |
| `/status` | Agent status (model, memories, tools) |
| `/model` | Show/switch model |
| `/clear` | Clear conversation |
| `/compact` | Compact conversation |
| `/think [level]` | Set thinking budget |
| `/exit` | Exit CLI |

---

## Telegram Commands

| Command | Description | Access |
|---------|-------------|--------|
| `/start` | Welcome message | All |
| `/help` | Available commands | All |
| `/version` | Agent version | All |
| `/status` | Agent status | All |
| `/memory` | Memory statistics | All |
| `/identity` | Agent identity | All |
| `/compact` | Compact conversation | Owner |
| `/think [level]` | Set thinking (off/low/medium/high/max) | Owner |
| `/reasoning [on/off]` | Toggle reasoning visibility | Owner |
| `/autocapture [on/off]` | Toggle auto memory capture | Owner |
| `/model` | Show/switch model | Owner |
| `/embedding` | Show/switch embedding model | Owner |
| `/forget` | Clear conversation | Owner |
| `/restart` | Restart agent | Owner |

---

## Architecture

```
+------------------------------------------------------------+
|                       SYNE AGENT                           |
|                                                            |
|  +------------------------------------------------------+  |
|  |                 CORE (Protected)                     |  |
|  |                                                      |  |
|  |  [Chat]  [Memory]  [Compaction]  [Channels]  [Sub]   |  |
|  |  (LLM)   (pgvec)    (context)   (TG + CLI)  agent   |  |
|  |                                                      |  |
|  |  Core Tools (13):                                    |  |
|  |  exec · memory · web · config · source · sub-agents  |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |              ABILITIES (Pluggable)                   |  |
|  |                                                      |  |
|  |  [image_gen]  [image_analysis]  [maps]  [custom...]  |  |
|  |                                                      |  |
|  |  Self-Created: Syne adds new abilities at runtime    |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |              PostgreSQL + pgvector                   |  |
|  |  12 tables: identity · soul · rules · users ·       |  |
|  |  memory · sessions · messages · config · abilities · |  |
|  |  groups · subagent_runs · capabilities               |  |
|  +------------------------------------------------------+  |
+------------------------------------------------------------+
```

---

## Database Schema

| Table | Purpose |
|-------|---------|
| `identity` | Agent name, motto, personality |
| `soul` | Behavioral rules by category |
| `rules` | Hard/soft rules with severity |
| `users` | Multi-user with access levels |
| `groups` | Group chat configuration |
| `memory` | Semantic memory with pgvector embeddings |
| `sessions` | Conversation sessions |
| `messages` | Full message history |
| `abilities` | Registered abilities + config |
| `config` | Runtime configuration (key-value) |
| `subagent_runs` | Sub-agent execution history |
| `capabilities` | System capabilities registry |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| Database | PostgreSQL 16 + pgvector |
| Chat LLM | Google Gemini 2.5 Pro (OAuth), Claude, GPT (multi-driver) |
| Embedding | Together AI (bge-base-en-v1.5) |
| Telegram | python-telegram-bot |
| HTTP | httpx (async) |
| CLI | Click + Rich |

---

## Project Structure

```
syne/
├── syne/
│   ├── main.py              # Entry point
│   ├── agent.py             # Main agent coordinator
│   ├── boot.py              # System prompt builder
│   ├── config.py            # Settings loader
│   ├── conversation.py      # Session management
│   ├── context.py           # Context window management
│   ├── compaction.py        # Conversation summarization
│   ├── security.py          # Rule 700/760 enforcement
│   ├── cli.py               # CLI commands (init, start, repair, etc.)
│   ├── auth/
│   │   └── google_oauth.py  # Google CCA OAuth PKCE
│   ├── llm/
│   │   ├── provider.py      # Abstract LLM interface
│   │   ├── drivers.py       # Driver registry + model system
│   │   ├── google.py        # Gemini (OAuth)
│   │   ├── codex.py         # ChatGPT/Codex (OAuth)
│   │   ├── openai.py        # OpenAI-compatible (Groq, etc.)
│   │   ├── anthropic.py     # Claude (OAuth)
│   │   ├── together.py      # Together AI (embedding)
│   │   └── hybrid.py        # Chat + Embed from different providers
│   ├── memory/
│   │   ├── engine.py        # Store, recall, dedup, conflict resolution
│   │   └── evaluator.py     # Auto-evaluate (3-layer filter)
│   ├── channels/
│   │   ├── telegram.py      # Telegram bot adapter
│   │   └── cli_channel.py   # Interactive CLI (REPL)
│   ├── tools/               # 13 core tools
│   ├── abilities/           # Bundled + self-created abilities
│   └── db/
│       ├── schema.sql       # Database schema (12 tables)
│       ├── connection.py    # Async connection pool
│       └── models.py        # Data access layer
├── tests/                   # 247 tests
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/riyogarta/syne.git
cd syne
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

---

## Roadmap

- [x] Core memory system with pgvector
- [x] Google OAuth (free Gemini access)
- [x] Telegram channel
- [x] Anti-hallucination memory (3-layer)
- [x] Conflict resolution (3-zone similarity)
- [x] Ability system (bundled + self-created)
- [x] Self-modification (abilities only)
- [x] Multi-model support (5 drivers: Google, OpenAI, Anthropic, Groq, Together AI)
- [x] Interactive CLI mode
- [x] Source code introspection (read_source)
- [x] Systemd service auto-setup
- [x] Sub-agents
- [x] Multi-user access control
- [ ] Ollama support (local embedding — zero cost, but requires more CPU/RAM)
- [ ] Ability marketplace
- [ ] Web UI

---

## License

Apache 2.0

---

**Author:** [Riyogarta Pratikto](https://github.com/riyogarta)

*"I remember, therefore I am"*
