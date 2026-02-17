# Syne 🧠

**AI Agent Framework with Unlimited Memory**

*"I remember, therefore I am"*

---

Named after [Mnemosyne](https://en.wikipedia.org/wiki/Mnemosyne), the Greek goddess of memory and mother of the Muses.

Syne is a standalone, open-source AI agent framework built in Python. It features **PostgreSQL-native memory** with semantic search, an **ability-based architecture** for extensibility, and **self-evolution** capabilities where the agent can create new abilities for itself.

## Why Syne?

Most AI assistants forget everything between sessions. They have no persistent memory, no learning, no growth. Syne is different:

- **Unlimited memory** — Semantic search over millions of memories using pgvector
- **Anti-hallucination** — 3-layer defense ensures only user-confirmed facts are stored
- **Self-evolving** — Syne can create new abilities for itself (with your permission)
- **Near-zero cost** — Chat uses Google Gemini via OAuth (free). Embedding via Together AI (~$0.008/1M tokens). Typical monthly cost < $1
- **PostgreSQL-native** — Everything in the database, no file-based config drift

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         SYNE AGENT                               │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                        CORE                                 │ │
│  │  (Cannot be modified by Syne)                               │ │
│  │                                                             │ │
│  │  ┌────────┐ ┌────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐│ │
│  │  │  Chat  │ │ Memory │ │Compaction│ │Channels │ │Sub-agents││ │
│  │  │ (LLM)  │ │(pgvec) │ │(context) │ │(Telegram│ │(max: 2)  ││ │
│  │  └────────┘ └────────┘ └──────────┘ └─────────┘ └──────────┘│ │
│  │                                                             │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │  Core Tools: web_search │ web_fetch                   │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                      ABILITIES                              │ │
│  │  (Pluggable — can be ON/OFF, added, created)                │ │
│  │                                                             │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │ │
│  │  │image_gen│ │  maps   │ │   tts   │ │  exec   │  ...     │ │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘          │ │
│  │                                                             │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │              Self-Created Abilities                  │  │ │
│  │  │  (Syne can create new abilities with exec ON)        │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   PostgreSQL + pgvector                     │ │
│  │  identity │ soul │ rules │ users │ memory │ sessions       │ │
│  │  messages │ abilities │ config                              │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Core vs Abilities

| Layer | What it does | Modifiable? |
|-------|--------------|-------------|
| **Core** | Chat, Memory, Compaction, Channels, Sub-agents | ❌ Protected |
| **Abilities** | Everything else (image gen, search, maps, etc.) | ✅ Pluggable |

---

## No Config Files — Just Talk

Most agent frameworks require editing files like `SOUL.md`, `AGENTS.md`, or `CONFIG.yaml` to change behavior. **Syne has none of that.** Everything lives in PostgreSQL, and you change it through conversation.

| Traditional File | Syne Equivalent | How to change it |
|------------------|-----------------|------------------|
| `SOUL.md` | `soul` table | *"Be more casual and use humor"* |
| `AGENTS.md` | `rules` table | *"Add a rule: never share my location"* |
| `IDENTITY.md` | `identity` table | *"Change your name to Atlas"* |
| `CONFIG.yaml` | `config` table | *"Switch to GPT-4 for chat"* |

### Example: Changing personality

```
You:  Your tone is too formal. Be more casual and funny.
Syne: Got it — updated my personality to casual + humor. ✅

You:  What's your current personality?
Syne: Here's my soul:
      - Tone: casual, witty, helpful
      - Language: adapts to yours
      - Boundaries: respect privacy, ask before external actions
      Want to change anything?

You:  Add a rule: always respond in Indonesian if I write in Indonesian.
Syne: Done — added as a communication rule. ✅
```

Fresh install comes with sensible defaults. You can override anything through conversation — no SQL, no files, no commands to memorize.

> **For advanced users:** You can also modify tables directly via SQL or the CLI (`syne identity`, `syne prompt`). But you'll never *need* to.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for PostgreSQL 16+ with pgvector)

### Installation

```bash
# Clone the repository
git clone https://github.com/riyogarta/syne.git
cd syne

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -e .

# Start PostgreSQL with pgvector
docker compose up -d db

# Run interactive setup
syne init
```

### Setup Flow

```
syne init
    │
    ├─ Detect existing credentials (OpenClaw, Gemini CLI)?
    │   ├─ YES → Reuse OAuth tokens
    │   └─ NO → Open browser for Google OAuth
    │
    ├─ Configure Telegram bot token
    │
    ├─ Initialize database schema
    │
    └─ Ready to start!
```

### Start the Agent

```bash
syne start
```

---

## Configuration

Minimal environment variables (`.env` file):

```bash
# Database (required)
SYNE_DATABASE_URL=postgresql://syne:syne@localhost:5433/syne

# LLM Provider (default: google)
SYNE_PROVIDER=google
```

**All credentials** (Telegram token, OAuth, API keys) are stored in PostgreSQL, configured via `syne init` or chat. This keeps secrets in one secure place instead of scattered across env files.

### Why OAuth, Not API Keys?

| Method | Cost | Notes |
|--------|------|-------|
| Google AI Studio API Key | **PAID** per token | $0.35/1M input, $1.05/1M output |
| Google CCA OAuth | **FREE** | Rate limited, but sufficient for personal use |

Syne uses OAuth by default because it's **free**. This is a core design principle — the agent should be usable without paying per token.

---

## Configuration Reference

All configuration lives in the `config` table. You can change any setting by talking to Syne or via SQL.

### Provider Settings

| Key | Default | Description |
|-----|---------|-------------|
| `provider.primary` | `{"name": "google", "auth": "oauth"}` | LLM provider. Options: `google` (OAuth/free), `openai` |
| `provider.chat_model` | `gemini-2.5-pro` | Model used for chat. Any Gemini model name |
| `provider.embedding_model` | `BAAI/bge-base-en-v1.5` | Model for memory embeddings (Together AI) |
| `provider.embedding_dimensions` | `768` | Embedding vector dimensions |

### Memory Settings

| Key | Default | Description |
|-----|---------|-------------|
| `memory.auto_capture` | `false` | Auto-evaluate incoming messages for memory storage |
| `memory.auto_evaluate` | `true` | Use LLM to judge what's worth storing (only when auto_capture is ON) |
| `memory.recall_limit` | `10` | Max memories to recall per query |
| `memory.max_importance` | `1.0` | Maximum importance score (0.0–1.0) |

### Session Settings

| Key | Default | Description |
|-----|---------|-------------|
| `session.compaction_threshold` | `80000` | Token count before triggering context compaction |
| `session.max_messages` | `100` | Max messages before compaction |

### Sub-agent Settings

| Key | Default | Description |
|-----|---------|-------------|
| `subagents.enabled` | `true` | Master switch for sub-agent spawning |
| `subagents.max_concurrent` | `2` | Max sub-agents running at once |
| `subagents.timeout_seconds` | `300` | Sub-agent timeout (5 min default) |

### Changing Configuration

**Via conversation (recommended):**
```
You:  Increase the memory recall limit to 20
Syne: Done — memory.recall_limit updated to 20. ✅

You:  Disable sub-agents
Syne: Done — subagents.enabled set to false. ✅

You:  What model are you using?
Syne: I'm using gemini-2.5-pro for chat via Google OAuth (free).
      Embeddings use BAAI/bge-base-en-v1.5 via Together AI.
```

**Via SQL (advanced):**
```sql
-- View all config
SELECT key, value FROM config ORDER BY key;

-- Change a setting
UPDATE config SET value = '20' WHERE key = 'memory.recall_limit';

-- Add a new setting
INSERT INTO config (key, value, description)
VALUES ('custom.key', '"value"', 'My custom setting');
```

### Identity & Soul (in database)

| Table | What it controls | Example |
|-------|-----------------|---------|
| `identity` | Name, motto, personality | `name` = "Syne", `personality` = "Helpful, direct, resourceful" |
| `soul` | Behavioral rules by category | `tone` = "casual and witty", `privacy` = "never share user info" |
| `rules` | Hard/soft rules with severity | `[700] Server Access: owner only` (hard) |
| `abilities` | Enabled abilities + config | `image_gen` enabled, `exec` disabled |

---

## Sub-agents (Core)

Syne can spawn isolated background agents for parallel tasks — just like delegating work to an assistant.

| Config | Default | Description |
|--------|---------|-------------|
| `subagents.enabled` | `true` | Master ON/OFF switch |
| `subagents.max_concurrent` | `2` | Max simultaneous sub-agents |

```
User: "Write full documentation for the project"
    │
    ├─ Syne spawns sub-agent (background)
    ├─ Main session continues chatting
    │
    └─ Sub-agent completes → results delivered back
```

Sub-agents inherit abilities and memory access but run in isolated sessions. They cannot spawn other sub-agents.

---

## Ability System

Everything beyond core functionality is an **Ability**. Abilities can be:

| Source | Description | Location |
|--------|-------------|----------|
| `bundled` | Ships with Syne | `syne/abilities/` |
| `installed` | Downloaded from marketplace | Database |
| `self_created` | Created by Syne itself | `user_abilities/` |

### Core Tools (Always Available)

| Tool | Description | Provider |
|------|-------------|----------|
| `web_search` | Search the web | Brave Search API |
| `web_fetch` | Fetch and extract content from URLs | Built-in |

### Bundled Abilities (v1)

| Ability | Description | Provider | Default |
|---------|-------------|----------|---------|
| `image_gen` | Generate images from text | Together AI (FLUX.1-schnell) | ON |
| `image_analysis` | Analyze and describe images | Google Gemini vision | ON |
| `maps` | Places, directions, geocoding | Google Maps/Places | ON |
| `exec` | Execute shell commands | Local system | **OFF** |
| `tts` | Text-to-speech | ElevenLabs / Google | ON |

### Ability Interface

All abilities implement a standard interface:

```python
class Ability:
    name: str           # Unique identifier
    description: str    # Human-readable description
    version: str        # Semantic version
    
    async def execute(self, params: dict, context: dict) -> dict:
        """Execute the ability."""
        ...
    
    def get_schema(self) -> dict:
        """Return JSON schema for LLM function calling."""
        ...
```

### Managing Abilities

```bash
syne ability list              # List all abilities
syne ability enable image_gen  # Enable an ability
syne ability disable exec      # Disable an ability
syne ability info web_search   # Show ability details
```

---

## Self-Modification

When the `exec` ability is enabled, Syne can create new abilities for itself:

```
User: "I wish you could check Bitcoin prices"
    │
    ├─ Syne recognizes the need
    ├─ Creates user_abilities/crypto_price.py
    ├─ Registers as source='self_created', enabled=false
    │
    └─ Asks for owner approval:
        "I created a 'crypto_price' ability. Enable it?"
        │
        ├─ Owner approves → Enabled
        └─ Owner rejects → Remains disabled
```

### Safety Rules

| Rule | Description |
|------|-------------|
| 1 | Syne can CREATE files in `user_abilities/` only |
| 2 | Syne CANNOT modify `syne/` (core code) |
| 3 | Syne CANNOT modify `syne/abilities/` (bundled) |
| 4 | Owner approval required for new abilities |
| 5 | `exec` is OFF by default — owner must enable |

---

## Memory System

Syne's memory is designed to **never store hallucinations**. Only user-confirmed facts are stored.

### Three-Layer Anti-Hallucination

```
User message
    │
    ├─ Layer 1: Quick Filter (no LLM call)
    │   └─ Skip: short messages, greetings, questions-only
    │
    ├─ Layer 2: LLM Evaluation
    │   └─ Analyze: Is this worth remembering?
    │
    └─ Layer 3: Similarity Dedup
        └─ Check: Does this already exist?
```

### Conflict Resolution — Three Zones

When storing a new memory, similarity to existing memories determines the action:

```
    0.0            0.70            0.85            1.0
     │──────────────│───────────────│───────────────│
     │   INSERT     │    UPDATE     │     SKIP      │
     │  New topic   │  Same topic,  │   Duplicate   │
     │              │  updated info │               │
```

| Similarity | Action | Example |
|------------|--------|---------|
| < 0.70 | **Insert** new memory | "I have a dog" (no prior pet info) |
| 0.70–0.84 | **Update** existing | "I moved to Bandung" updates "lives in Jakarta" |
| ≥ 0.85 | **Skip** duplicate | "I live in Jakarta" (already stored) |

### How Memory Storage Works

Syne has two memory storage modes, controlled by `memory.auto_capture`:

**`auto_capture = false` (default):**
Memory is only stored when the user explicitly asks — e.g. "remember this", "save that", "note that down". This gives the user full control over what enters long-term memory.

**`auto_capture = true`:**
Syne automatically evaluates every conversation turn using the LLM and stores what it considers important. The evaluator uses a 3-layer anti-hallucination defense:

1. **Only user-confirmed statements** — assistant suggestions are never stored
2. **Conflict resolution** — new facts are checked against existing memories
3. **Importance scoring** — trivial messages are filtered out

> ⚠️ **Cost warning:** Enabling `auto_capture` adds **one extra LLM call per message** (for evaluation) plus **one embedding call per stored memory**. On free-tier OAuth this means faster rate-limit exhaustion. On paid providers this increases cost per turn. Keep it OFF if cost or rate limits are a concern.

### What Gets Stored (when auto_capture is ON)?

✅ **STORE:**
- Personal facts (name, job, family)
- Preferences (likes, dislikes)
- Important events/milestones
- Decisions and commitments
- Health information
- Lessons learned

❌ **NEVER STORE:**
- Casual greetings ("hi", "thanks")
- Questions without new info
- **Assistant suggestions** (only user-confirmed)
- Temporary/transient info
- Vague statements

### Source Tracking

Every memory has a `source` field:

```sql
source = 'user_confirmed'  -- Only this is stored by evaluator
```

This prevents the common AI problem of storing assistant interpretations as user facts.

---

## CLI Commands

```bash
# Setup & Running
syne init                  # Interactive setup wizard
syne start                 # Start the agent
syne start --debug         # Start with debug logging
syne status                # Show status

# Database
syne db init               # Initialize schema
syne db reset              # Reset database (destructive!)

# Identity
syne identity              # View current identity
syne identity name "Syne"  # Set identity value
syne prompt                # Show current system prompt

# Memory
syne memory stats          # Memory statistics
syne memory search "query" # Semantic search
syne memory add "info" -c fact  # Manually add memory

# Abilities
syne ability list          # List all abilities
syne ability enable <name> # Enable an ability
syne ability disable <name># Disable an ability
syne ability info <name>   # Show ability details
```

---

## Telegram Commands

```
/start     — Welcome message
/help      — Available commands
/status    — Agent status (memories, sessions)
/memory    — Memory statistics
/compact   — Compact conversation history (owner only)
/forget    — Clear current conversation
/identity  — Show agent identity
```

**Group behavior:** Syne only responds when @mentioned or replied to.

---

## Database Schema

All data lives in PostgreSQL:

| Table | Purpose |
|-------|---------|
| `identity` | Agent identity (name, motto, personality) |
| `soul` | Behavioral rules by category |
| `rules` | Hard/soft rules with severity |
| `users` | Multi-user with access levels |
| `memory` | Semantic memory with embeddings |
| `sessions` | Conversation sessions |
| `messages` | Full message history |
| `abilities` | Registered abilities |
| `config` | Runtime configuration |

---

## Cost

| Component | Model | Cost | Notes |
|-----------|-------|------|-------|
| Chat LLM | `gemini-2.5-pro` (Google) | **$0** | Free via CCA OAuth (rate-limited) |
| Embedding | `BAAI/bge-base-en-v1.5` (Together AI) | **~$0.008/1M tokens** | Google CCA OAuth doesn't support embedding API. Together AI is the cheapest alternative |
| Image Gen | `FLUX.1-schnell` (Together AI) | **~$0.003/image** | Optional ability, not required for core |
| PostgreSQL | — | **$0** | Self-hosted via Docker |
| Telegram Bot | — | **$0** | Telegram Bot API is free |
| **Typical monthly** | | **< $1** | Embedding is the only paid component for core usage |

> **Why Together AI for embedding?** Google's CCA OAuth (used for free Gemini access) returns 403 on the embedding endpoint (`generativelanguage.googleapis.com`). Together AI's `BAAI/bge-base-en-v1.5` is open-source, multilingual, and costs ~$0.008 per million tokens — effectively free for personal use.

---

## Project Structure

```
syne/
├── syne/
│   ├── main.py              # Entry point
│   ├── agent.py             # Main agent coordinator
│   ├── boot.py              # System prompt builder
│   ├── config.py            # Settings
│   ├── conversation.py      # Session management
│   ├── context.py           # Context window management
│   ├── compaction.py        # Conversation summarization
│   ├── cli.py               # CLI commands
│   ├── auth/
│   │   └── google_oauth.py  # Google OAuth PKCE
│   ├── llm/
│   │   ├── provider.py      # Abstract interface
│   │   ├── google.py        # Gemini CCA
│   │   ├── together.py      # Together AI
│   │   └── hybrid.py        # Chat + Embed hybrid
│   ├── memory/
│   │   ├── engine.py        # Store, recall, dedup
│   │   └── evaluator.py     # Auto-evaluate
│   ├── channels/
│   │   └── telegram.py      # Telegram adapter
│   ├── tools/               # Core tools
│   │   ├── web_search.py    # Brave Search
│   │   ├── web_fetch.py     # URL content extraction
│   │   └── registry.py      # Tool registry
│   ├── abilities/           # Bundled abilities
│   │   ├── base.py
│   │   ├── image_gen.py
│   │   └── ...
│   └── db/
│       ├── schema.sql
│       ├── connection.py
│       └── models.py
├── user_abilities/          # Self-created abilities
├── tests/
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| Database | PostgreSQL 16 + pgvector |
| Chat LLM | Google Gemini 2.5 Pro (CCA OAuth) |
| Embedding | Together AI (BAAI/bge-base-en-v1.5) |
| Image Gen | Together AI (FLUX.1-schnell) |
| Telegram | python-telegram-bot |
| HTTP | httpx (async) |
| CLI | Click + Rich |

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/riyogarta/syne.git
cd syne
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check .
```

---

## Roadmap

- [x] Core memory system with pgvector
- [x] Google OAuth (free Gemini access)
- [x] Telegram channel
- [x] Anti-hallucination memory
- [x] Conflict resolution (3-zone)
- [x] Ability system
- [x] Self-modification (abilities only — Syne can create/edit abilities but never touch core)
- [ ] Multi-model support (OpenAI, Anthropic, local models via configurable providers)
- [ ] Ability marketplace

---

## License

Apache 2.0

---

## Author

**Riyogarta Pratikto** — [@riyogarta](https://github.com/riyogarta)

---

*"I remember, therefore I am"*
