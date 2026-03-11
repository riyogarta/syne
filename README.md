# Syne 🧠

**AI Agent Framework with Unlimited Memory**

*"I remember, therefore I am"*

Named after [Mnemosyne](https://en.wikipedia.org/wiki/Mnemosyne), the Greek goddess of memory and mother of the Muses.

Syne is a standalone, open-source AI agent framework built in Python. It features **PostgreSQL-native memory** with semantic search, an **ability-based architecture** for extensibility, and **self-evolution** capabilities where the agent can create new abilities for itself.

---

## Why Syne?

Most AI assistants forget everything between sessions. They have no persistent memory, no learning, no growth. Syne is different:

- **Unlimited memory** — Semantic search over millions of memories using pgvector + knowledge graph for entity-relation traversal
- **Anti-hallucination** — 3-layer defense ensures only user-confirmed facts are stored
- **Self-evolving** — Syne can create new abilities for itself (with your permission)
- **No runtime config files** — No SOUL.md or CONFIG.yaml. All runtime behavior lives in PostgreSQL, changed through conversation
- **Near-zero cost** — Chat via Google Gemini OAuth (free). Embedding + memory evaluator via Ollama (local, $0). Entire stack can run at $0/month
- **Linux-style permissions** — 3-digit octal permissions (owner/family/public) for every tool and ability
- **Interactive CLI** — Terminal-based chat with persistent memory and tools

---

## Minimum Requirements

| Requirement | Details |
|-------------|---------|
| **CPU** | 1 vCPU minimum (2+ recommended for Ollama embedding + evaluator) |
| **OS** | Linux (Ubuntu 22.04+, Debian 12+) |
| **Python** | 3.11+ |
| **RAM** | 2 GB minimum (Ollama loads one model at a time, ~1.3 GB per model). 4 GB recommended for smooth operation |
| **Storage** | 1 GB base + ~1 GB for Ollama models (embedding + evaluator). ~2.5 GB total recommended |
| **Docker** | Required — PostgreSQL 16 + pgvector runs in Docker |
| **Network** | Outbound HTTPS to: LLM provider (Google/OpenAI/Anthropic), Telegram API, Brave Search (optional). Ollama runs locally — no network needed for embedding and memory evaluation |

---

## 🚀 Quick Start — 3 Commands

```bash
git clone https://github.com/riyogarta/syne.git
cd syne
bash install.sh
```

**That's it.** `install.sh` is fully automated:

| Step | What happens |
|------|---|
| ① | Choose AI provider (OAuth free / API key) |
| ② | Auto-detect hardware → recommend models |
| ③ | Enter Telegram bot token |
| ④ | Docker + PostgreSQL + pgvector |
| ⑤ | Ollama + embedding + evaluator models |
| ⑥ | Database schema + systemd service |

> When install finishes, Syne is running.

### Server Tiers

During `syne init`, the installer detects your server's CPU and RAM to recommend the best embedding and evaluator models. Higher-tier models produce better memory recall quality.

| Server | CPU | RAM | Embedding | Evaluator |
|--------|-----|-----|-----------|-----------|
| No Ollama | <2 core | <2 GB | Together AI (cloud) | - (no auto-capture) |
| Minimal | 2+ core | 2-4 GB | qwen3-embedding:0.6b (1024d) | qwen3:0.6b |
| Moderate | 2+ core | 4-8 GB | qwen3-embedding:0.6b (1024d) | qwen3:1.7b |
| Strong | 4+ core | 8-16 GB | qwen3-embedding:4b (2560d) | qwen3:1.7b |
| Beast | 4+ core | 16+ GB | qwen3-embedding:8b (4096d) | qwen3:4b |

> **Embedding model is permanent** — changing it later requires resetting all memories (re-embedding is not yet supported). The evaluator model can be changed anytime via `/evaluator` in Telegram.

### Verify Installation

```bash
syne repair     # Diagnose: DB, OAuth, Telegram, abilities
syne status     # Quick status check
```

---

## No Runtime Config Files — Just Talk

Most agent frameworks require editing `SOUL.md`, `AGENTS.md`, or `CONFIG.yaml` to change behavior. Syne has none of that — all runtime behavior lives in PostgreSQL, and you change it through conversation:

| Traditional File | Syne Equivalent | Just say... |
|------------------|-----------------|-------------|
| `SOUL.md` | `soul` table | *"Be more casual and witty"* |
| `AGENTS.md` | `rules` table | *"Add a rule: never share my location"* |
| `IDENTITY.md` | `identity` table | *"Change your name to Atlas"* |
| `CONFIG.yaml` | `config` table | *"Turn on auto memory capture"* |

Fresh install comes with sensible defaults. Override anything through conversation — no SQL, no files.

---

## Memory System

Syne's memory is **unlimited** — not by storing everything, but by intelligently deciding what to remember, how to find it, and when to forget. Three components work together: the **evaluator** decides what's worth storing, the **embedding model** makes memories searchable, and the **decay engine** ensures only relevant memories survive.

### The Three Engines

#### 1. Evaluator — "Is this worth remembering?"

A small, local LLM (default: Ollama qwen3:0.6b, **$0**) evaluates every message through a 3-layer filter:

```
User message
    │
    ├─ Layer 1: Quick Filter (no LLM call)
    │   └─ Skip: greetings, short messages, questions-only, technical noise
    │
    ├─ Layer 2: LLM Evaluation (evaluator model)
    │   └─ Is this a fact, preference, event, decision, or lesson?
    │   └─ Assigns: category, importance score (0.0–1.0), cleaned content
    │
    └─ Layer 3: Similarity Dedup (embedding model)
        └─ Does this already exist? (cosine similarity check)
```

The evaluator runs **asynchronously** — it never blocks the chat response. Only user-confirmed facts are stored, never assistant suggestions.

| Evaluator Driver | Model | Cost | Speed | Accuracy |
|------------------|-------|------|-------|----------|
| **Ollama** (default) | qwen3:0.6b | **$0** | Fast (local) | Good for most content |
| Provider (main LLM) | Same as chat model | Tokens + 40s delay | Slower | Higher accuracy |

#### 2. Embedding — "How do I find this later?"

Every memory is converted to a high-dimensional vector (embedding) for semantic search. When the user asks a question, the question is also embedded, and pgvector finds the most similar memories using cosine distance with an HNSW index.

This is what makes memory **unlimited** — you can store millions of memories and still find the right one in milliseconds, because the search is by meaning, not keywords.

| Embedding Provider | Model | Dimensions | Cost | Requirements |
|--------------------|-------|------------|------|-------------|
| **Ollama** (recommended) | qwen3-embedding:0.6b | 1024 | **$0** | 2+ CPU, 2 GB+ RAM |
| **Ollama** | qwen3-embedding:4b | 2560 | **$0** | 4+ CPU, 8 GB+ RAM |
| **Ollama** | qwen3-embedding:8b | 4096 | **$0** | 4+ CPU, 16 GB+ RAM |
| Together AI | bge-base-en-v1.5 | 768 | ~$0.008/1M tokens | API key |
| OpenAI | text-embedding-3-small | 1536 | ~$0.02/1M tokens | API key |

**Ollama** is auto-installed during `syne init` — binary, server, and model are all set up automatically. The installer recommends the best model for your hardware (see [Server Tiers](#server-tiers)).

> **Switching embedding providers deletes all stored memories.** Different models produce incompatible vector spaces. Use the `/embedding` command in Telegram to switch.

#### 3. Decay Engine — "What should I forget?"

Non-permanent memories fade naturally, mimicking human forgetting:

- Every **50 conversations** (configurable), `recall_count` decreases by 1
- When `recall_count` reaches 0, the memory is **deleted**
- Memories that are **recalled** (used in context) get a **+2 boost** each time
- **Permanent** memories (explicit "remember this") **never decay**

This creates a natural selection: useful memories that keep getting recalled survive indefinitely, while irrelevant ones fade away. You don't need to manually clean up — the system self-maintains.

### Memory Types

| Type | Decay | Created by | Example |
|------|-------|------------|---------|
| **Permanent** | Never | "Remember: I'm allergic to shellfish" | Important facts, preferences, decisions |
| **Transient** | Fades over time | Auto-capture evaluation | Casual mentions, observations |
| **Conversation history** | Compacted when long | Every message automatically | Full chat logs in `messages` table |

### Conflict Resolution

When storing a new memory, similarity to existing memories determines the action:

| Similarity | Action | Example |
|------------|--------|---------|
| < 0.70 | **Insert** new memory | "I have a dog" (no prior pet info) |
| 0.70–0.84 | **Update** existing | "I moved to Bandung" updates "lives in Jakarta" |
| ≥ 0.85 | **Skip** duplicate | "I live in Jakarta" (already stored) |

Source priority resolves conflicts: `user_confirmed` > `observed` > `auto_captured` > `system`.

### Auto Capture vs Manual

| Mode | Trigger | Cost impact |
|------|---------|-------------|
| `auto_capture = false` (default) | Only when user says "remember this" | No extra calls |
| `auto_capture = true` | Every message evaluated | +1 evaluator call + 1 embedding per message |

With Ollama as both evaluator and embedding provider, auto-capture costs **$0** — both run locally.

### Managing Memories

```
You:  Remember: I'm allergic to shellfish.
Syne: Stored.

              [3 days later]

You:  Suggest dinner for tonight.
Syne: How about rendang or soto ayam? Avoiding shellfish as noted.

You:  What do you remember about my family?
Syne: You have a partner and a child. You're allergic to shellfish.
      Anything else you'd like me to note?

You:  Forget that I like sushi.
Syne: Removed from memory.
```

Memory recall respects the permission system — only the owner and family-level users can access memories. Public users cannot:

```
Stranger: What do you know about your owner's family?
Syne:     I can't share that. That's private information.
```

Via CLI: `syne memory stats`, `syne memory search "query"`, `syne memory add "info"`

### Knowledge Graph

On top of semantic search, Syne builds a **knowledge graph** from permanent memories — extracting entities (people, places, organizations) and their relationships automatically.

```
User:  Remember: Riyogarta is married to Yuliazmi. Agha is their daughter.
Syne:  Stored.
       → Graph: Riyogarta --married_to--> Yuliazmi
                Agha --child_of--> Riyogarta
                Agha --child_of--> Yuliazmi
```

When answering questions, Syne uses **both** semantic search and graph traversal:

| Source | How it works | Best for |
|--------|-------------|----------|
| **Embedding** | Cosine similarity over memory vectors | Fuzzy recall ("what do you know about my work?") |
| **Knowledge Graph** | Entity lookup → 1-hop relation traversal | Structured facts ("who is Agha's mother?") |

Both results are injected into the conversation context — the LLM sees everything and decides the best answer.

**Graph extraction** runs automatically when permanent memories are stored. The extractor uses the main chat LLM by default, but can be switched to a local Ollama model via `/graph` in Telegram.

| Setting | Default | Description |
|---------|---------|-------------|
| `graph.enabled` | `true` | Enable/disable graph extraction |
| `graph.extractor_driver` | `"provider"` | `"provider"` (main LLM) or `"ollama"` |

Manage via `/graph` in Telegram: add/remove extractors, toggle on/off, reprocess existing memories, view stats.

---

## Remote Node

Syne can run on multiple machines. A **remote node** connects to a Syne server via WebSocket, allowing you to use Syne's tools (exec, file access) on your laptop or other machines — all controlled from Telegram.

### Setup

On the server (one-time):
```bash
# Gateway is built into Syne — no extra setup needed
# It starts automatically with the main service
```

On the remote machine:
```bash
git clone https://github.com/riyogarta/syne.git
cd syne
bash install.sh          # Minimal install (no DB, no Telegram)
syne node init           # Pair with server (enter pairing token)
```

Pairing is done via Telegram: use `/nodes` to generate a one-time token, then enter it on the remote machine. After pairing, the node daemon starts automatically and reconnects on reboot.

### How It Works

```
+------------------+        WebSocket        +------------------+
|  Remote Node     | ◄--------------------► |  Syne Server     |
|  (your laptop)   |   persistent conn      |  (cloud VPS)     |
|                  |   auto-reconnect       |                  |
|  syne-node.service|                       |  gateway (built-in)|
|  exec, files     |                       |  DB, Telegram, LLM|
+------------------+                        +------------------+
```

| Feature | Detail |
|---------|--------|
| **Auto-reconnect** | Exponential backoff (5s → 60s max), survives network changes |
| **Auto-start** | systemd user service with linger, starts on boot |
| **Suspend/resume** | Reconnects automatically when laptop wakes from sleep |
| **Coexistence** | A machine can be both server and node simultaneously |

### Node CLI Commands

```bash
syne node init           # Pair with server
syne node start          # Start node daemon
syne node stop           # Stop node daemon
syne node restart        # Restart node daemon
syne node status         # Show connection status
```

---

## Permission System

Syne uses a Linux-inspired 3-digit octal permission system. Every tool and ability declares its own permission (e.g., `0o770` = owner + family can use).

### Access Levels

| Level | Who | Description |
|-------|-----|-------------|
| **owner** | System administrator | Full access to everything |
| **family** | Trusted users (household, close friends) | Access based on tool/ability permission |
| **public** | Anyone else who messages the bot | Limited to safe, read-only tools |
| **blocked** | Denied users | No access to anything |

The first user to message Syne automatically becomes `owner`.

### Permission Format: `0oOFP`

Each digit controls access for Owner / Family / Public. Bits: `r`(4) read, `w`(2) write, `x`(1) execute.

| Permission | Meaning | Example tools |
|------------|---------|---------------|
| `0o700` | Owner only | exec, db_query, file_write, update_config |
| `0o770` | Owner + family | send_message, memory_store, manage_schedule |
| `0o555` | Everyone (read/exec) | web_search, web_fetch |
| `0o550` | Owner + family (read/exec) | memory_search, subagent_status |

Manage users via conversation: *"Make @alice family"*, *"Remove @bob's access"*

---

## Core Tools (23)

| Tool | Permission | Description |
|------|-----------|-------------|
| `web_search` | 555 | Search the web (Brave Search API) |
| `web_fetch` | 555 | Fetch and extract content from URLs |
| `exec` | 700 | Execute shell commands |
| `db_query` | 700 | Read-only SQL queries (SELECT only, credentials redacted) |
| `file_read` | 500 | Read files in workspace (max 100KB) |
| `file_write` | 700 | Write files (restricted to safe directories) |
| `read_source` | 500 | Read Syne's own source code (for self-healing) |
| `send_message` | 770 | Send messages to any chat |
| `send_file` | 770 | Send files/media to chat |
| `send_voice` | 770 | Text-to-speech voice messages |
| `send_reaction` | 771 | Emoji reactions on messages |
| `manage_schedule` | 770 | Create/list/delete scheduled tasks |
| `spawn_subagent` | 750 | Spawn background sub-agents |
| `subagent_status` | 550 | Check sub-agent status |
| `memory_search` | 550 | Semantic search over memories |
| `memory_store` | 770 | Store new memories |
| `memory_delete` | 700 | Delete memories |
| `manage_group` | 700 | Manage group chat settings |
| `manage_user` | 700 | Manage user access levels |
| `update_config` | 700 | Change runtime configuration |
| `update_ability` | 700 | Enable/disable/create abilities |
| `update_soul` | 700 | Modify personality and behavioral rules |
| `check_auth` | 700 | OAuth token management |

---

## Bundled Abilities

| Ability | Permission | Description | Requires |
|---------|-----------|-------------|----------|
| `image_gen` | 777 | Generate images from text (FLUX.1-schnell via Together AI) | Together AI API key |
| `image_analysis` | 555 | Analyze and describe images (Gemini 2.0 Flash default) | Google Gemini / Together AI / OpenAI |
| `maps` | 555 | Places, directions, geocoding | Google Maps API key |
| `pdf` | 770 | Generate PDF documents from HTML | wkhtmltopdf (auto-installed) |
| `website_screenshot` | 550 | Capture website screenshots | Playwright + Chromium (auto-installed) |
| `whatsapp` | 700 | WhatsApp bridge (send/receive via wacli) | wacli binary |

Each ability manages its own dependencies via `ensure_dependencies()` — external binaries and packages are auto-installed when you enable the ability.

---

## Self-Modification

Syne can create new abilities at runtime — no restart required:

### Flow

```
User: "I wish you could check Bitcoin prices"
    │
    ├─ Syne writes syne/abilities/custom/crypto_price.py
    ├─ Validates syntax, structure, and schema
    ├─ Registers via update_ability (source='self_created')
    ├─ Ability is immediately available
    │
    └─ "Created 'crypto_price' ability. Try: what's BTC now?"
```

### Safety Rules

| Rule | Description |
|------|-------------|
| CAN | Create/edit files in `syne/abilities/custom/` |
| CANNOT | Modify core code (engine, tools, channels, db, llm, security) |
| CANNOT | Modify `syne/db/schema.sql` |
| REPORTS | Core bugs → formats GitHub issue for owner to post |

### Ability Interface

```python
class Ability:
    name: str
    description: str
    version: str
    permission: int = 0o700  # 3-digit octal (owner/family/public)

    async def execute(self, params: dict, context: dict) -> dict: ...
    def get_schema(self) -> dict: ...
    def get_guide(self, enabled: bool, config: dict) -> str: ...
    async def ensure_dependencies(self) -> tuple[bool, str]: ...
```

### Exec Security

The `exec` tool gives Syne shell access on the host system:

- **Owner-only** (permission 700) — only the owner can trigger exec
- **Configurable timeout** — `exec.timeout_max` (default: 300 seconds)
- **Output limit** — `exec.output_max_chars` (default: 4000 chars)
- **Sub-agents** — inherit exec access but run in isolated sessions
- **Your responsibility** — review what Syne executes, especially on production systems

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
| `subagents.timeout_seconds` | `900` | Sub-agent timeout (15 min) |

Sub-agents inherit abilities and memory access but run in isolated sessions. They cannot spawn other sub-agents or use config/management tools.

---

## Configuration Reference

All configuration lives in the `config` table. Change via conversation or `update_config` tool.

### Provider Settings

| Key | Default | Description |
|-----|---------|-------------|
| `provider.active_model` | `"gemini-pro"` | Active chat model key |
| `provider.active_embedding` | *(set during init)* | Active embedding model key (auto-detected by hardware tier) |

### Memory Settings

| Key | Default | Description |
|-----|---------|-------------|
| `memory.auto_capture` | `false` | Auto-evaluate messages for storage |
| `memory.recall_limit` | `5` | Max memories per query |
| `memory.decay_interval` | `50` | Decay every N conversations |
| `memory.decay_amount` | `1` | Recall count decrease per decay cycle |
| `memory.initial_recall_count` | `1` | Starting durability for new memories |
| `memory.evaluator_driver` | `"ollama"` | Evaluator: "ollama" (local) or "provider" (main LLM) |
| `memory.evaluator_model` | `"qwen3:0.6b"` | Ollama model for evaluation |

### Session Settings

| Key | Default | Description |
|-----|---------|-------------|
| `session.compaction_threshold` | `80000` | Characters before auto-compaction |
| `session.compaction_keep_recent` | `40` | Messages kept after compaction |
| `session.max_messages` | `100` | Messages before suggesting compaction |
| `session.thinking_budget` | `null` | Global default only — per-model thinking is set via `/models` |

### Claude OAuth (Optional)

| Key | Default | Description |
|-----|---------|-------------|
| `claude.oauth_client_id` | *(built-in)* | Override OAuth client_id for Anthropic Claude |

### Rate Limiting

| Key | Default | Description |
|-----|---------|-------------|
| `ratelimit.max_requests` | `4` | Max requests per user per window |
| `ratelimit.window_seconds` | `60` | Rate limit window |
| `ratelimit.owner_exempt` | `true` | Owner exempt from rate limits |

### Telegram Settings

| Key | Default | Description |
|-----|---------|-------------|
| `telegram.dm_policy` | `"approval"` | DM policy: "approval" or "open" |
| `telegram.group_policy` | `"allowlist"` | Group policy: "allowlist" or "open" |
| `telegram.require_mention` | `true` | Require @mention in groups |

### Exec & Web Settings

| Key | Default | Description |
|-----|---------|-------------|
| `exec.timeout_max` | `300` | Max exec timeout (seconds) |
| `exec.output_max_chars` | `4000` | Max output characters |
| `web_fetch.timeout` | `30` | HTTP fetch timeout (seconds) |

### Sub-agent Settings

| Key | Default | Description |
|-----|---------|-------------|
| `subagents.enabled` | `true` | Enable sub-agents |
| `subagents.max_concurrent` | `2` | Max concurrent sub-agents |
| `subagents.timeout_seconds` | `900` | Sub-agent timeout (15 min) |

---

## CLI Commands

```bash
# Setup & Running
syne init                  # Interactive setup (fully automated)
syne start                 # Start Telegram agent
syne start --debug         # Start with debug logging
syne cli                   # Interactive CLI chat (resumes per-directory)
syne cli -n                # Start fresh conversation (clear history)
syne cli --yolo            # Skip file write approvals (auto-yes)
syne status                # Show status
syne repair                # Diagnose and repair
syne restart               # Restart agent
syne stop                  # Stop agent

# Updates
syne update                # Update to latest release (includes restart)
syne updatedev             # Force pull + reinstall (includes restart)

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

# Backup & Restore
syne backup                # Backup database
syne restore               # Restore from backup
```

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
| `/clear` | Clear conversation history | Owner |
| `/compact` | Compact conversation | Owner |
| `/reasoning [on/off]` | Toggle reasoning visibility | Owner |
| `/autocapture [on/off]` | Toggle auto memory capture | Owner |
| `/models` | Manage LLM models (thinking, reasoning, context, params) | Owner |
| `/embedding` | Manage embedding models | Owner |
| `/evaluator` | Manage evaluator model | Owner |
| `/browse` | Browse directories (share session with CLI) | Owner |
| `/groups` | Manage groups & members | Owner |
| `/members` | Manage global user access levels | Owner |
| `/wamembers` | Manage WhatsApp allowlist | Owner |
| `/backup` | Backup database | Owner |
| `/restore` | Restore database from backup | Owner |
| `/graph` | Manage knowledge graph extractors | Owner |
| `/nodes` | Manage remote nodes | Owner |
| `/update` | Update Syne to latest version | Owner |
| `/quit` | End conversation | Owner |
| `/cancel` | Cancel active operation | Owner |
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
|  |  [Graph]  [Gateway]                                  |  |
|  |  (KG)    (remote)                                    |  |
|  |                                                      |  |
|  |  Core Tools (23):                                    |  |
|  |  exec · db_query · memory · web · config · files     |  |
|  |  send · schedule · reactions · voice · subagent      |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |              ABILITIES (Pluggable)                   |  |
|  |                                                      |  |
|  |  [image_gen]  [image_analysis]  [maps]  [pdf]        |  |
|  |  [website_screenshot]  [whatsapp]  [custom...]       |  |
|  |                                                      |  |
|  |  Self-Created: Syne adds new abilities at runtime    |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |              PERMISSION LAYER                        |  |
|  |  Linux-style 3-digit octal: owner/family/public      |  |
|  |  Every tool and ability has its own permission        |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |              PostgreSQL + pgvector                   |  |
|  |  17 tables — all state in one database              |  |
|  |  memory · sessions · messages · config · abilities   |  |
|  |  users · groups · identity · soul · capabilities     |  |
|  |  rules · scheduled_tasks · subagent_runs             |  |
|  |  kg_entities · kg_relations · gateway_nodes          |  |
|  |  pairing_tokens                                      |  |
|  +------------------------------------------------------+  |
+------------------------------------------------------------+
```

---

## Database Schema

| Table | Purpose |
|-------|---------|
| `identity` | Agent name, motto, personality |
| `soul` | Behavioral directives by category |
| `rules` | Hard/soft rules with severity |
| `users` | Multi-user with access levels (owner/family/public/blocked) |
| `groups` | Group chat configuration and member access |
| `memory` | Semantic memory with pgvector embeddings |
| `sessions` | Conversation sessions |
| `messages` | Full message history |
| `capabilities` | Tool capability registry |
| `abilities` | Registered abilities + config |
| `config` | Runtime configuration (key-value) |
| `subagent_runs` | Sub-agent execution history |
| `scheduled_tasks` | Cron jobs and scheduled task definitions |
| `kg_entities` | Knowledge graph entities (name, type, aliases) |
| `kg_relations` | Knowledge graph relationships (subject → predicate → object) |
| `gateway_nodes` | Registered remote nodes |
| `pairing_tokens` | One-time pairing tokens for remote nodes |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Database | PostgreSQL 16 + pgvector |
| Chat LLM | Google Gemini, Claude, ChatGPT, OpenAI, Together AI, Ollama (7 drivers) |
| Embedding | Ollama (qwen3-embedding, local) or Together AI / OpenAI (cloud) |
| Telegram | python-telegram-bot |
| HTTP | httpx (async) |
| CLI | Click + Rich |

---

## Project Structure

```
syne/
├── syne/
│   ├── main.py              # Entry point
│   ├── agent.py             # Main agent: tool registration, OAuth, startup
│   ├── boot.py              # System prompt builder (identity, soul, rules, guides)
│   ├── conversation.py      # Conversation loop: context → LLM → tools → response
│   ├── context.py           # Context window manager (token counting)
│   ├── compaction.py        # Conversation summarization
│   ├── security.py          # Permission system, SSRF protection, credential masking
│   ├── ratelimit.py         # Per-user rate limiting
│   ├── scheduler.py         # Cron/scheduled task runner
│   ├── subagent.py          # Background sub-agent task runner
│   ├── config_guide.py      # Config reference (injected into system prompt)
│   ├── system_guide.py      # Architecture guide (injected into system prompt)
│   ├── update_checker.py    # GitHub version checker
│   ├── communication/
│   │   ├── inbound.py       # InboundContext (source of truth per message)
│   │   ├── formatting.py    # Output formatting, tag stripping
│   │   └── errors.py        # Error classification
│   ├── channels/
│   │   ├── telegram.py      # Telegram bot (commands, groups, photos, voice, docs)
│   │   └── cli_channel.py   # Interactive CLI (REPL)
│   ├── auth/
│   │   ├── google_oauth.py  # Google OAuth2 (for Gemini CCA)
│   │   ├── codex_oauth.py   # ChatGPT/Codex OAuth
│   │   └── claude_oauth.py  # Anthropic Claude OAuth
│   ├── llm/
│   │   ├── provider.py      # Abstract LLM interface
│   │   ├── drivers.py       # Driver registry + embedding drivers
│   │   ├── google.py        # Gemini (OAuth + API key)
│   │   ├── codex.py         # ChatGPT/Codex (OAuth)
│   │   ├── anthropic.py     # Claude (OAuth + API key)
│   │   ├── openai.py        # OpenAI-compatible
│   │   ├── together.py      # Together AI
│   │   ├── ollama.py        # Local Ollama models
│   │   └── hybrid.py        # Multi-provider with failover
│   ├── memory/
│   │   ├── engine.py        # Store, recall, decay, dedup, conflict resolution
│   │   ├── evaluator.py     # Auto-evaluate (3-layer filter)
│   │   └── graph.py         # Knowledge graph extraction, storage, recall
│   ├── tools/               # 23 core tools
│   ├── abilities/           # Bundled + self-created abilities
│   │   ├── custom/          # User-created abilities (only writable dir)
│   │   └── ...              # 6 bundled abilities
│   ├── db/
│   │   ├── schema.sql       # Database schema (17 tables)
│   │   ├── connection.py    # Async connection pool (asyncpg)
│   │   ├── models.py        # Data access layer
│   │   └── credentials.py   # Encrypted credential storage
│   ├── gateway/             # Remote node support
│   │   └── server.py        # WebSocket gateway server
│   └── cli/                 # CLI commands package
│       ├── cmd_init.py      # syne init
│       ├── cmd_start.py     # syne start
│       ├── cmd_status.py    # syne status
│       ├── cmd_repair.py    # syne repair
│       ├── cmd_update.py    # syne update
│       ├── cmd_db.py        # syne db
│       ├── cmd_memory.py    # syne memory
│       ├── cmd_node.py      # syne node (init/start/stop/restart/status)
│       ├── cmd_backup.py    # syne backup/restore
│       └── ...
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

- [x] Interactive CLI (`syne cli`)
- [x] Knowledge Graph (entity-relation extraction from memories)
- [x] Remote Node (multi-machine via WebSocket gateway)
- [ ] Ability marketplace

---

## License

Apache 2.0

---

**Author:** [Riyogarta Pratikto](https://github.com/riyogarta)

*"I remember, therefore I am"*
