# Syne 🧠

**Self-hosted personal AI assistant on Telegram with persistent long-term memory.**

Runs on your own server. Stores everything in your own PostgreSQL. Near-zero API cost when paired with free OAuth providers or local Ollama. Designed for the case where you want an assistant that *remembers across sessions*, not a fresh chat each time.

*"I remember, therefore I am"* — named after [Mnemosyne](https://en.wikipedia.org/wiki/Mnemosyne), Greek goddess of memory.

---

## What Syne actually is

A Telegram bot you self-host. You chat with it. It remembers what you tell it to remember, across days, weeks, restarts. Built around:

- **Persistent long-term memory** — semantic search via pgvector + a small knowledge graph for entity-relation queries. Subject to decay (non-permanent memories fade with disuse) and conflict resolution (newer, higher-source-priority entries win).
- **User-confirmed memory filtering** — a 3-layer filter (quick rules → small local LLM evaluator → similarity dedup) decides what's worth storing **at write time**. This reduces what enters memory; it is not a guard against the chat model hallucinating at read time.
- **Multi-tier permissions** — owner / family / public with Linux-style 3-digit octal per tool. Public users by default get nothing from memory unless you explicitly open specific categories (Rule 765).
- **Runtime configuration in PostgreSQL** — no `SOUL.md` or `CONFIG.yaml`. Identity, behavior, rules, and config all live in tables, changeable through conversation.
- **Cost floor: Ollama local + optional paid API** — Ollama embeddings + small local evaluator are free and self-contained. Chat goes through Anthropic / Google / OpenAI / Together / Vertex / Codex — paid API keys for resilience, OAuth options as bonus for personal use.

## Advanced capabilities (opt-in)

Beyond the core assistant, Syne ships with optional extensions. They live in `syne/abilities/` and can be enabled/disabled per install:

- **Document I/O** — read & create PDF, Word, Excel, PowerPoint
- **Image generation & vision OCR** — via Vertex Imagen / Together FLUX / OpenAI DALL-E for generation; Vertex Gemini / Ollama vision for analysis
- **Memory file attachments** — store images, PDFs, voice notes alongside their description; retrieve or re-analyze on demand (up to 50 MB per blob)
- **Voice in & out** — Groq Whisper STT for inbound voice notes, Edge TTS for outbound voice messages
- **Remote nodes** — extend Syne across multiple machines via WebSocket gateway
- **Sub-agents** — spawn background workers for parallel tasks
- **Self-modification** — the LLM can write new abilities (off by default, see [SECURITY.md](SECURITY.md))
- **Maps, screenshots, WhatsApp bridge, and more**

These are deliberately separate from the core. The headline use case is chat + memory; everything else is plumbing you can ignore until you need it.

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

Syne's memory is **unlimited** — not by storing everything, but by intelligently deciding what to remember, how to find it, and when to forget. Three components work together: the **evaluator** decides what's worth storing, **embedding + knowledge graph** make memories retrievable through both semantic search and entity-relation traversal, and the **decay engine** ensures only relevant memories survive.

### Memory Flow

Two modes, depending on whether `auto_capture` is enabled:

#### Auto Capture OFF (default)

```
Chat ──→ "Ingat/Remember" detected?
          │
          ├─ Yes → memory_store tool → permanent=true → Embed + KG
          │
          └─ No  → nothing stored (chat history only)
```

Only explicit commands ("ingat ini", "remember this", "catat", "jangan lupa") trigger memory storage. Zero cost when not storing.

#### Auto Capture ON

```
Chat ──→ "Ingat/Remember" detected?
          │
          ├─ Yes → memory_store tool → permanent=true → Embed + KG
          │        (evaluator also processes as safety net — dedup prevents double store)
          │
          └─ No  → Evaluator processes message:
                    │
                    ├─ Layer 1: Quick Filter (no LLM call)
                    │   └─ Skip: greetings, 1-word messages, questions-only, technical noise
                    │
                    ├─ Layer 2: LLM Evaluation (Ollama, $0)
                    │   └─ Worth storing? Assigns category, importance, cleaned content
                    │
                    └─ Layer 3: Similarity Dedup (embedding)
                        └─ New? → permanent=false → Embed (no KG) → subject to decay
                        └─ Duplicate? → skip
```

The evaluator runs **asynchronously** — it never blocks the chat response. Only user-confirmed facts are stored, never assistant suggestions.

| Evaluator Driver | Model | Cost | Speed | Accuracy |
|------------------|-------|------|-------|----------|
| **Ollama** (default) | qwen3:0.6b | **$0** | Fast (local) | Good for most content |
| Provider (main LLM) | Same as chat model | Tokens + 40s delay | Slower | Higher accuracy |

#### Memory Recall

When the user sends a message (2+ words), Syne retrieves relevant memories:

```
User message (2+ words)
    │
    ├─ Embedding Search — cosine similarity over all memory vectors (pgvector HNSW)
    │
    └─ Knowledge Graph — entity name match → 1-hop relation traversal
```

Both results are injected into the conversation context — the LLM sees everything and decides the best answer. Single-word messages ("ok", "ya", "thanks") skip recall entirely.

### The Three Engines

#### 1. Evaluator — "Is this worth remembering?"

A small, local LLM (default: Ollama qwen3:0.6b, **$0**) evaluates messages when `auto_capture` is enabled. Explicit "remember" commands bypass the evaluator entirely — the LLM's `memory_store` tool handles them directly as permanent memories.

#### 2. Embedding & Knowledge Graph — "How do I find this later?"

Two retrieval systems work together to find relevant memories:

**Semantic Search (Embedding)** — Every memory is converted to a high-dimensional vector. When the user asks a question, pgvector finds the most similar memories using cosine distance with an HNSW index. This is what makes memory **unlimited** — millions of memories, millisecond lookup by meaning.

**Knowledge Graph** — Permanent memories are automatically parsed into entities (people, places, organizations) and their relationships. When a question mentions a known entity, Syne traverses 1-hop relations to find connected facts. KG extraction only runs for **permanent** memories — transient memories are too short-lived to justify the cost.

| Source | How it works | Best for |
|--------|-------------|----------|
| **Embedding** | Cosine similarity over memory vectors | Fuzzy recall ("what do you know about my work?") |
| **Knowledge Graph** | Entity lookup → relation traversal | Structured facts ("who is Agha's mother?") |

```
User:  Remember: Alex is married to Sarah. Emma is their daughter.
Syne:  Stored.
       → Graph: Alex --married_to--> Sarah
                Emma --child_of--> Alex
                Emma --child_of--> Sarah
```

| Embedding Provider | Model | Dimensions | Cost | Requirements |
|--------------------|-------|------------|------|-------------|
| **Ollama** (recommended) | qwen3-embedding:0.6b | 1024 | **$0** | 2+ CPU, 2 GB+ RAM |
| **Ollama** | qwen3-embedding:4b | 2560 | **$0** | 4+ CPU, 8 GB+ RAM |
| **Ollama** | qwen3-embedding:8b | 4096 | **$0** | 4+ CPU, 16 GB+ RAM |
| Together AI | bge-base-en-v1.5 | 768 | ~$0.008/1M tokens | API key |
| OpenAI | text-embedding-3-small | 1536 | ~$0.02/1M tokens | API key |

**Ollama** is auto-installed during `syne init` — binary, server, and model are all set up automatically. The installer recommends the best model for your hardware (see [Server Tiers](#server-tiers)).

> **Switching embedding providers deletes all stored memories.** Different models produce incompatible vector spaces. Use the `/embedding` command in Telegram to switch.

Graph extraction uses the main chat LLM by default, switchable to Ollama via `/graph` in Telegram. Manage extractors, toggle on/off, reprocess existing memories, and view stats from the same command.

#### 3. Decay Engine — "What should I forget?"

Non-permanent memories fade naturally, mimicking human forgetting:

- New memories start with `recall_count = 5` (configurable via `memory.initial_recall_count`)
- Every **50 conversations** (configurable), `recall_count` decreases by 1
- When `recall_count` reaches 0, the memory is **deleted**
- Memories that are **recalled** (used in context) get a **+2 boost** each time
- **Permanent** memories (explicit "remember this") **never decay**

This creates a natural selection: useful memories that keep getting recalled survive, while irrelevant ones fade away. No manual cleanup needed.

### Memory Types

| Type | Decay | KG | Created by | Example |
|------|-------|-----|------------|---------|
| **Permanent** | Never | Yes | "Remember: I'm allergic to shellfish" | Important facts, preferences, decisions |
| **Transient** | Fades over time | No | Auto-capture evaluation | Casual mentions, observations |
| **Conversation history** | Compacted when long | No | Every message automatically | Full chat logs in `messages` table |

### Memory File Attachments

A memory can have a **binary file attached** — image, PDF, document, anything up to 50 MB. The file is stored as a BYTEA blob in the `memory_blobs` table, foreign-keyed to the memory row with `ON DELETE CASCADE`. The text description is embedded for semantic search; the file is preserved as-is for later retrieval.

**Two use cases:**
- **Image memories** — gambar yang tidak bisa direduksi jadi teks. Syne describes the image (via `image_analysis` ability), embeds the description, and stores the original image alongside.
- **Source reference** — saat baca PDF/dokumen, simpan teks hasil ekstrak DAN file aslinya. User bisa minta dokumen asli kembali kapan saja.

**Behavior:**
- **Storage is explicit** — same as `memory_store`. Syne won't auto-attach files. User has to ask: *"simpan beserta filenya"* / *"save with the file"*.
- **Retrieval is explicit** — `memory_search` results show a 📎 marker when attachment exists, but the file is NOT auto-sent. LLM tells the user "this memory has an attachment, want me to show it?". User confirms → Syne fetches the blob and sends.
- **CASCADE on decay** — transient memories whose `recall_count` decays to 0 are deleted along with their attached blobs.

**Tools:**

| Tool | Permission | Purpose |
|---|---|---|
| `memory_store_file` | 0o770 | Store memory + attached file in one call |
| `memory_get_file` | 0o555 | Retrieve attached file, deliver as document to user |
| `memory_analyze_file` | 0o555 | Re-extract content for LLM analysis (PDF → text + vision OCR, Office → text, image → vision). User-triggered only. |
| `memory_search` | 0o555 | Now indicates `📎` + filename/size per result |

**Re-analysis vs Retrieve:**
- `memory_get_file` — sends the file to user as a document. User-facing.
- `memory_analyze_file` — re-reads the file and pipes the extracted content back to the LLM context. LLM-facing. Use when the original description isn't enough (e.g., *"cek tanggal di kontrak yang dulu kusimpan"* → LLM needs to re-read the PDF). Triggered only when the user explicitly asks for re-analysis, not on every recall.

**Limits:** 50 MB per attachment (Telegram Bot API cap). Larger files get a friendly error suggesting `memory_store` (description only) instead.

**Example flow:**

```
You:  [upload kontrak.pdf] simpan beserta filenya, ini kontrak sewa rumah 2026
Syne: Stored (memory #42 with attachment kontrak.pdf, 1.2 MB).

           [3 months later]

You:  Cariin dokumen kontrak rumah
Syne: Ada satu memori dengan lampiran:
      - [fact] Kontrak sewa rumah 2026 📎 kontrak.pdf (1.2 MB)
      Mau saya kirim filenya?

You:  Iya
Syne: [sends kontrak.pdf as document]

           [later that day]

You:  Tanggal mulai sewanya kapan ya? Cek kontraknya
Syne: [calls memory_analyze_file(42)]
      Berdasarkan kontrak (PDF, 5 halaman): tanggal mulai sewa
      adalah 1 Januari 2026, berakhir 31 Desember 2026.
```

### Memory Access Control

Two security rules govern who can access memories:

**Rule 760 — Family Privacy**: All memories are private by default. Only `owner` and `family` access levels can read memories.

**Rule 765 — Public Category Exception**: Specific memory categories can be opened for public access via the `memory.public_categories` config (JSON array).

```
# Example: allow public to search Al-Quran and Hadith memories
Set config memory.public_categories to ["alquran","bukhari","muslim","fiqih"]
```

| Requester | Allowed categories | Rule |
|-----------|-------------------|------|
| **Owner** | All | 760 |
| **Family** | All | 760 |
| **Public** | Only categories in `memory.public_categories` | 765 |

When a public user searches memories, results outside the allowed categories are silently filtered — they appear as if they don't exist.

### History & Compaction

Syne uses **Limit N** history loading — only the last N messages (default: 100, configurable via `session.history_limit`) are loaded from the database per chat turn. This leverages PostgreSQL's ability to query with `LIMIT` efficiently, regardless of total session size.

**Adaptive reduction**: If the loaded messages still exceed the context window, the oldest non-system messages are dropped 4 at a time until they fit. This is transparent — no data is lost in the database.

**Compaction** summarizes old messages when the session grows large:
- Triggered automatically when total messages in DB exceed thresholds
- Creates a detailed narrative summary preserving conversation context, facts, and decisions
- Adaptive: if the batch to summarize exceeds the LLM's context, reduces by 4 messages at a time — remaining messages stay for the next compact round
- All compact paths (auto, manual `/compact`, emergency) use the same `run_compact()` method

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
bash install.sh          # Same installer — select option 2 (Remote Node) during syne init
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

## Consent Gate & Shell Guard

Beyond the permission tier (who can invoke), Syne has a **per-action confirmation layer** for destructive tools and a **deterministic parser** for every shell command. Together they make prompt-injection attacks operationally hard: an attacker cannot press a Telegram button, and a hard-denied command never runs even if the model was convinced to try.

### Consent Gate — Per-Action Yes/No

When the LLM emits a destructive tool call (any tool whose caller-class digit has the `x` bit — `shell`, `file_write`, `memory_delete`, `update_config`, etc.), the gate:

1. Suspends the call
2. Sends a `⚠️ Confirm this?` prompt to Telegram with inline **Yes** / **No** buttons
3. Only runs the tool if the owner clicks Yes (buttons are owner-only)

Skips the gate on a **clean turn** from a trusted caller (owner or family) — a conscious, direct command doesn't need friction. A turn is considered *tainted* and re-arms the gate when:

- The message carries an attachment (image / document / uploaded file) or a URL
- A tool that fetches external content ran mid-turn (`web_search`, `fetch_url`, `website_screenshot`, `image_analysis`, `file_read`, `pdf`, `office`)

The Yes button becomes proof of conscious approval that content injected from a web page, an uploaded file, or an image cannot forge — it cannot press a Telegram button. Enforced by code + platform, not by the model's discipline.

**Anti-double-execution** (v1.18.20): the grant is single-use per approval — if a transcript race makes the LLM re-emit the same call, it hits the gate again and prompts fresh rather than running silently on a cached grant. Different-payload re-emits while pending are deterministically blocked, so a small edit ("try again with the flag") can't sneak an unapproved second action past.

Toggle globally via `/consent on|off` (owner-only). Off means destructive tools run immediately without asking — meant for headless / trusted-terminal contexts.

### Shell Guard — Deterministic Command Parser

Every shell command — whether from the LLM, a sub-agent, or an ability shell-out — routes through `run_shell()`, which calls `shell_guard.analyze()` for a verdict before any subprocess is spawned:

| Verdict | Meaning |
|---------|---------|
| `ALLOW` | Every segment's binary is on the allowlist and no danger signal → runs directly |
| `CONSENT` | Allowlisted binary but a danger signal (write flag, redirect, pager exec, etc.) → gate fires |
| `HARD_DENY` | A haram pattern (`rm -rf /`, `mkfs`, `curl | sh`, fork bomb), an unknown binary, or a parse failure → never runs, not even the owner can approve |

Non-negotiable properties:

- **Fail-closed** — any parse error / exception / tokenize failure = `HARD_DENY`, never `ALLOW`
- **Default-deny** — unknown binaries never run; the allowlist is the only thing that opens the gate
- **Denylist first** — haram patterns are checked before the allowlist, so a bad entry in the runtime allowlist can't launder a forbidden command
- **Whole-command** — a compound (`cmd1 && cmd2`) is only as safe as its weakest segment (aggregate = strictest)

**Read-only tiering** — routine inspection doesn't gate:

| Binary | ALLOW form | Escalates to CONSENT when |
|---|---|---|
| `git` | `log`, `show`, `status`, `diff`, `branch`, `tag`, `remote`, `rev-parse/list`, `describe`, `ls-files/tree`, `blame`, `shortlog`, `reflog`, `stash`, `grep`, `cat-file` — with no write flags | write flag (`-d/-D/-M/-c/-C/--force`), unknown sub-command, config-injection (`-c ...`), output redirect |
| `sed` | `-n` (print-only) present, no `-i` | in-place edit (`-i`), no `-n`, or any write signal |
| `awk` | no `-i` (inline edit), no `-f` (script file) | inline edit, script file, or `system()` (danger regex) |
| `pip` | `list`, `show`, `freeze`, `check`, `help`, `config get`, `--version` | `install`, `uninstall`, `download`, `wheel`, `config set/unset` |
| `cat`, `grep`, `head`, `tail`, `wc`, `diff`, `df`, `du`, etc. | Any use | Danger signal (redirect, pager escape) |

**Runtime allowlist/denylist** — owner can promote binaries or add pattern denies via Telegram:

```
/allowlist                            → show current allowlist + candidates
/allowlist add npm                    → promote npm to ALLOW
/allowlist remove foo                 → revoke
/denylist                             → show denies
/denylist add-pattern 'gnome-keyring' → block anything matching the regex
```

Runtime lists layer over the hardcoded floor — floor is authoritative for security; runtime is the operator's carve-outs.

Shell guard is standalone: no imports from Syne internals, no DB, no I/O in `analyze()`. Pure so it can be unit-tested exhaustively. The DB-backed runtime allowlist/denylist is injected as arguments — a DB hiccup fails closed to the floor, never widens the gate.

---

## Core Tools (26)

| Tool | Permission | Description |
|------|-----------|-------------|
| `web_search` | 555 | Search the web (Brave Search API) |
| `web_fetch` | 555 | Fetch and extract content from URLs |
| `shell` | 700 | Execute shell commands (routed through shell_guard for parse + consent). Alias: `exec` |
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
| `memory_search` | 555 | Semantic search over memories (Rule 760/765 filters by category) |
| `memory_store` | 770 | Store new text memory |
| `memory_store_file` | 770 | Store memory + binary file attachment (image, PDF, etc.) |
| `memory_get_file` | 555 | Retrieve a memory's attached file and deliver it as document |
| `memory_analyze_file` | 555 | Re-analyze a memory's attached file (PDF/Office text extract or image vision) — returns content to LLM, not media |
| `memory_delete` | 700 | Delete memories (owner only) |
| `update_config` | 700 | Change runtime configuration |
| `update_ability` | 700 | Enable/disable/create abilities |
| `update_soul` | 700 | Modify personality and behavioral rules |
| `check_auth` | 700 | OAuth token management |

`manage_group` and `manage_user` are owner-only tools registered in the engine but **not exposed to the LLM** — group and member changes are more reliable via the `/groups` and `/members` Telegram menus. The tools exist for future automation but the model doesn't see them today.

Since v1.18.18, the LLM sees the **full toolset every turn** (subject to the access-level filter). Prior versions filtered tools by regex-matching the user's message, which caused false-negative "I don't have that tool" reports for meta-tools like `update_soul` when users phrased around the trigger keywords. Full-toolset delivery also holds the Anthropic prompt-cache breakpoint on system+tools, so the extra schema tokens are absorbed by the cache after the first turn.

---

## Bundled Abilities

| Ability | Permission | Description | Requires |
|---------|-----------|-------------|----------|
| `image_gen` | 777 | Generate images from text (FLUX.1-schnell via Together AI / Imagen via Vertex / DALL-E via OpenAI) | API key per provider |
| `image_analysis` | 555 | Analyze and describe images (Gemini 2.5 Flash default; Vertex / Ollama / OpenAI) | Provider-specific (Vertex auto, OpenAI key) |
| `maps` | 555 | Places, directions, geocoding | Google Maps API key |
| `pdf` | 770 | **Read** PDFs (text + vision OCR for scanned/CAD pages) and **create** PDFs from text or URL | PyMuPDF, reportlab, beautifulsoup4 (auto-installed) |
| `office` | 770 | **Create and read** Microsoft Office documents — Word (.docx), Excel (.xlsx), PowerPoint (.pptx) | python-docx, openpyxl, python-pptx (auto-installed) |
| `website_screenshot` | 550 | Capture website screenshots | Playwright + Chromium (auto-installed) |
| `whatsapp` | 700 | WhatsApp bridge (send/receive via wacli) | wacli binary |

Each ability manages its own dependencies via `ensure_dependencies()` — external binaries and packages are auto-installed when you enable the ability.

### Document Workflows

The `pdf` and `office` abilities together cover most document workflows. Both **auto-extract content** when a user uploads a file via Telegram — the LLM sees the document content as plain text without needing any tool call.

| Format | Read | Create | Notes |
|---|---|---|---|
| PDF | ✓ | ✓ | Hybrid mode: text + vision OCR for scanned/drawing pages (uses `/vision` provider) |
| DOCX (Word) | ✓ | ✓ | Markdown-style content for create; auto-extracts paragraphs + tables on read |
| XLSX (Excel) | ✓ | ✓ | JSON sheets array for create; renders as markdown table on read |
| PPTX (PowerPoint) | ✓ | ✓ | JSON slides array for create; extracts per-slide text on read |

Examples (LLM-callable):
```
office(action='create_docx', title='Laporan', content='# Ringkasan\n\nIsi paragraf.')
office(action='create_xlsx', sheets='[{"name":"Q1","headers":["Item","Total"],"rows":[["Gaji",10000]]}]')
office(action='create_pptx', title='Proposal', slides='[{"title":"Slide 1","bullets":["A","B"]}]')
pdf(action='make_from_text', title='Doc', text='...')
pdf(action='read_from_url', url='https://example.com/file.pdf')
```

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

### Shell Security

The `shell` tool (formerly `exec`) gives Syne shell access on the host system. Every command — from the LLM, from a sub-agent, or from an ability shell-out (LibreOffice conversion, WhatsApp bridge, etc.) — routes through a single chokepoint `run_shell()` that calls **shell_guard** for a verdict (see [Consent Gate & Shell Guard](#consent-gate--shell-guard)).

- **Owner-only** (permission 700) — only the owner can trigger the LLM to call `shell`
- **Configurable timeout** — `exec.timeout_max` (default: 300 seconds)
- **Output limit** — `exec.output_max_chars` (default: 4000 chars)
- **Sub-agents** — inherit shell access but gate through shell_guard the same way (v1.18.0 Phase 5 closed the headless bypass); a sub-agent has no human to approve, so any `CONSENT`-tier command in a sub-agent context becomes an effective denial
- **Fail-closed everywhere** — parse errors, tokenize failures, DB hiccups on the allowlist load — every uncertainty resolves to HARD_DENY, never ALLOW
- **rollback.sh** — a terminal-only recovery helper. Hard-denied via `_HARAM_PATTERNS` so no agent path can invoke it; runs only when you type it yourself in a real terminal
- **Your responsibility** — the parser is a defense, not a magic barrier. Review what you approve, especially on production systems

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
| `memory.initial_recall_count` | `5` | Starting durability for new memories |
| `memory.promotion_threshold` | `10` | Promote to permanent when recall_count exceeds this |
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
syne memory stats                    # Memory statistics
syne memory search "query"           # Semantic search
syne memory add "info"               # Manually add memory
syne memory delete <id>              # Delete a memory by id
syne memory prune '%pattern%'        # Bulk-delete matching ILIKE pattern (with preview + confirm)

# Config
syne config                          # List all config keys
syne config list [prefix]            # Filter list by prefix
syne config get <key>                # Print a single value
syne config set <key> <value>        # Upsert (value auto-parsed: bool/int/JSON/str)
syne config delete <key>             # Remove a key

# Backup & Restore
syne backup                          # Backup database
syne restore                         # Restore from backup
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
| `/consent [on/off]` | Toggle consent gate for destructive tools | Owner |
| `/allowlist` | Manage shell_guard runtime allowlist (add/remove binaries) | Owner |
| `/denylist` | Manage shell_guard runtime denylist (binaries + patterns) | Owner |
| `/models` | Manage LLM models (thinking, reasoning, context, params) | Owner |
| `/embedding` | Manage embedding models | Owner |
| `/evaluator` | Manage evaluator model | Owner |
| `/vision` | Manage vision provider (for OCR / image analysis) | Owner |
| `/imagegen` | Manage image generation provider | Owner |
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
|  |  Core Tools (26):                                    |  |
|  |  exec · db_query · memory · web · config · files     |  |
|  |  send · schedule · reactions · voice · subagent      |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |              ABILITIES (Pluggable)                   |  |
|  |                                                      |  |
|  |  [image_gen]  [image_analysis]  [maps]  [pdf]        |  |
|  |  [office]  [website_screenshot]  [whatsapp]          |  |
|  |  [custom...]                                         |  |
|  |                                                      |  |
|  |  Self-Created: Syne adds new abilities at runtime    |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |              PERMISSION LAYER                        |  |
|  |  Linux-style 3-digit octal: owner/family/public      |  |
|  |  Every tool and ability has its own permission       |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |          CONSENT GATE  +  SHELL GUARD                |  |
|  |  Provenance-aware Yes/No confirmation for destructive|  |
|  |  tools; deterministic parser (ALLOW / CONSENT /      |  |
|  |  HARD_DENY) for every shell command; fail-closed.    |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |              PostgreSQL + pgvector                   |  |
|  |  18 tables — all state in one database              |  |
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
| `memory_blobs` | Binary file attachments for memories (CASCADE on memory delete) |
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
│   ├── consent.py           # Per-action Yes/No gate (owner/family clean-turn skip, provenance-aware)
│   ├── shell_guard.py       # Deterministic shell parser (ALLOW/CONSENT/HARD_DENY, fail-closed)
│   ├── shell_exec.py        # run_shell() — the single subprocess chokepoint for every shell path
│   ├── ratelimit.py         # Per-user rate limiting
│   ├── scheduler.py         # Cron/scheduled task runner
│   ├── subagent.py          # Background sub-agent task runner
│   ├── config_guide.py      # Config reference (injected into system prompt)
│   ├── system_guide.py      # Architecture guide (injected into system prompt)
│   ├── communication/
│   │   ├── telegram.py      # Telegram bot (commands, groups, photos, voice, docs)
│   │   ├── inbound.py       # InboundContext (source of truth per message)
│   │   ├── outbound.py      # Outbound post-processing (path strip, narration strip, MEDIA extract)
│   │   ├── formatting.py    # Output formatting, markdown → Telegram HTML
│   │   ├── tags.py          # [[reply_to]] / [[react]] tag parsing
│   │   └── errors.py        # Error classification
│   ├── channels/
│   │   ├── telegram.py      # Deprecated re-export → syne.communication.telegram
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
│   ├── tools/               # 26 core tools
│   ├── abilities/           # Bundled + self-created abilities
│   │   ├── custom/          # User-created abilities (only writable dir)
│   │   └── ...              # 7 bundled abilities (image_gen, image_analysis, maps, pdf, office, website_screenshot, whatsapp)
│   ├── db/
│   │   ├── schema.sql       # Database schema (18 tables)
│   │   ├── migrations.py    # Versioned schema migrations
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
│       ├── cmd_config.py    # syne config (list/get/set/delete)
│       ├── cmd_memory.py    # syne memory (stats/search/add/delete/prune)
│       ├── cmd_node.py      # syne node (init/start/stop/restart/status)
│       ├── cmd_backup.py    # syne backup/restore
│       └── ...
├── rollback.sh              # Terminal-only recovery helper (hard-denied via shell_guard)
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
- [x] Consent gate (per-action Yes/No with provenance-aware skip)
- [x] Shell guard (deterministic parser: ALLOW / CONSENT / HARD_DENY)
- [ ] Ability marketplace

---

## License

Apache 2.0

---

**Author:** [Riyogarta Pratikto](https://github.com/riyogarta)

*"I remember, therefore I am"*
