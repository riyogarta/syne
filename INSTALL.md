# Syne — Installation Guide

## Prerequisites

- Python 3.11+
- Docker (for PostgreSQL + pgvector)
- A Telegram bot token (from [@BotFather](https://t.me/BotFather))
- Google account (for free Gemini OAuth)

## Quick Install

```bash
# 1. Clone
git clone https://github.com/riyogarta/syne.git
cd syne

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install
pip install -e ".[dev]"

# 4. Interactive setup (DB, auth, Telegram)
syne init

# 5. Start
syne start
```

That's it. Syne is running on Telegram.

## Manual Setup (Step by Step)

### 1. Database

Syne runs its own isolated PostgreSQL container. This is **not optional** — your API keys, OAuth tokens, and personal memories are stored in the database. An isolated container ensures they stay private.

```bash
# Start PostgreSQL with pgvector via Docker
docker compose up -d db

# Verify
docker exec syne-db psql -U syne -d syne -c "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
```

Connection (auto-configured): `postgresql://syne:syne@localhost:5433/syne`

> ⚠️ **Security**: External/shared PostgreSQL is not supported. A misconfigured public database would expose your OAuth tokens, API keys, and personal memories.

### 2. Environment

Create `.env` in the project root (copy from `.env.example`):

```env
SYNE_DATABASE_URL=postgresql://syne:syne@localhost:5433/syne
SYNE_PROVIDER=google
```

That's it! All other credentials (Telegram token, API keys) are configured via `syne init` or chat, and stored securely in PostgreSQL.

### 3. Google OAuth

```bash
syne init  # Follow the OAuth flow
```

Or if you already have OpenClaw installed, Syne auto-detects and reuses OpenClaw's Google credentials.

Credentials saved at: `~/.syne/google_credentials.json`

### 4. Initialize Database Schema

```bash
syne db init
```

This creates all tables and seed data (identity, rules, soul, config).

### 5. Start

```bash
syne start          # Foreground (Ctrl+C to stop)
syne start --debug  # With debug logging
```

## Running as a Service

### Start / Stop / Restart

```bash
syne start     # Start in foreground
syne stop      # Stop running Syne
syne restart   # Stop + start (background)
```

### Enable Autostart (systemd)

```bash
syne autostart           # Enable autostart on boot
syne autostart --disable # Disable autostart
```

This creates a systemd user service at `~/.config/systemd/user/syne.service`.

After enabling:
```bash
systemctl --user start syne      # Start now
systemctl --user stop syne       # Stop
systemctl --user restart syne    # Restart
systemctl --user status syne     # Check status
journalctl --user -u syne -f     # Live logs
```

## Health Check & Repair

```bash
syne status          # Quick status overview
syne repair          # Diagnose issues (check only)
syne repair --fix    # Diagnose + auto-fix
```

`syne repair` checks:
1. **Database** — Connection, pgvector, tables, seed data
2. **Google OAuth** — Credentials, token validity
3. **Telegram Bot** — Token validity, bot info
4. **Abilities** — All loaded, config status (ready vs needs API key)
5. **Process** — Running? Autostart enabled?
6. **Docker** — syne-db container status

`syne repair --fix` will attempt to:
- Install pgvector extension
- Create missing tables
- Seed empty config/identity/rules
- Start stopped Docker containers

## Other CLI Commands

```bash
syne identity                   # View agent identity
syne identity name "MyAgent"    # Set agent name
syne prompt                     # Show current system prompt
syne memory stats               # Memory statistics
syne memory search "query"      # Search memories
syne memory add "info" -c fact  # Add a memory manually
syne db init                    # Initialize schema
syne db reset                   # ⚠️ DROP ALL + re-init
```

## Docker Compose (Full Stack)

Run everything in Docker:

```bash
docker compose up -d     # Start DB + Syne
docker compose logs -f   # Follow logs
docker compose down      # Stop all
```

## Configuration via Chat

After Syne is running, configure it by chatting on Telegram:

```
You: "Here's my Brave API key: your_api_key_here"
Syne: [stores it → web_search ability ready]

You: "Change your name to Atlas"
Syne: [updates identity in DB]

You: "Add a rule: always respond in Indonesian"
Syne: [adds rule to DB]
```

No config files to edit. Just talk.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `syne: command not found` | Run `pip install -e .` in the project dir |
| DB connection refused | `docker compose up -d db` |
| OAuth token expired | `syne repair --fix` or `syne init` |
| 429 rate limit | Wait 30s, Syne has auto-retry (2s → 4s → 8s) |
| Bot not responding | Check `syne status` and `tail -f /tmp/syne.log` |
| Ability not working | Chat: "list my abilities" — check config status |

## Architecture

```
syne/
├── agent.py          # Main agent (ties everything together)
├── boot.py           # System prompt builder
├── conversation.py   # Session & context management
├── main.py           # Entry point
├── cli.py            # CLI commands
├── config.py         # Settings loader
├── context.py        # Context window management
├── compaction.py     # Message compaction
├── subagent.py       # Sub-agent manager
├── auth/
│   └── google_oauth.py  # Google CCA OAuth
├── db/
│   ├── connection.py    # DB pool
│   ├── models.py        # Data access
│   └── schema.sql       # Table definitions
├── llm/
│   ├── provider.py      # Base provider
│   ├── google.py        # Gemini CCA + API
│   ├── openai.py        # OpenAI
│   ├── together.py      # Together AI
│   └── hybrid.py        # Chat + Embed split
├── memory/
│   ├── engine.py        # Store, recall, conflict resolution
│   └── evaluator.py     # LLM-based memory evaluation
├── tools/
│   └── registry.py      # Built-in tool registry
├── abilities/
│   ├── base.py          # Ability base class
│   ├── registry.py      # Ability management
│   ├── loader.py        # Auto-discovery & DB sync
│   ├── image_gen.py     # Together AI FLUX.1
│   ├── web_search.py    # Brave Search
│   ├── image_analysis.py # Gemini vision
│   └── maps.py          # Google Maps
└── channels/
    └── telegram.py      # Telegram adapter
```
