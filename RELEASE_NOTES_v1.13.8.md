# Syne v1.13.8 — Long-term memory becomes the product

Since the previous release (v0.3.0 in February), Syne has been
re-grounded around a single, sharp positioning:

> **Self-hosted personal AI assistant on Telegram with persistent long-term memory.**

This release captures roughly nine months of work in one cut. Most of
the headline features below didn't exist in v0.3.0.

---

## 🧠 Memory system

- **Knowledge graph** alongside semantic vectors — entity-relation
  extraction for permanent memories, queried in parallel with pgvector
  during recall.
- **Memory file attachments** — store binary files (images, PDFs,
  voice notes, Office docs, up to 50 MB) as BYTEA blobs linked to a
  memory by FK. Description is embedded for search; the file is
  preserved for retrieval (`memory_get_file`) or re-analysis
  (`memory_analyze_file`).
- **Decay & conflict resolution** — non-permanent memories fade with
  disuse (recall_count counter, configurable interval). New memories
  similar to existing ones are routed to insert / update / skip based
  on similarity threshold + source-priority rules.
- **Anti-zombie defense** — 3 layers: SQL filter on recall (skip rows
  with NULL embedding), Python null-guard, refuse to insert rows
  without a vector.
- **Rule 760 / 765** — memories are private by default; specific
  categories can be opened for public access via
  `memory.public_categories` config.

## 📄 Document I/O

- **PDF**: read (text extraction + vision OCR fallback for scanned /
  CAD pages, via the configured vision provider) and create
  (text → PDF, URL → PDF, comparison tables).
- **Office**: read & create `.docx`, `.xlsx`, `.pptx`. Created files
  use a bundled skeleton template — Syne auto-picks the right layout
  per slide (cover / section / two-column / icon rows / stat callouts /
  2x2 grid / timeline / comparison / closing) from a 9-layout library.

## 🤖 Multi-provider LLM

- **7 drivers**: Anthropic (OAuth + API key), Google CCA, Vertex AI,
  Codex (ChatGPT), OpenAI-compatible, Together AI, Ollama.
- **Adaptive thinking** for Anthropic Opus 4.6 / Sonnet 4.6.
- **Prompt caching** on Anthropic system blocks + last user message.
- **HTTP/2 + persistent client** + transient-429 recovery (recreate
  client on each retry, longer backoff).
- **Per-user and per-group model overrides** via `/models` Telegram
  menu.

## 🎙️ Voice in & out

- **Inbound voice notes** transcribed via Groq Whisper (default
  `whisper-large-v3`). Original audio preserved in metadata so
  `memory_store_file` can save the voice recording as a blob.
- **Outbound voice replies** via Edge TTS (free, local). Voice picker
  per personality.

## 🛡️ Security & privacy

- **`SECURITY.md`** with threat model, permission tiers, prompt
  injection mitigations, and hardening checklist.
- **Self-modification OFF by default** —
  `abilities.self_modification_enabled` gates
  `update_ability(action='create')`. Closes the prompt-injection-to-RCE
  vector. Existing custom abilities continue to run.
- **Multi-tier permissions**: owner / family / public with Linux-style
  3-digit octal per tool. Group chat filter strips owner-only tools
  from the LLM's schema.

## ⏰ Scheduler & automation

- **Group reminders work** — scheduled tasks track `target_chat_id`
  and `target_chat_type`. Payload is wrapped at fire time with explicit
  delivery instruction so the LLM knows where to send.
- **Cron / interval / once** schedule types with optional end date.

## 🔌 Remote nodes

- **WebSocket gateway** — extend Syne to multiple machines. Each node
  has its own CLI, one shared memory, controlled from one Telegram bot.
- **Auto-reconnect** with exponential backoff, survives sleep/wake.

## 📦 Other extensions

- Image generation (Vertex Imagen / Together FLUX / OpenAI DALL-E).
- Image analysis with vision provider selection (`/vision`) — Vertex /
  OpenAI / Ollama.
- Web search (Tavily + Brave) with throttle + retry.
- Maps (Google Maps API).
- WhatsApp bridge (via `wacli`).
- Website screenshot (Playwright).

## 🛠️ Infrastructure & ops

- **Versioned schema migrations** —
  `schema_version` integer in config + ordered MIGRATIONS list with
  advisory lock. Complements the idempotent `schema.sql`.
- **CI parity test** — fresh-install schema vs upgrade-path schema
  must match (catches dual-path drift).
- **Three-layer migration**: install / `syne update` / startup all run
  schema migrations.
- **Auto-pull Ollama models** during install + update for abilities
  that need them.
- **`syne repair`** — diagnose + auto-fix common issues, including
  zombie memories (rows with NULL embedding).
- **Bundled office templates** — `skeleton.docx` and `skeleton.pptx`
  shipped with the package.

## 🤝 Telegram-specific

- **Smart message handling** — Indonesian/English language detection,
  voice notes, photos, documents, replies, group mentions.
- **`/vision`, `/imagegen`, `/models`, `/embedding`, `/evaluator`,
  `/graph`, `/groups`, `/members`, `/nodes`, `/backup`, `/restore`,
  `/update`** — owner menus for runtime configuration.
- **Inline approval** for new users and group memberships.

## 🐛 Notable bug fixes

- KG extraction now correctly parses tool calls from Vertex/Gemini
  drivers (function.arguments JSON string format).
- Memory engine: zombie rows with NULL embedding no longer crash
  `memory_search` with `TypeError: '>=' not supported between
  NoneType and float`.
- Anthropic transient 429 errors retried with fresh HTTP client.
- Telegram delivery timeouts now correctly labeled (not blamed on the
  LLM).
- Token refresh on Anthropic OAuth recovers from transient failures
  (uses expired token, lets 401 handler refresh on actual API call).
- `office` ability handles Arabic glyphs with proper Complex Script
  font + RTL settings.
- KG recall optimized 188ms → 12ms via two-step query + GIN trigram
  index; memory + graph recall now run in parallel.

---

## ⚙️ Honest framing

This release also rewrites the positioning. Previously Syne was
described as an "AI agent framework with unlimited memory" — which
over-promised. Decay forgets; "unlimited" was wrong. "Anti-hallucination"
was misleading (the filter operates at memory write time, not at
chat-model read time). Both are now corrected to accurate, narrower
claims.

The cost framing has also been corrected: the resilience floor is
Ollama local + paid API keys. OAuth gray-zone providers (Google CCA,
Anthropic OAuth, Codex) remain supported but framed as bonus for
personal use, not foundation.

---

## 🔄 Upgrade

Existing installs:

```bash
syne update
```

Schema migration runs automatically. The new `memory_blobs` table is
created via `CREATE TABLE IF NOT EXISTS`. Service restarts at the
end of update; `_run_startup_migration` re-applies the schema and runs
any pending versioned migrations on the next boot.

Existing custom abilities continue to work. To allow the LLM to create
new ones, the owner must now explicitly enable:

```
update_config(key='abilities.self_modification_enabled', value='true')
```

See [`SECURITY.md`](SECURITY.md) before turning that on.

---

**Full changelog:**
[`v0.9.2...v1.13.8`](https://github.com/riyogarta/syne/compare/v0.9.2...v1.13.8)
(618 commits — 137 features, 341 fixes, 5 perf improvements,
30 documentation updates.)
