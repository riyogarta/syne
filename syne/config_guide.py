"""Configuration Guide — operational manual for the bot.

Provides a comprehensive reference of all user-facing configuration keys,
organized by use case. Injected into the system prompt so the bot can
advise the owner on config changes with full awareness of impacts.

NOTE: Credential keys (credential.*) and internal counters (memory.conversation_counter,
update_check.*) are excluded — they are not user-facing operational configs.
"""

CONFIG_GUIDE = """\
# Configuration Guide

Use `update_config(action='list')` to see current values.
Use `update_config(action='set', key='...', value='...')` to change.
Always inform the owner of **both positive and negative impacts** before changing a config.

## Memory System

### Auto-Capture
| Key | Default | Type |
|-----|---------|------|
| `memory.auto_capture` | `false` | boolean |
| `memory.auto_evaluate` | `true` | boolean |

- `auto_capture` — automatically evaluates every message for memory-worthy content.
- `auto_evaluate` — use LLM to judge what is worth storing (only applies when auto_capture is ON).
  If false, auto_capture stores everything without evaluation.
- **Enable auto_capture when**: Owner wants the bot to learn from conversations without explicit "remember" commands.
- **Disable when**: Privacy is priority, or conversations are mostly transactional.
- **Positive**: Bot learns preferences, facts, and context over time without manual effort.
- **Negative**: Uses extra compute per message (evaluator model call). May store unwanted info.
  Non-permanent auto-captured memories decay naturally, so low-value ones will fade.

### Evaluator Model
| Key | Default | Type |
|-----|---------|------|
| `memory.evaluator_driver` | `"ollama"` | string |
| `memory.evaluator_model` | `"qwen3:0.6b"` | string |
| `memory.evaluator_models` | *(registry)* | JSON array |
| `memory.active_evaluator` | `"qwen3-0-6b"` | string |

Which model judges whether a message is worth remembering.
- `evaluator_driver` / `evaluator_model` — legacy shorthand (still works).
- `evaluator_models` — registry of available evaluator models (managed via `/evaluator` command).
- `active_evaluator` — key of the currently active evaluator from the registry.
- `"ollama"` driver — local model, free, fast, no API cost. Recommended.
- `"provider"` driver — uses the main chat model. More accurate but costs tokens.
- **Change when**: Auto-captured memories are low quality (switch to better model) or
  evaluator is too slow (switch to smaller/local model).
- **Warning**: Evaluator models must be pre-pulled in Ollama. Use `/evaluator` to manage.

### Memory Decay
| Key | Default | Type |
|-----|---------|------|
| `memory.decay_interval` | `50` | integer |
| `memory.decay_amount` | `1` | integer |
| `memory.initial_recall_count` | `1` | integer |

Controls how fast non-permanent memories fade.
- `decay_interval` — every N conversations, decay triggers. Lower = memories fade faster.
- `decay_amount` — how much `recall_count` decreases per decay cycle.
- `initial_recall_count` — starting durability of new memories. Higher = lasts more cycles.
- **Increase interval when**: Memories disappear too fast, bot forgets useful context.
- **Decrease interval when**: Too many stale memories accumulate, bot recalls irrelevant things.
- **Increase initial_recall_count when**: Auto-captured memories should last longer by default.
- **Warning**: With default values (interval=50, amount=1, initial=1), a memory that is never
  recalled will be deleted after 50 conversations. Frequently recalled memories get +2 boost
  each time, so useful memories survive naturally.

### Recall & Importance
| Key | Default | Type |
|-----|---------|------|
| `memory.recall_limit` | `5` | integer |
| `memory.max_importance` | `1.0` | float |

- `recall_limit` — max memories injected into context per message.
- `max_importance` — ceiling for importance scores (0.0–1.0).
- **Increase recall_limit when**: Bot lacks context, forgets things mentioned recently.
- **Decrease when**: Context window is tight, or too many irrelevant memories appear.
- **Warning**: Each recalled memory uses tokens. Setting too high wastes context window.

## Session & Compaction

### Compaction Threshold
| Key | Default | Type |
|-----|---------|------|
| `session.compaction_threshold` | `80000` | integer (chars) |
| `session.compaction_keep_recent` | `40` | integer (messages) |
| `session.max_messages` | `100` | integer |

Controls when conversation history is compacted (summarized) to save context.
- `compaction_threshold` — character count that triggers auto-compaction.
- `compaction_keep_recent` — number of recent messages preserved verbatim after compaction.
- `max_messages` — soft limit before suggesting compaction.
- **Increase threshold when**: Owner wants longer uncompacted conversations (needs large context model).
- **Decrease threshold when**: Running into context limits, responses slowing down.
- **Increase keep_recent when**: Bot loses too much recent context after compaction.
- **Warning**: Very high thresholds may cause API errors if the model's context window is exceeded.

### Thinking Budget
| Key | Default | Type |
|-----|---------|------|
| `session.thinking_budget` | `null` | integer or null |

Global fallback for extended thinking/reasoning. Per-model `thinking_budget` in `provider.models`
takes priority — this only applies when the per-model value is null.
- `null` — use model's default behavior.
- `0` — disable thinking entirely (fastest, cheapest).
- `1024`-`24576` — thinking token budget (low to max).
- **Increase when**: Complex reasoning tasks, math, code, multi-step analysis.
- **Decrease when**: Simple conversations, cost reduction, faster responses.
- **Warning**: Higher budgets significantly increase token usage and response time.
  Not all models support thinking budgets — ignored by models without this feature.

## Time & Date Formatting

| Key | Default | Type |
|-----|---------|------|
| `time.locale` | `"id"` | string |
| `time.format.full` | `"{day_name}, {date} {time}"` | string |
| `time.format.date` | `"{date}"` | string |
| `time.format.time` | `"{time}"` | string |
| `time.format.day` | `"{day_name}"` | string |

Controls how the runtime time context is formatted when injected into conversations.
- `time.locale` affects day/month names (currently `id` and `en`).
- Formats use Python `str.format` templates with fields: `{day_name}`, `{date}`, `{time}`, `{year}`, `{month}`, `{day}`, `{hour}`, `{minute}`, `{second}`.
- **Warning**: Invalid templates fall back to ISO/full string.

## System Timezone

| Key | Default | Type |
|-----|---------|------|
| `system.timezone` | `"UTC"` | string (IANA timezone) |

Controls what timezone the agent uses for:
- Time context injected into conversations (ground truth for LLM)
- Cron schedule interpretation ("0 9 * * *" = 9 AM in this timezone)

Examples: `"Asia/Jakarta"`, `"America/New_York"`, `"Europe/London"`, `"UTC"`
- **Change when**: Bot operator is not in UTC and wants local time awareness.
- **Warning**: Changing timezone does NOT retroactively adjust existing scheduled tasks'
  `next_run` values. Existing cron tasks will pick up the new timezone on their next
  recalculation (after the current `next_run` fires).

## Rate Limiting

| Key | Default | Type |
|-----|---------|------|
| `ratelimit.max_requests` | `4` | integer |
| `ratelimit.window_seconds` | `60` | integer |
| `ratelimit.owner_exempt` | `true` | boolean |

Prevents users from flooding the bot.
- **Increase max_requests when**: Users complain about being rate-limited in normal use.
- **Decrease when**: Bot is being abused or API costs are too high.
- **owner_exempt** — when `true`, owner is never rate-limited. Disable only if owner
  wants to test rate limiting behavior on themselves.
- **Warning**: Setting too high removes abuse protection. Setting too low frustrates users.

## Telegram Channel

### DM Policy
| Key | Default | Type |
|-----|---------|------|
| `telegram.dm_policy` | `"approval"` | string |

Controls who can DM the bot.
- `"approval"` — new users must be approved by owner before they can chat.
- `"open"` — anyone can DM the bot immediately.
- **Use "approval" when**: Bot is private/personal, owner wants to vet users.
- **Use "open" when**: Bot is a public service, open to all.
- **Warning**: "open" combined with permissive tool permissions could allow strangers
  to use expensive abilities. Review ability permissions before switching to open.

### Group Policy
| Key | Default | Type |
|-----|---------|------|
| `telegram.group_policy` | `"allowlist"` | string |

Controls which groups the bot responds in.
- `"allowlist"` — only responds in registered groups.
- `"open"` — responds in any group it's added to.
- **Warning**: "open" means anyone who adds the bot to a group gets access.
  Combined with public tool permissions, this could be a security risk.

### Mention & Trigger
| Key | Default | Type |
|-----|---------|------|
| `telegram.require_mention` | `true` | boolean |
| `telegram.bot_trigger_name` | `null` | string or null |

- `require_mention` — whether bot requires @mention or reply to respond in groups.
  `true` recommended for busy groups; `false` makes bot respond to every message.
- `bot_trigger_name` — custom name that triggers the bot (e.g., "Syne", "hey bot").
  `null` = auto-reads from `identity.name`. Set this if the bot should respond to a
  different name than its identity name.
- **Warning**: Disabling require_mention in active groups will cause the bot to respond
  to every message, which is expensive and potentially annoying.

## Execution & Tools

### Exec Limits
| Key | Default | Type |
|-----|---------|------|
| `exec.timeout_max` | `300` | integer (seconds) |
| `exec.output_max_chars` | `4000` | integer |

Safety limits for the `exec` tool (shell command execution).
- **Increase timeout when**: Running long tasks (compilation, large file processing).
- **Decrease timeout when**: Want faster failure for stuck commands.
- **Increase output_max_chars when**: Commands produce large output that gets truncated.
- **Warning**: Very high timeouts could let a runaway process block the bot for minutes.
  The exec tool is owner-only (permission 700) so this is a self-imposed risk.

### Web Tools
| Key | Default | Type |
|-----|---------|------|
| `web_search.api_key` | `""` | string |
| `web_search.driver` | `""` | string |
| `web_fetch.timeout` | `30` | integer (seconds) |

- `web_search.api_key` — Web search API key. Supports Tavily (tvly-...) or Brave Search.
  Tavily: get a free key at https://app.tavily.com (1,000 searches/month free).
  Brave: https://brave.com/search/api/ (paid).
  Empty string = web search disabled.
- `web_search.driver` — `"tavily"` or `"brave"`. Auto-detected from key prefix if empty.
- `web_fetch.timeout` — HTTP request timeout for the `web_fetch` tool.
- **Increase timeout when**: Fetching from slow servers or large pages.
- **Warning**: High timeouts delay bot responses while waiting for external servers.

### File Operations
| Key | Default | Type |
|-----|---------|------|
| `file_ops.max_read_size` | `102400` | integer (bytes) |

Maximum file size the bot can read via the `read_file` tool (default 100KB).
- **Increase when**: Owner needs the bot to read larger files (logs, data files).
- **Decrease when**: Want to limit memory usage from large file reads.
- **Warning**: Very large values may cause context window overflow or slow responses.

### Voice / Speech-to-Text
| Key | Default | Type |
|-----|---------|------|
| `voice.stt_provider` | `"groq"` | string |
| `voice.stt_model` | `"whisper-large-v3"` | string |

Controls how voice messages are transcribed.
- `"groq"` — fast, free-tier available via Groq Cloud. Requires `credential.groq_api_key`.
- `stt_model` — Whisper model variant to use.
- **Change when**: Current provider is down, or want to use a different transcription service.
- **Warning**: Requires a valid API key for the chosen provider.

## Sub-Agents

| Key | Default | Type |
|-----|---------|------|
| `subagents.enabled` | `true` | boolean |
| `subagents.max_concurrent` | `2` | integer |
| `subagents.timeout_seconds` | `900` | integer (15 min) |

Controls background task delegation.
- **Disable when**: Not using sub-agents, want to save resources.
- **Increase max_concurrent when**: Multiple background tasks needed simultaneously.
- **Decrease timeout when**: Sub-agents should fail faster if stuck.
- **Warning**: Each sub-agent uses its own LLM calls. More concurrent = more API cost.

## Scheduler

| Key | Default | Type |
|-----|---------|------|
| `scheduler.misfire_grace_seconds` | `300` | integer |

Grace period for scheduled tasks that run late (e.g., bot was offline).
- **Increase when**: Bot may be offline for extended periods but tasks should still run.
- **Decrease when**: Time-sensitive tasks should be skipped if late.
- **Warning**: A large grace period means old tasks execute when the bot comes back online,
  which might surprise the user if the context has changed.

## Provider & Model

### Active Models
| Key | Default | Type |
|-----|---------|------|
| `provider.active_model` | `"gemini-pro"` | string |
| `provider.active_embedding` | `"together-bge"` | string |

Active chat and embedding models. These reference keys in `provider.models` and
`provider.embedding_models` registries.
- **Change chat model when**: Owner wants different quality/cost tradeoff, or current model
  has issues (rate limits, downtime).
- **Change embedding model when**: Switching embedding providers. **CRITICAL**: changing
  embedding model requires re-embedding ALL memories. This is automatic but takes time
  and may temporarily affect memory recall quality.
- **Warning**: Changing models mid-conversation does not affect existing sessions —
  only new sessions use the new model.

### Model Registries
| Key | Type |
|-----|------|
| `provider.models` | JSON array |
| `provider.embedding_models` | JSON array |

Full model registries. Each entry contains: key, label, driver, model_id, auth, context_window,
params (temperature, max_tokens, thinking_budget, top_p, top_k, frequency_penalty, presence_penalty),
and reasoning_visible.
- Managed via `/model` and `/embedding` commands — avoid editing raw JSON directly.
- Each model has its own `thinking_budget` and `reasoning_visible` per-model settings.
- **Warning**: Invalid JSON in these registries will break model selection.

### Legacy Provider Keys
| Key | Default | Type |
|-----|---------|------|
| `provider.primary` | `{"name": "google", "auth": "oauth"}` | JSON object |
| `provider.chat_model` | `"gemini-2.5-pro"` | string |
| `provider.embedding_model` | `"text-embedding-004"` | string |
| `provider.embedding_dimensions` | `768` | integer |

Legacy keys from before the model registry system. Still read as fallback by some components.
- `provider.primary` — primary LLM provider name and auth method.
- `provider.chat_model` / `provider.embedding_model` — model identifiers (superseded by registries).
- `provider.embedding_dimensions` — vector dimensions for embeddings.
- **Do not change directly** — use `/model` and `/embedding` commands instead.

### Claude OAuth Client ID
| Key | Default | Type |
|-----|---------|------|
| `claude.oauth_client_id` | *(built-in)* | string (UUID) |

Overrides the OAuth client_id used for Claude/Anthropic OAuth flows.
- Default uses the built-in client_id — no need to set this unless customizing.
- **Change when**: Owner wants to use a different OAuth client registration.
- **After changing**: must re-authenticate Claude OAuth via `/models` (existing tokens are tied to the old client_id).
- **Warning**: Using Claude OAuth in third-party applications may violate Anthropic's Terms of Service.

### Auth Refresh (Advanced)
| Key | Default | Type |
|-----|---------|------|
| `auth.refresh_buffer_seconds` | `600` | integer (10 min) |
| `auth.refresh_interval_seconds` | `1800` | integer (30 min) |

Controls OAuth token refresh timing.
- `refresh_buffer_seconds` — how early before expiry to refresh tokens (safety margin).
- `refresh_interval_seconds` — how often the background refresh loop runs.
- **Increase buffer when**: Token refresh is failing intermittently (gives more retry time).
- **Decrease interval when**: Tokens expire quickly and need more frequent refresh.
- **Warning**: Setting buffer too low risks token expiry mid-conversation.
  Setting interval too high may miss refresh windows.

## WhatsApp Bridge

| Key | Default | Type |
|-----|---------|------|
| `whatsapp.wacli_path` | `"wacli"` | string |
| `whatsapp.allowed_chat_jids` | `[]` | JSON array |

WhatsApp integration settings.
- `wacli_path` — path to the wacli binary. Only change if installed in non-standard location.
- `allowed_chat_jids` — whitelist of WhatsApp chat JIDs the bot monitors.
  Empty = no WhatsApp monitoring.
- **Warning**: Adding JIDs to the allowlist means the bot will read and respond to those chats.

## Impact Summary Table

For quick reference when deciding permission level:

| Change | Risk | Reversible |
|--------|------|------------|
| Enable auto_capture | Low — memories decay naturally | Yes — disable anytime |
| Increase decay_interval | Low — memories last longer | Yes |
| Decrease decay_interval | Medium — memories deleted faster | Partially — deleted memories gone |
| Change chat model | Medium — different response quality | Yes — switch back anytime |
| Change embedding model | High — requires full re-embedding | Yes but slow |
| Change timezone | Low — affects new cron calculations | Yes — switch back anytime |
| Set DM policy to "open" | High — anyone can message the bot | Yes — switch back to "approval" |
| Set group policy to "open" | High — bot responds in any group | Yes — switch back to "allowlist" |
| Disable require_mention | Medium — bot responds to everything in groups | Yes |
| Increase exec timeout | Low — owner-only tool | Yes |
| Disable subagents | Low — just disables feature | Yes |
| Change auth refresh timing | Medium — may cause token expiry | Yes — restore defaults |
"""
