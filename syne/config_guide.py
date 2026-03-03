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

Automatically evaluates every message for memory-worthy content.
- **Enable when**: Owner wants the bot to learn from conversations without explicit "remember" commands.
- **Disable when**: Privacy is priority, or conversations are mostly transactional.
- **Positive**: Bot learns preferences, facts, and context over time without manual effort.
- **Negative**: Uses extra compute per message (evaluator model call). May store unwanted info.
  Non-permanent auto-captured memories decay naturally, so low-value ones will fade.

### Evaluator Model
| Key | Default | Type |
|-----|---------|------|
| `memory.evaluator_driver` | `"ollama"` | string |
| `memory.evaluator_model` | `"qwen3:0.6b"` | string |

Which model judges whether a message is worth remembering.
- `"ollama"` — local model, free, fast, no API cost. Recommended.
- `"provider"` — uses the main chat model. More accurate but costs tokens and adds 40s delay
  (to avoid rate limits on some providers).
- **Change when**: Auto-captured memories are low quality (switch to better model) or
  evaluator is too slow (switch to smaller/local model).

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

### Recall Limit
| Key | Default | Type |
|-----|---------|------|
| `memory.recall_limit` | `5` | integer |

Max memories injected into context per message.
- **Increase when**: Bot lacks context, forgets things mentioned recently.
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

Controls extended thinking/reasoning for supported models.
- `null` — use model's default behavior.
- `0` — disable thinking entirely (fastest, cheapest).
- `1024`-`24576` — thinking token budget (low to max).
- **Increase when**: Complex reasoning tasks, math, code, multi-step analysis.
- **Decrease when**: Simple conversations, cost reduction, faster responses.
- **Warning**: Higher budgets significantly increase token usage and response time.
  Not all models support thinking budgets — ignored by models without this feature.

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

### Mention Requirement
| Key | Default | Type |
|-----|---------|------|
| `telegram.require_mention` | `true` | boolean |

Whether bot requires @mention or reply to respond in groups.
- `true` — bot only responds when mentioned or replied to. Recommended for busy groups.
- `false` — bot responds to every message in the group.
- **Warning**: Disabling this in active groups will cause the bot to respond to every
  message, which is expensive and potentially annoying.

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

### Web Fetch Timeout
| Key | Default | Type |
|-----|---------|------|
| `web_fetch.timeout` | `30` | integer (seconds) |

HTTP request timeout for the `web_fetch` tool.
- **Increase when**: Fetching from slow servers or large pages.
- **Warning**: High timeouts delay bot responses while waiting for external servers.

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

## Provider & Model (Advanced)

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
| Decrease decay_interval | Medium — memories deleted faster | Partially — deleted memories cannot be recovered |
| Change chat model | Medium — different response quality | Yes — switch back anytime |
| Change embedding model | High — requires full re-embedding | Yes but slow (re-embedding takes time) |
| Set DM policy to "open" | High — anyone can message the bot | Yes — switch back to "approval" |
| Set group policy to "open" | High — bot responds in any group | Yes — switch back to "allowlist" |
| Disable require_mention | Medium — bot responds to everything in groups | Yes |
| Increase exec timeout | Low — owner-only tool | Yes |
| Disable subagents | Low — just disables feature | Yes |
"""
