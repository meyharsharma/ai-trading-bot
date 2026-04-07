# Lessons learned

Running log of non-obvious things we discovered building the agent. If you find
yourself solving the same problem twice, write it down here. New entries go at
the **top** so the freshest pain is the easiest to find.

---

## Kraken CLI (`kraken-cli` v0.3.0)

**Source:** https://github.com/krakenfx/kraken-cli — install with the official
shell script (`curl ... | sh`), lands in `~/.cargo/bin/kraken`. Verify with
`kraken --version` and `kraken ticker BTCUSD`.

### `kraken mcp` runs in `guarded` mode by default
The server boots with **38 tools**, and the only writable surface is the
`kraken_paper_*` family (`kraken_paper_buy`, `kraken_paper_sell`, etc.). There
is **no `kraken_add_order` exposed in guarded mode** — the server is physically
incapable of placing real orders. This is great for safety:
- We don't need a paper-vs-live config flag for the MCP path; guarded mode
  enforces it for us.
- `KrakenLiveExecutionAdapter` should target `kraken_paper_buy` /
  `kraken_paper_sell` by default.
- "Real money" mode would require an unguarded flag (TBD; not for the
  hackathon demo).

### Tool naming + arg shape gotchas
The MCP tools all use the `kraken_` prefix (not `add_order` style). And the
schemas don't always match what you'd guess from the Kraken REST docs:

| Tool | Surprise |
|---|---|
| `kraken_ticker` | takes `pairs` (**array**), not `pair` (singular). |
| `kraken_ohlc` | `interval` is a **string of minutes** (`"5"`, `"60"`, `"1440"`), not `"5m"` or `"1h"`. **No `limit` parameter** — use `since` for pagination. |
| `kraken_paper_buy` | `volume` and `price` are **strings**, not numbers. |
| `kraken_positions` | this is what our `get_open_positions` should resolve to (not `kraken_open_positions`). |
| pair format | `BTCUSD` works directly; you do **not** need `XBTUSD` translation when calling MCP tools. The CLI handles the asset code mapping internally. |

`additionalProperties: false` is set on every schema, so passing extra args
(like our old `limit=200`) **rejects the call**. Stick to exactly what the
schema declares.

### Paper trading is exposed via MCP
`kraken_paper_init`, `kraken_paper_buy`, `kraken_paper_sell`,
`kraken_paper_balance`, `kraken_paper_status`, `kraken_paper_history`,
`kraken_paper_orders`, `kraken_paper_cancel`, `kraken_paper_cancel_all`,
`kraken_paper_reset`. This is a real tradeable surface that runs against live
market data — strictly stronger than our in-process `PaperExecutionAdapter`
for the demo narrative ("we use Kraken's official paper engine").

---

## Claude Code as the LLM brain (no API key)

### `--bare` is the wrong flag for OAuth
The `--bare` option **strictly disables OAuth** and requires
`ANTHROPIC_API_KEY` or `apiKeyHelper`. If you're a Claude Pro/Max subscriber
logged in via `/login`, `--bare` will fail with "Not logged in". Use the
narrower flags instead:

```
--print --no-session-persistence --disable-slash-commands --max-turns 3
```

This gets you most of `--bare`'s noise reduction (no skill auto-load, no
session pollution) while keeping OAuth alive.

### `--max-turns 1` is too tight
The CLI's natural minimum is 2 turns (one for setup, one for the answer). Use
`--max-turns 3` or higher. `1` returns `error_max_turns` immediately.

### `--json-schema` enforces shape, not numeric bounds
You can give it `{"size_pct": {"type": "number", "minimum": 0, "maximum": 1}}`
and the model will happily return `25`. The CLI honors `enum`, `required`, and
shape, but treats numeric `minimum`/`maximum` as a hint. **Validate downstream
with pydantic** (which we do) — the schema is guidance, not a guarantee.

### The envelope has BOTH `result` and `structured_output`
When `--json-schema` is in play, the JSON envelope carries the parsed dict in
`structured_output` and a stringified copy in `result`. Prefer
`structured_output` — it skips a re-parse and is type-safe. Fall back to
`result` for older CLI builds.

### Parse the envelope BEFORE checking the exit code
`claude --print` often exits **non-zero AND** prints a perfectly good JSON
envelope on stdout (e.g. `{"is_error": true, "result": "Not logged in · Please
run /login"}`). If you bail at `returncode != 0` you'll lose the human-readable
message. Always: parse stdout first → check `envelope.is_error` → check
returncode last.

### Default user prompt fights JSON mode
Our shared prompts in `agent/brain/prompts.py` end with `"Call \`submit_decision\`
with your decision."` because they were designed for tool-use mode in the
Anthropic SDK. In CLI mode there is no tool, so the model interprets this as
"summarize what you submitted in plain English" and skips the JSON. Strip that
trailing line and append a JSON-only instruction. See
`claude_code_strategist._strip_tool_call_line()`.

### Max plan billing is safe… as long as `ANTHROPIC_API_KEY` is unset
- OAuth-authed `claude --print` calls **count against the Max 5-hour quota**,
  same as interactive sessions. Not separately billed.
- The `total_cost_usd` field in the envelope is **theoretical** (token usage at
  API rates). It's not what you're charged on a subscription.
- **TRAP:** if `ANTHROPIC_API_KEY` is set anywhere in the env (`.env`, shell
  rc, CI), the CLI silently routes everything to pay-per-token API billing,
  bypassing the Max quota entirely. Always run `echo "$ANTHROPIC_API_KEY"` to
  confirm it's empty before the agent goes live.
- Hitting the Max quota mid-loop just blocks until the window resets — the
  strategist gracefully falls back to HOLD on the rejected calls.

---

## Repository / worktree workflow

### Three parallel worktrees worked, with one critical rule
Splitting the four layers across three worktrees (`feat/kraken-mcp`,
`feat/brain-risk`, `feat/erc8004`) was the single biggest speed multiplier.
What made it work:

1. **Lock the contracts on `main` first.** `src/agent/state/models.py` was
   committed before any worktree was created. None of the three worktrees
   touched it (verified with `git diff origin/main -- src/agent/state/models.py`
   in each one before merging). Without that, the merge would have been hell.
2. **One owner per directory.** No two worktrees ever wrote into the same
   subpackage. The only collisions were `scripts/` (different files inside,
   no conflict) and `uv.lock` (identical, gitignored after the merge).
3. **Empty `__init__.py` files pre-created on `main`.** Each worktree could
   add to the package without fighting over module-init creation.

### Always TDD the risk gate
The risk gate is the only thing standing between the LLM's hallucinations and
real (or paper-real) capital. It got TDD'd from the start — and that's the
test file that's caught the most "the LLM said size 25%" type bugs in
backtests. Same energy applies to anything that wraps LLM output: don't trust
the model to honor numeric constraints, enforce them yourself.

### `uv.lock` should be gitignored if `pyproject.toml` is pip-flavored
Two of the three worktrees ran `uv pip install` which generated identical
`uv.lock` files. We're not standardizing on `uv` (pyproject is plain pip), so
those locks shouldn't be committed. Added to `.gitignore` post-merge.

---

## Generic engineering takeaways

### The cheapest debugging tool is dumping the raw payload
Every Kraken/Claude bug in this session was solved by running the underlying
command directly with `python3 -m json.tool` and reading the actual envelope.
Not by reading docs, not by reasoning about what *should* happen. Always:
1. Mock-replay the failure path in a test.
2. Run the real command directly and capture the raw output.
3. Diff the two.

### Graceful degradation > strict failure for autonomous loops
Every external call in the loop falls back to a conservative default:
- LLM call fails → HOLD this cycle
- Risk gate rejects → log + continue
- Exec fails → record decision, no fill, next cycle
- Chain submit fails → mark `unverified`, retry next tick
- Top-level cycle crash → record on cycle row, sleep, try again

A bot that crashes at 2am loses. A bot that HOLDs for an hour because Kraken
hiccupped just gives up some upside. Pick the second every time.

### Don't reach for skills before the code exists
We were tempted to scaffold a `trade-audit` Claude Code skill on Day 1. The
right call was to defer until after `state/store.py` existed, because the
skill would have invented schema details we didn't have yet. Skills are
better at being thin wrappers over real things than at sketching what the
real thing should look like.
