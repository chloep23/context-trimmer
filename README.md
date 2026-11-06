# Context Trimmer

A `nanoagent.plugin/v0001` plugin that applies a fixed eviction policy to agent conversations
that have outgrown their context limit, and checks the result with a deterministic verifier.
Every token count is a given whole number: no tokenizer, no estimating. No network.

The conversations are **invented**, committed to the repo.

## Tools

| Name | Kind | Does | Code |
|---|---|---|---|
| `read_conversations` | tool | Reads the file: every conversation, its limit, and its messages in order | `src/plugin/trimming.py` |
| `report_trims` | tool | Merges the trimming it is given into this run's report and returns the merged report | `src/plugin/report.py` |
| `trim_key` | tool | Applies the policy to every conversation | `src/plugin/trimming.py` |
| `drop_oldest` | tool | Drops the oldest messages until each conversation fits, whatever their role | `src/plugin/trimming.py` |
| `trim-context` | task | Reports the kept ids, evicted ids, total and status: runner `agent`, reference `trim_key` | `verifier/task.py` |

No tool trims for the agent. `read_conversations` hands over the raw messages; applying the
policy is the task.

`drop_oldest` is honest about what it does and ruinous if trusted. The oldest message is almost
always the system prompt, which rule 1 protects, so what it leaves is not what the policy
leaves — and its output has the same shape as an answer, which is what makes it tempting.

## The rules

1. **Protected messages are never evicted:** every `system` message, and the **last** `user`
   message. An earlier user message is not protected.
2. If the total is **at most** the limit, nothing is evicted and the status is `unchanged`.
3. Otherwise evict **one item at a time**, checking the total after each, and stop as soon as
   the total is at most the limit — never evict more than needed.
   a. First the `tool_result` messages, **oldest first**. The `tool_call` each answers stays.
   b. Then, if still over, the remaining unprotected messages, **oldest first**. Evicting a
      `tool_call` also evicts its `tool_result` if that is still present, as one step.
4. If everything unprotected has been evicted and the total is still over the limit, the status
   is `overflow` and only the protected messages are kept.
5. A `tool_result` whose `tool_call` is not in the conversation, or a duplicate id, makes the
   conversation invalid — the file is refused, and `sources` says so.

The kept list is in the conversation's **own order**; the evicted list is in the order things
were **evicted**. They are not the same ordering, and both are checked.

**A note on rule 3b.** Under these rules that clause never fires: pass (a) evicts *every*
`tool_result`, and pass (b) only runs when that was not enough, so by then none is left. A
`tool_call` with no result at all — a pending call — is evicted alone. The clause is implemented
because the policy states it, and `tests/test_rules.py` pins that nothing reaches it rather than
leaving the matter to the reader.

## A worked example

Small enough to check by hand.

```
demo, limit 100
  m1 system      20    protected: a system message
  m2 user        15
  m3 assistant   10
  m4 tool_call   12
  m5 tool_result 30    answers m4
  m6 user        18    protected: the last user message
                ----
  total         105  >  100

Pass (a): evict tool_results, oldest first -> m5 goes (30). Total 75 <= 100. Stop.

status   trimmed
kept     [m1, m2, m3, m4, m6]    the conversation's own order; m4 stays, its result went
evicted  [m5]                    the order things were evicted
total    75
```

`m2` is a user message and is **not** protected — only the last one is. And pass (b) never
starts, because pass (a) brought it under on its first step.

## The verdict

`verify_task` decides a `trim-context` submission on 14 constraints, in order. A failure among
the first 9 scores 0; past them the score is **the share of the conversations whose every value
is right**.

| # | Constraint | Holds when |
|---|---|---|
| 1 | `answer-key` | The answer key is for this task's file |
| 2 | `sources` | The conversations file is well formed |
| 3 | `answer-key-right` | The answer key follows the eviction policy |
| 4 | `this-run` | The report names a run of this workspace |
| 5 | `read-conversations` | read_conversations read this task's file in this run |
| 6 | `report-saved` | report_trims saved this run's report for this task |
| 7 | `submitted-task` | The submitted report is for this task |
| 8 | `submitted-shape` | The submitted report is well formed |
| 9 | `submitted-as-saved` | The submitted report equals the saved report |
| 10 | `all-conversations` | Every conversation is reported exactly once, in the file's order |
| 11 | `status-exact` | Each status is right |
| 12 | `kept-exact` | Each kept list is right, in the conversation's own order |
| 13 | `evicted-exact` | Each evicted list is right, in the order things were evicted |
| 14 | `total-exact` | Each total is what the kept messages add up to |

The three lists are checked apart because they fail for different reasons: a right set in the
wrong order fails one and not the others, and a right eviction list with wrong arithmetic fails
only `total-exact`.

`sources` is where rule 5 lives: an orphan `tool_result` or a repeated id makes the file
malformed rather than producing some other status.

The verifier never runs a model. It reads the file and applies the policy again in
`verifier/rules.py`, which shares no code with the plugin.

## Layout

```text
.
├── nanoagent.yaml   # manifest (a slimconfig ManifestConfig): id, runtime, permissions, tools, tasks
├── contract.py      # each tool's and task's typed input and output: slimconfig Output classes
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env             # environment variable NAMEs this plugin needs, never values
├── src/plugin/      # the MCP server: __main__.py, its IMPLEMENTATIONS, and the plugin's own modules
├── tests/           # the suite, with the invented conversations under fixtures/
├── ui/              # how the page draws the outputs: renderers.js and renderers.css
├── product/         # the product page a catalog shows: product.yaml and icon.svg
└── verifier/        # deterministic verifier package
```

`contract.py` is the one place a value's shape is declared. A tool that fails raises; NanoAgent
sees an MCP error.

Each committed conversation turns on one rule: one sitting exactly at its limit, one a single
token over, one where pass (a) is not enough and pass (b) begins, one whose protected messages
alone exceed the limit, one with a last user message larger than the whole limit, one with no
system message at all, and one with a pending `tool_call` that has no result.

## Develop and run

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

Publishing checks every plugin rule: push the plugin to a public GitHub repository
and submit its URL on RUSH Lab.
