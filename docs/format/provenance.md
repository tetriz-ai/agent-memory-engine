# provenance

`provenance` is `{confidence, corroboration}`. Code writes both. They label a memory and
never drop one.

## authorship and confidence

Every quote carries `authorship`, the author of the quoted bytes. Code reconstructs it from Claude
Code's own event fields, such as how the prompt was sent and whether the turn is a tool result.

| Authorship | Means |
|---|---|
| `human_prose` | The person typed it |
| `model_generated` | The model wrote it, or a sub-agent's report |
| `tool_output` | A tool's output |
| `template_boilerplate` | Text injected by the harness, a command template, a skill body |
| `unknown` | No reliable signal |

`authorship` is `null` when code could not attribute the quote's turn at all. Confidence treats it
like `unknown`.

`confidence` compares the quotes' authorship with the memory's type.

| Value | When |
|---|---|
| `high` | Every quote is admissible. For `preference`, `convention` and `intention` only `human_prose` is admissible. For the other types, `model_generated` and `tool_output` are too. |
| `uncertain` | No quote is inadmissible, but at least one has no signal (`unknown` or `null`) |
| `low` | At least one quote is inadmissible, such as harness text, or a tool output behind a preference |

## corroboration

| Field | Means |
|---|---|
| `distinct_conversations` | How many sessions back the memory. Only this counts as independent support. |
| `distinct_arms` | How many arms produced a member |
| `source_conversation_ids` | Those sessions, sorted |
| `source_arms` | Those arms, in the order `facts`, `preferences` |

`memories.jsonl` is ordered by `distinct_conversations`, then `distinct_arms`, highest first.

A value looks like:

```json
{"provenance": {"confidence": "high",
                "corroboration": {"distinct_arms": 2, "distinct_conversations": 1,
                                  "source_arms": ["facts", "preferences"],
                                  "source_conversation_ids": ["cc:7d3c9a10-0b1e-4c55-9a8e-2f6b1c0d4e11"]}}}
```

To pick the memories that both arms found, for example with `jq`:

```sh
jq -r 'select(.provenance.corroboration.distinct_arms == 2) | .statement' examples/find-memories-by-tag/memories.jsonl
```

```text
In shop, run pnpm install --frozen-lockfile in each worktree instead of symlinking node_modules from the main checkout, because Turbopack refuses a node_modules outside the project root.
```
