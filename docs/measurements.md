# Measurements

Every measured number in the README and `docs/` comes from one of the measurements on this page. They were
taken on real Claude Code sessions, and those session sets are not published.

## What was compared

- **The default:** `gemini-3.7-flash` at thinking level `medium` on both extraction arms and the
  merge, with explicit prompt caching on.
- **The comparison:** the same pipeline at thinking level `high`, with no cache.
- The prompts and the output schema were byte-identical in both. Only the thinking level and the
  cache differed.

## The session sets

| Set | Source | Windows |
|---|---|---|
| The 72-window set | 30 sessions, one developer's | 72 |
| The 239-window set | 30 sessions, another developer's | 239 |
| The long-session run | The first developer's five most recent sessions, about 19 windows each | 96 |

All three runs used the same response schema. The two 30-session sets used an earlier prompt revision
with a different illustrative name in the reconcile and merge examples; the extraction and merge
instructions were otherwise unchanged.

## Cost

Cost is each call's billed tokens (input, cached input, and output with thinking) at Gemini's price
card as read on 2026-08-26, summed over every call of all three stages. A call answered from the
local cache is counted at its original price, so the figure is what a fresh run would pay. Holding
the cached prompts is billed separately, as `explicit_cache_storage_usd`.

| Set | Default | Comparison |
|---|---|---|
| The 72-window set | $1.29 | $3.09 |
| The 239-window set | $4.51 | $11.38 |

The default cost 42% and 40% as much as the comparison. It also found fewer memories:

| Set | Default | Comparison |
|---|---|---|
| The 72-window set | 113 memories | 172 memories |
| The 239-window set | 343 memories | 524 memories |

Per window, the default cost 1.8 cents on the 72-window set, 1.9 cents on the 239-window set and
2.9 cents on the long-session run. On that run, the facts arm spent 2 to 4 times as many thinking
tokens per call as on the two sets, and found more memories per window.

## Matching two memories

Several measurements below ask whether a memory from one run "is" a memory from another. Two
memories match when the cosine similarity of their statements' embeddings is at least 0.80.

## Blind grade

This asks whether the memories that only one setting finds are worse than the ones only the other
finds. It used the merged outputs of the 239-window set.

1. Memories with no match in the other setting formed two piles: 195 from the default, 376 from the comparison.
2. Each pile was sampled to 60, stratified by `type`, and the 120 were shuffled together.
3. Each showed its statement, description, types, generality, scope and one quote, with no setting named.
4. `claude-opus-5`, through `claude -p` with no tools, graded each as keep or not useful, valued 1 to 5.

| Pile | Keep rate | Mean value |
|---|---|---|
| Found only by the default | 93.3% | 3.58 |
| Found only by the comparison | 95.0% | 3.48 |

The keep-rate difference is −1.7 percentage points, with a 95% confidence interval of −10.0 to
+6.7. The mean-value difference is +0.10, with a standard error of 0.14 (t = 0.72). Neither is
statistically significant.

The grader kept about 95% of everything, so this grade would catch a large quality gap and could
miss a small one.

## Run-to-run variance

**Extraction.** The facts arm was run twice with identical settings, at temperature 0, over the five
sessions with the most windows in the 239-window set. The second run changed only the local cache
key, so every call was made again with the same request. 46.6% of the memories matched. The two
runs' costs differed by 2.3%, with 176 and 165 memories. This is one pair, on one arm.

**Merge.** The same 218 extracted memories, merged at levels `medium` and `high`, gave 181 memories
each, and 95.6% of them matched.

## Narrow decisions

On those same five sessions, the default's facts arm recovered 32.3% of the comparison's memories
overall, but 20% of its `decision` memories about one specific subject (generality
`instance-specific`). That gap of about 12 percentage points rests on 30 memories, and is within
the run-to-run variance above. It showed up in two runs of the default.
