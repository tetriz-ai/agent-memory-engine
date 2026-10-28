# Limitations

These limits matter before you load the memories into an agent or share them. Each one is inherent
to the design and pinned by a test, or measured as [measurements.md](measurements.md) describes.

## Memories

### Keeps more than it should (can leak or mislead)

**Quotes are verbatim.** Memories carry exact quotes from your sessions, secrets included if you
pasted any. The prompts tell the model to skip secrets, but no code removes them. Treat the
memories as you treat the transcripts. Pinned by `VerbatimQuotesAreKept` in
`tests/test_contract.py`.

**`speaker: user` is the slot, not the typist.** Text in a user turn that the person did not type
still has `speaker: user`. To keep only what they typed, require `authorship == "human_prose"` on
every quote, as the second filter in [usage.md](usage.md#pick-memories-by-tag) does. Pinned by
`ParsedAuthorship` and `MixedToolResultAndText` in `tests/test_contract.py`.

### Misses (loses a lesson)

**Long sessions.** A session is read only up to its first 300,000 characters, which is 30 windows
or more. A turn over 8,000 characters keeps only its first 6,000 and last 1,500. Pinned by
`TranscriptBudget` and `TurnTruncation` in `tests/test_transcript.py`.

**Long sub-agent reports.** A quote from the cut middle of a report over 8,000 characters fails the
evidence gate. A quote from its kept head or tail is labelled
`template_boilerplate`, so its `provenance.confidence` is low. Pinned by `LongSubAgentReports` in `tests/test_contract.py`.

**Run-to-run variance.** Two identical runs of the facts arm over five sessions shared 46.6% of their
memories. Run-to-run differences alone do not establish a regression
([method](measurements.md#run-to-run-variance)).

## Where your transcripts go

| Where | What goes there |
|---|---|
| Your machine | `sessions`, the split into windows, the evidence gate and the grouping of similar memories all run locally. `sessions` sends nothing anywhere. |
| Gemini | Each window of transcript text goes to `generativelanguage.googleapis.com`, the code's only network endpoint. When the run combines similar memories, the memories follow, with up to three quotes each. |
| Gemini's prompt cache | The system prompt, never a transcript. It is deleted when the run ends, or expires within an hour if the delete fails. |
| Your `claude` CLI | The same text as Gemini gets. The CLI decides where it goes. |
| Anywhere else | Nothing. There is no telemetry and no other upload. |

On Gemini’s unpaid tier, Google may use submitted content to improve its products, and human
reviewers may read it. Use a key from a billing-enabled project for real transcripts and review
[Gemini’s data-use terms](https://ai.google.dev/gemini-api/terms), including regional exceptions.
Paid services still retain data for abuse monitoring as described there.

## Platform and providers

- **Windows is not supported,** because the tool uses POSIX file locks.
- **The `claude` provider is not measured** for cost or quality. It prints the cost the CLI reports.
- **The measurements cover the pinned Gemini model only.** `--model` runs another one unmeasured.
