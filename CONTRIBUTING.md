# Contributing

## Set up

You need [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer, on macOS or Linux.

```sh
uv sync --locked
```

## Checks

Run these checks locally before submitting a pull request:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
scripts/run-examples.sh
scripts/check-package.sh
uv run python tests/fixtures/make_storefront_session.py --check
```

The linter is ruff 0.16.9 and the type checker is mypy 2.3.1, both pinned in the `dev` group of
`pyproject.toml`, so `uv run` uses the versions CI uses. Fix what they report rather than silencing
it. A rule that has to be off is off in `pyproject.toml`, with a comment saying why.

`uv run pytest` runs the suite offline. It covers the evidence gate, authorship, merge, the schema,
the cache and the command's exit codes. `scripts/run-examples.sh` runs the examples, and
`tests/test_readme.py` checks that the README and `docs/usage.md` show what their commands print.

No test calls a model or reads your real sessions. Every test runs with a temporary `HOME`, and
the network and subprocesses are blocked (`tests/conftest.py`). Use the stub clients in the tests,
or `tests/scripted.py`, for anything that would call a model.

## Changes

- Every behaviour change comes with a test that fails before the change and passes after it.
- A change a user can notice adds a line under `[Unreleased]` in `CHANGELOG.md`.
- A change to a prompt under `src/agent_memory_engine/prompts/` changes what the model returns and
  what a run costs. Say in the pull request how you checked it.
- Commit messages say what changed and why, in the imperative.
- A maintainer releases a tag `vX.Y.Z` equal to `__version__`, after running all checks above.
  Move the `[Unreleased]` entries into the new version’s changelog section before tagging,
  run `uv build`, and attach the wheel and source archive to a GitHub Release from the tag.
  GitHub Actions is disabled; release checks run locally.

## Licensing of contributions

No sign-off line and no separately signed agreement are needed. By opening a pull request you
agree that your contribution is released under the project's [LICENSE](LICENSE).
