# Readiness fixes

This records the scoped response to the 27 October 2026 agent-memory-engine review, based on
the pre-publication source snapshot. It is not a full release-readiness audit.
The public documentation targets `tetriz-ai/agent-memory-engine`; the PyPI distribution and unique
CLI name are `tetriz-agent-memory-engine`. It is the only command; the Python import remains
`agent_memory_engine`.

## Findings

| Finding | Resolution |
|---|---|
| AME-B1 | Removed access-required wording and switched source instructions and their command check to the public repository. |
| AME-S1 | Kept the distinct PyPI name, made it the only CLI command, and documented installation from PyPI. |
| AME-S2 | Named the exact package and the local `pip install .` alternative in the examples. |
| AME-S7 | Added the 0.1.0 section and release link, with documentation written for the public release. |
| AME-S8 | Documented the public release sequence in CONTRIBUTING. |
| AME-S10 | Linked Gemini’s data-use terms and recommended a billing-enabled project for real transcripts. |
| AME-M1 | Retained `tetriz-ai` as project author. The author and publisher identity may differ from the copyright holder in LICENSE. |
| AME-M2 | Added copyright and SPDX headers to the example Python and shell scripts. |
| AME-M3 | Removed unshipped card references while retaining the worked examples and extraction rules. Prompt text changes invalidate cached responses keyed by its content. |
| AME-M4 | Replaced the command illustration’s number with `123`. |
| AME-M5 | Pointed the schema identifier at the raw public source file and updated its regression check. |
| AME-M6 | Described the measured prompt revision without scrub-related wording. |
| AME-M9 | Retained the generic, synthetic storefront/payments examples and their pinned outputs. |
| AME-M10 | Contributor agreement was confirmed by the project owner. |

The owner excluded GitHub settings and Actions, security changes, Dependabot and Code of Conduct
findings. AME-B2, B3, S3–S6, S9, M7, M8 and M11 are outside this change. The frontend-web-os
findings, including demo CI, are outside this repository’s scope.

## Validation scope

The checks cover synthetic sessions and stub providers. No real transcripts or live model calls are
used. `scripts/check-package.sh` builds and installs a wheel in isolation, checks that only the intended
command is installed, exercises it on a fixture and checks dependency licensing.
`scripts/run-examples.sh` builds a separate wheel and compares all four examples with their documented
output. The unit suite also executes the documented replay and Python usage examples.

The initial public release installs from its GitHub wheel; PyPI publication is deferred.
GitHub Actions is disabled by the project owner, and release checks run locally.
Full secret/history scanning, media review and the full independent release-review panels were not
repeated for this scoped fix. Validation and scoped review results are reported with the change.

## Initial public import

The 0.1.0 changelog contains a single initial release dated 28 October 2026. The README uses
static license, Python and platform badges. The bug-report template names the installed command.
Local validation passed 463 tests on Python 3.12 and 3.15, ruff lint and formatting, mypy,
fixture generation, all four examples, wheel installation and an independent source-archive
installation. Gitleaks 8.30.1 found no leaks in the publication tree. Hosted CI and live-provider
calls were not run. The existing banners and examples are retained; no new demo was created.
