# List sessions in a date range

This is the README's "See what a run would read" task. It lists the sessions a run with the same
flags would read, with each one's turns and windows, and calls no model.

The script copies a small sample session home into a temporary `HOME`, so the output is the same on
every machine.

The storefront session holds a pasted CI log, so it spans three windows. A window holds up to 10,000
characters of whole turns, and a turn over 8,000 characters keeps its first 6,000 and last 1,500.

To run it, clone the repo and, from its root, run
`uv run examples/list-sessions-in-a-date-range/run.sh`. uv installs the package first.

Put the command on your `PATH` with `uv tool install .` from a clone. Then run:

```sh
examples/list-sessions-in-a-date-range/run.sh
```

```text
DATE        PROJECT                              SESSION                               TURNS  WINDOWS
2026-09-03  /home/dev/src/shop                   7d3c9a10-0b1e-4c55-9a8e-2f6b1c0d4e11      7        1
2026-09-08  /home/dev/src/storefront             3f6a2c81-9d47-4b0e-a5c3-1e8b7d2f4a60      8        3
2026-09-15  /home/dev/src/shop/services/billing  c41f8e22-6b9d-4f3a-8d71-90e5a2b7c013      4        1
2026-09-20  /home/dev/src/shop-admin             e2a7d5b0-3c18-4e6f-9b42-7f0c1d8a6e35      2        1
2026-10-01  -home-dev-notes                      5b9e0c47-a1d2-4f83-b6c5-d4e3f2a1b0c9      2        1
5 sessions, 7 windows.
```
