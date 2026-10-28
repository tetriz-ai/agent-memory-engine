# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""Pick memories out of a run's memories.jsonl by their typed tags and type.

memories.jsonl here is the output of the extract-from-a-recorded-session example, copied as is.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "memories.jsonl"), encoding="utf-8") as f:
    memories = [json.loads(line) for line in f]


def find(domain=None, activity=None, concept=None, types=None):
    for m in memories:
        # Only memories safe to load as they are: no intention waiting for review, nothing borderline.
        if m["review"]["status"] != "active" or m["disposition"]["level"] != "confident":
            continue
        tags = m["tags"]
        if domain and tags["domain"] != domain:
            continue
        if activity and tags["activity"] != activity:
            continue
        if concept and concept not in tags["concept"]:
            continue
        if types and m["type"] not in types:
            continue
        yield m


print("Before touching billing retries (domain=payments, concept=idempotency):")
for m in find(domain="payments", concept="idempotency"):
    print(f"  [{m['type']}] {m['statement']}")

print("While debugging or in an incident (cautions and failed attempts only):")
for activity in ("debugging", "incident"):
    for m in find(activity=activity, types={"caution", "failed-attempt"}):
        print(f"  [{activity}] {m['statement']}")

print("Before a review:")
for m in find(activity="reviewing"):
    print(f"  {m['statement']}")
