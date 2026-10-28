# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""The merge stage: loading and arm-tagging, statement clustering, member conversation ids and
evidence, uncovered-member passthrough, dedup folding, corroboration counting and ranking, the
output schema, ordering at any worker count, fatal stops and spend. No test calls a model:
synthesis and dedup use stub clients."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import unittest
from unittest import mock

from agent_memory_engine import cache as cache_mod
from agent_memory_engine import merge
from agent_memory_engine.providers import USAGE_METRIC_KEYS, ExtractResult
from agent_memory_engine.providers import gemini as mcl


def _single(arm, stmt, cid, evidence):
    """An arm memory whose origin carries one conversation_id and whose evidence items carry none."""
    return {
        "_arm": arm,
        "statement": stmt,
        "type": "preference",
        "scope": "personal",
        "tags": {**merge.mc.empty_tags(), "keyword": ["t"]},
        "memory_id": merge.mc.memory_id(stmt),
        "origin": {"conversation_id": cid, "tool": "claude-code"},
        "evidence": [{"quote": q, "turn": t, "speaker": "user"} for q, t in evidence],
    }


def _multi(arm, stmt, cids, ev):
    """A member whose origin lists several conversation_ids and whose evidence items each name
    their own conversation_id."""
    return {
        "_arm": arm,
        "statement": stmt,
        "type": "convention",
        "scope": "personal",
        "tags": {**merge.mc.empty_tags(), "keyword": ["t"]},
        "memory_id": merge.mc.memory_id(stmt),
        "origin": {"conversation_ids": cids, "tool": "claude-code"},
        "evidence": [{"quote": q, "turn": t, "speaker": "user", "conversation_id": c} for q, t, c in ev],
    }


class LoadTag(unittest.TestCase):
    def test_load_arms_tags_by_file(self):
        d = tempfile.mkdtemp()
        paths = {}
        for arm in merge.ARM_ORDER:
            paths[arm] = os.path.join(d, arm, "memories.jsonl")
            os.makedirs(os.path.dirname(paths[arm]))
            with open(paths[arm], "w") as f:
                f.write(json.dumps({"statement": f"s-{arm}", "evidence": [], "origin": {}}) + "\n")
                f.write("\n")  # blank: skipped silently
                f.write("not json\n")  # malformed: skipped and counted
                f.write(json.dumps({"statement": "  "}) + "\n")  # no statement: skipped and counted
        logged = []
        mems = merge.load_arms(paths, log=logged.append)
        self.assertEqual([m["_arm"] for m in mems], merge.ARM_ORDER)  # loaded in ARM_ORDER
        self.assertEqual({m["statement"] for m in mems}, {f"s-{a}" for a in merge.ARM_ORDER})
        self.assertEqual(logged, ["merge: skipped 4 malformed arm line(s) lacking a statement"])

    def test_missing_arm_file_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            merge.load_arms({a: "/nonexistent/memories.jsonl" for a in merge.ARM_ORDER})


class ConvIdsAndEvidence(unittest.TestCase):
    def test_single_source_conv_ids(self):
        m = _single("facts", "x", "cc:1", [("q", 3)])
        self.assertEqual(merge.member_conversation_ids(m), ["cc:1"])
        ev = merge.member_evidence(m)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["conversation_id"], "cc:1")  # stamped from origin
        self.assertEqual(ev[0]["arm"], "facts")

    def test_conv_ids_come_from_origin_and_evidence_only(self):
        # A member whose evidence names its own conversations: those ids count; an origin
        # `conversation_ids` list is not a shape the arms write, and is not read.
        m = _multi(
            "preferences", "y", ["cc:1", "cc:2"], [("q1", 1, "cc:1"), ("q2", 2, "cc:3")]
        )  # cc:2 only in the unread origin list
        self.assertEqual(merge.member_conversation_ids(m), ["cc:1", "cc:3"])
        ev = merge.member_evidence(m)
        self.assertEqual([e["conversation_id"] for e in ev], ["cc:1", "cc:3"])


class Clustering(unittest.TestCase):
    def test_similar_statements_group_distinct_singleton(self):
        mems = [
            _single("facts", "always run ci before merging a PR in the acme shop repo", "cc:1", [("a", 1)]),
            _single(
                "preferences", "run ci on the pull request before merge in acme shop", "cc:2", [("b", 1)]
            ),
            _single("facts", "use sentence case in doc headings", "cc:3", [("c", 1)]),
        ]
        clusters, singletons = merge.cluster_memories(mems, threshold=0.10)
        joined = [set(c) for c in clusters]
        self.assertIn({0, 1}, joined)  # the two ci statements group
        self.assertIn(2, singletons)  # the section-symbol one stands alone


class Passthrough(unittest.TestCase):
    def test_passthrough_generality_and_fields(self):
        mems = [_single("facts", "prefer worktrees over the main checkout", "cc:9", [("q", 2)])]
        pt = merge._passthrough(mems, 0)
        self.assertEqual(pt["_member_idxs"], [0])
        self.assertEqual(pt["generality"], "general-principle")  # preference -> principle
        self.assertEqual(pt["statement"], "prefer worktrees over the main checkout")


class DedupFold(unittest.TestCase):
    def test_merge_dedup_groups_unions_members(self):
        synth = [
            {
                "statement": "A",
                "type": "fact",
                "scope": "personal",
                "generality": "instance-specific",
                "review": {"status": "active"},
                "_member_idxs": [0],
            },
            {
                "statement": "A-restated but longer text",
                "type": "fact",
                "scope": "personal",
                "generality": "instance-specific",
                "review": {"status": "active"},
                "_member_idxs": [1],
            },
            {
                "statement": "B",
                "type": "fact",
                "scope": "personal",
                "generality": "instance-specific",
                "review": {"status": "active"},
                "_member_idxs": [2],
            },
        ]
        merged = merge.merge_dedup_groups(synth, [[0, 1]])
        # one folded (members {0,1}) + the ungrouped B
        idxsets = sorted(tuple(s["_member_idxs"]) for s in merged)
        self.assertEqual(idxsets, [(0, 1), (2,)])

    def test_dedup_group_keeps_intention_representative(self):
        """If a dedup group mixes an intention with a non-intention, the merged entry MUST keep
        type=intention + review=hold-for-review (the safety gate) — even when the non-intention
        would otherwise win the (member-count, longest-statement) tiebreak."""
        synth = [
            # a non-intention that would win on richness alone (richer + longer statement).
            {
                "statement": "a much longer non-intention statement that would win a richness tiebreak",
                "type": "fact",
                "scope": "personal",
                "generality": "instance-specific",
                "review": {"status": "active"},
                "intention_meta": None,
                "_member_idxs": [0, 1],
            },
            # the intention — shorter, fewer members, but must be preferred as the representative.
            {
                "statement": "wants X done",
                "type": "intention",
                "scope": "personal",
                "generality": "instance-specific",
                "review": {"status": "hold-for-review"},
                "intention_meta": {"kind": "todo"},
                "_member_idxs": [2],
            },
        ]
        merged = merge.merge_dedup_groups(synth, [[0, 1]])
        self.assertEqual(len(merged), 1)
        rep = merged[0]
        self.assertEqual(rep["type"], "intention")
        self.assertEqual(rep["review"], {"status": "hold-for-review"})
        self.assertEqual(rep["intention_meta"], {"kind": "todo"})
        # members are still the UNION across the group.
        self.assertEqual(rep["_member_idxs"], [0, 1, 2])

    def test_dedup_group_without_intention_unchanged(self):
        """No intention in the group -> ranking is the original (member-count, longest statement)."""
        synth = [
            {
                "statement": "short",
                "type": "fact",
                "scope": "personal",
                "generality": "instance-specific",
                "review": {"status": "active"},
                "_member_idxs": [0],
            },
            {
                "statement": "a longer richer statement",
                "type": "fact",
                "scope": "personal",
                "generality": "instance-specific",
                "review": {"status": "active"},
                "_member_idxs": [1, 2],
            },
        ]
        merged = merge.merge_dedup_groups(synth, [[0, 1]])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["statement"], "a longer richer statement")
        self.assertEqual(merged[0]["_member_idxs"], [0, 1, 2])


class Corroboration(unittest.TestCase):
    def test_counts_and_ranking(self):
        mems = [
            _single("facts", "same lesson", "cc:1", [("q", 1)]),
            _single("preferences", "same lesson", "cc:2", [("q", 1)]),
            _multi("preferences", "same lesson", ["cc:2", "cc:3"], [("q2", 1, "cc:2"), ("q", 1, "cc:3")]),
            _single("facts", "lonely lesson", "cc:9", [("q", 1)]),
        ]
        strong = {
            "statement": "unified lesson",
            "type": "fact",
            "scope": "personal",
            "generality": "general-principle",
            "review": {"status": "active"},
            "_member_idxs": [0, 1, 2],
        }
        weak = {
            "statement": "lonely lesson",
            "type": "fact",
            "scope": "personal",
            "generality": "instance-specific",
            "review": {"status": "active"},
            "_member_idxs": [3],
        }
        f_strong = merge.build_final(strong, mems)
        f_weak = merge.build_final(weak, mems)
        c = f_strong["provenance"]["corroboration"]
        self.assertEqual(c["distinct_arms"], 2)  # facts, preferences
        self.assertEqual(c["source_arms"], ["facts", "preferences"])  # in ARM_ORDER
        self.assertEqual(c["distinct_conversations"], 3)  # cc:1, cc:2, cc:3
        self.assertEqual(c["source_conversation_ids"], ["cc:1", "cc:2", "cc:3"])
        self.assertEqual(len(f_strong["evidence"]), 4)  # 1 + 1 + 2 preserved
        self.assertEqual(
            f_weak["provenance"]["corroboration"],
            {
                "distinct_arms": 1,
                "distinct_conversations": 1,
                "source_arms": ["facts"],
                "source_conversation_ids": ["cc:9"],
            },
        )
        # ranking: strong (3 convs) sorts before weak (1 conv)
        finals = [f_weak, f_strong]
        finals.sort(
            key=lambda f: (
                -f["provenance"]["corroboration"]["distinct_conversations"],
                -f["provenance"]["corroboration"]["distinct_arms"],
                f["memory_id"],
            )
        )
        self.assertEqual(finals[0]["statement"], "unified lesson")


class SynthesizeStub(unittest.TestCase):
    """Exercise synthesize_group end-to-end with a STUB client (no real LLM): the model splits a
    2-member group into one unified memory (M1+M2) plus leaves nothing uncovered."""

    class _StubResult:
        def __init__(self, raw):
            self.ok = True
            self.raw_text = raw
            self.cost_usd = 0.01
            self.duration_ms = 5
            self.model_id = "stub"
            self.meta = {"canonical_model": "stub"}

    class _StubClient:
        model_id = "stub"
        cache_model_id = "stub"  # mirrors the ModelClient interface (used by the cache fingerprint)

        def __init__(self, raw):
            self._raw = raw

        def extract(self, sysp, text):
            return SynthesizeStub._StubResult(self._raw)

    def test_split_and_generality(self):

        mems = [
            _single("facts", "run ci before merge", "cc:1", [("q1", 1)]),
            _single("preferences", "ci must pass before merge", "cc:2", [("q2", 1)]),
        ]
        raw = (
            '{"memories":[{"members":["M1","M2"],"statement":"Run ci and let it fully pass '
            'before merging.","type":"convention","scope":"repo/acme-shop",'
            '"tags":["ci","ci","merge"],"generality":"general-principle"}]}'
        )
        client = self._StubClient(raw)
        cache = cache_mod.Cache(tempfile.mkdtemp())
        metrics = merge._fresh_group_metrics()
        out = merge.synthesize_group(mems, [0, 1], "sysprompt", client, cache, metrics)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["_member_idxs"], [0, 1])
        self.assertEqual(out[0]["generality"], "general-principle")
        # the prompt still asks for `tags`; code discards the answer and the merge takes the members'
        self.assertNotIn("tags", out[0])
        self.assertEqual(merge.build_final(out[0], mems)["tags"], {**merge.mc.empty_tags(), "keyword": ["t"]})
        self.assertEqual(metrics["members_uncovered"], 0)
        self.assertEqual(metrics["llm_calls"], 1)
        # second call hits the cache -> no new llm call
        merge.synthesize_group(mems, [0, 1], "sysprompt", client, cache, metrics)
        self.assertEqual(metrics["llm_calls"], 1)
        self.assertEqual(metrics["cache_hits"], 1)


class ReviewSurvival(unittest.TestCase):
    """Regression: the merged memories.jsonl is the consumed artifact, so an intention's
    review=hold-for-review gate (and its intention_meta) must survive through _passthrough
    and build_final — dropping them would silently make intentions loadable without a human check."""

    def _intention_member(self):
        return {
            "_arm": "facts",
            "statement": "revisit tz handling holistically",
            "type": "intention",
            "scope_level": "repo",
            "scope_ref": "acme/shop",
            "scope": "repo/acme/shop",
            "tags": {**merge.mc.empty_tags(), "keyword": ["tz"]},
            "memory_id": merge.mc.memory_id("revisit tz handling holistically"),
            "review": {"status": "hold-for-review"},
            "intention_meta": {"what": "holistic tz audit", "done_check": "a tz audit PR exists"},
            "origin": {"conversation_id": "cc:1", "tool": "claude-code"},
            "evidence": [{"quote": "we should revisit tz", "turn": 1, "speaker": "user"}],
        }

    def test_passthrough_carries_review_and_meta(self):
        mems = [self._intention_member()]
        pt = merge._passthrough(mems, 0)
        self.assertEqual(pt["review"]["status"], "hold-for-review")
        self.assertEqual(pt["intention_meta"]["what"], "holistic tz audit")

    def test_build_final_carries_review_and_meta(self):
        mems = [self._intention_member()]
        final = merge.build_final(merge._passthrough(mems, 0), mems)
        self.assertEqual(final["type"], "intention")
        self.assertEqual(
            final["review"]["status"],
            "hold-for-review",
            "intention lost its hold-for-review gate at the merged artifact",
        )
        self.assertIsNotNone(final["intention_meta"])
        self.assertEqual(final["intention_meta"]["done_check"], "a tz audit PR exists")


def _mem_temporal(arm, stmt, cid, when, valid_as_of, temporal_kind, volatility, tags):
    """A single-source member carrying the temporal rich-meta + `when` (conversation date) that the
    merge code-carries onto the merged memory. `tags` is given as a list-form answer and stored the way
    an arm line carries it: decoded at the gate into the typed object."""
    return {
        "_arm": arm,
        "statement": stmt,
        "type": "fact",
        "scope": "personal",
        "tags": merge.mc.union_tags([merge.mc.typed_from_flat(tags)]),
        "memory_id": merge.mc.memory_id(stmt),
        "when": when,
        "valid_as_of": valid_as_of,
        "temporal_kind": temporal_kind,
        "volatility": volatility,
        "origin": {"conversation_id": cid, "tool": "claude-code"},
        "evidence": [{"quote": "q", "turn": 1, "speaker": "user"}],
    }


class EvidenceRows(unittest.TestCase):
    def test_two_arms_citing_one_quote_keep_two_rows(self):
        row = {"quote": "q", "turn": 1, "speaker": "user", "conversation_id": "cc:1"}
        rows = [{**row, "arm": "facts"}, {**row, "arm": "preferences"}, {**row, "arm": "facts"}]
        self.assertEqual([e["arm"] for e in merge._dedupe_evidence(rows)], ["facts", "preferences"])


class TemporalTagsDispositionOutcome(unittest.TestCase):
    """The cross-conversation MERGE code-carries, from the member set, the temporal rich-meta
    (valid_as_of, temporal_kind, volatility), the dimensioned tags (union — never the flattening
    synthesis model), disposition, and a decision's `decision_outcome`."""

    def test_passthrough_singleton_carries_temporal_and_tags(self):
        m = _mem_temporal(
            "facts",
            "gofmt runs on save",
            "cc:1",
            "2026-05-01",
            "2026-05",
            "dated",
            "low",
            ["domain:backend", "fmt"],
        )
        final = merge.build_final(merge._passthrough([m], 0), [m])
        # one member -> its three temporal fields pass straight through (grouped under `temporal`)
        self.assertEqual(final["temporal"]["valid_as_of"], "2026-05")
        self.assertEqual(final["temporal"]["kind"], "dated")
        self.assertEqual(final["temporal"]["volatility"], "low")
        # its tags survive; the bare `fmt` is coerced into the keyword catch-all (nothing dropped)
        self.assertEqual(final["tags"]["domain"], "backend")
        self.assertEqual(final["tags"]["keyword"], ["fmt"])

    def test_cluster_carries_most_recent_member_temporal(self):
        older = _mem_temporal("facts", "s1", "cc:1", "2026-01-01", "2026-01", "dated", "high", ["t"])
        newer = _mem_temporal(
            "preferences", "s2", "cc:2", "2026-06-01", "2026-06", "until-event", "med", ["t"]
        )
        mems = [older, newer]
        synth = {
            "statement": "unified",
            "type": "fact",
            "scope": "personal",
            "generality": "instance-specific",
            "review": {"status": "active"},
            "_member_idxs": [0, 1],
        }
        final = merge.build_final(synth, mems)
        # latest-wins: the most-recent conversation date (newer) supplies ALL THREE fields as a unit
        self.assertEqual(final["temporal"]["valid_as_of"], "2026-06")
        self.assertEqual(final["temporal"]["kind"], "until-event")
        self.assertEqual(final["temporal"]["volatility"], "med")

    def test_temporal_null_when_no_member_carries_it(self):
        # a member without temporal fields -> merged fields default to null
        m = _mem_temporal("facts", "s", "cc:1", "2026-01-01", None, None, None, ["t"])
        final = merge.build_final(merge._passthrough([m], 0), [m])
        self.assertIsNone(final["temporal"]["valid_as_of"])
        self.assertIsNone(final["temporal"]["kind"])
        self.assertIsNone(final["temporal"]["volatility"])

    def test_open_axis_tags_union_across_members(self):
        a = _mem_temporal(
            "facts",
            "s1",
            "cc:1",
            "2026-02-01",
            None,
            None,
            None,
            ["domain:backend", "component:cart"],
        )
        b = _mem_temporal(
            "preferences",
            "s2",
            "cc:2",
            "2026-01-01",
            None,
            None,
            None,
            ["domain:backend", "component:rate-limiter", "component:cart"],
        )
        synth = {
            "statement": "unified",
            "type": "fact",
            "scope": "personal",
            "generality": "instance-specific",
            "review": {"status": "active"},
            "_member_idxs": [0, 1],
        }
        final = merge.build_final(synth, [a, b])
        # open axis: union of distinct component values, deduped, first-seen
        self.assertEqual(final["tags"]["component"], ["cart", "rate-limiter"])
        self.assertEqual(final["tags"]["keyword"], [])

    def test_controlled_domain_takes_majority_on_conflict(self):
        a = _mem_temporal("facts", "s1", "cc:1", None, None, None, None, ["domain:backend"])
        b = _mem_temporal("preferences", "s2", "cc:2", None, None, None, None, ["domain:backend"])
        c = _mem_temporal("facts", "s3", "cc:3", None, None, None, None, ["domain:database"])
        synth = {
            "statement": "u",
            "type": "fact",
            "scope": "personal",
            "generality": "instance-specific",
            "review": {"status": "active"},
            "_member_idxs": [0, 1, 2],
        }
        final = merge.build_final(synth, [a, b, c])
        self.assertEqual(final["tags"]["domain"], "backend")  # exactly one; the majority value wins
        self.assertEqual(final["tags"]["keyword"], ["database"])  # the loser is kept in keyword

    def test_bare_tag_goes_to_keyword_never_promoted_to_controlled(self):
        # A bare word is NEVER promoted to a controlled axis (that is the prompt's job); it lands in
        # keyword so a non-compliant extraction stays visible instead of masquerading as a real domain.
        m = _mem_temporal("facts", "s", "cc:1", None, None, None, None, ["database", "postgres", "debugging"])
        final = merge.build_final(merge._passthrough([m], 0), [m])
        self.assertEqual(final["tags"]["keyword"], ["database", "postgres", "debugging"])
        self.assertIsNone(final["tags"]["domain"])  # no invented domain
        self.assertIsNone(final["tags"]["activity"])  # "debugging" stays a keyword

    def test_bare_word_cannot_evict_an_explicit_domain(self):
        # the corruption the simplicity review flagged: a bare "api" must NOT beat an explicit domain
        a = _mem_temporal("facts", "s1", "cc:1", None, None, None, None, ["domain:database"])
        b = _mem_temporal("preferences", "s2", "cc:2", None, None, None, None, ["api"])  # bare
        synth = {
            "statement": "u",
            "type": "fact",
            "scope": "personal",
            "generality": "instance-specific",
            "review": {"status": "active"},
            "_member_idxs": [0, 1],
        }
        final = merge.build_final(synth, [a, b])
        self.assertEqual(final["tags"]["domain"], "database")
        self.assertEqual(final["tags"]["keyword"], ["api"])  # the bare word is preserved, non-authoritative

    def test_unknown_prefix_and_case_and_degenerate_values(self):
        m = _mem_temporal(
            "facts",
            "s",
            "cc:1",
            None,
            None,
            None,
            None,
            ["topic:stacks", "Domain:Mobile", "", ":", "component: Search "],
        )
        final = merge.build_final(merge._passthrough([m], 0), [m])
        self.assertEqual(
            final["tags"]["keyword"], ["topic:stacks"]
        )  # non-axis prefix kept whole; empties dropped
        self.assertEqual(final["tags"]["domain"], "mobile")  # prefix + value both lowercased
        self.assertEqual(final["tags"]["component"], ["search"])  # value trimmed + lowercased

    def test_controlled_vote_counts_distinct_members_not_occurrences(self):
        # Discriminating fixture: the REPEATED value must LOSE under distinct counting. Member A repeats
        # "database" twice; member B has "backend" once. Under OCCURRENCE counting database=2 beats
        # backend=1 (wrong). Under DISTINCT-MEMBER counting it is 1-vs-1 -> tie -> lexicographic winner
        # "domain:backend". Asserting the winner is backend fails if the dedup regresses to occurrences.
        a = _mem_temporal(
            "facts", "s1", "cc:1", None, None, None, None, ["domain:database", "domain:database"]
        )
        b = _mem_temporal("preferences", "s2", "cc:2", None, None, None, None, ["domain:backend"])
        synth = {
            "statement": "u",
            "type": "fact",
            "scope": "personal",
            "generality": "instance-specific",
            "review": {"status": "active"},
            "_member_idxs": [0, 1],
        }
        final = merge.build_final(synth, [a, b])
        self.assertEqual(final["tags"]["domain"], "backend")  # distinct-member tie, NOT a 2:1 database win
        # the losing controlled value is preserved in keyword, not dropped
        self.assertEqual(final["tags"]["keyword"], ["database"])

    def test_disposition_borderline_survives_single_source(self):
        m = _mem_temporal("facts", "s", "cc:1", None, None, None, None, ["domain:backend"])
        m["disposition"] = {"level": "borderline", "note": "induced from one act"}
        final = merge.build_final(merge._passthrough([m], 0), [m])
        self.assertEqual(final["disposition"]["level"], "borderline")
        self.assertEqual(final["disposition"]["note"], "induced from one act")

    def test_disposition_promoted_by_cross_conversation_corroboration(self):
        a = _mem_temporal("facts", "s1", "cc:1", None, None, None, None, ["domain:backend"])
        b = _mem_temporal("facts", "s2", "cc:2", None, None, None, None, ["domain:backend"])
        a["disposition"] = {"level": "borderline", "note": "n"}
        b["disposition"] = {"level": "confident", "note": ""}
        synth = {
            "statement": "u",
            "type": "fact",
            "scope": "personal",
            "generality": "instance-specific",
            "review": {"status": "active"},
            "_member_idxs": [0, 1],
        }
        final = merge.build_final(synth, [a, b])
        # 2 distinct conversations corroborate -> the single-source doubt is resolved
        self.assertEqual(final["disposition"]["level"], "confident")

    def test_two_arms_one_conversation_does_NOT_promote(self):
        # the arms read the SAME conversation, so two arms agreeing on one conversation is one source
        # seen twice, not independent corroboration -> stays borderline
        a = _mem_temporal("facts", "s1", "cc:1", None, None, None, None, ["domain:backend"])
        b = _mem_temporal(
            "preferences", "s2", "cc:1", None, None, None, None, ["domain:backend"]
        )  # SAME conv
        a["disposition"] = {"level": "borderline", "note": "induced from one act"}
        b["disposition"] = {"level": "borderline", "note": "induced from one act"}
        synth = {
            "statement": "u",
            "type": "fact",
            "scope": "personal",
            "generality": "instance-specific",
            "review": {"status": "active"},
            "_member_idxs": [0, 1],
        }
        final = merge.build_final(synth, [a, b])
        self.assertEqual(final["provenance"]["corroboration"]["distinct_arms"], 2)
        self.assertEqual(final["provenance"]["corroboration"]["distinct_conversations"], 1)
        self.assertEqual(final["disposition"]["level"], "borderline")  # NOT promoted on arm count

    def test_decision_outcome_carried_for_decision_only(self):
        d = _mem_temporal(
            "facts", "chose keyset pagination", "cc:1", None, None, None, None, ["domain:database"]
        )
        d["type"] = "decision"
        d["decision_outcome"] = "adopted keyset, rejected offset"
        final = merge.build_final(merge._passthrough([d], 0), [d])
        self.assertEqual(final["type"], "decision")
        self.assertEqual(final["decision_outcome"], "adopted keyset, rejected offset")
        # a non-decision carries null decision_outcome
        f = _mem_temporal("facts", "a fact", "cc:2", None, None, None, None, ["domain:backend"])
        final_f = merge.build_final(merge._passthrough([f], 0), [f])
        self.assertIsNone(final_f["decision_outcome"])

    def test_a_resolution_key_is_not_an_outcome(self):
        # only `decision_outcome` carries a member's outcome; any other key is not read as one
        d = _mem_temporal("facts", "chose keyset", "cc:1", None, None, None, None, ["domain:database"])
        d["type"] = "decision"
        d["resolution"] = "adopted keyset"  # no decision_outcome key present
        final = merge.build_final(merge._passthrough([d], 0), [d])
        self.assertIsNone(final["decision_outcome"])

    def test_decision_outcome_latest_wins_and_flags_divergence(self):
        # conflicting outcomes must NOT be silently first-by-load-order; latest-by-date wins + is flagged
        older = _mem_temporal("facts", "s1", "cc:1", "2026-01-01", None, None, None, ["domain:database"])
        newer = _mem_temporal(
            "preferences", "s2", "cc:2", "2026-06-01", None, None, None, ["domain:database"]
        )
        older["type"] = newer["type"] = "decision"
        older["decision_outcome"] = "adopted offset pagination"
        newer["decision_outcome"] = "adopted keyset pagination"  # the decision flipped later
        synth = {
            "statement": "u",
            "type": "decision",
            "scope": "personal",
            "generality": "instance-specific",
            "review": {"status": "active"},
            "_member_idxs": [0, 1],
        }
        final = merge.build_final(synth, [older, newer])
        self.assertIn(
            "adopted keyset pagination", final["decision_outcome"]
        )  # latest by date, not first-loaded
        self.assertIn("differing outcomes", final["decision_outcome"])  # divergence flagged, not dropped

    def test_decision_outcome_ignores_non_decision_members_and_non_string(self):
        # a stray outcome on a NON-decision member must not win (even if newer); a non-string outcome
        # must be skipped, never reach the string concat (which would TypeError and abort the merge)
        dec = _mem_temporal("facts", "d", "cc:1", "2026-01-01", None, None, None, ["domain:database"])
        dec["type"] = "decision"
        dec["decision_outcome"] = "adopted keyset"
        stray = _mem_temporal("preferences", "f", "cc:2", "2026-09-01", None, None, None, ["domain:backend"])
        stray["type"] = "fact"
        stray["decision_outcome"] = "IGNORE ME (newer, but not a decision)"
        bad = _mem_temporal("facts", "d2", "cc:3", "2026-08-01", None, None, None, ["domain:database"])
        bad["type"] = "decision"
        bad["decision_outcome"] = {"not": "a string"}  # malformed model output
        synth = {
            "statement": "u",
            "type": "decision",
            "scope": "personal",
            "generality": "instance-specific",
            "review": {"status": "active"},
            "_member_idxs": [0, 1, 2],
        }
        final = merge.build_final(synth, [dec, stray, bad])  # must NOT raise TypeError
        self.assertEqual(final["decision_outcome"], "adopted keyset")  # only the valid decision-member string

    def test_leading_colon_and_degenerate_values_dropped(self):
        m = _mem_temporal(
            "facts", "s", "cc:1", None, None, None, None, ["::", ":x", ":::", "  ", "keyword:ok"]
        )
        final = merge.build_final(merge._passthrough([m], 0), [m])
        # ":x" is kept whole (no axis before its colon); "::", ":::" and blanks carry nothing
        self.assertEqual(final["tags"]["keyword"], [":x", "ok"])


class DescriptionField(unittest.TestCase):
    """The OPTIONAL `description` rides with the (model-rewritten) statement through the merge: the
    SYNTHESIS model writes a unified description; when it omits one, description falls back to the
    first covered member (as type/scope/tags do); a passthrough keeps its own; build_final emits it;
    a dedup fold keeps the representative's."""

    class _StubClient:
        model_id = "stub"
        cache_model_id = "stub"

        def __init__(self, raw):
            self._raw = raw

        def extract(self, sysp, text):
            return SynthesizeStub._StubResult(self._raw)

    def _member(self, arm, stmt, cid, description=None):
        m = {
            "_arm": arm,
            "statement": stmt,
            "type": "fact",
            "scope": "personal",
            "tags": {**merge.mc.empty_tags(), "keyword": ["t"]},
            "memory_id": merge.mc.memory_id(stmt),
            "origin": {"conversation_id": cid, "tool": "claude-code"},
            "evidence": [{"quote": "q", "turn": 1, "speaker": "user"}],
        }
        if description is not None:
            m["description"] = description
        return m

    def _synth(self, raw, mems, idxs):

        cache = cache_mod.Cache(tempfile.mkdtemp())
        metrics = merge._fresh_group_metrics()
        return merge.synthesize_group(mems, idxs, "sysprompt", self._StubClient(raw), cache, metrics)

    def test_model_emitted_description_flows(self):
        mems = [
            self._member("facts", "run ci before merge", "cc:1", "member desc a"),
            self._member("preferences", "ci must pass before merge", "cc:2", "member desc b"),
        ]
        raw = (
            '{"memories":[{"members":["M1","M2"],"statement":"Run ci and let it pass before '
            'merging.","description":"Unified prose: why ci is a hard gate before any merge.",'
            '"type":"convention","scope":"repo/acme-shop","tags":["ci"],'
            '"generality":"general-principle"}]}'
        )
        out = self._synth(raw, mems, [0, 1])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["description"], "Unified prose: why ci is a hard gate before any merge.")
        # build_final carries it onto the final merged memory.
        self.assertEqual(
            merge.build_final(out[0], mems)["description"],
            "Unified prose: why ci is a hard gate before any merge.",
        )

    def test_description_falls_back_to_first_member_when_model_omits(self):
        mems = [
            self._member("facts", "run ci before merge", "cc:1", "first member desc"),
            self._member("preferences", "ci must pass before merge", "cc:2", "second member desc"),
        ]
        raw = (
            '{"memories":[{"members":["M1","M2"],"statement":"Run ci before merging.",'
            '"type":"convention","scope":"repo/x","tags":["ci"],'
            '"generality":"general-principle"}]}'
        )  # NO description emitted
        out = self._synth(raw, mems, [0, 1])
        self.assertEqual(out[0]["description"], "first member desc")  # gidxs[0] fallback

    def test_description_null_when_no_member_carries_it(self):
        mems = [
            self._member("facts", "run ci before merge", "cc:1"),
            self._member("preferences", "ci must pass before merge", "cc:2"),
        ]
        raw = (
            '{"memories":[{"members":["M1","M2"],"statement":"Run ci before merging.",'
            '"type":"convention","scope":"repo/x","tags":["ci"],'
            '"generality":"general-principle"}]}'
        )
        out = self._synth(raw, mems, [0, 1])
        self.assertIsNone(out[0]["description"])

    def test_passthrough_carries_own_description(self):
        mems = [self._member("facts", "prefer worktrees", "cc:9", "why worktrees are safer")]
        pt = merge._passthrough(mems, 0)
        self.assertEqual(pt["description"], "why worktrees are safer")
        self.assertEqual(merge.build_final(pt, mems)["description"], "why worktrees are safer")

    def test_dedup_fold_keeps_representative_description(self):
        synth = [
            {
                "statement": "A short",
                "type": "fact",
                "scope": "personal",
                "generality": "instance-specific",
                "description": "desc A",
                "review": {"status": "active"},
                "_member_idxs": [0],
            },
            {
                "statement": "A restated but longer text",
                "type": "fact",
                "scope": "personal",
                "generality": "instance-specific",
                "description": "desc B (rep)",
                "review": {"status": "active"},
                "_member_idxs": [1],
            },
        ]
        merged = merge.merge_dedup_groups(synth, [[0, 1]])
        self.assertEqual(len(merged), 1)
        # rep is the longer-statement entry -> its description survives the fold.
        self.assertEqual(merged[0]["description"], "desc B (rep)")


def _mem_typed(arm, stmt, cid, tags):
    """A single-source member carrying the typed tag object that merge unions onto the merged memory."""
    return {
        "_arm": arm,
        "statement": stmt,
        "type": "fact",
        "scope": "personal",
        "tags": tags,
        "memory_id": merge.mc.memory_id(stmt),
        "origin": {"conversation_id": cid, "tool": "claude-code"},
        "evidence": [{"quote": "q", "turn": 1, "speaker": "user"}],
    }


class PassthroughIntention(unittest.TestCase):
    def test_a_passthrough_intention_keeps_its_meta_and_hold(self):
        meta = {"what": "w", "done_check": "d"}
        m = {
            "_arm": "facts",
            "statement": "revisit tz",
            "type": "intention",
            "scope": "personal",
            "tags": {**merge.mc.empty_tags(), "keyword": ["t"]},
            "memory_id": "m0",
            "origin": {"conversation_id": "cc:1", "tool": "claude-code"},
            "evidence": [{"quote": "q", "turn": 1, "speaker": "user"}],
            "review": {"status": "active"},
            "intention_meta": meta,
        }
        final = merge.build_final(merge._passthrough([m], 0), [m])
        self.assertEqual(
            (final["type"], final["review"], final["intention_meta"]),
            ("intention", {"status": "hold-for-review"}, meta),
        )


class ReviewIsStampedFromType(unittest.TestCase):
    """Review comes from the merged memory's type alone: an intention member keeps the group an
    intention, held for review with its meta, whatever order the model lists members in or type it
    names; a status the model or a member sends is never copied."""

    @staticmethod
    def _member(stmt, cid, mtype, status, meta=None):
        return {
            "_arm": "facts",
            "statement": stmt,
            "type": mtype,
            "scope": "personal",
            "tags": {**merge.mc.empty_tags(), "keyword": ["t"]},
            "memory_id": merge.mc.memory_id(stmt),
            "origin": {"conversation_id": cid, "tool": "claude-code"},
            "evidence": [{"quote": "q", "turn": 1, "speaker": "user"}],
            "review": {"status": status},
            "intention_meta": meta,
        }

    def _synth(self, out, mems):
        cache = cache_mod.Cache(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, cache.root)
        client = DescriptionField._StubClient(json.dumps({"memories": [out]}))
        return merge.synthesize_group(
            mems, list(range(len(mems))), "sysprompt", client, cache, merge._fresh_group_metrics()
        )

    def test_a_group_with_an_intention_stays_an_intention(self):
        meta = {"what": "revisit tz", "done_check": "tz test passes"}
        mems = [
            self._member("revisit tz handling", "cc:1", "intention", "hold-for-review", meta),
            self._member("tz handling is broken", "cc:2", "fact", "active"),
        ]
        for members in (["M1", "M2"], ["M2", "M1"]):
            with self.subTest(members=members):
                (m,) = self._synth({"statement": "fix tz handling", "type": "fact", "members": members}, mems)
                self.assertEqual(m["type"], "intention")
                self.assertEqual(m["review"], {"status": "hold-for-review"})
                self.assertEqual(m["intention_meta"], meta)

    def test_a_status_sent_by_the_model_or_a_member_is_not_copied(self):
        mems = [self._member("tz handling is broken", "cc:2", "fact", "hold-for-review")]
        out = {
            "statement": "tz is broken",
            "type": "fact",
            "members": ["M1"],
            "review": {"status": "hold-for-review"},
        }
        (m,) = self._synth(out, mems)
        self.assertEqual(m["review"], {"status": "active"})
        self.assertEqual(merge._passthrough(mems, 0)["review"], {"status": "active"})


class TagsTypedMerge(unittest.TestCase):
    """The typed tag object survives the cross-conversation MERGE as the memory's one `tags` field:
    build_final unions the members' objects (single domain/activity by majority, losers demoted to
    keyword; open axes unioned)."""

    def test_build_final_unions_typed_objects(self):
        a = _mem_typed(
            "facts",
            "s1",
            "cc:1",
            {
                "domain": "backend",
                "activity": "coding",
                "component": ["cart"],
                "keyword": ["retry"],
            },
        )
        b = _mem_typed(
            "preferences",
            "s2",
            "cc:2",
            {"domain": "backend", "component": ["rate-limiter", "cart"], "tech": ["go"]},
        )
        synth = {
            "statement": "unified",
            "type": "fact",
            "scope": "personal",
            "generality": "instance-specific",
            "review": {"status": "active"},
            "_member_idxs": [0, 1],
        }
        final = merge.build_final(synth, [a, b])
        self.assertNotIn("tags_typed", final)
        self.assertEqual(
            final["tags"],
            {
                "domain": "backend",
                "activity": "coding",
                "component": ["cart", "rate-limiter"],  # union, first-seen
                "tech": ["go"],
                "concept": [],
                "keyword": ["retry"],
            },
        )

    def test_build_final_typed_domain_majority_demotes_loser_to_keyword(self):
        # domain conflict: two members say backend, one says database -> backend wins (majority), the
        # losing "database" is demoted into keyword (never dropped).
        a = _mem_typed("facts", "s1", "cc:1", {"domain": "backend"})
        b = _mem_typed("preferences", "s2", "cc:2", {"domain": "backend"})
        c = _mem_typed("facts", "s3", "cc:3", {"domain": "database"})
        synth = {
            "statement": "u",
            "type": "fact",
            "scope": "personal",
            "generality": "instance-specific",
            "review": {"status": "active"},
            "_member_idxs": [0, 1, 2],
        }
        final = merge.build_final(synth, [a, b, c])
        self.assertEqual(final["tags"]["domain"], "backend")
        self.assertEqual(final["tags"]["keyword"], ["database"])

    def test_list_tags_member_is_decoded(self):
        # An arm line in the v2 shape (flat `tags` list) still merges with its tags.
        a = _mem_typed("facts", "s1", "cc:1", ["domain:payments", "tech:go"])
        b = _mem_typed("preferences", "s2", "cc:2", {**merge.mc.empty_tags(), "domain": "payments"})
        final = merge.build_final(merge._passthrough([a, b], 0) | {"_member_idxs": [0, 1]}, [a, b])
        self.assertEqual(final["tags"], {**merge.mc.empty_tags(), "domain": "payments", "tech": ["go"]})

    def test_build_final_with_no_member_tags_emits_the_empty_object(self):
        a = _mem_typed("facts", "s1", "cc:1", merge.mc.empty_tags())
        final = merge.build_final(merge._passthrough([a], 0), [a])
        self.assertEqual(final["tags"], merge.mc.empty_tags())


class OutputSchemaMatchesCode(unittest.TestCase):
    """The packaged memory.schema.json describes the merged output memory. This guards it from
    drifting from the code: it must be valid JSON (draft 2020-12), and its top-level `properties`
    must be exactly the keys build_final emits."""

    SCHEMA_PATH = os.path.join(os.path.dirname(merge.__file__), "schema", "memory.schema.json")

    def _a_final_memory(self):
        m = _mem_typed(
            "facts", "s1", "cc:1", {"domain": "backend", "activity": "coding", "component": ["cart"]}
        )
        return merge.build_final(merge._passthrough([m], 0), [m])

    def _a_list_answer_final_memory(self):
        # A member whose model answered its tags in the list form.
        m = _mem_temporal(
            "facts", "s2", "cc:2", "2026-05-01", "2026-05", "dated", "low", ["domain:backend", "tech:go"]
        )
        return merge.build_final(merge._passthrough([m], 0), [m])

    def _an_intention_final_memory(self):
        m = {
            "_arm": "facts",
            "statement": "revisit tz handling holistically",
            "type": "intention",
            "scope_level": "repo",
            "scope_ref": "acme/shop",
            "scope": "repo/acme/shop",
            "tags": {**merge.mc.empty_tags(), "keyword": ["tz"]},
            "memory_id": merge.mc.memory_id("revisit tz handling holistically"),
            "review": {"status": "hold-for-review"},
            "intention_meta": {"what": "holistic tz audit", "done_check": "a tz audit PR exists"},
            "origin": {"conversation_id": "cc:1", "tool": "claude-code"},
            "evidence": [{"quote": "we should revisit tz", "turn": 1, "speaker": "user"}],
        }
        return merge.build_final(merge._passthrough([m], 0), [m])

    def _load_schema(self):
        with open(self.SCHEMA_PATH, encoding="utf-8") as f:
            return json.load(f)  # raises on invalid JSON -> test fails

    def test_schema_enums_match_code_vocabularies(self):
        # Enum VALUES in the packaged schema must equal the code's vocabularies directly. The
        # build_final-sample validation only exercises `fact`/`intention`, so a rename or drop of
        # any other `type` (or a generality value) would otherwise slip through unnoticed.
        schema = self._load_schema()
        props = schema["properties"]
        self.assertEqual(set(props["type"]["enum"]), set(merge.mc.VALID_TYPES))
        self.assertEqual(
            {x for x in props["secondary_type"]["enum"] if x is not None}, set(merge.mc.VALID_TYPES)
        )
        self.assertEqual(set(props["generality"]["enum"]), set(merge.mc.GENERALITY_VALUES))

    def test_null_disposition_note_rejected(self):
        # A merged memory always emits disposition.note as a string ("" when confident), so the
        # schema types it non-null. A null note must fail validation.
        import jsonschema

        schema = self._load_schema()
        mem = self._a_final_memory()
        mem["disposition"]["note"] = None
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(mem, schema)

    def test_schema_is_valid_json_and_matches_output_keys(self):
        with open(self.SCHEMA_PATH, encoding="utf-8") as f:
            schema = json.load(f)  # raises on invalid JSON -> test fails
        self.assertEqual(schema.get("$schema"), "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema.get("type"), "object")
        schema_props = set(schema.get("properties", {}).keys())
        output_keys = set(self._a_final_memory().keys())
        self.assertEqual(
            schema_props,
            output_keys,
            f"schema drift: only-in-schema={schema_props - output_keys}, "
            f"only-in-output={output_keys - schema_props}",
        )
        # the tags sub-schema documents the six axes; domain carries NO enum (open value)
        # The canonical identifier resolves to the shipped schema in the public repository.
        self.assertEqual(
            schema["$id"],
            "https://raw.githubusercontent.com/tetriz-ai/agent-memory-engine/"
            "main/src/agent_memory_engine/schema/memory.schema.json",
        )
        tt = schema["properties"]["tags"]
        axis_props = set(tt.get("properties", {}).keys())
        self.assertEqual(axis_props, {"domain", "activity", "component", "tech", "concept", "keyword"})
        self.assertNotIn("enum", tt["properties"]["domain"])
        # every documented field carries a plain-language description
        for name, sub in schema["properties"].items():
            self.assertTrue(sub.get("description"), f"{name} is missing a description")

    def test_nested_keysets_and_axis_shapes_match_schema(self):
        # Guard NESTED drift the top-level name-set check is blind to. Two tiers, because the objects
        # differ in whether they are always fully emitted:
        #   _exact  — the grouped scope/temporal/provenance objects + evidence items are emitted with a
        #             FIXED key-set unconditionally (build_final / member_evidence build them literally),
        #             so their key-set must EQUAL the schema's declared one. Equality catches drift in
        #             BOTH directions: an emitted key the schema omits AND a schema-declared key the code
        #             stops emitting (the subset check below is blind to the latter).
        #   _subset — review/disposition/intention_meta are conditionally shaped (keys vary by type),
        #             so only the one-directional check applies: no emitted key may be undocumented.
        # additionalProperties:false on every nested object is what makes an undocumented key real drift.
        # Also assert the tag-axis SHAPES in a real sample: single-value axes scalar-or-null, open axes lists.
        schema = self._load_schema()
        props = schema["properties"]
        typed = self._a_final_memory()
        intention = self._an_intention_final_memory()

        def _subset(sample, declared, ctx):
            self.assertTrue(
                set(sample.keys()) <= declared,
                f"{ctx}: emitted keys {set(sample.keys()) - declared} absent from schema",
            )

        def _exact(sample, declared, ctx):
            self.assertEqual(
                set(sample.keys()),
                declared,
                f"{ctx} drift: only-in-output={set(sample.keys()) - declared}, "
                f"only-in-schema={declared - set(sample.keys())}",
            )

        # conditionally-shaped top-level nested objects: one-directional (no undocumented key)
        for field, sample in (("review", typed), ("disposition", typed), ("intention_meta", intention)):
            _subset(sample[field], set(props[field]["properties"].keys()), field)
        # scope: always {identified, applicable}; identified always {level, ref}; applicable null in the
        # common case (its non-null {level, ref} shape is pinned statically below, not via this sample).
        scope_props = props["scope"]["properties"]
        _exact(typed["scope"], set(scope_props.keys()), "scope")
        _exact(
            typed["scope"]["identified"],
            set(scope_props["identified"]["properties"].keys()),
            "scope.identified",
        )
        self.assertIsNone(typed["scope"]["applicable"])
        # applicable mirrors identified's {level, ref} shape and is nullable — a static schema check so a
        # divergent-applicable memory (rare) validates without needing a crafted runtime fixture here.
        self.assertEqual(
            set(scope_props["applicable"]["properties"].keys()),
            set(scope_props["identified"]["properties"].keys()),
            "scope.applicable must declare the same {level, ref} shape as scope.identified",
        )
        self.assertIn("null", scope_props["applicable"]["type"])
        # temporal: kind/volatility/valid_as_of
        _exact(typed["temporal"], set(props["temporal"]["properties"].keys()), "temporal")
        # provenance: confidence + nested corroboration
        prov_props = props["provenance"]["properties"]
        _exact(typed["provenance"], set(prov_props.keys()), "provenance")
        _exact(
            typed["provenance"]["corroboration"],
            set(prov_props["corroboration"]["properties"].keys()),
            "provenance.corroboration",
        )
        # evidence.items: fixed 6-key shape (authorship_carried is gone from both schema and output)
        ev_declared = set(props["evidence"]["items"]["properties"].keys())
        self.assertNotIn("authorship_carried", ev_declared)
        for ev in typed["evidence"]:
            _exact(ev, ev_declared, "evidence item")
            self.assertNotIn("authorship_carried", ev)
        # tag-axis shapes in a real sample: single-value scalar-or-null, open axes lists
        tt = typed["tags"]
        _exact(tt, set(props["tags"]["properties"].keys()), "tags")
        for axis in ("domain", "activity"):
            self.assertTrue(
                tt[axis] is None or isinstance(tt[axis], str),
                f"{axis} must be scalar-or-null in output, got {tt[axis]!r}",
            )
        for axis in ("component", "tech", "concept", "keyword"):
            self.assertIsInstance(tt[axis], list, f"{axis} must be a list in output")

    def test_schema_rejects_the_old_and_malformed_tag_shapes(self):
        import jsonschema

        schema = self._load_schema()
        path = os.path.join(
            os.path.dirname(__file__), "..", "examples", "find-memories-by-tag", "memories.jsonl"
        )
        with open(path, encoding="utf-8") as f:
            line = json.loads(f.readline())
        jsonschema.validate(line, schema)
        bad = {
            "a tags_typed key": {**line, "tags_typed": line["tags"]},
            "a flat tags list": {**line, "tags": ["domain:developer-tooling"]},
            "a seventh axis": {**line, "tags": {**line["tags"], "team": ["x"]}},
        }
        for name, sample in bad.items():
            with self.subTest(name), self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(sample, schema)

    def test_build_final_samples_validate_against_schema(self):
        # Full validation of build_final samples (typed, list answer, and an intention), catching
        # type, nullability and enum drift the key-set checks cannot see. jsonschema is a dev
        # dependency, so this never skips.
        import jsonschema

        schema = self._load_schema()
        jsonschema.Draft202012Validator.check_schema(schema)
        for sample in (
            self._a_final_memory(),
            self._a_list_answer_final_memory(),
            self._an_intention_final_memory(),
        ):
            jsonschema.validate(sample, schema)


class ProvenanceConfidenceInMerge(unittest.TestCase):
    """build_final RECOMPUTES provenance_confidence from the FINAL merged evidence set against
    the FINAL type — per-quote authorship is the single source of truth, not a stored member value."""

    def _member(self, mtype, cid, evidence):
        return {
            "_arm": "facts",
            "statement": f"stmt {mtype} {cid}",
            "type": mtype,
            "scope_level": "repo",
            "scope_ref": "acme/shop",
            "scope": "repo/acme/shop",
            "tags": merge.mc.empty_tags(),
            "memory_id": merge.mc.memory_id(f"stmt {mtype} {cid}"),
            "origin": {"conversation_id": cid, "tool": "claude-code"},
            "evidence": evidence,
        }

    def _pconf(self, mtype, evidence):
        m = self._member(mtype, "cc:1", evidence)
        return merge.build_final(merge._passthrough([m], 0), [m])["provenance"]["confidence"]

    def _ev(self, quote, authorship):
        return {"quote": quote, "turn": 1, "speaker": "user", "authorship": authorship}

    def test_all_admissible_is_high(self):
        self.assertEqual(self._pconf("preference", [self._ev("q", "human_prose")]), "high")
        self.assertEqual(self._pconf("fact", [self._ev("q", "model_generated")]), "high")

    def test_unknown_is_uncertain(self):
        self.assertEqual(self._pconf("preference", [self._ev("q", "unknown")]), "uncertain")

    def test_inadmissible_is_low(self):
        # template_boilerplate is inadmissible for any type -> low
        self.assertEqual(self._pconf("fact", [self._ev("q", "template_boilerplate")]), "low")
        # model_generated is inadmissible for a human-established type -> low
        self.assertEqual(self._pconf("preference", [self._ev("q", "model_generated")]), "low")
        # tool_output is inadmissible for a human-established type -> low
        self.assertEqual(self._pconf("preference", [self._ev("q", "tool_output")]), "low")

    def test_tool_output_admissible_for_knowledge_types(self):
        # Calibration fix: tool_output (schema DDL, code+line citations, proto defs, issue records)
        # is admissible for the knowledge types -> high, not low.
        self.assertEqual(self._pconf("fact", [self._ev("q", "tool_output")]), "high")

    def test_missing_authorship_reads_as_uncertain(self):
        # A member whose evidence carries no authorship key -> null -> uncertain.
        self.assertEqual(self._pconf("fact", [{"quote": "q", "turn": 1, "speaker": "user"}]), "uncertain")

    def test_worst_quote_wins_across_members(self):
        # One human_prose quote + one template_boilerplate quote on a preference -> low (inadmissible wins).
        a = self._member("preference", "cc:1", [self._ev("qa", "human_prose")])
        b = self._member("preference", "cc:2", [self._ev("qb", "template_boilerplate")])
        synth = {
            "statement": a["statement"],
            "type": "preference",
            "scope_level": "repo",
            "scope_ref": "acme/shop",
            "scope": "repo/acme/shop",
            "generality": "general-principle",
            "review": {"status": "active"},
            "_member_idxs": [0, 1],
        }
        self.assertEqual(merge.build_final(synth, [a, b])["provenance"]["confidence"], "low")


class _MergeStubClient:
    """A no-network stand-in for the merge client. For a SYNTHESIS call it leaves every member
    uncovered (returns no memories), so each member becomes its own passthrough keeping its own
    statement — a deterministic, order-preserving synthesized list. For the DEDUP call it returns
    no duplicate groups and RECORDS the exact dedup_input it received, so a test can assert the
    [S#] list was assembled in groups_idx order (never completion order). A per-group delay derived
    from the group's LOWEST member index makes completion order the OPPOSITE of submission order, so
    a 'write in completion order' regression would scramble the synthesized list."""

    model_id = "stub"
    cache_model_id = "stub"

    class _R:
        def __init__(self, raw, ok=True, err=None):
            self.ok = ok
            self.raw_text = raw
            self.cost_usd = 0.01
            self.duration_ms = 5
            self.model_id = "stub"
            self.error = err
            self.meta = {"canonical_model": "stub"}

    def __init__(
        self, delay_unit: float = 0.0, fail_on_stmt: str | None = None, raise_on_stmt: str | None = None
    ):
        self.delay_unit = delay_unit
        self.fail_on_stmt = fail_on_stmt
        self.raise_on_stmt = raise_on_stmt  # a group whose synthesis RAISES (unexpected exception)
        self.dedup_inputs: list[str] = []

    def extract(self, system_prompt, text):
        import re
        import time

        if text.startswith("Identify duplicate"):  # the DEDUP call
            self.dedup_inputs.append(text)
            return self._R('{"duplicate_groups":[]}')
        # a SYNTHESIS call: delay by the group's lowest member index (earlier groups finish LAST)
        member_idxs = [int(x) for x in re.findall(r"statement: mem-(\d+)", text)]
        if self.delay_unit and member_idxs:
            time.sleep(self.delay_unit * (10 - min(member_idxs)))
        if self.raise_on_stmt is not None and f"statement: {self.raise_on_stmt}" in text:
            raise RuntimeError(f"boom on {self.raise_on_stmt}")
        if self.fail_on_stmt is not None and f"statement: {self.fail_on_stmt}" in text:
            return self._R("", ok=False, err="synth-fail")
        return self._R('{"memories":[]}')  # nothing covered -> all passthroughs


def _write_merge_arms(n_mems: int):
    """Write `n_mems` memories (mem-0..mem-(n-1)) under ONE arm, and return (data_dir, arm_paths,
    mems). ARM_ORDER is set to that one arm by the caller."""
    data_dir = tempfile.mkdtemp(prefix="merge_conc_")
    path = os.path.join(data_dir, "arms", "facts", "memories.jsonl")
    os.makedirs(os.path.dirname(path))
    mems = []
    with open(path, "w") as f:
        for k in range(n_mems):
            m = _single("facts", f"mem-{k}", f"cc:{k}", [(f"q{k}", 1)])
            mems.append(m)
            f.write(json.dumps(m) + "\n")
    return data_dir, {"facts": path}, mems


class _MergeHarness(unittest.TestCase):
    """Base: pin ARM_ORDER to one arm and monkeypatch cluster_memories to a FIXED group layout so the
    concurrency behavior is tested deterministically, independent of the TF-IDF clusterer."""

    GROUPS = [[0, 1], [2, 3], [4, 5]]  # 3 clusters -> 3 synthesis groups, 6 members total

    def setUp(self):
        self._saved_arm_order = merge.ARM_ORDER
        self._saved_cluster = merge.cluster_memories
        merge.ARM_ORDER = ["facts"]
        merge.cluster_memories = lambda mems, threshold: ([list(g) for g in self.GROUPS], [])

    def tearDown(self):
        merge.ARM_ORDER = self._saved_arm_order
        merge.cluster_memories = self._saved_cluster

    def _run(self, client, workers, n_mems=6):
        data_dir, arm_paths, mems = _write_merge_arms(n_mems)
        out_dir = os.path.join(data_dir, "runs", f"merge_out_w{workers}")
        metrics, ok = merge.run(
            arm_paths,
            out_dir,
            client,
            cache_dir=os.path.join(data_dir, "cache"),
            workers=workers,
            log=lambda m: None,
        )
        memp = os.path.join(out_dir, "memories.jsonl")
        lines = []
        if os.path.exists(memp):
            with open(memp, encoding="utf-8") as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
        return metrics, ok, lines, mems


class MergeGroupParallelDeterminism(_MergeHarness):
    """workers=1 and workers>1 must produce a byte-identical memories.jsonl and stable metrics, and
    the dedup input must be assembled in groups_idx order regardless of synthesis completion order."""

    def _stable(self, m):
        return {k: v for k, v in m.items() if k not in ("wall_clock_s",)}

    def test_workers_1_vs_4_identical_output_and_dedup_order(self):
        for workers in (2, 4):
            with self.subTest(workers=workers):
                c1 = _MergeStubClient(delay_unit=0.0)
                cp = _MergeStubClient(delay_unit=0.01)
                m1, ok1, lines1, _ = self._run(c1, workers=1)
                mp, okp, linesp, _ = self._run(cp, workers=workers)
                self.assertTrue(ok1 and okp)
                self.assertEqual(lines1, linesp)  # byte-identical merged output
                self.assertEqual(self._stable(m1), self._stable(mp))
                # the dedup [S#] list is built in groups_idx order (mem-0..mem-5), never completion
                # order — a 'write in completion order' regression would reorder these.
                self.assertEqual(len(cp.dedup_inputs), 1)
                order = [
                    int(x) for x in __import__("re").findall(r"\[S\d+\][^\n]*?mem-(\d+)", cp.dedup_inputs[0])
                ]
                self.assertEqual(order, [0, 1, 2, 3, 4, 5])


class MergeGroupFatalStop(_MergeHarness):
    """A fatal in one synthesis group must STOP cleanly: ok=False, fatal_error names that group's
    key, only groups strictly BEFORE it are folded into metrics, and no memories.jsonl is written —
    identical whether the pool ran 1 or 4 wide (the later groups that ran in parallel are not folded)."""

    def test_fatal_group_folds_only_earlier_groups(self):
        # fail the SECOND group (idx 1 -> members [2,3]); group 0 ([0,1]) must fold, groups >1 must not.
        for workers in (1, 4):
            with self.subTest(workers=workers):
                client = _MergeStubClient(delay_unit=0.0, fail_on_stmt="mem-2")
                metrics, ok, lines, mems = self._run(client, workers=workers)
                ctx = f"workers={workers}"
                self.assertFalse(ok, ctx)
                self.assertEqual(lines, [], ctx)  # nothing written
                fe = metrics.get("fatal_error") or {}
                self.assertEqual(fe.get("group"), merge._group_key(mems, self.GROUPS[1]), ctx)
                # only group 0's two members were folded as uncovered passthroughs; groups >1 not folded
                self.assertEqual(metrics["members_uncovered"], 2, ctx)


class MergeGroupFatalOrdering(_MergeHarness):
    """Fix: with >1 worker a LATER group's raised exception must NOT beat an EARLIER group's
    controlled `None` fatal. The parallel path collects every outcome by INDEX and acts on the first
    failing group only, so a controlled fatal at group 0 wins over an exception at group 2 even when
    the exception's future completes first — exactly as the sequential runner (which fails at the
    first group in index order, whatever the failure kind) would order them."""

    def test_earlier_controlled_fatal_wins_over_later_raise(self):
        # group 0 ([0,1]) -> controlled None fatal (mem-0); group 2 ([4,5]) -> RAISES (mem-4). The
        # delay makes the LATER (raising) group finish FIRST, so completion-order handling
        # would let the exception propagate out of run(); the run must instead take the clean fatal
        # path at group 0 and never raise.
        for workers in (2, 4):
            with self.subTest(workers=workers):
                client = _MergeStubClient(delay_unit=0.01, fail_on_stmt="mem-0", raise_on_stmt="mem-4")
                metrics, ok, lines, mems = self._run(client, workers=workers)
                ctx = f"workers={workers}"
                self.assertFalse(ok, ctx)  # clean fatal return, no propagated raise
                self.assertEqual(lines, [], ctx)  # nothing written
                fe = metrics.get("fatal_error") or {}
                self.assertEqual(fe.get("group"), merge._group_key(mems, self.GROUPS[0]), ctx)


class MergeWorkersClamp(_MergeHarness):
    """--workers is clamped to MERGE_MAX_WORKERS inside run() for every caller (run-level, not just
    the CLI), so an over-large value never spawns an unbounded pool."""

    def test_workers_clamped_to_cap(self):
        captured = {}
        real_pool = merge.ThreadPoolExecutor

        def _spy(*args, **kwargs):
            captured["max_workers"] = kwargs.get("max_workers", args[0] if args else None)
            return real_pool(*args, **kwargs)

        merge.ThreadPoolExecutor = _spy
        try:
            _metrics, ok, _lines, _ = self._run(_MergeStubClient(), workers=99)
        finally:
            merge.ThreadPoolExecutor = real_pool
        self.assertTrue(ok)
        self.assertEqual(captured["max_workers"], merge.MERGE_MAX_WORKERS)  # 99 -> 6


class MergeStageTokenAccounting(unittest.TestCase):
    """The merge stage must persist each call's usage into its cache record and report the stage's
    token anatomy — and a warm replay must report the same tokens as the cold call did, the same
    way it replays the cost."""

    USAGE = {
        "promptTokenCount": 900,
        "candidatesTokenCount": 120,
        "thoughtsTokenCount": 3_000,
        "cachedContentTokenCount": 400,
    }

    class _Result:
        def __init__(self, raw, usage, service_tier="standard"):
            self.ok = True
            self.raw_text = raw
            self.cost_usd = 0.01
            self.duration_ms = 5
            self.model_id = "stub"
            self.meta = {"canonical_model": "stub", "usage": usage}

    class _Client:
        model_id = "stub"
        cache_model_id = "stub"

        def __init__(self, raw, usage):
            self._raw, self._usage, self.calls = raw, usage, 0

        def extract(self, sysp, text):
            self.calls += 1
            return MergeStageTokenAccounting._Result(self._raw, self._usage)

    RAW = (
        '{"memories":[{"members":["M1","M2"],"statement":"Run ci and let it fully pass before '
        'merging.","type":"convention","scope":"repo/acme-shop","tags":["ci"],'
        '"generality":"general-principle"}]}'
    )

    def _synth(self, client, cache, metrics):
        mems = [
            _single("facts", "run ci before merge", "cc:1", [("q1", 1)]),
            _single("preferences", "ci must pass before merge", "cc:2", [("q2", 1)]),
        ]
        return merge.synthesize_group(mems, [0, 1], "sysprompt", client, cache, metrics)

    def test_usage_is_persisted_and_replays_identically(self):

        cache = cache_mod.Cache(tempfile.mkdtemp(prefix="merge_usage_"))
        client = self._Client(self.RAW, self.USAGE)

        cold = merge._fresh_group_metrics()
        self.assertIsNotNone(self._synth(client, cache, cold))
        self.assertEqual(cold["llm_calls"], 1)
        self.assertEqual(cold["prompt_tokens_total"], 900)
        self.assertEqual(cold["thinking_tokens_total"], 3_000)
        self.assertEqual(cold["cached_tokens_total"], 400)
        self.assertEqual(cold["usage_missing_calls"], 0)
        # The stored cost is IGNORED in favour of re-pricing from the record's own usage block, so
        # the stub's placeholder $0.01 does not reach the metric.
        self.assertAlmostEqual(cold["cost_usd_total"], mcl.gemini_call_cost(self.USAGE), places=9)
        self.assertNotAlmostEqual(cold["cost_usd_total"], 0.01, places=4)

        warm = merge._fresh_group_metrics()
        self.assertIsNotNone(self._synth(client, cache, warm))
        self.assertEqual(client.calls, 1)  # the second run replayed the cache
        self.assertEqual(warm["cache_hits"], 1)
        for k in USAGE_METRIC_KEYS:
            self.assertEqual(warm[k], cold[k], k)

    def test_both_merge_prompts_caches_are_recorded_and_summed(self):
        """The merge stage calls under TWO prompts — per-group synthesis and the cross-group dedup —
        so one client can hold two resources and both bill storage. Recording only the first
        under-reports the arm."""

        class _Client:
            explicit_cache = True
            explicit_caches = [
                {"name": "cachedContents/synth", "tokens": 4_700, "lifetime_s": 600.0, "live": False},
                {"name": "cachedContents/dedup", "tokens": 4_200, "lifetime_s": 200.0, "live": False},
            ]
            explicit_cache_prompts = [
                {
                    "role": "synthesis",
                    "status": "cached",
                    "tokens": 4_700,
                    "name": "cachedContents/synth",
                    "reason": None,
                },
                {
                    "role": "dedup",
                    "status": "cached",
                    "tokens": 4_200,
                    "name": "cachedContents/dedup",
                    "reason": None,
                },
            ]

        metrics = {
            "arm": "merge",
            "explicit_caches": [],
            "explicit_cache_prompts": [],
            "explicit_cache_tokens": 0,
            "explicit_cache_storage_usd": 0.0,
        }
        merge._record_explicit_caches(metrics, (_Client(),))
        out = tempfile.mkdtemp(prefix="merge_caches_")
        merge._write_metrics(out, metrics, 0.0)
        with open(os.path.join(out, "metrics.json"), encoding="utf-8") as f:
            on_disk = json.load(f)

        # IN-MEMORY == ON-DISK. Folding the fields inside _write_metrics' local copy left the
        # returned and printed metrics reporting zeros for a run whose metrics.json had them —
        # exactly the split that gets read as "the lever did nothing".
        for key in (
            "explicit_caches",
            "explicit_cache_prompts",
            "explicit_cache_tokens",
            "explicit_cache_storage_usd",
        ):
            self.assertEqual(metrics[key], on_disk[key], key)
        self.assertEqual(
            [c["name"] for c in metrics["explicit_caches"]], ["cachedContents/synth", "cachedContents/dedup"]
        )
        self.assertEqual([p["role"] for p in metrics["explicit_cache_prompts"]], ["synthesis", "dedup"])
        self.assertEqual(metrics["explicit_cache_tokens"], 4_700 + 4_200)
        # each resource on ITS OWN lifetime (600s and 200s), not one shared elapsed
        self.assertAlmostEqual(
            metrics["explicit_cache_storage_usd"],
            mcl.gemini_cache_storage_cost(4_700, 600.0) + mcl.gemini_cache_storage_cost(4_200, 200.0),
            places=6,
        )
        self.assertLess(metrics["explicit_cache_storage_usd"], mcl.gemini_cache_storage_cost(8_900, 3_600.0))

    def test_a_stage_with_no_cache_records_nothing(self):
        metrics = {
            "explicit_caches": [],
            "explicit_cache_prompts": [],
            "explicit_cache_tokens": 0,
            "explicit_cache_storage_usd": 0.0,
        }
        merge._record_explicit_caches(metrics, (None,))
        self.assertEqual(metrics["explicit_caches"], [])
        self.assertEqual(metrics["explicit_cache_storage_usd"], 0.0)

    def test_a_call_without_usage_is_counted_not_silently_zeroed(self):

        cache = cache_mod.Cache(tempfile.mkdtemp(prefix="merge_usage_none_"))
        metrics = merge._fresh_group_metrics()
        self._synth(self._Client(self.RAW, None), cache, metrics)
        self.assertEqual(metrics["usage_missing_calls"], 1)
        self.assertEqual(metrics["prompt_tokens_total"], 0)

    def test_fold_group_metrics_carries_the_token_counters(self):
        metrics = {
            k: 0
            for k in (
                "llm_calls",
                "cache_hits",
                "members_uncovered",
                "cost_usd_total",
                "model_duration_ms_total",
                "http_retries",
                *USAGE_METRIC_KEYS,
            )
        }
        gm = merge._fresh_group_metrics()
        gm["prompt_tokens_total"] = 900
        gm["usage_missing_calls"] = 2
        gm["http_retries"] = 3
        merge._fold_group_metrics(metrics, gm)
        merge._fold_group_metrics(metrics, gm)
        self.assertEqual(metrics["prompt_tokens_total"], 1_800)
        self.assertEqual(metrics["usage_missing_calls"], 4)
        self.assertEqual(metrics["http_retries"], 6)
        for k in USAGE_METRIC_KEYS:
            self.assertIn(k, metrics)


class AFatalMergeReportsWhatItPaidFor(unittest.TestCase):
    """On a fatal, the failing group and the groups that had already run in parallel (billed,
    sitting in the cache) are all counted in the metrics, so a stopped merge reports the work it
    paid for, never $0.00."""

    def test_the_fatal_group_and_the_abandoned_ones_are_both_accounted(self):
        metrics = {
            k: 0
            for k in (
                "llm_calls",
                "cache_hits",
                "members_uncovered",
                "cost_usd_total",
                "model_duration_ms_total",
                "http_retries",
                *USAGE_METRIC_KEYS,
            )
        }
        metrics["canonical_model"] = None
        metrics["partial_groups"] = []

        def _gm(calls, cost, prompt_tokens):
            gm = merge._fresh_group_metrics()
            gm["llm_calls"] = calls
            gm["cost_usd_total"] = cost
            gm["prompt_tokens_total"] = prompt_tokens
            return gm

        ran = _gm(2, 0.04, 2_000)
        never = merge._fresh_group_metrics()
        self.assertTrue(merge._group_ran(ran))
        self.assertFalse(merge._group_ran(never), "a group the pool never reached owes nothing")

        merge._fold_group_spend(metrics, ran, "group:abc", "abandoned")
        self.assertEqual(metrics["llm_calls"], 2)
        self.assertAlmostEqual(metrics["cost_usd_total"], 0.04, places=9)
        self.assertEqual(metrics["prompt_tokens_total"], 2_000)
        self.assertEqual(
            metrics["partial_groups"],
            [
                {
                    "group": "group:abc",
                    "status": "abandoned",
                    "calls": 2,
                    "llm_calls": 2,
                    "cache_hits": 0,
                    "cost_usd": 0.04,
                }
            ],
        )
        # spend only: the quality counters stay with the groups that contributed output
        self.assertEqual(metrics["members_uncovered"], 0)

    def test_spend_folding_does_not_double_count_the_quality_counters(self):
        full = {
            k: 0
            for k in (
                "llm_calls",
                "cache_hits",
                "members_uncovered",
                "cost_usd_total",
                "model_duration_ms_total",
                "http_retries",
                *USAGE_METRIC_KEYS,
            )
        }
        full["canonical_model"] = None
        full["partial_groups"] = []
        gm = merge._fresh_group_metrics()
        gm["llm_calls"] = 1
        gm["cost_usd_total"] = 0.01
        gm["members_uncovered"] = 3
        merge._fold_group_spend(full, gm, "g", "abandoned")
        self.assertEqual(full["llm_calls"], 1)
        self.assertEqual(full["members_uncovered"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class FailedMergeCallsAreCounted(_MergeHarness):
    """A failed synthesis or dedup call is a call, and its billed spend counts, also for a group
    that failed while another group's failure stopped the merge."""

    GROUPS = [[0, 1], [2, 3]]

    @staticmethod
    def _result(ok, raw="", cost=0.0):
        r = ExtractResult(raw, cost, 1, "stub", ok=ok, error=None if ok else "http 500: busy")
        r.meta["http_retries"] = 0 if ok else 2
        return r

    def test_parallel_failed_synthesis_calls(self):
        both_in = threading.Barrier(2, timeout=10)

        class _Failing:
            model_id = cache_model_id = "stub"

            def extract(self, system_prompt, text):
                both_in.wait()
                return FailedMergeCallsAreCounted._result(False, cost=0.5)

        metrics, ok, _, _ = self._run(_Failing(), workers=4, n_mems=4)
        self.assertFalse(ok)
        self.assertAlmostEqual(metrics["cost_usd_total"], 1.0, places=9)
        self.assertEqual(metrics["llm_calls"], 2)
        self.assertEqual(metrics["http_retries"], 4)

    def test_a_failed_dedup_call(self):
        class _DedupFails:
            model_id = cache_model_id = "stub"

            def extract(self, system_prompt, text):
                if text.startswith("Identify duplicate"):
                    return FailedMergeCallsAreCounted._result(False, cost=0.5)
                return FailedMergeCallsAreCounted._result(True, raw='{"memories": []}')

        metrics, ok, _, _ = self._run(_DedupFails(), workers=1, n_mems=4)
        self.assertTrue(ok)
        self.assertEqual(metrics["dedup_error"], "http 500: busy")
        self.assertEqual(metrics["llm_calls"], 3)  # two synthesis calls and the failed dedup call
        self.assertAlmostEqual(metrics["cost_usd_total"], 0.5, places=9)


class MergeCacheWriteFailure(_MergeHarness):
    """A cache write that fails after a billed call stops the merge as a fatal, and metrics.json still
    reports what was spent."""

    class _Paid:
        model_id = cache_model_id = "stub"

        def __init__(self):
            self.dedup_calls = 0

        def extract(self, system_prompt, text):
            if text.startswith("Identify duplicate"):
                self.dedup_calls += 1
            return ExtractResult('{"memories": []}', 0.25, 1, "stub", ok=True)

    def _merge(self, client, cache_factory, workers):
        data_dir, arm_paths, _ = _write_merge_arms(6)
        out_dir = os.path.join(data_dir, "out")
        with mock.patch.object(merge.cache_mod, "Cache", cache_factory):
            metrics, ok = merge.run(
                arm_paths,
                out_dir,
                client,
                cache_dir=os.path.join(data_dir, "cache"),
                workers=workers,
                log=lambda m: None,
            )
        return metrics, ok, out_dir

    def test_a_merge_cache_write_failure_still_reports_its_spend(self):
        real = cache_mod.Cache

        class _Exploding(real):
            def put(self, fp, res):
                raise OSError("disk full")

        for workers, calls in ((1, 1), (4, 3)):
            with self.subTest(workers=workers):
                metrics, ok, out_dir = self._merge(self._Paid(), _Exploding, workers)
                self.assertFalse(ok)
                self.assertIn("disk full", metrics["fatal_error"]["error"])
                self.assertEqual(metrics["llm_calls"], calls)
                self.assertAlmostEqual(metrics["cost_usd_total"], 0.25 * calls, places=9)
                with open(os.path.join(out_dir, "metrics.json")) as f:
                    self.assertAlmostEqual(json.load(f)["cost_usd_total"], 0.25 * calls, places=6)

    def test_a_dedup_cache_write_failure_still_reports_its_spend(self):
        real = cache_mod.Cache
        client = self._Paid()

        class _DedupWriteFails(real):
            def put(self, fp, res):
                if client.dedup_calls:
                    raise OSError("disk full")
                return super().put(fp, res)

        metrics, ok, out_dir = self._merge(client, _DedupWriteFails, 1)
        self.assertFalse(ok)
        self.assertEqual(metrics["fatal_error"]["stage"], "dedup")
        self.assertEqual(metrics["llm_calls"], 4)  # three synthesis calls and the dedup call
        self.assertAlmostEqual(metrics["cost_usd_total"], 1.0, places=9)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "metrics.json")))


class FailedGroupsAndReruns(_MergeHarness):
    GROUPS = [[0, 1], [2, 3]]

    def test_a_failed_group_that_billed_nothing_is_still_named(self):
        both_in = threading.Barrier(2, timeout=10)

        class _FailsFree:
            model_id = cache_model_id = "stub"

            def extract(self, system_prompt, text):
                both_in.wait()
                return ExtractResult("", 0.0, 1, "stub", ok=False, error="http 500: busy")

        metrics, ok, _, _ = self._run(_FailsFree(), workers=4, n_mems=4)
        self.assertFalse(ok)
        self.assertEqual([p["status"] for p in metrics["partial_groups"]], ["fatal", "abandoned"])
        self.assertEqual(metrics["llm_calls"], 2)

    def test_a_rerun_after_a_failure_replays_only_what_was_cached(self):
        data_dir, arm_paths, _ = _write_merge_arms(4)

        class _SecondFails:
            model_id = cache_model_id = "stub"

            def __init__(self, fail):
                self.fail = fail

            def extract(self, system_prompt, text):
                if self.fail and "mem-2" in text:
                    return ExtractResult("", 0.0, 1, "stub", ok=False, error="http 500: busy")
                return ExtractResult('{"memories": []}', 0.0, 1, "stub", ok=True)

        kw = {"cache_dir": os.path.join(data_dir, "cache"), "workers": 1, "log": lambda m: None}
        _, ok1 = merge.run(arm_paths, os.path.join(data_dir, "a"), _SecondFails(True), **kw)
        second, ok2 = merge.run(arm_paths, os.path.join(data_dir, "b"), _SecondFails(False), **kw)
        self.assertEqual((ok1, ok2), (False, True))
        self.assertEqual(
            (second["cache_hits"], second["llm_calls"]), (1, 2)
        )  # group 1 replayed; group 2 and dedup


class ArmLinesSkippedAreCounted(_MergeHarness):
    def test_malformed_arm_lines_reach_the_metrics(self):
        data_dir, arm_paths, _ = _write_merge_arms(6)
        with open(arm_paths["facts"], "a") as f:
            f.write("not json\n")
            f.write(json.dumps({"statement": ""}) + "\n")
        metrics, ok = merge.run(
            arm_paths,
            os.path.join(data_dir, "out"),
            _MergeStubClient(),
            cache_dir=os.path.join(data_dir, "cache"),
            log=lambda m: None,
        )
        self.assertTrue(ok)
        self.assertEqual(metrics["arm_lines_skipped"], 2)
