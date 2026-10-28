# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""Statement clustering for the merge: same groups as the straightforward average-linkage
definition, at a cost that grows about quadratically rather than cubically."""

from __future__ import annotations

import copy
import json
import os
import random
import time
import unittest

from agent_memory_engine import cluster, merge


def _reference_cluster_documents(ids, docs_tokens, threshold):
    """The textbook form: recompute every pair's average link every round. Kept here as the
    definition the fast form must reproduce exactly."""
    vecs = cluster.tfidf_vectors(docs_tokens)
    n = len(ids)
    sims = {}
    for i in range(n):
        for j in range(i + 1, n):
            sims[(i, j)] = cluster.cosine(vecs[i], vecs[j])

    def pair_sim(a, b):
        return sims[(a, b)] if a < b else sims[(b, a)]

    members = {i: [i] for i in range(n)}

    def avg_link(a, b):
        ma, mb = members[a], members[b]
        return sum(pair_sim(x, y) for x in ma for y in mb) / (len(ma) * len(mb))

    while True:
        active = sorted(members)
        best, best_s = None, threshold
        for ai in range(len(active)):
            for bi in range(ai + 1, len(active)):
                a, b = active[ai], active[bi]
                s = avg_link(a, b)
                if s > best_s or (best is None and s >= best_s):
                    best_s, best = s, (a, b)
        if best is None:
            break
        a, b = best
        members[a] = sorted(members[a] + members[b])
        del members[b]
    clusters = sorted((sorted(g) for g in members.values() if len(g) >= 2), key=lambda g: g[0])
    singletons = sorted(g[0] for g in members.values() if len(g) == 1)
    return clusters, singletons, sims


def _topical_docs(seed, n, n_topics=12, dup_every=0):
    rng = random.Random(seed)
    topics = [[f"t{k}w{i}" for i in range(15)] for k in range(n_topics)]
    common = [f"c{i}" for i in range(120)]
    docs = []
    for i in range(n):
        if dup_every and i and i % dup_every == 0:
            docs.append(list(docs[i - 1]))  # an exact duplicate: equal similarities, a tie
        else:
            docs.append(rng.sample(topics[rng.randrange(n_topics)], 6) + rng.sample(common, 6))
    return docs


class SameGroupsAsTheDefinition(unittest.TestCase):
    def test_differential(self):
        cases = [
            ("topical", _topical_docs(1, 60), 0.25),
            ("with duplicate documents (ties)", _topical_docs(2, 60, dup_every=4), 0.25),
            ("all identical (every pair ties)", [["alpha", "beta", "gamma"]] * 12, 0.25),
            (
                "random vocabulary",
                [random.Random(3).sample([f"w{i}" for i in range(80)], 8) for _ in range(50)],
                0.10,
            ),
            ("threshold equal to a tie value", [["a", "b"], ["a", "b"], ["c", "d"], ["c", "d"]], 1.0),
            ("nothing clusters", [["x1"], ["x2"], ["x3"]], 0.25),
            ("empty", [], 0.25),
        ]
        for name, docs, threshold in cases:
            with self.subTest(name):
                ids = list(range(len(docs)))
                want = _reference_cluster_documents(ids, docs, threshold)
                got = cluster.cluster_documents(ids, docs, threshold)
                self.assertEqual(got[0], want[0])
                self.assertEqual(got[1], want[1])
                self.assertEqual(got[2], want[2])


class Boundaries(unittest.TestCase):
    def test_an_exact_tie_merges_the_earlier_pair(self):
        # docs 0 and 1 are each exactly as similar to doc 2; (0, 2) comes first, so it merges, and
        # the merged pair is then too far from doc 1 for another merge.
        docs = [["aa", "bb"], ["cc", "dd"], ["aa", "bb", "cc", "dd"]]
        clusters, singletons, sims = cluster.cluster_documents([0, 1, 2], docs, 0.5)
        self.assertEqual(sims[(0, 2)], sims[(1, 2)])
        self.assertEqual((clusters, singletons), ([[0, 2]], [1]))
        self.assertEqual(_reference_cluster_documents([0, 1, 2], docs, 0.5)[:2], (clusters, singletons))

    def test_a_link_equal_to_the_threshold_merges(self):
        docs = [["aa", "bb"], ["cc", "dd"], ["aa", "bb", "cc", "dd"]]
        s = cluster.cluster_documents([0, 1, 2], docs, 0.9)[2][(0, 2)]
        self.assertEqual(cluster.cluster_documents([0, 1, 2], docs, s)[0], [[0, 2]])
        self.assertEqual(cluster.cluster_documents([0, 1, 2], docs, s + 1e-9)[0], [])

    def test_many_merges_match_the_definition(self):
        # Low thresholds force long merge sequences, where a stale best-partner row would show.
        for seed in (11, 12, 13):
            for threshold in (0.05, 0.15):
                with self.subTest(seed=seed, threshold=threshold):
                    docs = _topical_docs(seed, 90, n_topics=6, dup_every=7)
                    ids = list(range(len(docs)))
                    self.assertEqual(
                        cluster.cluster_documents(ids, docs, threshold)[:2],
                        _reference_cluster_documents(ids, docs, threshold)[:2],
                    )


class GrowthIsQuadratic(unittest.TestCase):
    def test_doubling_the_input_costs_well_under_eight_times(self):
        # Cubic growth costs about 8x (measured ~13x) per doubling; quadratic about 4x. The ratio
        # is taken from the best of three runs so a busy machine does not decide it.
        def best_time(n):
            docs = _topical_docs(7, n, n_topics=40)
            ts = []
            for _ in range(3):
                t = time.perf_counter()
                cluster.cluster_documents(list(range(n)), docs, 0.25)
                ts.append(time.perf_counter() - t)
            return min(ts)

        small, large = best_time(150), best_time(300)
        self.assertLess(large / small, 6.5, f"150 -> {small:.3f}s, 300 -> {large:.3f}s")


class DefaultThreshold(unittest.TestCase):
    def test_the_merge_clusters_at_a_quarter(self):
        from agent_memory_engine import merge

        statements = [
            "Retry billing charges with the original idempotency key",
            "The billing worker retries failed charges twice",  # 0.26 to the first: groups
            "The search index rebuild runs nightly on the worker pool",
            "Rebuild the search cache after every schema migration",  # 0.22 to the third: does not
            "Use pnpm in each worktree for the shop frontend",
            "Rotate the staging database password every quarter",
        ]
        self.assertEqual(merge.DEFAULT_THRESHOLD, 0.25)
        clusters, _ = merge.cluster_memories([{"statement": s} for s in statements], merge.DEFAULT_THRESHOLD)
        self.assertEqual(clusters, [[0, 1]])


class ClusteringReadsStatementsOnly(unittest.TestCase):
    """The merge clusters on statements alone, so the shape of `tags` cannot move a memory between
    groups. Pinned over the sample corpus, the recorded-session example's output."""

    CORPUS = os.path.join(
        os.path.dirname(__file__), "..", "examples", "find-memories-by-tag", "memories.jsonl"
    )

    def test_groups_do_not_depend_on_tags(self):
        with open(self.CORPUS, encoding="utf-8") as f:
            mems = [json.loads(line) for line in f]
        self.assertEqual(len(mems), 5)
        # The same memories with the flat `axis:value` list and no tags at all.
        flat = copy.deepcopy(mems)
        for m in flat:
            typed = m["tags"]
            m["tags"] = [
                f"{a}:{v}" for a, vs in typed.items() for v in (vs if isinstance(vs, list) else [vs]) if v
            ]
        bare = copy.deepcopy(mems)
        for m in bare:
            del m["tags"]
        pinned = ((merge.DEFAULT_THRESHOLD, ([[1, 2]], [0, 3, 4])), (0.1, ([[1, 2, 3]], [0, 4])))
        for threshold, groups in pinned:
            for variant in (mems, flat, bare):
                self.assertEqual(merge.cluster_memories(variant, threshold), groups)
