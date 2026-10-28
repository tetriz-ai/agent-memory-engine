# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""Deterministic TF-IDF + average-linkage clustering (no model). The merge stage uses it to place
similar memory statements in one candidate group before the model unifies or splits them."""

from __future__ import annotations

import collections
import math
import re

from .transcript import Conversation

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]{2,}")
# Vocabulary common to almost every coding session — carries no topical signal, so it is
# excluded before TF-IDF rather than merely down-weighted (keeps vectors small and honest).
_STOP = set(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "is",
        "it",
        "be",
        "as",
        "at",
        "by",
        "for",
        "on",
        "if",
        "we",
        "you",
        "i",
        "this",
        "that",
        "with",
        "from",
        "into",
        "not",
        "no",
        "yes",
        "do",
        "does",
        "did",
        "done",
        "can",
        "will",
        "would",
        "should",
        "could",
        "have",
        "has",
        "had",
        "was",
        "were",
        "are",
        "am",
        "been",
        "being",
        "being",
        "use",
        "used",
        "using",
        "get",
        "got",
        "run",
        "running",
        "file",
        "files",
        "code",
        "let",
        "lets",
        "like",
        "just",
        "also",
        "then",
        "than",
        "so",
        "out",
        "up",
        "down",
        "off",
        "over",
        "here",
        "there",
        "what",
        "which",
        "who",
        "how",
        "why",
        "when",
        "where",
        "your",
        "our",
        "their",
        "them",
        "they",
    ]
)


def conversation_tokens(conv: Conversation) -> list[str]:
    """Salient tokens for similarity: identifier-ish words from the whole transcript, stopped and
    lowercased. Works on any rendered Conversation."""
    text = "\n".join(t.content for t in conv.turns).lower()
    return [w for w in _TOKEN_RE.findall(text) if w not in _STOP and not w.isdigit()]


def tfidf_vectors(docs_tokens: list[list[str]]) -> list[dict]:
    n = len(docs_tokens)
    df: collections.Counter = collections.Counter()
    for toks in docs_tokens:
        for t in set(toks):
            df[t] += 1
    vecs = []
    for toks in docs_tokens:
        tf = collections.Counter(toks)
        total = sum(tf.values()) or 1
        v: dict[str, float] = {}
        for t, c in tf.items():
            idf = math.log((1 + n) / (1 + df[t])) + 1.0  # smoothed idf
            v[t] = (c / total) * idf
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        for t in v:
            v[t] /= norm
        vecs.append(v)
    return vecs


def cosine(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


def cluster_documents(ids: list, docs_tokens: list[list[str]], threshold: float):
    """AVERAGE-LINKAGE agglomerative clustering over the TF-IDF cosine graph. Returns
    (clusters, singletons, pairwise_sims): clusters is a list of index-lists (each >= 2),
    singletons is the indices left alone. Deterministic: sorted output, deterministic tie-break.

    Why average-linkage and not single-linkage connected components: one person's memories share a
    lot of vocabulary, so a few vocabulary-rich items are similar to almost everything. Single-linkage
    chains through those hubs and collapses a whole batch into one blob. Average-linkage requires an
    item to be similar to a group's members on average before it joins, which keeps clusters
    topically coherent.

    Each round merges the pair of clusters with the highest average link, if it is at least
    `threshold`; on a tie, the pair that comes first in (lower id, higher id) order. A pair's average
    link is computed once and kept until one of its two clusters changes, and every cluster keeps its
    best partner among the clusters after it, so a round costs about one pass over the clusters
    instead of a pass over every pair. The values and the tie-break are exactly those of recomputing
    every pair every round, so the groups are the same.
    """
    vecs = tfidf_vectors(docs_tokens)
    n = len(ids)
    sims: dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            sims[(i, j)] = cosine(vecs[i], vecs[j])

    def pair_sim(a: int, b: int) -> float:
        return sims[(a, b)] if a < b else sims[(b, a)]

    members: dict = {i: [i] for i in range(n)}  # cluster-rep -> member indices

    def avg_link(a: int, b: int) -> float:
        ma, mb = members[a], members[b]
        return sum(pair_sim(x, y) for x in ma for y in mb) / (len(ma) * len(mb))

    link: dict = {}  # (a, b) with a < b -> average link, valid while neither cluster changes
    for i in range(n):
        for j in range(i + 1, n):
            link[(i, j)] = avg_link(i, j)

    best: dict = {}  # rep -> (link, partner): its best partner among the reps after it

    def recompute_row(a: int) -> None:
        top = None
        for b in sorted(members):
            if b <= a:
                continue
            s = link[(a, b)]
            if top is None or s > top[0]:
                top = (s, b)
        if top is None:
            best.pop(a, None)
        else:
            best[a] = top

    for a in range(n):
        recompute_row(a)

    while best:
        # The highest link overall; on a tie the lowest rep, and within a row the lowest partner.
        a = min(best)
        for r in sorted(best):
            if best[r][0] > best[a][0]:
                a = r
        s, b = best[a]
        if s < threshold:
            break
        for k in members:
            if k != b:
                link.pop((k, b) if k < b else (b, k), None)
        members[a] = sorted(members[a] + members[b])
        del members[b]
        best.pop(b, None)
        for k in members:
            if k != a:
                link[(k, a) if k < a else (a, k)] = avg_link(k, a) if k < a else avg_link(a, k)
        recompute_row(a)
        for k in sorted(members):
            if k == a:
                continue
            if k > b:
                break
            row = best.get(k)
            if row is None or row[1] in (a, b):
                recompute_row(k)
            elif k < a:
                # The merged link is a weighted mean of k's links to the two parts, so it cannot pass
                # k's current best except by float rounding; this keeps the result exact if it does.
                s_new = link[(k, a)]
                if s_new > row[0] or (s_new == row[0] and a < row[1]):
                    best[k] = (s_new, a)

    clusters = sorted((sorted(g) for g in members.values() if len(g) >= 2), key=lambda g: g[0])
    singletons = sorted(g[0] for g in members.values() if len(g) == 1)
    return clusters, singletons, sims
