# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""The memory format, its validation, and the evidence gate.

The model proposes; plain code disposes. The model emits statement, type, scope, evidence,
decision_outcome, tags and disposition; code stamps identity (memory_id), origin (which
conversation) and when (its date).

The evidence gate is the load-bearing check: for every cited quote, code confirms the quote really
appears in the conversation as sent, at the claimed turn, spoken by the claimed speaker. Position
and speaker matter as much as the text: the same words can recur in another turn, and an
assistant's words must never pass as the person's own. A memory any of whose quotes fails is
dropped. Plain validation, no third-party dependency.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

MEMORY_FORMAT_VERSION = "v3"

# "type": what kind of memory. (decision carries its adopted-vs-rejected outcome via
# `decision_outcome`; an intention is a "we should do X soon" nudge/TODO, captured but held for
# review, never loaded without a human check.) A memory that is genuinely two kinds at once (e.g. a
# cache config choice that is both a decision AND a fact) may carry ONE optional `secondary_type`
# from this same set — a single field, never a list — with the primary `type` staying authoritative.
VALID_TYPES = {"fact", "preference", "decision", "convention", "failed-attempt", "caution", "intention"}
# Scope is a fixed enum (`scope_level`) plus a free target string (`scope_ref`). The org/repo/service
# names are whatever the conversation reveals (never assumed); an `org` may be the user's own or a
# separate one. personal carries no ref (scope_ref is null).
# `scope_level`/`scope_ref` are the IDENTIFIED scope — where the memory was OBSERVED. A memory may
# also carry an optional APPLICABLE scope (`scope_applicable_level` + `scope_applicable_ref`) for the
# case where it applies more broadly (or narrowly) than where it was seen — e.g. a dir-level
# convention that actually holds repo-wide. It is stored ONLY when applicability differs from the
# identified scope; otherwise both applicable fields are null and the identified scope is the whole story.
#
# scope_ref CONVENTION — what the target string holds at each level (ONE consistent format per level):
#   personal -> null (no target)
#   dir      -> the directory path (e.g. "src/api/handlers")
#   service  -> the BARE service name, lowercased, no owner/repo qualifier (e.g. "gateway", "billing")
#   repo     -> "<owner>/<repo>"
#   team     -> the team name
#   org      -> the org name (e.g. a cloud provider like "aws", or "<owner>") — an org ref need not be
#               the user's own org
VALID_SCOPE_LEVELS = {"personal", "dir", "service", "repo", "team", "org"}
VALID_DISPOSITIONS = {"confident", "borderline"}
VALID_SPEAKERS = {"user", "assistant"}
# OPTIONAL temporal metadata (model-emitted). All three default to null when a memory does not
# carry them. An
# unknown enum value is coerced to null (not a drop), mirroring how an unspecified disposition is
# softened rather than rejected. `valid_as_of` is a free anchor string (e.g. "2026-05"), not enum.
VALID_TEMPORAL_KINDS = {"timeless", "dated", "episodic", "until-event"}
VALID_VOLATILITY = {"low", "med", "high"}


def _clean_enum(raw, valid: set) -> str | None:
    """A SOFT enum label: keep it only if it is a known value, else null. Never drops the memory —
    an out-of-vocab value (e.g. temporal_kind="banana") coerces to null and the memory continues."""
    v = raw.strip().lower() if isinstance(raw, str) else ""
    return v if v in valid else None


def _clean_free_str(raw) -> str | None:
    """A free anchor string kept only if it says something; whitespace/empty/non-string -> null. A literal
    "none"/"null"/"n/a" is a model artifact, not a real value, so it also coerces to null (guards a stray
    scope_applicable_ref="None" from surfacing as `repo/None`)."""
    if not (isinstance(raw, str) and raw.strip()):
        return None
    s = raw.strip()
    return None if s.lower() in ("none", "null", "n/a") else s


def derive_scope(scope_level, scope_ref, scope_text) -> tuple[str | None, str | None]:
    """Return (scope_level, scope_ref) for the fixed-enum + target-ref shape.

    Inputs carry `scope_level` (+ optional `scope_ref`). As a fallback for a model that answers with
    one free `scope` string instead (e.g. "repo/<owner>/<repo>", "personal"; nothing enforces the
    shape of a `claude` answer), that string is parsed best-effort.
    Returns (None, None) when nothing usable is present (the caller drops the candidate as
    missing-scope)."""
    lvl = (scope_level or "").strip().lower() if isinstance(scope_level, str) else ""
    if lvl in VALID_SCOPE_LEVELS:
        if lvl == "personal":
            return "personal", None  # personal never carries a ref
        ref = scope_ref.strip() if isinstance(scope_ref, str) and scope_ref.strip() else None
        return lvl, ref
    # Fallback for one free scope string: "<level>/<ref>" (ref may itself contain slashes) or a level.
    if isinstance(scope_text, str) and scope_text.strip():
        head, _, tail = scope_text.strip().partition("/")
        head = head.strip().lower()
        tail = tail.strip()
        if head == "personal":
            return "personal", None
        if head in VALID_SCOPE_LEVELS:
            return head, (tail or None)
    return None, None


def clean_secondary_type(raw, primary_type) -> str | None:
    """A SOFT optional second memory type. Keep it only if it is a valid type AND differs from the
    primary `type` (a secondary equal to the primary carries no signal). Out-of-vocab or equal-to-primary
    coerces to null — never a drop, mirroring the other soft enums. Primary `type` stays authoritative."""
    v = _clean_enum(raw, VALID_TYPES)
    return v if (v and v != primary_type) else None


def clean_applicable_scope(raw_level, raw_ref, identified_level, identified_ref):
    """Resolve the optional APPLICABLE scope (where the memory APPLIES, distinct from the identified
    scope where it was observed). Returns (level, ref) or (None, None). Softly validated: an invalid
    level yields (None, None); personal carries no ref. Stored ONLY when it genuinely differs from the
    identified scope — an applicable scope identical to the identified one is redundant and dropped to
    null, so the field means exactly 'applies somewhere other than where it was seen'."""
    lvl = _clean_enum(raw_level, VALID_SCOPE_LEVELS)
    if lvl is None:
        return None, None
    ref = None if lvl == "personal" else _clean_free_str(raw_ref)
    if lvl == identified_level and ref == identified_ref:
        return None, None
    return lvl, ref


def _clean_intention_meta(raw) -> dict | None:
    """An intention may carry {what, done_check}; keep the object only if it says something."""
    if not isinstance(raw, dict):
        return None
    what = raw.get("what")
    done = raw.get("done_check")
    what = what.strip() if isinstance(what, str) and what.strip() else None
    done = done.strip() if isinstance(done, str) and done.strip() else None
    if what is None and done is None:
        return None
    return {"what": what, "done_check": done}


def _review_for(mtype: str) -> dict:
    """Review status: whether a memory is safe to load without a human check. It comes from the type
    alone: an `intention` is "hold-for-review", everything else "active". A status the model or a
    member sends is never copied."""
    return {"status": "hold-for-review" if mtype == "intention" else "active"}


@dataclass
class GatedMemory:
    statement: str
    type: str
    scope_level: str  # fixed enum: personal|dir|service|repo|team|org
    scope_ref: str | None  # the specific target (e.g. "<owner>/<repo>"); null for personal
    evidence: list[dict]  # [{quote, turn, speaker}] — all verified
    decision_outcome: str | None  # for type=decision: what was adopted vs rejected
    # Typed tag object over the six axes, in the canonical output shape (see `union_tags`). A
    # list-form answer is decoded into it, so every memory carries the same shape.
    tags: dict
    disposition: str  # confident | borderline
    disposition_note: str | None
    memory_id: str  # sha256(norm_statement)[:24]
    intention_meta: dict | None = None  # for type=intention: {what, done_check}
    review: dict = field(default_factory=lambda: {"status": "active"})  # {status}
    # OPTIONAL second memory type — a single field (not a list) for a memory genuinely of two kinds
    # (e.g. decision AND fact). Null unless the model emits a valid, distinct second type. Primary
    # `type` remains authoritative (provenance routing keys on the primary, not this).
    secondary_type: str | None = None
    # OPTIONAL applicable scope — where the memory APPLIES when that differs from the identified
    # scope_level/scope_ref (where it was observed). Both null when applicability == identified scope.
    scope_applicable_level: str | None = None
    scope_applicable_ref: str | None = None
    # OPTIONAL temporal metadata (all default null; see VALID_TEMPORAL_KINDS / VALID_VOLATILITY).
    valid_as_of: str | None = None  # explicit anchor the fact was asserted true (e.g. "2026-05")
    temporal_kind: str | None = None  # timeless | dated | episodic | until-event — decay class
    volatility: str | None = None  # low | med | high — coarse rot-speed (SOFT/heuristic)
    # OPTIONAL longer prose companion to `statement` (2-4 sentences: the why/symptom/consequence/
    # nuance the terse statement cannot carry). Free string; whitespace/empty/non-string -> null.
    description: str | None = None
    # The tags exactly as the reconcile request shows them: the answer's strings (such as `tech:go`) in the
    # order the model wrote them, joined by ", ". Request text only, never written to output.
    tags_line: str = ""
    # CODE-STAMPED provenance annotation: high | uncertain | low over the memory's quotes'
    # effective authorships (see `provenance_confidence`). It annotates; it does not gate.
    provenance_confidence: str = "uncertain"

    @property
    def scope(self) -> str:
        """Display form: "<level>/<ref>", or the bare level (e.g. "repo/<owner>/<repo>", "personal")."""
        if not self.scope_ref or self.scope_level == "personal":
            return self.scope_level
        return f"{self.scope_level}/{self.scope_ref}"


@dataclass
class GateReport:
    memories: list[GatedMemory] = field(default_factory=list)
    n_raw: int = 0  # candidates the model emitted
    n_kept: int = 0  # survived validation + evidence gate
    n_dropped_schema: int = 0  # failed field validation
    n_dropped_evidence: int = 0  # failed the quote/turn/speaker check
    n_folded_duplicate: int = 0  # exact restatements folded within one conversation
    # Tag-encoding telemetry: how each kept memory's tags arrived, so extraction
    # non-compliance / malformation stays VISIBLE instead of masked. object=typed tag object; list=
    # list[str]; empty=no usable tags; malformed=tags neither list nor dict; unknown=coined non-axis keys.
    n_tags_object: int = 0
    n_tags_list: int = 0
    n_tags_empty: int = 0
    n_tags_malformed: int = 0
    n_unknown_tag_keys: int = 0
    drop_reasons: list[str] = field(default_factory=list)
    episode_summary: str | None = None


# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    # Curly quotes compare equal to straight ones, so a quote the model retyped still matches.
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip()


_CURLY = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def _fold_with_offsets(s: str) -> tuple[str, list[int], list[int]]:
    """`s` folded the way the evidence match compares text, with each folded character's source
    span in `s`. Each base character with its combining marks is folded as one unit: NFKC, then
    lowercase, then curly quotes to straight ones. Runs of whitespace become one space, and
    leading and trailing whitespace is dropped."""
    out: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    space = True  # drops leading whitespace
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and unicodedata.combining(s[j]):
            j += 1
        for ch in unicodedata.normalize("NFKC", s[i:j]).lower().translate(_CURLY):
            if ch.isspace():
                if space:
                    continue
                ch, space = " ", True
            else:
                space = False
            out.append(ch)
            starts.append(i)
            ends.append(j)
        i = j
    while out and out[-1] == " ":
        out.pop()
        starts.pop()
        ends.pop()
    return "".join(out), starts, ends


def _fold(s: str) -> str:
    """`s` folded the way the evidence match compares text (see `_fold_with_offsets`)."""
    return _fold_with_offsets(s)[0]


def _matched_span(quote: str, text: str) -> str | None:
    """The stretch of `text` that `quote` matches once both are folded, as `text` holds it. When
    the quote occurs more than once, the first match is stored. A source character whose fold
    expands to several characters is kept whole, so the span can be wider than the quote: "f"
    against the ligature "\ufb01" stores "\ufb01"."""
    q = _fold_with_offsets(quote)[0]
    if not q:
        return None
    folded, starts, ends = _fold_with_offsets(text)
    k = folded.find(q)
    if k < 0:
        return None
    return text[starts[k] : ends[k + len(q) - 1]]


def norm_statement(s: str) -> str:
    return _norm(s)


def memory_id(statement: str) -> str:
    return hashlib.sha256(norm_statement(statement).encode()).hexdigest()[:24]


def _turn_index(conv) -> dict:
    return {t.i: t for t in conv.turns}


# Text the code inserts into a rendered turn. A quote may not include it or span across it.
_CODE_MARKER_RE = re.compile(r"\[\.\.\.turn truncated\.\.\.\]|\[Tool result\]|\[Tool: [^\]\n]*\]")


def _check_evidence(ev: dict, tindex: dict) -> tuple[bool, str]:
    """Verify one evidence item against the transcript AS SENT."""
    quote = ev.get("quote")
    turn = ev.get("turn")
    speaker = (ev.get("speaker") or "").lower()
    if not isinstance(quote, str) or not quote.strip():
        return False, "empty-quote"
    # Speaker is a mandatory part of the citation: an omitted speaker cannot be waved through, or
    # an assistant-turn quote could back a person-attributed memory unchecked. It must be a valid
    # speaker and must equal the actual role of the cited turn.
    if speaker not in VALID_SPEAKERS:
        return False, f"bad-speaker:{speaker!r}"
    # A JSON bool would else read as turn 1/0 (bool is an int subclass); a fractional value
    # would silently floor onto a real adjacent turn. Accept only an integral turn index.
    if turn is None or isinstance(turn, bool):
        return False, f"bad-turn:{turn!r}"
    if isinstance(turn, int):
        pass
    elif isinstance(turn, float):
        if not turn.is_integer():
            return False, f"bad-turn:{turn!r}"
        turn = int(turn)
    elif isinstance(turn, str) and turn.strip().isdigit():
        turn = int(turn.strip())
    else:
        return False, f"bad-turn:{turn!r}"
    ev["turn"] = turn
    t = tindex.get(turn)
    if t is None:
        return False, f"turn-out-of-range:{turn}"
    if speaker != t.role:
        return False, f"speaker-mismatch: claimed {speaker}, actual {t.role} @turn {turn}"
    user_text = getattr(t, "user_text", None)
    if speaker == "user" and user_text is not None and _matched_span(quote, user_text) is None:
        return False, f"speaker-mismatch: quote outside the person's own text @turn {turn}"
    # The quote must sit inside one stretch of the person's or model's own text: markers the code
    # inserts ("[...turn truncated...]", "[Tool result]", "[Tool: <name>]") are never quotable, and
    # the kept head and tail of a truncated turn were never adjacent.
    span = next(
        (m for part in _CODE_MARKER_RE.split(t.content) if (m := _matched_span(quote, part)) is not None),
        None,
    )
    if span is None:
        return False, f"quote-not-in-turn:{turn}"
    ev["quote"] = span  # the session's own bytes, never the model's retyping
    ev["speaker"] = speaker  # store the canonical lowercased speaker
    return True, ""


# --------------------------------------------------------------------------- #
# Per-quote PROVENANCE annotation. The evidence gate proves a quote was really said, at the claimed
# turn, by the claimed speaker. It trusts the transcript role, and a machine-authored turn (harness
# boilerplate, a tool result, a delegated sub-agent's report) can sit in the `user` slot. So every
# quote also carries the EFFECTIVE authorship of its turn (the carried inner authorship when the quote
# falls inside an envelope's inner span, else the turn's own), and the memory carries a confidence
# derived from those authorships against its TYPE.
#
# Routing:
#   - a memory the human ESTABLISHED (preference / convention / intention) is backed only by
#     human_prose; a tool result is never the user's stance;
#   - a knowledge memory (fact / decision / caution / failed-attempt) also accepts model_generated
#     (assistant analysis is legitimate content) and tool_output (schema DDL, a code citation, an
#     issue record are among the best-grounded evidence a fact can rest on);
#   - template_boilerplate and unknown back nothing, for any type.
# Interpretive tool prose is credited like a raw artifact; telling them apart would need a classifier.
_HUMAN_ONLY_TYPES = {"preference", "convention", "intention"}
_KNOWLEDGE_TYPES = {"fact", "decision", "caution", "failed-attempt"}
ALLOWED_AUTHORSHIP: dict[str, set] = {
    **{t: {"human_prose"} for t in _HUMAN_ONLY_TYPES},
    **{t: {"human_prose", "model_generated", "tool_output"} for t in _KNOWLEDGE_TYPES},
}
# Every memory type MUST route (a type with no entry can back nothing) — enforce that the table
# covers the enum at import so a newly-added type cannot silently fall through to "backs nothing"
# unnoticed. A hard raise (not `assert`) so it still fires under `python -O`.
if set(ALLOWED_AUTHORSHIP) != VALID_TYPES:
    raise RuntimeError(
        f"ALLOWED_AUTHORSHIP must cover every VALID_TYPE; "
        f"missing={VALID_TYPES - set(ALLOWED_AUTHORSHIP)} extra={set(ALLOWED_AUTHORSHIP) - VALID_TYPES}"
    )


def _effective_authorship_for_quote(t, quote: str) -> tuple[str | None, bool]:
    """Resolve the EFFECTIVE authorship of one quote at turn `t`, and whether it came from the
    carried (inner) authorship. The carried authorship applies ONLY to a quote that falls SOLELY
    within the turn's carved inner span (`carried_text`); a quote of the wrapper boilerplate falls
    back to the turn's own `authorship`.

    Offset-free ambiguity guard: we know a quote's TEXT, not its character offset, so a token that
    appears in BOTH the carried span and the wrapper cannot be attributed to the inner content — it
    is credited to the turn's own authorship (fail-safe). We test this by removing the carried span
    once from the turn content and checking whether the quote still appears in that wrapper remainder.

    Fail-safe throughout: an envelope whose inner span could not be isolated (`carried_text`
    None/empty), or whose span cannot be located in the (possibly truncated) content, grants the
    carried authorship to NO quote. Folds text with the gate's own `_fold_with_offsets`, so a quote
    matches here exactly when it matches in the gate."""
    own = getattr(t, "authorship", None) or None
    carried = getattr(t, "carried_authorship", None)
    carried_text = getattr(t, "carried_text", None)
    if not (carried and isinstance(carried_text, str) and carried_text.strip()):
        return own, False
    nq = _fold(quote)
    nct = _fold(carried_text)
    if not nq or nq not in nct:
        return own, False  # quote is not in the inner span at all
    # Wrapper = the turn content with the carried span removed once. If the quote still appears
    # there, its location is ambiguous -> do NOT grant the carried (inner) authorship.
    nc = _fold(getattr(t, "content", "") or "")
    i = nc.find(nct)
    wrapper = (nc[:i] + nc[i + len(nct) :]) if i >= 0 else nc  # span unlocatable -> all-wrapper (safe)
    if nq in wrapper:
        return own, False
    return carried, True


def provenance_confidence(mtype: str, authorships) -> str:
    """The memory-level provenance confidence over its quotes' effective authorships. An
    annotation, not a gate: computed for every evidence-passing memory. Three values:
      - high      — every quote is admissibly, POSITIVELY authored for the type (`human_prose` for the
                    human-established types; `human_prose`/`model_generated`/`tool_output` for the
                    knowledge types).
      - uncertain — no positively-inadmissible quote, but at least one lacks signal (`unknown`, or a
                    missing/None authorship).
      - low       — at least one quote is positively inadmissible: `template_boilerplate` for any type,
                    or `model_generated`/`tool_output` for a human-established type (`tool_output` is
                    admissible for the knowledge types, so it is not low there).
    Keyed on the routing table `ALLOWED_AUTHORSHIP`."""
    allowed = ALLOWED_AUTHORSHIP.get(mtype) or set()
    if not authorships:
        return "uncertain"  # no signal at all -> fail-safe to uncertain, never "high"
    has_inadmissible = False
    has_unknown = False
    for a in authorships:
        if a in allowed:
            continue  # positively attributed AND admissible
        if a is None or a == "unknown":
            has_unknown = True  # lack of signal
        else:
            has_inadmissible = True  # a known authorship, inadmissible for this type
    if has_inadmissible:
        return "low"
    if has_unknown:
        return "uncertain"
    return "high"


# The six tag axes. domain/activity are single-valued; component/tech/concept/keyword are open,
# multi-valued; keyword: is the catch-all. `domain` is OPEN vocab (inferred, prompt-guided only);
# `activity` is CLOSED — its enum is `ACTIVITY_VALUES` below, which drives the responseSchema and must
# mirror the prompts' activity list.
TAG_AXES = ("domain", "activity", "component", "tech", "concept", "keyword")
SINGLE_VALUE_TAG_AXES = ("domain", "activity")


def _decode_tags(raw):
    """Decode a candidate's `tags` into (shown_strings, typed_object|None, encoding, unknown_key_count).
    Accepts EITHER a list of strings such as `tech:go` OR the typed tag object. The typed object is what the
    memory carries; a list answer is decoded into one by `typed_from_flat`. The shown strings are the
    answer's tags in answer order, which the reconcile request prints. Robust to the malformed shapes a
    model emits: a scalar where a list is expected is never iterated per character; a list handed to a
    single-value axis takes the first element; empty/null entries drop; an echoed axis prefix is
    stripped; an unknown/coined key's string values are swept into keyword: and counted. Anything
    that is neither a list nor a dict (a string, a number) is 'malformed'.
    encoding is one of 'list', 'object', 'empty', 'malformed'."""
    if raw is None:
        return [], None, "empty", 0
    # List form, disambiguated purely by type. Keep only non-empty STRING entries (a null or a
    # number would otherwise str()-repr into a junk tag such as "None").
    if isinstance(raw, list):
        listed = [t.strip() for t in raw if isinstance(t, str) and t.strip()]
        typed_list, unknown_list = _typed_from_list(listed)
        return listed, typed_list, ("list" if listed else "empty"), unknown_list
    if isinstance(raw, dict):
        typed: dict[str, list[str]] = {}
        flat: list[str] = []
        unknown = 0

        def _add(axis: str, value, demote: bool = False) -> None:
            if isinstance(value, (dict, list)):  # never str()-repr a nested value into a tag
                return
            v = str(value).strip().lower()
            if v.startswith(axis + ":"):  # strip an echoed prefix: domain "domain:x" -> "x"
                v = v[len(axis) + 1 :].strip()
            if not v:
                return
            if demote:  # kept in keyword, and not shown: the request line stays the answer's first value
                typed.setdefault("keyword", []).append(v)
                return
            typed.setdefault(axis, []).append(v)
            flat.append(f"{axis}:{v}")

        for key, val in raw.items():
            axis = str(key).strip().lower()
            if axis in SINGLE_VALUE_TAG_AXES:
                extra: list = []
                if isinstance(val, list):  # a list on a single-value axis: the first value is the axis's
                    val, extra = (val[0], val[1:]) if val else (None, [])
                if val is not None and str(val).strip():
                    _add(axis, val)
                for v in extra:  # the rest move to keyword, never dropped
                    if v is not None:
                        _add(axis, v, demote=True)
            elif axis in TAG_AXES:  # open axis: coerce a scalar to [scalar], never iterate a str
                vals = val if isinstance(val, list) else ([val] if val is not None else [])
                for v in vals:
                    if v is not None and str(v).strip():  # skip null entries (str(None) is truthy)
                        _add(axis, v)
            else:  # unknown/coined key -> COUNT it, sweep scalars into keyword:
                vals = val if isinstance(val, list) else ([val] if val is not None else [])
                for v in vals:
                    if v is None:
                        continue
                    unknown += 1
                    _add("keyword", v)
        return flat, (typed or None), ("object" if flat else "empty"), unknown
    return [], None, "malformed", 0


def _iter_axis_values(typed, axis):
    """Yield the cleaned string values one typed tag object holds on one axis, tolerant of the
    shape: a scalar, a list, or absent (on the gate a single-value axis is a one-element list).
    Empty/None entries and an echoed `axis:` prefix are dropped, mirroring `_decode_tags`."""
    if not isinstance(typed, dict):
        return
    val = typed.get(axis)
    if val is None:
        return
    vals = val if isinstance(val, list) else [val]
    for v in vals:
        if v is None:
            continue
        s = str(v).strip().lower()
        if s.startswith(axis + ":"):
            s = s[len(axis) + 1 :].strip()
        if s:
            yield s


def _typed_from_list(flat_tags) -> tuple[dict | None, int]:
    """Decode a list-form tag answer into (typed object | None, count of non-axis colon entries).
    Text before the first colon is an axis only when it is one of `TAG_AXES`; any other entry,
    a bare word or one such as `std::vector` or `https://x`, is kept whole in keyword, and one with a
    colon is counted. A second value for domain or activity moves to keyword. Entries that are empty
    or only colons carry nothing and drop."""
    if not isinstance(flat_tags, (list, tuple)):
        return None, 0
    typed: dict[str, list[str]] = {}
    unknown = 0

    def _keep(axis: str, value: str) -> None:
        bucket = typed.setdefault(axis, [])
        if value not in bucket:
            bucket.append(value)

    for raw in flat_tags:
        if not isinstance(raw, str):
            continue
        t = raw.strip().lower()
        if not t.replace(":", "").strip():
            continue
        i = t.find(":")
        if i > 0 and t[:i] in TAG_AXES:
            axis, value = t[:i], t[i + 1 :].strip()
            if not value:
                continue
            if axis in SINGLE_VALUE_TAG_AXES and typed.get(axis) and typed[axis][0] != value:
                _keep("keyword", value)
            elif axis in SINGLE_VALUE_TAG_AXES:
                typed[axis] = [value]
            else:
                _keep(axis, value)
        else:
            unknown += ":" in t
            _keep("keyword", t)
    return (typed or None), unknown


def typed_from_flat(flat_tags) -> dict | None:
    """Decode a list-form tag answer (strings such as `tech:go`) into a typed tag object, so a memory
    carries the typed object whichever form the model answered in. See `_typed_from_list` for the
    rules. None when nothing usable is present."""
    return _typed_from_list(flat_tags)[0]


def empty_tags() -> dict:
    """The tag object of a memory that carries no tags: every axis present and empty."""
    return {a: (None if a in SINGLE_VALUE_TAG_AXES else []) for a in TAG_AXES}


def union_tags(typed_objs, single_value: str = "first") -> dict:
    """Union several members' typed tag objects into ONE canonical typed object.

    Output shape: all six axes present; domain and activity as a scalar or None; component, tech,
    concept and keyword as (possibly empty) lists. `single_value` governs how domain and activity
    resolve across members:
      - "first"    : the first non-null value in member order (the reconcile rule).
      - "majority" : the value backed by the most members (ties -> lexicographically smallest) (the
                     merge rule).
    Under either rule every other value, from another member or a second value within one member,
    moves to keyword, so nothing is dropped. Open axes union all distinct values in first-seen order.
    A list-form member (strings such as `tech:go`, as a v2 arm line carries) is decoded first.

    With no usable value on any member, every axis is empty (`empty_tags`)."""
    objs = [
        t
        for t in ((typed_from_flat(o) if isinstance(o, list) else o) for o in (typed_objs or []))
        if isinstance(t, dict) and t
    ]
    merged: dict = empty_tags()
    demoted: list[str] = []
    for axis in TAG_AXES:
        if axis in SINGLE_VALUE_TAG_AXES:
            continue
        seen: set[str] = set()
        for t in objs:
            for v in _iter_axis_values(t, axis):
                if v not in seen:
                    seen.add(v)
                    merged[axis].append(v)
    for axis in SINGLE_VALUE_TAG_AXES:
        per_member: list[str] = []  # each member's first value: its vote
        every: set[str] = set()
        for t in objs:
            vals = list(_iter_axis_values(t, axis))
            if vals:
                per_member.append(vals[0])
                every.update(vals)
        if not per_member:
            continue
        if single_value == "majority":
            counts = Counter(per_member)
            winner = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        else:
            winner = per_member[0]
        merged[axis] = winner
        demoted.extend(sorted(every - {winner}))
    for loser in demoted:
        if loser not in merged["keyword"]:
            merged["keyword"].append(loser)
    return merged


# The closed activity vocabulary: it feeds the Gemini responseSchema enum below and must match the
# `activity` list in both extraction prompts. It covers non-software work too (data, research, ops,
# design, writing). A model that emits an unlisted value on the list-form path keeps it; under the
# responseSchema it cannot emit one.
ACTIVITY_VALUES = (
    "coding",
    "reviewing",
    "debugging",
    "deploying",
    "planning",
    "incident",
    "onboarding",
    "researching",
    "analyzing",
    "experimenting",
    "training",
    "operating",
    "writing",
    "designing",
)
GENERALITY_VALUES = ("instance-specific", "general-principle")


def structured_output_schema(valid_types) -> dict:
    """Gemini responseSchema for the whole extraction output (constrained decoding). Permissive for
    content (statement/description/scope_ref/valid_as_of are free strings, nullable where optional),
    enums only where the set is genuinely closed, and a strict tag object: domain a required string
    with an open value, activity a scalar-or-null enum, the four open axes arrays of strings.
    `valid_types` is the arm's allowed memory types. Gemini accepts an OpenAPI-3.0 subset
    (type/properties/items/enum/nullable/required/propertyOrdering)."""

    def enum(vals):
        return {"type": "string", "enum": sorted(vals)}

    def nstr():
        return {"type": "string", "nullable": True}

    tags = {
        "type": "object",
        "properties": {
            "domain": {"type": "string"},
            "activity": {"type": "string", "nullable": True, "enum": list(ACTIVITY_VALUES)},
            "component": {"type": "array", "items": {"type": "string"}},
            "tech": {"type": "array", "items": {"type": "string"}},
            "concept": {"type": "array", "items": {"type": "string"}},
            "keyword": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["domain"],
        "propertyOrdering": ["domain", "activity", "component", "tech", "concept", "keyword"],
    }
    memory = {
        "type": "object",
        "properties": {
            "statement": {"type": "string"},
            "description": nstr(),
            "type": enum(valid_types),
            # optional SECOND type (single field, not a list) for a genuinely two-kind memory; null otherwise
            "secondary_type": {"type": "string", "nullable": True, "enum": sorted(valid_types)},
            "scope_level": enum(VALID_SCOPE_LEVELS),
            "scope_ref": nstr(),
            # optional APPLICABLE scope — where it applies when that differs from where it was observed
            "scope_applicable_level": {
                "type": "string",
                "nullable": True,
                "enum": sorted(VALID_SCOPE_LEVELS),
            },
            "scope_applicable_ref": nstr(),
            "generality": {"type": "string", "nullable": True, "enum": list(GENERALITY_VALUES)},
            "valid_as_of": nstr(),
            "temporal_kind": {"type": "string", "nullable": True, "enum": sorted(VALID_TEMPORAL_KINDS)},
            "volatility": {"type": "string", "nullable": True, "enum": sorted(VALID_VOLATILITY)},
            "decision_outcome": nstr(),
            "intention_meta": {
                "type": "object",
                "nullable": True,
                "properties": {"what": {"type": "string"}, "done_check": {"type": "string"}},
            },
            "tags": tags,
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "quote": {"type": "string"},
                        "turn": {"type": "integer"},
                        "speaker": enum(VALID_SPEAKERS),
                    },
                    "required": ["quote", "turn", "speaker"],
                },
            },
            "disposition": {
                "type": "object",
                "properties": {"level": enum(VALID_DISPOSITIONS), "note": nstr()},
                "required": ["level"],
            },
        },
        "required": ["statement", "type", "scope_level", "tags", "evidence", "disposition"],
        "propertyOrdering": [
            "statement",
            "description",
            "type",
            "secondary_type",
            "scope_level",
            "scope_ref",
            "scope_applicable_level",
            "scope_applicable_ref",
            "generality",
            "valid_as_of",
            "temporal_kind",
            "volatility",
            "decision_outcome",
            "intention_meta",
            "tags",
            "evidence",
            "disposition",
        ],
    }
    return {
        "type": "object",
        "properties": {"window_summary": {"type": "string"}, "memories": {"type": "array", "items": memory}},
        "required": ["memories"],
        "propertyOrdering": ["window_summary", "memories"],
    }


def _parse_candidate(c: dict, rep: GateReport):
    """Schema-validate one raw candidate (everything except the evidence quote/turn/speaker check).
    Returns a dict of cleaned fields, or None having incremented rep.n_dropped_schema and recorded
    a drop reason."""
    if not isinstance(c, dict):
        rep.n_dropped_schema += 1
        rep.drop_reasons.append("candidate-not-object")
        return None
    stmt = c.get("statement")
    mtype = (c.get("type") or "").lower()
    scope_level, scope_ref = derive_scope(c.get("scope_level"), c.get("scope_ref"), c.get("scope"))
    tags_shown, tags_obj, tag_encoding, tag_unknown = _decode_tags(c.get("tags"))
    evidence = c.get("evidence") or []
    disp_raw = c.get("disposition")
    # disposition may be a string or {level, note}
    if isinstance(disp_raw, dict):
        disp = (disp_raw.get("level") or disp_raw.get("disposition") or "").lower()
        disp_note = disp_raw.get("note")
    else:
        disp = (disp_raw or "").lower()
        disp_note = c.get("disposition_note")

    if not isinstance(stmt, str) or not stmt.strip():
        rep.n_dropped_schema += 1
        rep.drop_reasons.append("missing-statement")
        return None
    if mtype not in VALID_TYPES:
        rep.n_dropped_schema += 1
        rep.drop_reasons.append(f"bad-type:{mtype}")
        return None
    if scope_level is None:
        rep.n_dropped_schema += 1
        rep.drop_reasons.append("missing-scope")
        return None
    if not isinstance(evidence, list) or not evidence:
        rep.n_dropped_schema += 1
        rep.drop_reasons.append("missing-evidence")
        return None
    if disp not in VALID_DISPOSITIONS:
        disp = "borderline"  # unspecified confidence is treated as needing review, not dropped
    # TAGS ARE ENRICHMENT, NOT A SURVIVAL CONDITION: a memory with a valid statement + evidence is kept
    # even when its tags are empty or malformed (recorded below), never dropped for tags alone. This
    # avoids a structured-but-sparse tag object turning a tag-quality problem into memory LOSS.
    rep.n_tags_object += tag_encoding == "object"
    rep.n_tags_list += tag_encoding == "list"
    rep.n_tags_malformed += tag_encoding == "malformed"
    rep.n_tags_empty += tag_encoding == "empty"
    rep.n_unknown_tag_keys += tag_unknown
    scope_app_level, scope_app_ref = clean_applicable_scope(
        c.get("scope_applicable_level"), c.get("scope_applicable_ref"), scope_level, scope_ref
    )
    return {
        "stmt": stmt,
        "mtype": mtype,
        "scope_level": scope_level,
        "scope_ref": scope_ref,
        "secondary_type": clean_secondary_type(c.get("secondary_type"), mtype),
        "scope_applicable_level": scope_app_level,
        "scope_applicable_ref": scope_app_ref,
        "tags_shown": tags_shown,
        "tags_obj": tags_obj,
        "evidence": evidence,
        "disp": disp,
        "disp_note": disp_note,
        "decision_outcome": (c.get("decision_outcome") or None),
        "intention_meta": _clean_intention_meta(c.get("intention_meta")),
        "review": _review_for(mtype),
        "valid_as_of": _clean_free_str(c.get("valid_as_of")),
        "temporal_kind": _clean_enum(c.get("temporal_kind"), VALID_TEMPORAL_KINDS),
        "volatility": _clean_enum(c.get("volatility"), VALID_VOLATILITY),
        "description": _clean_free_str(c.get("description")),
    }


def validate_and_gate(raw_obj: dict, conv, model_id: str) -> GateReport:
    """Validate a parsed model response against the memory format and run the evidence gate.
    `conv` is the rendered transcript.Conversation whose turns are exactly what was sent.
    Single-source semantics: ALL of a memory's quotes must pass, or the memory is dropped."""
    rep = GateReport()
    if not isinstance(raw_obj, dict):
        rep.drop_reasons.append("response-not-object")
        return rep
    rep.episode_summary = raw_obj.get("episode_summary")
    cands = raw_obj.get("memories")
    if not isinstance(cands, list):
        rep.drop_reasons.append("no-memories-list")
        return rep
    rep.n_raw = len(cands)
    tindex = _turn_index(conv)
    seen_ids = set()

    for c in cands:
        pc = _parse_candidate(c, rep)
        if pc is None:
            continue
        evidence = pc["evidence"]

        # --- evidence gate: ALL quotes must pass ---
        gate_ok = True
        for ev in evidence:
            if not isinstance(ev, dict):
                gate_ok = False
                rep.drop_reasons.append("evidence-not-object")
                break
            ok, why = _check_evidence(ev, tindex)
            if not ok:
                gate_ok = False
                rep.drop_reasons.append(why)
                break
        if not gate_ok:
            rep.n_dropped_evidence += 1
            continue

        # --- provenance: annotate each quote's effective authorship, then the memory's confidence ---
        for ev in evidence:
            t = tindex.get(ev["turn"])
            eff, _ = (
                _effective_authorship_for_quote(t, ev.get("quote") or "") if t is not None else (None, False)
            )
            ev["authorship"] = eff
        prov_conf = provenance_confidence(pc["mtype"], [ev.get("authorship") for ev in evidence])

        mid = memory_id(pc["stmt"])
        if mid in seen_ids:  # exact-restatement within one conversation -> fold
            rep.n_folded_duplicate += 1
            kept = next(m for m in rep.memories if m.memory_id == mid)
            # The restatement's quotes back the folded memory too.
            have = {(e["quote"], e["turn"], e["speaker"]) for e in kept.evidence}
            for e in evidence:
                if (e["quote"], e["turn"], e["speaker"]) not in have:
                    kept.evidence.append(
                        {
                            "quote": e["quote"],
                            "turn": e["turn"],
                            "speaker": e["speaker"],
                            "authorship": e.get("authorship"),
                        }
                    )
            # An intention restated as another type keeps the folded memory an intention, and the
            # latest intention with meta gives it its meta.
            if pc["mtype"] == "intention":
                if kept.type != "intention":
                    kept.type = "intention"
                    kept.review = _review_for("intention")
                    kept.decision_outcome = None  # only a decision carries an outcome
                    if kept.secondary_type == "intention":
                        kept.secondary_type = None
                if pc["intention_meta"]:
                    kept.intention_meta = pc["intention_meta"]
            # The restatement's tags join the kept memory's; its own request line is not shown.
            kept.tags = union_tags([kept.tags, pc["tags_obj"]])
            # Confidence follows the final type and the full set of quotes.
            kept.provenance_confidence = provenance_confidence(
                kept.type, [e.get("authorship") for e in kept.evidence]
            )
            continue
        seen_ids.add(mid)

        rep.memories.append(
            GatedMemory(
                statement=pc["stmt"].strip(),
                type=pc["mtype"],
                scope_level=pc["scope_level"],
                scope_ref=pc["scope_ref"],
                # turn + speaker were normalized onto each ev by the gate above; the effective authorship
                # was resolved by the provenance step (carried inner authorship already folded into it) —
                # kept so the provenance a memory rests on is visible downstream.
                evidence=[
                    {
                        "quote": e["quote"],
                        "turn": e["turn"],
                        "speaker": e["speaker"],
                        "authorship": e.get("authorship"),
                    }
                    for e in evidence
                ],
                decision_outcome=pc["decision_outcome"],
                tags=union_tags([pc["tags_obj"]]),
                tags_line=", ".join(pc["tags_shown"]),
                provenance_confidence=prov_conf,
                disposition=pc["disp"],
                disposition_note=pc["disp_note"],
                memory_id=mid,
                secondary_type=pc["secondary_type"],
                scope_applicable_level=pc["scope_applicable_level"],
                scope_applicable_ref=pc["scope_applicable_ref"],
                intention_meta=pc["intention_meta"],
                review=pc["review"],
                valid_as_of=pc["valid_as_of"],
                temporal_kind=pc["temporal_kind"],
                volatility=pc["volatility"],
                description=pc["description"],
            )
        )
    rep.n_kept = len(rep.memories)
    return rep


def extract_json(raw_text: str):
    """Pull the first JSON object out of a model response (tolerant of code fences / prose)."""
    if not raw_text:
        return None
    s = raw_text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    # fall back: decode the first JSON object embedded in prose. raw_decode is string-aware,
    # so a brace inside a quoted value (e.g. a quote containing "{var}") does not end it early.
    start = s.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[start:])
        return obj
    except Exception:
        return None
