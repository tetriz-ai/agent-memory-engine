# Copyright 2026 Workloom Solutions Private Limited
# SPDX-License-Identifier: MIT
"""Parse one Claude Code session file into a turn-indexed transcript (no model involved).

A Turn is (i, role, content) where i is 1-based, role is user or assistant, and content is the
text exactly as it will be sent to the model (after per-turn truncation). The evidence gate checks
quotes against exactly this per-turn content, so this rendering is the ground truth.

Speaker integrity: tool output a "user" event carries (its tool_result blocks) is rendered as its
own `assistant` turn, so a quote of tool output cannot pass the evidence gate as the person's own
words. When the same event also carries typed text, that text becomes a separate `user` turn.

Authorship, per turn and deterministic: who wrote the bytes. One of human_prose, template_boilerplate,
model_generated, tool_output, or unknown (for a turn whose source cannot be positively attributed).
  - carried_authorship (nullable): the authorship of the inner payload when the turn is an
    envelope: a task notification wraps a sub-agent's model_generated result, and a slash command
    with appended free text wraps the person's human_prose.
  - The gate resolves authorship per quote (contract._effective_authorship_for_quote) and grants
    the carried authorship only to a quote inside carried_text.

Authorship is reconstructed by a per-agent adapter registered in _ADAPTERS. The Claude Code adapter
reads Claude Code's own event fields (promptSource, isMeta, origin.kind, block type), decided per
record. Two ordering fail-safes: a machine promptSource (system, sdk, any non-human value) is
classified as machine before any command or slash branch can grant carried human_prose (only the
task-notification envelope precedes it); and every text-heuristic human_prose label requires native
fields and no machine signal, otherwise it degrades to unknown. A source with no adapter degrades
wholesale to unknown. A turn is labelled human_prose only with a signal reliable on its own record:
uncertainty is `unknown`, not human.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

# Per-turn and whole-transcript truncation. The prompts assume this coverage shape.
PER_TURN_MAX = 8_000
PER_TURN_HEAD = 6_000
PER_TURN_TAIL = 1_500
DEFAULT_MAX_CHARS = 300_000

# Content markers used to reconstruct provenance. These are Claude Code's own injection wrappers.
_M_COMMAND = "<command-name>"  # human typed a /slash — human chose the workflow
_M_SKILL_BODY = "Base directory for this skill:"  # expanded skill body (same whoever invoked it)
_M_TASK_NOTIF = "<task-notification>"  # harness envelope wrapping a sub-agent result
_M_IMAGE_REF = "[Image: source:"  # injected attachment-path placeholder turn
# Harness boilerplate injected into the user slot (order-independent; none is human prose).
_HARNESS_MARKERS = (
    "<local-command-caveat>",
    "<local-command-stdout>",
    "<local-command-stderr>",
    "<user-prompt-submit-hook>",
    "<system-reminder>",
    "Continue from where you left off.",
    "[Request interrupted by user",
    "This session is being continued from a previous conversation",
)
# Shell output from Claude Code's `!` mode, which arrives in the user slot: a tool's output.
_SHELL_OUTPUT_MARKERS = ("<bash-stdout>", "<bash-stderr>")
# tool_result blocks that are control-flow acks, not real tool output — excluded from tool_output.
_PSEUDO_RESULT_MARKERS = ("Launching skill:", "Async agent launched successfully")

# promptSource values that positively indicate a HUMAN caused the turn. The values Claude Code is
# known to write are {None, typed, queued, suggestion_accepted, system, sdk}. Any promptSource that
# is SET but not in this set (system, sdk, or a future value) is a MACHINE signal, so an unknown
# human source is conservatively dropped, never falsely trusted. promptSource=None is ambiguous (a
# human /command turn carries no promptSource), so it is not machine.
_HUMAN_PROMPT_SOURCES = {"typed", "queued", "suggestion_accepted"}


@dataclass
class Turn:
    i: int
    role: str  # 'user' | 'assistant'
    content: str  # rendered text AS SENT (already per-turn truncated)
    authorship: str = ""  # who wrote the bytes (see module docstring)
    carried_authorship: str | None = None  # inner-payload authorship when the turn is an envelope
    # The carved INNER content span the carried_authorship refers to (the sub-agent <result> body,
    # the human /command args). The gate grants carried_authorship
    # ONLY to a quote that falls within this text; a quote of the WRAPPER boilerplate falls back to
    # `authorship`. None when the turn is not an envelope OR the inner span could not be isolated —
    # fail-safe: no inner span means no quote is granted the carried authorship.
    carried_text: str | None = None
    # When set, the only text of this user turn the person wrote: quotes outside it cannot cite the
    # user. Set for an envelope with typed arguments that also holds harness or shell output.
    user_text: str | None = None

    @property
    def effective_authorship(self) -> str | None:
        """WHOLE-TURN effective authorship: the carried (inner) authorship if this turn is an
        envelope, else the turn's own authorship. Returns 'unknown' when authorship is 'unknown'
        and there is no carried inner payload — a weak-signal turn that extraction must not trust
        as human.

        NON-GATING: this ignores WHICH span a specific quote falls in — it reports carried for the
        whole turn even for a quote that lies in the wrapper boilerplate. Gating code MUST instead
        use the quote-aware `contract._effective_authorship_for_quote`, which grants the
        carried authorship only to a quote isolated to the carved inner span."""
        return self.carried_authorship or self.authorship or None


@dataclass
class Conversation:
    conversation_id: str
    tool: str
    source: str
    date: str | None
    turns: list[Turn]
    truncated: bool  # True if whole-transcript budget cut some turns


# --------------------------------------------------------------------------- #
# content-block rendering
# --------------------------------------------------------------------------- #
def _render_blocks(content) -> tuple[str, bool]:
    """Render a message 'content' (str or list-of-blocks) to text.
    Returns (text, has_human_text) — has_human_text is True only when the message
    carries author-written prose (a plain string, or a 'text' block), NOT merely a
    tool_result. Used to keep the user/assistant speaker labeling honest."""
    if content is None:
        return "", False
    if isinstance(content, str):
        return content, True
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)[:2000], False
    parts: list[str] = []
    has_human_text = False
    for b in content:
        if isinstance(b, str):
            parts.append(b)
            has_human_text = True
            continue
        if not isinstance(b, dict):
            parts.append(str(b)[:1000])
            continue
        bt = b.get("type")
        if bt == "text":
            parts.append(b.get("text") or "")
            has_human_text = True
        elif bt == "thinking":
            parts.append(b.get("thinking") or "")
        elif bt == "tool_use":
            parts.append(f"[Tool: {b.get('name', '?')}]")
        elif bt == "tool_result":
            inner, _ = _render_blocks(b.get("content"))
            parts.append(f"[Tool result] {inner}")
        else:
            # unknown block: keep an obvious text field, else a short marker naming its type
            val = b.get("text") or b.get("content")
            parts.append(val if isinstance(val, str) and val else f"[{bt}]")
    return "\n".join(p for p in parts if p), has_human_text


def _block_flags(content) -> tuple[bool, bool]:
    """Scan a message's blocks for tool-output signal.
    Returns (has_tool_result, is_pseudo_result). is_pseudo_result marks the Skill 'Launching…' /
    Agent 'Async agent launched' control-flow acks that ride the tool_result channel but are NOT
    real tool output. It is True ONLY when EVERY tool_result in the message is a pseudo-result —
    a message carrying a real tool_result alongside a pseudo one is genuine tool output (masking
    fix): the pseudo ack must not launder real tool output into template_boilerplate."""
    if not isinstance(content, list):
        return False, False
    n_tool_results = 0
    n_pseudo = 0
    for b in content:
        if isinstance(b, dict) and b.get("type") == "tool_result":
            n_tool_results += 1
            inner, _ = _render_blocks(b.get("content"))
            if any(mk in inner for mk in _PSEUDO_RESULT_MARKERS):
                n_pseudo += 1
    has_tool_result = n_tool_results > 0
    is_pseudo = has_tool_result and n_pseudo == n_tool_results
    return has_tool_result, is_pseudo


_SLASH_TOKEN_RE = re.compile(r"^/[a-z][\w-]*(?=\s|$)")
# A machine-loop retry/iteration counter (e.g. 'attempt=2', 'retry #2', 'run-2', 'attempt no. 2',
# 'retry_count=2') marks the invocation as loop-generated, not human-typed. Longer keyword first
# so 'retry_count' wins over 'retry'. The separator allows '=', ':', '#', '-', 'no.', or a space.
_COUNTER_TOKEN_RE = re.compile(
    r"\b(retry_count|retrycount|attempt|retry|rerun|iteration|iter|run|try)\b"
    r"\s*(?:no\.?|#|[-=:])?\s*\d+",
    re.I,
)
# Tokens that are structured, not natural-language: numbers, PR-refs, flags, key=value.
_STRUCTURED_TOKEN_RE = re.compile(r"#?\d+|--?[\w-]+|[a-z_][\w-]*=\S+", re.I)


def _looks_machine_counter(args: str) -> bool:
    """True only when `args` is a short, structured string DOMINATED by a machine retry-counter
    token — not when a counter word is embedded in a sentence. So 'retry #2' / 'run-2' /
    '123 attempt=2' are machine, but '/review run 2 tests' ('run 2' followed by the word
    'tests') stays human. Rule: a counter token must be present, and after removing counter
    tokens no natural-language word (3+ letters that is not itself structured) may remain."""
    a = (args or "").strip()
    if not a or not _COUNTER_TOKEN_RE.search(a):
        return False
    residual = _COUNTER_TOKEN_RE.sub(" ", a)
    for tok in residual.split():
        t = tok.strip()
        if _STRUCTURED_TOKEN_RE.fullmatch(t):
            continue
        if re.search(r"[a-zA-Z]{3,}", t):  # a real word remains -> it is a sentence, not a counter
            return False
    return True


def _slash_envelope(text: str) -> tuple[bool, bool] | None:
    """A user turn typed as a builtin slash command with NO <command-name> markup
    (e.g. '/compact <intent>', '/clear'). Returns (is_slash, has_human_intent) or None.
    The leading /token is a control command (template); appended free text beyond it is the
    human's own intent, recovered as carried=human_prose (an envelope) — UNLESS the
    appended text is a machine retry-counter, in which case the whole invocation is loop-driven
    and carries no human prose. Native promptSource=typed turns are classified as human_prose
    before this is reached, so this only fires on the weak-signal tail."""
    stripped = text.strip()
    if not _SLASH_TOKEN_RE.match(stripped):
        return None
    remainder = _SLASH_TOKEN_RE.sub("", stripped, count=1).strip()
    has_intent = bool(remainder) and not _looks_machine_counter(remainder)
    return True, has_intent


def _command_args_intent(text: str) -> bool:
    """True when a <command-name> markup turn carries non-empty, human-typed <command-args>
    (e.g. '--plan'). A machine retry-counter (e.g. '123 attempt=2') is loop-generated, not human
    intent, so it returns False and no human_prose is carried."""
    m = re.search(r"<command-args>(.*?)</command-args>", text, re.DOTALL)
    if not (m and m.group(1).strip()):
        return False
    return not _looks_machine_counter(m.group(1))


def _classify_claude_code(
    *,
    role,
    text,
    has_human_text,
    has_tool_result,
    is_pseudo_result,
    is_meta,
    prompt_source,
    origin_kind,
    has_native_fields,
    is_sub_agent,
) -> tuple[str, str | None]:
    """CLAUDE CODE provenance adapter. Derives (authorship, carried_authorship) for one
    turn from Claude Code's native transcript conventions.

    Pure and deterministic. `has_native_fields` is True ONLY when THIS record genuinely carries
    the native provenance structure (a real session event read by the local parser). It is False
    for pre-rendered content and for any synthetic record: without native fields no text heuristic
    can be corroborated.

    Two fail-safes the branch ORDER enforces:
      1. MACHINE-SIGNAL-FIRST. A turn whose promptSource is SET but not human (system / sdk /
         any non-human source), or that is a sub-agent driver, is classified machine
         (template_boilerplate, carried=None) BEFORE any command/slash carried-human_prose branch
         can fire. Only the task-notification envelope runs ahead of this catch, because it
         legitimately carries inner model content.
      2. NATIVE-CORROBORATION. Every text-heuristic human_prose emission (command-args, slash,
         own or carried) requires has_native_fields AND the absence of a machine
         signal. So '/deploy rotate-keys' with native structure and no machine field stays
         human_prose; a promptSource=system command, or the same bytes from a field-incomplete
         record, does NOT. Anything still unattributable degrades to authorship=unknown, never
         human_prose. A non-claude-code source is handled by _classify_unknown_agent."""
    native = has_native_fields  # this record carries the native provenance structure
    weak = not native  # no native corroboration
    # A promptSource that is SET but not one of the human sources is a machine signal. None is
    # ambiguous (a human /command turn carries no promptSource) and is NOT treated as machine.
    machine_prompt = prompt_source is not None and prompt_source not in _HUMAN_PROMPT_SOURCES

    # Assistant reasoning / tool_use / thinking — the model wrote it, the agent drove it.
    if role == "assistant":
        return "model_generated", None

    # An unresolved / non-user role is not run through the user text-heuristic gauntlet — it
    # cannot be positively attributed, so it degrades to unknown.
    if role != "user":
        return "unknown", None

    # --- user role from here ---

    # Tool output routed back through the user role. Must precede the content-marker checks
    # below: a system-reminder nested INSIDE a tool_result is still tool output, not a harness
    # injection. is_pseudo_result is True only when EVERY tool_result is a Skill/Agent control
    # ack — a real tool_result alongside a pseudo one is genuine tool output.
    if has_tool_result and not has_human_text:
        if is_pseudo_result:
            return "template_boilerplate", None
        return "tool_output", None

    # Shell output (`!` mode) in the user slot is a tool's output, whoever typed the command.
    if any(mk in text for mk in _SHELL_OUTPUT_MARKERS):
        return "tool_output", None

    # RECOGNISED ENVELOPES (run before the machine-signal catch, because they carry inner content).
    # Harness envelope wrapping a delegated sub-agent's model output. carried=model_generated
    # so extraction corroborates the inner report rather than trusting the envelope as fact.
    if origin_kind == "task-notification" or _M_TASK_NOTIF in text:
        return "template_boilerplate", "model_generated"

    # MACHINE-SIGNAL CATCH. A non-human promptSource (system / sdk / unknown) or a sub-agent
    # driver is machine-authored boilerplate — classified here, BEFORE the command/slash branches,
    # so a machine field can never be laundered into human_prose by a text heuristic.
    if machine_prompt or is_sub_agent:
        return "template_boilerplate", None

    # Explicit /slash markup — the human chose the workflow (human_command), and the expanded
    # bytes are Claude Code markup (template). carried=human_prose only from human-typed
    # <command-args>, and only with native fields (machine promptSource was already caught above,
    # and a field-incomplete record cannot corroborate the args as human-typed). A machine
    # retry-counter (e.g. 'attempt=2') is excluded by _command_args_intent.
    if _M_COMMAND in text:
        carried = "human_prose" if (native and _command_args_intent(text)) else None
        return "template_boilerplate", carried

    # Expanded skill body: boilerplate whoever invoked it.
    if _M_SKILL_BODY in text:
        return "template_boilerplate", None

    # Other harness injections (caveats, slash-command stdout echoes, system-reminders,
    # continuation, interrupt/compact markers) — machine boilerplate to ignore.
    if any(mk in text for mk in _HARNESS_MARKERS):
        return "template_boilerplate", None

    # Injected image-path placeholder turn (distinct from a human turn that attached an image,
    # which carries promptSource=typed and lands on the human_prose branch below). Gate on
    # isMeta natively; without native fields require the placeholder to lead the turn.
    if _M_IMAGE_REF in text and (is_meta or (weak and text.lstrip().startswith(_M_IMAGE_REF))):
        return "template_boilerplate", None

    # Human accepted a suggested reply — a native, reliable human-endorsement signal (only ever
    # present when the record carries native fields). Labeled human_prose because the human chose
    # these exact bytes, BUT the bytes were model-drafted and human-accepted, not free-typed:
    # a reader may trust the intent, not quote them as the human's verbatim words.
    if prompt_source == "suggestion_accepted":
        return "human_prose", None

    # Typed / queued human prose (origin.kind=human corroborates when promptSource is absent).
    # Both signals come from native fields, so this cannot fire on a field-incomplete record.
    if prompt_source in ("typed", "queued") or origin_kind == "human":
        return "human_prose", None

    # Remaining isMeta turns are harness-injected boilerplate (system promptSource was caught by
    # the machine-signal catch above).
    if is_meta:
        return "template_boilerplate", None

    # Builtin slash command typed literally, with no <command-name> markup and no promptSource
    # (e.g. '/compact <intent>'). NATIVE-ONLY: this rests entirely on the text shape, which
    # without native fields is indistinguishable from a rendered tool result that happens to start
    # with '/' (e.g. '/tmp is full'), so it must not fire there — it falls through to unknown. On a
    # native record (reached here only with no machine promptSource / isMeta) the slash reading is
    # reliable; the /token is a control command and any appended free text is the human's intent,
    # carried human_prose (unless it is a machine retry-counter, excluded by _slash_envelope).
    if native:
        slash = _slash_envelope(text)
        if slash is not None:
            _, has_intent = slash
            return "template_boilerplate", ("human_prose" if has_intent else None)

    # Graceful degradation. No positive signal identified this turn. We refuse to guess human vs
    # machine: a turn carrying unattributed author text degrades to authorship=unknown (NEVER
    # human_prose — the hard rule); a turn with no author text at all is residual control
    # boilerplate.
    if has_human_text:
        return "unknown", None
    return "template_boilerplate", None


def _classify_unknown_agent(*, role, **_) -> tuple[str, str | None]:
    """Fallback adapter for a source with no per-agent provenance model (a source whose `tool` is
    not claude-code). Its native conventions are unknown, so we do not guess: an assistant turn is
    still safely model_generated, but any user turn degrades to authorship=unknown rather than
    inventing a human/machine attribution."""
    if role == "assistant":
        return "model_generated", None
    return "unknown", None


# Per-agent provenance adapters. Reconstruction depends on an agent's native transcript
# conventions, so each supported agent gets its own adapter; register a new one here to add
# support. An unregistered agent routes to _classify_unknown_agent (unknown).
_ADAPTERS = {"claude-code": _classify_claude_code}


def _classify_turn(*, agent, **kw) -> tuple[str, str | None]:
    """Dispatch one turn to its agent's provenance adapter."""
    return _ADAPTERS.get(agent, _classify_unknown_agent)(**kw)


# --------------------------------------------------------------------------- #
# envelope inner-span carving (feeds Turn.carried_text)
# --------------------------------------------------------------------------- #
# The carried_authorship of an ENVELOPE turn describes only the INNER payload the envelope wraps,
# never the wrapper boilerplate. To let the gate tell an inner quote from a wrapper quote, we carve
# the inner span here and hand it to the Turn as `carried_text`. Every carver is fail-safe: it
# returns None when the span cannot be isolated, and the gate then grants the carried authorship to
# NO quote of that turn (the quote falls back to the turn's own template_boilerplate — backs nothing).
_RESULT_RE = re.compile(r"<result>(.*?)</result>", re.DOTALL)
_CMD_ARGS_RE = re.compile(r"<command-args>(.*?)</command-args>", re.DOTALL)


def _result_inner(text: str) -> str | None:
    """The <result>…</result> payload of a task-notification envelope — the delegated sub-agent's
    own model output. Requires a CLOSING </result>: without it the span cannot be isolated (trailing
    injected text would otherwise be credited to EOF as model output), so this returns None and the
    envelope's carried authorship backs nothing (fail-safe). None too for a bare notification that
    carries no result block at all."""
    m = _RESULT_RE.search(text)
    if m:
        return m.group(1).strip() or None
    return None


def _command_args_inner(text: str) -> str | None:
    """The human-typed <command-args>…</command-args> body of a /slash-markup turn."""
    m = _CMD_ARGS_RE.search(text)
    if m:
        return m.group(1).strip() or None
    return None


def _slash_inner(text: str) -> str | None:
    """The human intent appended after a builtin /slash token typed with no markup."""
    stripped = text.strip()
    if not _SLASH_TOKEN_RE.match(stripped):
        return None
    return _SLASH_TOKEN_RE.sub("", stripped, count=1).strip() or None


def _carve_carried_text(text: str, carried: str | None) -> str | None:
    """Isolate the inner content span a turn's carried_authorship refers to, so the gate can tell an
    inner quote from wrapper boilerplate. Keyed on the carried value (which the classifier already
    decided) + the envelope's content markers, so it stays co-located with — and can never diverge
    from — the classification. Fail-safe: returns None when no span isolates, and the gate then
    grants the carried authorship to no quote of the turn.

    The envelope shapes are non-overlapping: a task-notification carries model_generated inner; the
    two human-carried envelopes (/command args, bare /slash) carry human_prose, and a markup command
    is tried before a bare slash."""
    if carried == "model_generated":
        return _result_inner(text)
    if carried == "human_prose":
        return _command_args_inner(text) or _slash_inner(text)
    return None


def _truncate_turn(text: str) -> str:
    if len(text) <= PER_TURN_MAX:
        return text
    return text[:PER_TURN_HEAD] + "\n[...turn truncated...]\n" + text[-PER_TURN_TAIL:]


def _norm_role(raw: str | None) -> str:
    # raw may be None: an event can omit the role. `(raw or '')` turns a missing role into
    # 'unknown' rather than raising on .lower().
    r = (raw or "").lower()
    if "user" in r:
        return "user"
    if "assistant" in r:
        return "assistant"
    return r or "unknown"


def _split_tool_results(content):
    """A user event's content split into (tool_result blocks, other blocks), or None when it does not
    mix the two. The tool output becomes its own turn, so it never shares a turn with typed text."""
    if not isinstance(content, list):
        return None
    tools = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
    rest = [b for b in content if not (isinstance(b, dict) and b.get("type") == "tool_result")]
    if not tools or not rest:
        return None
    return tools, rest


def _turns_from_messages(msgs, source: str = "local", agent: str | None = None) -> list[Turn]:
    """msgs: iterable of raw records. A record is either a (raw_role, content) tuple or a dict
    carrying native fields {role, content, isMeta, promptSource, originKind, isSubAgent, native}.
    Applies speaker integrity, per-turn authorship, and truncation.
    `source` is an informational hint only; native-field availability is decided per record (the
    explicit `native` flag the session parser stamps), never from the label.
    `agent` selects the provenance adapter and MUST be passed explicitly by a caller that knows
    the agent (the session parser passes 'claude-code'). It defaults to None so that a caller which
    does NOT know the agent gets the safe unknown adapter rather than being treated as claude-code."""
    expanded = []
    for rec in msgs:
        if isinstance(rec, dict) and _norm_role(rec.get("role")) == "user":
            parts = _split_tool_results(rec.get("content"))
            if parts is not None:
                expanded.append({**rec, "content": parts[0]})
                expanded.append({**rec, "content": parts[1]})
                continue
        expanded.append(rec)
    turns: list[Turn] = []
    i = 0
    for rec in expanded:
        if isinstance(rec, dict):
            raw_role = rec.get("role")
            content = rec.get("content")
            is_meta = bool(rec.get("isMeta"))
            prompt_source = rec.get("promptSource")
            origin_kind = rec.get("originKind")
            is_sub_agent = bool(rec.get("isSubAgent"))
            # has_native_fields marks a record that came from the session parser: an explicit flag
            # it stamps (`native: True`), not a key-presence test. The parser fills absent
            # promptSource/originKind/isMeta keys with None, so a key-presence test would always pass
            # and a field-incomplete machine record could mint carried=human_prose. The flag tells a
            # real event (native=True even when its native values are all None, as on a genuine
            # human /command turn) from tuples and synthetic dicts (no flag, treated conservatively).
            has_native_fields = bool(rec.get("native"))
        else:
            raw_role, content = rec
            is_meta = False
            prompt_source = None
            origin_kind = None
            is_sub_agent = False
            has_native_fields = False

        text, has_human_text = _render_blocks(content)
        if not text.strip():
            continue
        has_tool_result, is_pseudo_result = _block_flags(content)
        role = _norm_role(raw_role)

        authorship, carried = _classify_turn(
            agent=agent,
            role=role,
            text=text,
            has_human_text=has_human_text,
            has_tool_result=has_tool_result,
            is_pseudo_result=is_pseudo_result,
            is_meta=is_meta,
            prompt_source=prompt_source,
            origin_kind=origin_kind,
            has_native_fields=has_native_fields,
            is_sub_agent=is_sub_agent,
        )
        # The user role is kept only for words the person may have written. A user-slot message with
        # no authored prose, or one that is tool output or harness text and carries no human prose,
        # is assistant context, so the evidence gate cannot take it as the person's words.
        if role == "user" and (
            not has_human_text
            or (authorship in ("tool_output", "template_boilerplate") and carried != "human_prose")
        ):
            role = "assistant"
        # Carve the envelope inner span from the FULL rendered text (before per-turn truncation) so
        # the gate can match an inner quote against it.
        carried_text = _carve_carried_text(text, carried)
        user_text = None
        if (
            role == "user"
            and carried == "human_prose"
            and any(mk in text for mk in _HARNESS_MARKERS + _SHELL_OUTPUT_MARKERS)
        ):
            user_text = carried_text or ""
        i += 1
        turns.append(
            Turn(
                i=i,
                role=role,
                content=_truncate_turn(text),
                authorship=authorship,
                carried_authorship=carried,
                carried_text=carried_text,
                user_text=user_text,
            )
        )
    return turns


# --------------------------------------------------------------------------- #
# session file -> Turns
# --------------------------------------------------------------------------- #
def parse_claude_code_session(path: str, native_id: str, agent: str = "claude-code"):
    """Parse one Claude Code session file (one JSON event per line) into Turns. Returns
    (turns, first_timestamp). Only user and assistant events of this session are read; sub-agent
    sidechain events are skipped (they belong to the sub-agent, not the person); a line that is
    not valid JSON is skipped."""
    msgs = []
    started = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except Exception:
                continue
            if not isinstance(o, dict):
                continue
            if o.get("type") not in ("user", "assistant"):
                continue
            if o.get("isSidechain"):
                continue
            if o.get("sessionId") and o.get("sessionId") != native_id:
                continue
            if started is None:
                started = o.get("timestamp")
            m = o.get("message") or {}
            if isinstance(m, dict):
                # The native provenance signals sit on the event, not on the message:
                # promptSource + isMeta + origin.kind mark who sent a turn and injected turns.
                msgs.append(
                    {
                        "role": m.get("role") or o.get("type"),
                        "content": m.get("content"),
                        "isMeta": o.get("isMeta"),
                        "promptSource": o.get("promptSource"),
                        "originKind": (o.get("origin") or {}).get("kind"),
                        # A real session event: native even when the values above are all None.
                        "native": True,
                    }
                )
    return _turns_from_messages(msgs, source="local", agent=agent), started


def resolve(row: dict) -> Conversation | None:
    """Parse the session a selection row names, or None when it cannot be used (see
    `resolve_with_reason`); the caller counts it skipped."""
    return resolve_with_reason(row)[0]


def _encodes(text: str | None) -> bool:
    try:
        (text or "").encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate from a JSON escape
        return False
    return True


def resolve_with_reason(row: dict) -> tuple[Conversation | None, str | None]:
    """Parse the session a selection row names. Returns (conversation, None), or (None, reason)
    when the row lacks an id or a path, the file cannot be read, its text is not valid UTF-8, it
    fails to parse, or it holds no usable turn."""
    tool = row.get("tool")
    native_id = row.get("native_id")
    conversation_id = row.get("conversation_id")
    path = row.get("path")
    if not conversation_id or not native_id or not path:
        return None, "no session id or path"
    if tool != "claude-code":
        return None, "not a Claude Code session"
    try:
        turns, started = parse_claude_code_session(path, native_id, agent=tool)
    except UnicodeDecodeError:
        return None, "not valid UTF-8"
    except OSError as e:
        return None, f"cannot read: {e.strerror or type(e).__name__}"
    except Exception as e:  # any other malformed shape skips this session, never the run
        return None, f"cannot parse: {type(e).__name__}"
    if not all(_encodes(t.content) and _encodes(t.carried_text) for t in turns):
        return None, "not valid UTF-8"
    if not turns:
        return None, "no readable turns"
    return Conversation(
        conversation_id=conversation_id,
        tool=tool or "",
        source=row.get("source") or "local",
        date=row.get("date") or started,
        turns=turns,
        truncated=False,
    ), None


def render_transcript(conv: Conversation, max_chars: int = DEFAULT_MAX_CHARS) -> Conversation:
    """Apply the whole-transcript char budget at turn boundaries and return a Conversation
    whose turns are exactly what is sent (so the evidence gate sees the same text)."""
    kept: list[Turn] = []
    budget = max_chars
    truncated = False
    for t in conv.turns:
        line_cost = len(t.content) + 32  # + room for the "[i] role:" marker
        if budget - line_cost < 0:
            truncated = True
            break
        budget -= line_cost
        kept.append(t)
    return Conversation(conv.conversation_id, conv.tool, conv.source, conv.date, kept, truncated)


def transcript_text(conv: Conversation) -> str:
    """The turn-indexed transcript block handed to the model."""
    lines = ["--- TRANSCRIPT (turn-indexed; [i] role: content) ---"]
    for t in conv.turns:
        lines.append(f"[{t.i}] {t.role}: {t.content}")
    if conv.truncated:
        lines.append("[...transcript truncated for length...]")
    return "\n".join(lines)
