#!/usr/bin/env python3
# Copyright 2026
#
# Unit tests for utils.parse_output and its repair/salvage chain -- the
# selector-output parser used by the auto-hint agent loop (and vendored from
# selector/run_hint_selection_model.py).
#
# Regression target: the 20260723 dolci async run applied ~5% of its hints with
# EMPTY text ("(the selector returned an empty hint)" injected, budget charged)
# even though the raw gpt-5-mini responses contained both a valid hint_id and
# the full hint text. Chain of causes, each covered below:
#   1. the selector mixes escape styles in one blob (invalid \( next to valid
#      \\frac); _repair_json_escapes doubled the SECOND backslash of the valid
#      pair (\\le -> \\\le), so the repaired JSON stayed unparseable;
#   2. json.loads rejected raw newlines the selector copies verbatim into
#      "quote" strings (fixed by strict=False);
#   3. _extract_last_json_object then salvaged a nested completed_hints
#      citation item {hint_id, quote, why} as the "selection" -- a plausible
#      hint_id with no hint text;
#   4. _hard_parse's first-match "hint_id" grab also returned a cited id, not
#      the selected one.
#
# Run:  python test_selector_parse.py   (or pytest test_selector_parse.py)
from __future__ import annotations

from utils import (
    _as_selection_dict,
    _hard_parse,
    _repair_json_escapes,
    hint_id_of,
    parse_output,
)


def _wrap(block: str) -> str:
    return "<output>\n" + block + "\n</output>"


# A faithful miniature of the failing gpt-5-mini pattern: multi-line raw
# newline inside a quote, single-escaped \( \) LaTeX next to double-escaped
# \\le -- and clean nested citation items ready to be mis-salvaged.
MIXED_ESCAPES = """{
  "completed_hints": [
    {
      "hint_id": "1.1",
      "quote": "Factoring away:

\\\\[ 81 - 18 = 63 \\\\]",
      "why": "The student subtracted \\(2\\cdot 9\\) from \\(81\\)."
    }
  ],
  "hint_id": "1.2",
  "reasoning_of_hint": "They never justified an upper bound like \\(a\\\\le 15\\) via \\(\\\\frac{1}{5}\\\\le\\\\frac{3}{a}\\).",
  "hint": "Using a <= b <= c gives 1/5 <= 3/a, so a <= 15."
}"""


def test_repair_preserves_valid_pairs():
    # \\le is ALREADY a valid JSON escape pair; the old repair turned it into
    # \\\le (invalid). The pair must survive untouched while lone \( doubles.
    assert _repair_json_escapes(r'"\\le"') == r'"\\le"'
    assert _repair_json_escapes(r'"\("') == r'"\\("'
    assert _repair_json_escapes(r'"\(a\\le b\)"') == r'"\\(a\\le b\\)"'
    # idempotent: repairing repaired text changes nothing
    once = _repair_json_escapes(r'"\(x\\frac{1}{2}\)"')
    assert _repair_json_escapes(once) == once


def test_mixed_escapes_recover_hint_text_and_id():
    sel, err = parse_output(_wrap(MIXED_ESCAPES))
    assert err is None and isinstance(sel, dict)
    assert sel.get("hint_id") == "1.2"
    assert sel.get("hint", "").startswith("Using a <= b <= c")
    assert isinstance(sel.get("completed_hints"), list) and len(sel["completed_hints"]) == 1


def test_raw_newline_in_string_parses():
    block = '{"hint_id": "2.1", "completed_hints": [], "hint": "line one\nline two"}'
    sel, err = parse_output(_wrap(block))
    assert err is None and sel["hint_id"] == "2.1"
    assert sel["hint"] == "line one\nline two"


def test_citation_item_is_never_the_selection():
    # Blob so broken that only the nested citation object parses: the old
    # fallback returned {hint_id, quote, why} as the selection (empty hint
    # downstream). Now it must be rejected and _hard_parse must recover the
    # TOP-LEVEL id (nearest the "hint" key), not the cited one.
    broken = (
        '{\n  "completed_hints": [\n'
        '    {"hint_id": "1.1", "quote": "ok text", "why": "done"}\n'
        '  ],\n'
        '  "hint_id": "3.2", "invalid \\q here": unquoted,\n'
        '  "hint": "Take squared moduli to get 2^x5^y=10000."\n}'
    )
    sel, err = parse_output(_wrap(broken))
    assert isinstance(sel, dict)
    assert "quote" not in sel and "why" not in sel
    assert sel.get("hint_id") == "3.2"
    assert "moduli" in sel.get("hint", "")


def test_as_selection_dict_guards():
    cite = {"hint_id": "1.1", "quote": "q", "why": "w"}
    assert _as_selection_dict(cite) is None
    assert _as_selection_dict([cite, {"hint_id": "2.1", "hint": "h"}]) == {
        "hint_id": "2.1", "hint": "h"}
    # real selections still pass, including hint-less status-only ones
    ok = {"hint_id": "2.1", "completed_hints": [], "hint": "h"}
    assert _as_selection_dict(ok) is ok
    status_only = {"hint_id": "", "hint": "", "completed_hints": [cite]}
    assert _as_selection_dict(status_only) is status_only


def test_hard_parse_prefers_id_near_selection_keys():
    text = (
        '"completed_hints": [{"hint_id": "1.1", "quote": "q", "why": "w"},'
        ' {"hint_id": "1.2", "quote": "q2", "why": "w2"}],'
        ' "hint_id": "4.1", "reasoning_of_hint": "r", "hint": "the hint"'
    )
    obj = _hard_parse(text)
    assert obj["hint_id"] == "4.1"
    assert obj["hint"] == "the hint"


def test_plain_valid_json_unchanged():
    block = ('{"completed_hints": [], "hint_id": "1.1", '
             '"reasoning_of_hint": "r", "hint": "h"}')
    sel, err = parse_output(_wrap(block))
    assert err is None and sel["hint_id"] == "1.1" and sel["hint"] == "h"


def test_completed_hint_progress_survives_parse():
    block = (
        '{"completed_hints": [{"hint_id": "1.1", "quote": "q", "why": "w", '
        '"progress": "I used \\\\frac{a}{b} correctly."}], '
        '"hint_id": "1.2", "reasoning_of_hint": "r", "hint": "next"}'
    )
    sel, err = parse_output(_wrap(block))
    assert err is None
    assert sel["completed_hints"][0]["quote"] == "q"
    assert sel["completed_hints"][0]["progress"] == r"I used \frac{a}{b} correctly."


def test_hard_parse_recovers_completed_hint_progress():
    raw = _wrap(
        '{\n'
        '  "completed_hints": [\n'
        '    {"hint_id": "1.1", "quote": "student work", "why": "done", '
        '"progress": "I completed step 1."}\n'
        '  ],\n'
        '  "hint_id": "1.2",\n'
        '  "broken": unquoted,\n'
        '  "hint": "Use the next identity."\n'
        '}'
    )
    sel, err = parse_output(raw)
    assert err is None
    assert sel["hint_id"] == "1.2"
    assert sel["completed_hints"] == [{
        "hint_id": "1.1",
        "quote": "student work",
        "why": "done",
        "progress": "I completed step 1.",
    }]


def test_array_coercion_and_garbage():
    sel, _ = parse_output(_wrap('[{"hint_id": "1.1", "hint": "h"}, {"x": 1}]'))
    assert sel == {"hint_id": "1.1", "hint": "h"}
    sel, err = parse_output("no json here at all")
    assert sel is None and err


# ---------------------------------------------------------------------------
# Fixtures below are distilled from the 20260724 dolci qwen3 run's
# selector_calls corpus (618,056 calls replayed; 18,619 empty-hint instances).
# ---------------------------------------------------------------------------

def test_latex_shadow_escapes_survive_valid_json():
    # \frac / \boxed / \theta / \right are VALID JSON escape prefixes, so a
    # strictly-well-formed blob silently decoded them to control chars
    # (\x0crac, \x08oxed, \theta, \right): 3.6% of this run's applied hints
    # carried such corruption. The pre-parse normalizer must keep the LaTeX.
    block = ('{"completed_hints": [], "hint_id": "3.1", '
             '"hint": "Use \\frac{1}{2} and \\boxed{42} with \\theta, '
             '\\times k, \\binom{n}{2} and \\rfloor."}')
    sel, err = parse_output(_wrap(block))
    assert err is None
    assert sel["hint"] == ("Use \\frac{1}{2} and \\boxed{42} with \\theta, "
                           "\\times k, \\binom{n}{2} and \\rfloor.")
    # intended JSON escapes still decode: \n before a non-letter, \t at end
    sel2, _ = parse_output(_wrap('{"hint_id": "1.1", "hint": "a\\n(b)"}'))
    assert sel2["hint"] == "a\n(b)"


def test_u_nonhex_is_latex_not_unicode_escape():
    # \underbrace: "\u" + "nder..." is an invalid \uXXXX escape that used to
    # fail the whole blob into the salvage chain; now it parses as literal LaTeX.
    block = ('{"completed_hints": [], "hint_id": "2.1", '
             '"hint": "Write N as \\underbrace{0\\cdots0}_{100} and reduce."}')
    sel, err = parse_output(_wrap(block))
    assert err is None
    assert sel["hint_id"] == "2.1"
    assert sel["hint"].startswith("Write N as \\underbrace{0")
    # a REAL \uXXXX escape still decodes (U+2260 with a surrogate-pair friend)
    block2 = ('{"hint_id": "2.2", "hint": "so a \\u2260 b for \\ud835\\udc65"}')
    sel2, _ = parse_output(_wrap(block2))
    assert sel2["hint"] == "so a ≠ b for \U0001d465"


def test_nl_commands_restored_but_prose_newlines_kept():
    block = ('{"hint_id": "4.1", '
             '"hint": "Show x \\neq 0 and 3 \\nmid k.\\nThen conclude."}')
    sel, err = parse_output(_wrap(block))
    assert err is None
    assert sel["hint"] == "Show x \\neq 0 and 3 \\nmid k.\nThen conclude."


def test_decline_answers_yield_no_id_and_no_placeholder_fuel():
    # The selector's explicit "all pending hints already achieved" answers come
    # as hint_id ""/"none" with hint "". The dict is parsed faithfully but
    # hint_id_of maps the id to None, so the loop's decline guard (benign stop,
    # no placeholder injection, no penalty, no "" in completed) fires.
    for hid in ('""', '"none"', '"None"', '"null"'):
        block = ('{"completed_hints": [{"hint_id": "3.1", "quote": "q", '
                 '"why": "w"}], "hint_id": %s, '
                 '"reasoning_of_hint": "all pending achieved", "hint": ""}' % hid)
        sel, err = parse_output(_wrap(block))
        assert err is None and isinstance(sel, dict), hid
        assert not str(sel.get("hint") or "").strip()
        assert hint_id_of(sel) is None, hid
    # real ids keep working, including float/whitespace forms
    assert hint_id_of({"hint_id": "2.1"}) == "2.1"
    assert hint_id_of({"hint_id": 2.1}) == "2.1"
    assert hint_id_of({"hint_id": " 2.1 "}) == "2.1"
    assert hint_id_of(None) is None


def test_truncated_citations_only_returns_none():
    # Response cut off mid-completed_hints: no top-level selection was ever
    # emitted. The old salvage attached a CITED id ({'hint_id': '1.2'}), which
    # the runtime back-fills from the pool -> injects an already-given hint.
    raw = ('<output>{\n "completed_hints": [\n'
           ' {"hint_id": "1.1", "quote": "factored x", "why": "did step 1"},\n'
           ' {"hint_id": "1.2", "quote": "then I sub')
    sel, err = parse_output(raw)
    assert sel is None and err


def test_idless_selection_never_borrows_citation_id():
    # Model omitted the top-level hint_id but wrote real hint text (and the
    # blob is broken enough to need regex salvage). The hint must come back
    # WITHOUT any id (the loop remaps id from the pool by text) rather than
    # with the cited 2.1.
    raw = _wrap(
        '{\n  "reasoning_of_hint": "the projection step is missing",\n'
        '  "hint": "Project D onto AC and simplify.",\n'
        '  "invalid \\q": unquoted,\n'
        '  "completed_hints": [\n'
        '    {"hint_id": "2.1", "quote": "set E=(3,0)", "why": "did the foot"}\n'
        '  ]\n}')
    sel, err = parse_output(raw)
    assert isinstance(sel, dict)
    assert sel.get("hint") == "Project D onto AC and simplify."
    assert hint_id_of(sel) is None


def test_contentless_dicts_rejected():
    assert parse_output(_wrap('{}'))[0] is None
    assert parse_output(_wrap('{"foo": 1}'))[0] is None


def test_ctrl_garbage_escapes_scrubbed_from_delivered_fields():
    # Real corpus records: the selector emitted "\\u0001frac{100}{1200}" (meant
    # \frac) and "\\u0003(u,v,w)\\u0003" (meant \(u,v,w\)). Intent is
    # unreconstructable; delivered text must at least carry no control chars
    # (a lone surrogate/control char in one hint once killed a whole run at
    # the tokenizer). Citation quotes stay verbatim for trace matching.
    block = ('{"hint_id": "3.1", '
             '"hint": "so k = \\u0001frac{100}{1200} for \\u0003(u,v,w)\\u0003"}')
    sel, err = parse_output(_wrap(block))
    assert err is None
    assert sel["hint"] == "so k = frac{100}{1200} for (u,v,w)"
    # lone surrogate scrubbed, valid pair kept
    block2 = '{"hint_id": "3.2", "hint": "keep \\ud835\\udc65 drop \\ud835 end"}'
    sel2, _ = parse_output(_wrap(block2))
    assert sel2["hint"] == "keep \U0001d465 drop  end"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:  # noqa: PERF203
                fails += 1
                print(f"FAIL {name}: {e}")
    raise SystemExit(1 if fails else 0)
