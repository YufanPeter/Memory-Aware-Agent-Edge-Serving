"""Parsing model output into (action, argument)."""
from __future__ import annotations

import re

_ACTION = re.compile(r"Action\s*\d*\s*:\s*([A-Za-z_]+)\s*\[(.*)\]", re.S)
_ACTION_LOOSE = re.compile(r"Action\s*\d*\s*:\s*([A-Za-z_]+)\s*[:(]?\s*(.+)", re.S)
_FINAL = re.compile(r"(?:Final Answer|Answer)\s*:\s*(.+)", re.I | re.S)


def parse_react(text: str) -> tuple[str, str] | None:
    """Return (action, arg) from the last `Action: name[arg]` line, or None."""
    lines = [ln for ln in text.splitlines() if re.match(r"\s*Action\s*\d*\s*:", ln)]
    if lines:
        last = lines[-1]
        if m := _ACTION.search(last):
            return m.group(1).lower(), m.group(2).strip()
        if m := _ACTION_LOOSE.search(last):
            return m.group(1).lower(), m.group(2).strip().rstrip(")]").strip()
    if m := _FINAL.search(text):
        return "finish", m.group(1).strip().splitlines()[0]
    return None


def parse_answer(text: str) -> str:
    if m := _FINAL.search(text):
        return m.group(1).strip().splitlines()[0].strip()
    if (a := parse_react(text)) and a[0] == "finish":
        return a[1]
    return text.strip().splitlines()[0].strip() if text.strip() else ""
