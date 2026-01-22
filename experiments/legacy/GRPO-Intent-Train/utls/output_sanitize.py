from __future__ import annotations

import json
from typing import Any, Optional, Tuple


def _find_json_span(text: str) -> Optional[Tuple[int, int]]:
    s = text.strip()
    if not s:
        return None

    start = None
    open_ch = None
    close_ch = None

    for i, ch in enumerate(s):
        if ch == "[":
            start, open_ch, close_ch = i, "[", "]"
            break
        if ch == "{":
            start, open_ch, close_ch = i, "{", "}"
            break

    if start is None:
        return None

    depth = 0
    in_str = False
    esc = False

    for j in range(start, len(s)):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return (start, j + 1)

    return None


def extract_first_json_str(text: str) -> Optional[str]:
    """
    Return the substring of the first complete JSON array/object.
    """
    span = _find_json_span(text)
    if span is None:
        return None
    a, b = span
    return text.strip()[a:b].strip()


def sanitize_to_json_str(
    text: str,
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
) -> Optional[str]:
    """
    Extract first JSON from text, parse it, and dump it back to a normalized JSON string.
    Returns None if extraction/parsing fails.
    """
    js = extract_first_json_str(text)
    if js is None:
        return None
    try:
        obj = json.loads(js)
    except json.JSONDecodeError:
        return None
    return json.dumps(obj, ensure_ascii=ensure_ascii, sort_keys=sort_keys)


def parse_sanitized_json(text: str) -> Optional[Any]:
    """
    Extract and parse the first JSON from text.
    Returns parsed object or None.
    """
    js = extract_first_json_str(text)
    if js is None:
        return None
    try:
        return json.loads(js)
    except json.JSONDecodeError:
        return None
