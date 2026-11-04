"""An operator's documentation page, as plain blocks.

``GET /operators`` gives each operator one line. The rest — what its arguments do, worked
calculations, the ``rettype`` table of ``ts_regression``, tips and example simulations — is on
the page its ``documentation`` path names, the operator reference's "Show more". That page is a
list of blocks: HTML text, tables, formula images and simulation examples. Here each becomes a
``{"type", "text"}`` block a person or a prompt can read; an image keeps its title and address,
since its formula is drawn, not written.
"""

from __future__ import annotations

import json
from html.parser import HTMLParser
from typing import Any

#: Tags that end a line of text.
_BREAKS = frozenset({"p", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "div"})


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # noqa: ARG002 - HTMLParser's signature
        if tag == "li":
            self.parts.append("\n- ")
        elif tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _BREAKS and tag != "br":
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def plain(html: str) -> str:
    """HTML as text, one paragraph or list item a line."""
    parser = _Text()
    parser.feed(html)
    parser.close()
    lines = [line.rstrip() for line in "".join(parser.parts).splitlines()]
    return "\n".join(line for line in lines if line.strip()).strip()


def _table(value: Any) -> str:
    data = json.loads(value) if isinstance(value, str) else value
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return ""
    return "\n".join(
        "| " + " | ".join(plain(str(cell)) for cell in row) + " |"
        for row in rows
        if isinstance(row, list)
    )


def _example(value: Any) -> str:
    data = json.loads(value) if isinstance(value, str) else value
    if not isinstance(data, dict):
        return ""
    s = data.get("settings") or {}
    market = f"{s.get('region')}/D{s.get('delay')}/{s.get('universe')}"
    knobs = f"{s.get('neutralization')}, decay {s.get('decay')}, truncation {s.get('truncation')}"
    return f"{market}, {knobs}: {str(data.get('regular') or '').strip()}"


def blocks(page: dict[str, Any]) -> list[dict[str, str]]:
    """The page's content blocks, each ``{"type", "text"}``; an image adds ``"url"``.

    A block that cannot be read is left out rather than failing the page.
    """
    out: list[dict[str, str]] = []
    for block in page.get("content") or []:
        if not isinstance(block, dict):
            continue
        kind, value = str(block.get("type") or "").upper(), block.get("value")
        try:
            if kind == "TEXT" and isinstance(value, str):
                out.append({"type": "text", "text": plain(value)})
            elif kind == "EQUATION":
                text = plain(value) if isinstance(value, str) else json.dumps(value)
                out.append({"type": "equation", "text": text})
            elif kind == "TABLE":
                out.append({"type": "table", "text": _table(value)})
            elif kind == "SIMULATION_EXAMPLE":
                out.append({"type": "example", "text": _example(value)})
            elif kind == "IMAGE" and isinstance(value, dict):
                out.append(
                    {
                        "type": "image",
                        "text": str(value.get("title") or ""),
                        "url": str(value.get("url") or ""),
                    }
                )
        except ValueError, TypeError:
            continue
    return [b for b in out if b["text"] or b.get("url")]
