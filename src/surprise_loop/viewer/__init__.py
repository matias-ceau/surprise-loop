"""Pages a person opens in a browser: plain code that embeds results computed in Python.

Both pages draw the clearing with the one shared fragment `clearing.html` (its style and script),
which `export_page` puts in place of the template's `@@CLEARING@@`, so the pages cannot drift
apart and each exported file stays self-contained.
"""

import json
from importlib import resources
from typing import Any

FRAGMENT = "clearing.html"
FRAGMENT_PLACEHOLDER = "@@CLEARING@@"


def _read(name: str) -> str:
    return resources.files("surprise_loop.viewer").joinpath(name).read_text(encoding="utf-8")


def export_page(template: str, placeholder: str, data: Any) -> str:
    """The template `template` with the shared fragment in place and `data` embedded as JSON at
    `placeholder`, inside a `<script type="application/json">` element."""
    page = _read(template).replace(FRAGMENT_PLACEHOLDER, _read(FRAGMENT).strip(), 1)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # nothing inside the script element may close it or open a comment
    payload = payload.replace("<", "\\u003c")
    return page.replace(placeholder, payload, 1)
