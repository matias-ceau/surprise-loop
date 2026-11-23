"""Pages a person opens in a browser: plain code that embeds results computed in Python.

Each page is one template in this package with its data embedded as JSON, so an exported file is
self-contained.
"""

import json
from importlib import resources
from typing import Any


def _read(name: str) -> str:
    return resources.files("surprise_loop.viewer").joinpath(name).read_text(encoding="utf-8")


def export_page(template: str, placeholder: str, data: Any) -> str:
    """The template `template` with `data` embedded as JSON at `placeholder`, inside a
    `<script type="application/json">` element."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # nothing inside the script element may close it or open a comment
    payload = payload.replace("<", "\\u003c")
    return _read(template).replace(placeholder, payload, 1)
