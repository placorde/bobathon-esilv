"""Package root for the Parkinson's UPDRS prediction workspace.

Exposes `PROJECT_ROOT`: the absolute path to the project root,
derived from this file's location. Any module that needs to resolve
a project-relative path imports this constant instead of hard-coding
a CWD-relative string.

Requires an editable install (`pip install -e .`) so `__file__`
resolves back into the source tree and `parents[2]` is the repo root.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
