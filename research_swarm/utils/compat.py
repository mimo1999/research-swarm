"""Python-version compatibility shims.

``enum.StrEnum`` was added in Python 3.11. This project targets 3.11+, but
the Hugging Face Space deployment's container image is pinned to Python
3.10 (its base image, independent of this repo's own version target), so
schemas importing ``StrEnum`` directly from ``enum`` break there. Import it
from here instead so the same source works on both.
"""
from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):  # noqa: D101 -- mirrors stdlib StrEnum exactly
        def __str__(self) -> str:
            return str.__str__(self)

        def __format__(self, format_spec: str) -> str:
            return str.__format__(self, format_spec)


__all__ = ["StrEnum"]
