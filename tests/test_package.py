from __future__ import annotations

import importlib.metadata

import lkio as m


def test_version() -> None:
    assert importlib.metadata.version("lkio") == m.__version__
