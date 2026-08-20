"""Fixtures: planted-ring network."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_network import generate


@pytest.fixture(scope="session")
def accounts():
    return generate(n_accounts=1200, n_rings=10, ring_size_range=(4, 8), seed=5)
