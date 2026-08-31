"""Unit tests for the Blackboard Architecture in fraudgraph."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from fraudgraph.blackboard.core import (
    BlackboardController,
    CommunityDetector,
    DeviceCorroborator,
    FraudBlackboard,
    IPWatcher,
    KnowledgeSource,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ring_accounts() -> pd.DataFrame:
    """3 ring accounts sharing device_id=DEV1, 2 legit accounts with unique devices."""
    return pd.DataFrame(
        [
            {"account_id": 1, "device_id": "DEV1", "address_id": "A1", "payee_id": "P1"},
            {"account_id": 2, "device_id": "DEV1", "address_id": "A2", "payee_id": "P1"},
            {"account_id": 3, "device_id": "DEV1", "address_id": "A3", "payee_id": "P1"},
            {"account_id": 4, "device_id": "DEV2", "address_id": "A4", "payee_id": "P2"},
            {"account_id": 5, "device_id": "DEV3", "address_id": "A5", "payee_id": "P3"},
        ]
    )


@pytest.fixture
def empty_blackboard(ring_accounts) -> FraudBlackboard:
    return FraudBlackboard(accounts=ring_accounts)


# ---------------------------------------------------------------------------
# FraudBlackboard: annotate / read / record_verdict / aggregate_score
# ---------------------------------------------------------------------------


def test_annotate_and_read(empty_blackboard):
    bb = empty_blackboard
    bb.annotate(1, "test_key", 42)
    assert bb.read(1, "test_key") == 42
    assert bb.read(1, "missing", "default") == "default"
    assert bb.read(99, "anything", None) is None


def test_record_verdict_and_aggregate(empty_blackboard):
    bb = empty_blackboard
    bb.record_verdict(1, "ks_a", 0.6, "reason A")
    bb.record_verdict(1, "ks_b", 0.9, "reason B")
    # max-of-evidence aggregation
    assert bb.aggregate_score(1) == pytest.approx(0.9)


def test_aggregate_score_no_verdicts(empty_blackboard):
    bb = empty_blackboard
    assert bb.aggregate_score(99) == 0.0


# ---------------------------------------------------------------------------
# DeviceCorroborator
# ---------------------------------------------------------------------------


def test_device_corroborator_flags_shared_device(ring_accounts):
    bb = FraudBlackboard(accounts=ring_accounts)
    ks = DeviceCorroborator(threshold=3)
    ks.contribute(bb)

    # Accounts 1,2,3 share DEV1 (group size = 3 = threshold) — should be flagged
    assert bb.aggregate_score(1) > 0
    assert bb.aggregate_score(2) > 0
    assert bb.aggregate_score(3) > 0


def test_device_corroborator_leaves_legit_clean(ring_accounts):
    bb = FraudBlackboard(accounts=ring_accounts)
    ks = DeviceCorroborator(threshold=3)
    ks.contribute(bb)

    # Accounts 4,5 have unique devices
    assert bb.aggregate_score(4) == 0.0
    assert bb.aggregate_score(5) == 0.0


def test_device_corroborator_above_threshold(ring_accounts):
    bb = FraudBlackboard(accounts=ring_accounts)
    # threshold=4 — DEV1 group has 3 members, below threshold
    ks = DeviceCorroborator(threshold=4)
    ks.contribute(bb)
    assert bb.aggregate_score(1) == 0.0


# ---------------------------------------------------------------------------
# IPWatcher
# ---------------------------------------------------------------------------


def test_ip_watcher_flags_shared_address():
    accounts = pd.DataFrame(
        [
            {"account_id": 10, "device_id": "D1", "address_id": "ADDR1"},
            {"account_id": 11, "device_id": "D2", "address_id": "ADDR1"},
            {"account_id": 12, "device_id": "D3", "address_id": "ADDR1"},
            {"account_id": 13, "device_id": "D4", "address_id": "ADDR1"},
            {"account_id": 14, "device_id": "D5", "address_id": "ADDR2"},
        ]
    )
    bb = FraudBlackboard(accounts=accounts)
    ks = IPWatcher(threshold=4)
    ks.contribute(bb)

    assert bb.aggregate_score(10) > 0
    assert bb.aggregate_score(14) == 0.0


# ---------------------------------------------------------------------------
# CommunityDetector
# ---------------------------------------------------------------------------


def test_community_detector_flags_dense_subgraph(ring_accounts):
    # Build a corroborated graph where accounts 1,2,3 all share device+payee
    g = nx.Graph()
    g.add_nodes_from([1, 2, 3, 4, 5])
    g.add_edge(1, 2, weight=2)
    g.add_edge(2, 3, weight=2)
    g.add_edge(1, 3, weight=2)

    bb = FraudBlackboard(accounts=ring_accounts, graph=g)
    ks = CommunityDetector(min_community_size=3, min_multi_edges=2)
    ks.contribute(bb)

    assert bb.aggregate_score(1) > 0
    assert bb.aggregate_score(2) > 0
    assert bb.aggregate_score(3) > 0
    # Isolated accounts 4,5 are not in the dense community
    assert bb.aggregate_score(4) == 0.0


def test_community_detector_empty_graph(ring_accounts):
    bb = FraudBlackboard(accounts=ring_accounts)
    ks = CommunityDetector()
    ks.contribute(bb)  # Should not raise; empty graph = no verdicts
    assert all(bb.aggregate_score(aid) == 0.0 for aid in range(1, 6))


# ---------------------------------------------------------------------------
# BlackboardController
# ---------------------------------------------------------------------------


def test_controller_runs_all_ks(ring_accounts):
    """Controller orchestrates all KSes and returns a score dict."""
    g = nx.Graph()
    g.add_nodes_from([1, 2, 3, 4, 5])
    g.add_edge(1, 2, weight=2)
    g.add_edge(2, 3, weight=2)
    g.add_edge(1, 3, weight=2)

    bb = FraudBlackboard(accounts=ring_accounts, graph=g)
    ctrl = BlackboardController(
        sources=[
            DeviceCorroborator(threshold=3),
            IPWatcher(threshold=4),
            CommunityDetector(min_community_size=3),
        ]
    )
    scores = ctrl.run(bb)

    assert set(scores.keys()) == {1, 2, 3, 4, 5}
    # Ring members should have non-zero scores from DeviceCorroborator + CommunityDetector
    assert scores[1] > 0
    assert scores[4] == 0.0


def test_controller_custom_ks_registration():
    """KSes can be registered dynamically — open/closed extension."""
    isolated_accounts = pd.DataFrame(
        [{"account_id": 100, "device_id": "DX"}, {"account_id": 101, "device_id": "DY"}]
    )

    class AlwaysZeroKS(KnowledgeSource):
        name = "always_zero"

        def contribute(self, blackboard: FraudBlackboard) -> None:
            pass  # writes nothing

    ctrl = BlackboardController(sources=[])
    ctrl.register(AlwaysZeroKS())
    bb = FraudBlackboard(accounts=isolated_accounts)
    scores = ctrl.run(bb)

    assert all(s == 0.0 for s in scores.values())
