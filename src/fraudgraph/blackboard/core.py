"""Blackboard Architecture — Shared Graph Blackboard + Knowledge Sources.

The Blackboard pattern uses a shared data structure (the blackboard) that is
inspected and enriched by independent Knowledge Sources (KS). No KS knows about
the others; they only read/write the blackboard. A controller decides which KS
to invoke next.

In fraudgraph:
  - The **Blackboard** wraps a NetworkX bipartite graph linking accounts to
    shared attributes (device, address, payee).
  - **Knowledge Sources** (`DeviceCorroborator`, `IPWatcher`, `CommunityDetector`)
    each enrich the blackboard with fraud-signal annotations.
  - A **BlackboardController** runs all KSes in sequence and collects their verdicts.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Blackboard (shared state)
# ---------------------------------------------------------------------------


@dataclass
class FraudBlackboard:
    """Shared, mutable blackboard enriched by Knowledge Sources.

    Attributes:
        graph:       Bipartite account<->attribute graph.
        accounts:    DataFrame with account metadata.
        annotations: KS-written dict of lists keyed by account_id.
        verdicts:    Final per-account fraud signals (written by KSes).
    """

    accounts: pd.DataFrame
    graph: nx.Graph = field(default_factory=nx.Graph)
    annotations: dict[int, dict[str, Any]] = field(default_factory=dict)
    verdicts: dict[int, dict[str, Any]] = field(default_factory=dict)

    def annotate(self, account_id: int, key: str, value: Any) -> None:
        """Knowledge Source writes a named annotation for an account."""
        if account_id not in self.annotations:
            self.annotations[account_id] = {}
        self.annotations[account_id][key] = value

    def read(self, account_id: int, key: str, default: Any = None) -> Any:
        """Read a named annotation for an account (returns default if absent)."""
        return self.annotations.get(account_id, {}).get(key, default)

    def record_verdict(self, account_id: int, source: str, signal: float, reason: str) -> None:
        """Knowledge Source records its fraud signal (0.0–1.0) for an account."""
        if account_id not in self.verdicts:
            self.verdicts[account_id] = {}
        self.verdicts[account_id][source] = {"signal": signal, "reason": reason}

    def aggregate_score(self, account_id: int) -> float:
        """Combine all KS signals via max-of-evidence (any KS flagging = suspect)."""
        signals = [v["signal"] for v in self.verdicts.get(account_id, {}).values()]
        return max(signals, default=0.0)


# ---------------------------------------------------------------------------
# Knowledge Source ABC
# ---------------------------------------------------------------------------


class KnowledgeSource(ABC):
    """Abstract Knowledge Source — inspects the blackboard and adds annotations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier used when writing to the blackboard."""

    @abstractmethod
    def contribute(self, blackboard: FraudBlackboard) -> None:
        """Read the blackboard, run analysis, write back signals."""


# ---------------------------------------------------------------------------
# Knowledge Source: DeviceCorroborator
# ---------------------------------------------------------------------------


class DeviceCorroborator(KnowledgeSource):
    """Flags accounts that share device_id with multiple other accounts.

    A device shared by N>=threshold accounts is a co-registration signal.
    """

    name = "device_corroborator"

    def __init__(self, threshold: int = 3) -> None:
        self.threshold = threshold

    def contribute(self, blackboard: FraudBlackboard) -> None:
        accounts = blackboard.accounts
        if "device_id" not in accounts.columns:
            return
        device_groups = accounts.groupby("device_id")["account_id"].apply(list)
        for device_id, acct_ids in device_groups.items():
            if len(acct_ids) >= self.threshold:
                for aid in acct_ids:
                    signal = min(1.0, (len(acct_ids) - self.threshold + 1) / 10.0)
                    blackboard.annotate(aid, "shared_device_count", len(acct_ids))
                    blackboard.record_verdict(
                        aid,
                        self.name,
                        signal,
                        f"device {device_id} shared by {len(acct_ids)} accounts",
                    )
                logger.debug(
                    "DeviceCorroborator: device %s shared by %d accounts",
                    device_id,
                    len(acct_ids),
                )


# ---------------------------------------------------------------------------
# Knowledge Source: IPWatcher
# ---------------------------------------------------------------------------


class IPWatcher(KnowledgeSource):
    """Detects accounts with repeated address overlap (proxy for IP/location).

    Accounts sharing the same address_id within a short registration window
    are suspicious — legitimate siblings occasionally share addresses but
    organised rings do it systematically.
    """

    name = "ip_watcher"

    def __init__(self, threshold: int = 4) -> None:
        self.threshold = threshold

    def contribute(self, blackboard: FraudBlackboard) -> None:
        accounts = blackboard.accounts
        if "address_id" not in accounts.columns:
            return
        addr_groups = accounts.groupby("address_id")["account_id"].apply(list)
        for addr_id, acct_ids in addr_groups.items():
            if len(acct_ids) >= self.threshold:
                for aid in acct_ids:
                    signal = min(1.0, (len(acct_ids) - self.threshold + 1) / 8.0)
                    blackboard.annotate(aid, "shared_address_count", len(acct_ids))
                    blackboard.record_verdict(
                        aid,
                        self.name,
                        signal,
                        f"address {addr_id} shared by {len(acct_ids)} accounts",
                    )


# ---------------------------------------------------------------------------
# Knowledge Source: CommunityDetector
# ---------------------------------------------------------------------------


class CommunityDetector(KnowledgeSource):
    """Applies Louvain-style community detection on the shared-attribute graph.

    Large, dense communities on the corroborated subgraph indicate organised rings.
    """

    name = "community_detector"

    def __init__(self, min_community_size: int = 4, min_multi_edges: int = 2) -> None:
        self.min_community_size = min_community_size
        self.min_multi_edges = min_multi_edges

    def contribute(self, blackboard: FraudBlackboard) -> None:
        g = blackboard.graph
        if g.number_of_nodes() == 0:
            return

        # Corroborated subgraph: only edges with >=2 shared attributes
        corroborated = nx.Graph(
            (u, v) for u, v, d in g.edges(data=True) if d.get("weight", 1) >= self.min_multi_edges
        )

        for comp in nx.connected_components(corroborated):
            if len(comp) >= self.min_community_size:
                subg = corroborated.subgraph(comp)
                density = nx.density(subg)
                signal = min(1.0, density + (len(comp) - self.min_community_size) / 20.0)
                for aid in comp:
                    blackboard.annotate(aid, "community_size", len(comp))
                    blackboard.annotate(aid, "community_density", round(density, 4))
                    blackboard.record_verdict(
                        aid,
                        self.name,
                        signal,
                        f"dense community of {len(comp)} accounts (density={density:.2f})",
                    )


# ---------------------------------------------------------------------------
# Blackboard Controller
# ---------------------------------------------------------------------------


class BlackboardController:
    """Runs all Knowledge Sources against the shared blackboard.

    KSes are independent — they don't know about each other.
    The controller is responsible for sequencing and final score aggregation.
    """

    def __init__(self, sources: list[KnowledgeSource] | None = None) -> None:
        self._sources: list[KnowledgeSource] = sources or [
            DeviceCorroborator(),
            IPWatcher(),
            CommunityDetector(),
        ]

    def register(self, source: KnowledgeSource) -> None:
        """Dynamically add a new Knowledge Source."""
        self._sources.append(source)

    def run(self, blackboard: FraudBlackboard) -> dict[int, float]:
        """Run all KSes, return per-account aggregated fraud scores."""
        for ks in self._sources:
            logger.info("Running KnowledgeSource: %s", ks.name)
            ks.contribute(blackboard)

        scores = {
            int(aid): blackboard.aggregate_score(int(aid))
            for aid in blackboard.accounts["account_id"]
        }
        return scores
