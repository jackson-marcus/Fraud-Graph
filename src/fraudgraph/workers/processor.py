"""Incrementally maintained account graph.

``graph.build`` rebuilds the whole shared-attribute graph from a DataFrame.
That is the right shape for training and hopeless for a registration desk: an
account opens, and you want to know within one request what it changed. This
keeps the same graph — same hub guard, same corroborated subgraph, same four
features — under single-account insertion, and reports which already-registered
accounts had their features disturbed so the caller can re-score exactly those.

Equivalence with the batch builder is not decoration: it is asserted directly in
``tests/test_live_arrivals.py`` after every arrival, because an index that
silently drifts from the trained feature distribution is worse than no index.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import networkx as nx
import pandas as pd

from fraudgraph.streams.schemas import (
    BEHAVIOUR_FIELDS,
    LINK_ATTRIBUTES,
    AccountOpened,
    HubEviction,
)

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ("degree", "component_size", "clustering_coef", "multi_attr_edges")


class LiveGraph:
    """Account↔account link graph kept current one registration at a time."""

    def __init__(self, accounts: pd.DataFrame, max_group_size: int) -> None:
        self.max_group_size = int(max_group_size)
        self.graph = nx.Graph()
        self.attr_members: dict[tuple[str, int], set[int]] = {}
        self.behaviour: dict[int, dict[str, float]] = {}
        self.attrs_of: dict[int, dict[str, int]] = {}
        for row in accounts.itertuples(index=False):
            aid = int(row.account_id)
            self.graph.add_node(aid)
            self.behaviour[aid] = {f: getattr(row, f) for f in BEHAVIOUR_FIELDS}
            self.attrs_of[aid] = {a: int(getattr(row, a)) for a in LINK_ATTRIBUTES}
            for attr, value in self.attrs_of[aid].items():
                self.attr_members.setdefault((attr, value), set()).add(aid)
        for (attr, value), members in self.attr_members.items():
            if len(members) <= self.max_group_size:
                self._relink(members, +1)
            else:
                logger.debug("hub guard excludes %s=%s (%d accounts)", attr, value, len(members))

    # -- graph maintenance -------------------------------------------------

    def _relink(self, members: Iterable[int], delta: int) -> int:
        """Add or remove one unit of shared-attribute evidence between members.

        Returns the number of account pairs whose link weight moved.
        """
        ids = sorted(members)
        moved = 0
        for i, u in enumerate(ids):
            for v in ids[i + 1 :]:
                moved += 1
                if self.graph.has_edge(u, v):
                    weight = self.graph[u][v]["weight"] + delta
                    if weight <= 0:
                        self.graph.remove_edge(u, v)
                    else:
                        self.graph[u][v]["weight"] = weight
                elif delta > 0:
                    self.graph.add_edge(u, v, weight=delta)
        return moved

    def corroborated_component(self, account_id: int) -> set[int]:
        """Accounts reachable using only links backed by >= 2 shared attributes."""
        if account_id not in self.graph:
            return set()
        seen, stack = {account_id}, [account_id]
        while stack:
            node = stack.pop()
            for nbr, data in self.graph[node].items():
                if data.get("weight", 1) >= 2 and nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        return seen

    def rescore_scope(self, event: AccountOpened) -> set[int]:
        """Every account whose features this arrival could possibly move.

        Structural change is confined to edges incident to the arriving account
        and to edges *within* the attribute groups it joins, so let ``C`` be the
        members of those groups. ``degree`` and ``multi_attr_edges`` can only
        move inside ``C``; ``clustering_coef`` can additionally move for a
        neighbour of ``C``; and a corroborated component can only gain or lose
        members it was already connected to, so it is covered by the pre-arrival
        components of ``C``. The union of those three is complete, which is what
        lets the caller re-score a handful of accounts instead of the book.
        """
        candidates: set[int] = set()
        for attr, value in event.attributes().items():
            candidates |= self.attr_members.get((attr, value), set())
        scope = set(candidates) | {event.account_id}
        for node in candidates:
            scope |= set(self.graph.neighbors(node))
            scope |= self.corroborated_component(node)
        return scope

    def open_account(self, event: AccountOpened) -> list[HubEviction]:
        """Insert one registration; return any hub guards it tripped."""
        aid = event.account_id
        if aid in self.graph:
            raise ValueError(f"account {aid} is already registered")

        self.graph.add_node(aid)
        self.behaviour[aid] = event.behaviour()
        self.attrs_of[aid] = event.attributes()
        evictions: list[HubEviction] = []

        for attr, value in event.attributes().items():
            members = self.attr_members.setdefault((attr, value), set())
            was = len(members)
            members.add(aid)
            if len(members) <= self.max_group_size:
                for other in members - {aid}:
                    if self.graph.has_edge(aid, other):
                        self.graph[aid][other]["weight"] += 1
                    else:
                        self.graph.add_edge(aid, other, weight=1)
            elif was == self.max_group_size:
                # The guard just tripped: this group stops counting as evidence,
                # and every link it was contributing has to come back out.
                dropped = self._relink(members - {aid}, -1)
                evictions.append(
                    HubEviction(
                        attribute=attr,
                        value=value,
                        group_size=len(members),
                        accounts=sorted(members),
                        edges_dropped=dropped,
                    )
                )
                logger.info("hub guard tripped on %s=%s at %d accounts", attr, value, len(members))
        return evictions

    # -- feature extraction ------------------------------------------------

    def features_for(self, account_ids: Iterable[int]) -> pd.DataFrame:
        """Graph + behavioural features for the given accounts.

        Column-for-column identical to ``graph.build.graph_features`` restricted
        to the same rows, except ``component_id``, which is the lowest account
        id in the component rather than a global enumeration order — the live
        graph has no snapshot to enumerate against.
        """
        rows = []
        for aid in account_ids:
            aid = int(aid)
            if aid not in self.graph:
                continue
            edges = self.graph.edges(aid, data=True)
            component = self.corroborated_component(aid)
            rows.append(
                {
                    "account_id": aid,
                    **self.behaviour[aid],
                    "degree": self.graph.degree(aid),
                    "component_size": len(component),
                    "component_id": min(component),
                    "clustering_coef": round(nx.clustering(self.graph, aid), 4),
                    "multi_attr_edges": sum(1 for *_, d in edges if d["weight"] >= 2),
                }
            )
        return pd.DataFrame(rows)

    def neighbourhood_frame(self, account_id: int) -> pd.DataFrame:
        """Accounts a detector needs in order to judge ``account_id`` honestly.

        The account's *whole* attribute groups (so a "shared by N accounts"
        count is the real N, not the count inside some window) plus its
        corroborated component (so community detection sees the ring).
        """
        scope = self.corroborated_component(account_id) | {account_id}
        for attr, value in self.attrs_of.get(account_id, {}).items():
            scope |= self.attr_members.get((attr, value), set())
        return pd.DataFrame(
            [{"account_id": aid, **self.attrs_of[aid]} for aid in sorted(scope)]
        )
