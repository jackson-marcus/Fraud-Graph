"""Build the shared-attribute account graph and derive per-account graph features.

Edges connect accounts sharing a device, address, or payee. Features:
degree, component size, clustering coefficient, and shared-attribute edge
multiplicity — the classic fraud-ring tells.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

from fraudgraph.settings import get_config


def build_graph(accounts: pd.DataFrame) -> nx.Graph:
    max_group = get_config()["graph"]["max_group_size"]
    g = nx.Graph()
    g.add_nodes_from(accounts["account_id"].tolist())
    for attr in ("device_id", "address_id", "payee_id"):
        for _, group in accounts.groupby(attr):
            ids = group["account_id"].tolist()
            # Hub guard: an attribute shared by many accounts (popular payee,
            # default device id) is not linking evidence — it wires the whole
            # graph into one giant component and drowns the rings.
            if len(ids) > max_group:
                continue
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if g.has_edge(ids[i], ids[j]):
                        g[ids[i]][ids[j]]["weight"] += 1
                    else:
                        g.add_edge(ids[i], ids[j], weight=1)
    return g


def graph_features(accounts: pd.DataFrame, g: nx.Graph) -> pd.DataFrame:
    # Components on the CORROBORATED subgraph (>=2 shared attributes): single
    # random collisions percolate a giant component across thousands of
    # accounts, but a double collision by chance is vanishingly rare — while
    # rings (and households) doubly-share by construction.
    corroborated = nx.Graph(
        (u, v) for u, v, d in g.edges(data=True) if d.get("weight", 1) >= 2
    )
    corroborated.add_nodes_from(g.nodes)
    component_of = {}
    component_size = {}
    for comp_id, comp in enumerate(nx.connected_components(corroborated)):
        for node in comp:
            component_of[node] = comp_id
            component_size[node] = len(comp)

    clustering = nx.clustering(g)
    rows = []
    for account_id in accounts["account_id"]:
        edges = g.edges(account_id, data=True)
        multi_edges = sum(1 for *_, d in edges if d["weight"] >= 2)
        rows.append(
            {
                "account_id": account_id,
                "degree": g.degree(account_id),
                "component_size": component_size.get(account_id, 1),
                "component_id": component_of.get(account_id, -1),
                "clustering_coef": round(clustering.get(account_id, 0.0), 4),
                "multi_attr_edges": multi_edges,
            }
        )
    return accounts.merge(pd.DataFrame(rows), on="account_id")


def suspicious_components(features: pd.DataFrame, min_size: int) -> pd.DataFrame:
    """Components large enough to look like rings, ranked by density signals."""
    stats = (
        features.groupby("component_id")
        .agg(
            size=("account_id", "count"),
            mean_degree=("degree", "mean"),
            mean_clustering=("clustering_coef", "mean"),
            total_multi=("multi_attr_edges", "sum"),
            ring_members=("is_ring", "sum"),
        )
        .reset_index()
    )
    stats = stats[(stats["component_id"] >= 0) & (stats["size"] >= min_size)]
    # Total multi-attribute mass, not the mean: a 7-account ring with many
    # doubled links outranks a chain of two households that happens to have a
    # perfect mean on 3 edges.
    return stats.sort_values(["total_multi", "mean_clustering"], ascending=False)
