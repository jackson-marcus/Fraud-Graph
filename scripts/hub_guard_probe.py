"""Measure how cheaply the hub guard can be used to hide a ring.

``configs/config.yaml -> graph.max_group_size`` drops any attribute shared by
more than N accounts, on the entity-resolution argument that a value held by
hundreds of accounts is a default or a popular merchant rather than evidence.
That is sound on a static snapshot. Under arrivals it is a lever: an attacker
who controls a ring can register throwaway accounts on the ring's own shared
device until the group crosses N, at which point the guard deletes every link
that device was contributing and the ring stops looking like a ring.

This script plays that attack against every planted ring in the bundled data
and reports how many throwaway registrations each one costs.

Usage:
    uv run python scripts/hub_guard_probe.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraudgraph.settings import get_config, resolve_path
from fraudgraph.streams.consumer import ArrivalWatch
from fraudgraph.streams.schemas import LINK_ATTRIBUTES

ALERT_THRESHOLD = 0.5
JUNK_ATTRIBUTE_BASE = 10_000_000  # ids no real account can collide with


def load() -> tuple[dict, pd.DataFrame]:
    art = resolve_path(get_config()["data"]["artifacts_dir"])
    with open(art / "model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle, pd.read_parquet(art / "features.parquet")


def new_watch(bundle, features, max_group) -> ArrivalWatch:
    return ArrivalWatch(
        accounts=features,
        model=bundle["model"],
        feature_columns=bundle["features"],
        max_group_size=max_group,
    )


def pick_target(watch: ArrivalWatch, members: list[int], max_group: int):
    """The still-linking attribute held by the most members of this ring."""
    best = None
    for attr in LINK_ATTRIBUTES:
        counts: dict[int, int] = {}
        for aid in members:
            value = watch.graph.attrs_of[aid][attr]
            counts[value] = counts.get(value, 0) + 1
        for value, held in counts.items():
            size = len(watch.graph.attr_members[(attr, value)])
            if size > max_group:
                continue  # already evicted, no lever here
            cost = max_group + 1 - size
            key = (-held, cost)
            if best is None or key < best[0]:
                best = (key, attr, value, size, cost)
    return None if best is None else best[1:]


def main() -> None:
    cfg = get_config()
    max_group = cfg["graph"]["max_group_size"]
    bundle, features = load()
    rings = features[features["ring_id"] >= 0].groupby("ring_id")["account_id"].apply(list)

    results = []
    for ring_id, members in rings.items():
        watch = new_watch(bundle, features, max_group)
        before = watch.score(members)
        flagged_before = sum(1 for p in before.values() if p >= ALERT_THRESHOLD)
        if flagged_before == 0:
            continue  # the graph never caught this ring; nothing to hide
        target = pick_target(watch, members, max_group)
        if target is None:
            continue
        attr, value, _size, cost = target
        junk = JUNK_ATTRIBUTE_BASE + int(ring_id) * 100
        for k in range(cost):
            payload = {a: junk + k for a in LINK_ATTRIBUTES}
            payload[attr] = value
            payload |= {"age_days": 30, "txn_count_30d": 20, "avg_txn_amount": 50.0}
            watch.admit(payload)
        after = watch.score(members)
        flagged_after = sum(1 for p in after.values() if p >= ALERT_THRESHOLD)
        results.append(
            {
                "ring_id": int(ring_id),
                "size": len(members),
                "attribute": attr,
                "junk_accounts": cost,
                "flagged_before": flagged_before,
                "flagged_after": flagged_after,
                "mean_p_before": round(sum(before.values()) / len(before), 4),
                "mean_p_after": round(sum(after.values()) / len(after), 4),
            }
        )

    df = pd.DataFrame(results)
    if df.empty:
        print("no ring was flagged by the graph signal; nothing to probe")
        return
    df["silenced"] = df["flagged_after"] < df["flagged_before"]
    fully = df[df["flagged_after"] == 0]
    print(df.to_string(index=False))
    print()
    print(f"max_group_size          : {max_group}")
    print(f"rings the graph flags   : {len(df)}")
    print(f"rings partly silenced   : {int(df['silenced'].sum())}")
    print(f"rings fully silenced    : {len(fully)}")
    print(f"junk accounts needed    : median {int(df['junk_accounts'].median())}, "
          f"min {int(df['junk_accounts'].min())}, max {int(df['junk_accounts'].max())}")
    print(f"alerts lost             : {int(df['flagged_before'].sum() - df['flagged_after'].sum())}"
          f" of {int(df['flagged_before'].sum())}")


if __name__ == "__main__":
    main()
