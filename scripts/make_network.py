"""Synthetic account network with planted fraud rings.

Fraud rings share infrastructure (devices, addresses, payout accounts) far
more than legitimate users do — that shared-attribute structure, not any
single account's behavior, is what graph features expose. Legit users also
occasionally share (family devices, apartment addresses) so the signal is
noisy on purpose.

Usage:
    uv run python scripts/make_network.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraudgraph.settings import get_config, resolve_path


def generate(n_accounts: int, n_rings: int, ring_size_range: tuple[int, int], seed: int = 42):
    rng = np.random.default_rng(seed)
    accounts = pd.DataFrame(
        {
            "account_id": np.arange(1, n_accounts + 1),
            "age_days": rng.integers(1, 2000, n_accounts),
            "txn_count_30d": rng.poisson(18, n_accounts),
            "avg_txn_amount": rng.lognormal(3.4, 0.9, n_accounts).round(2),
            "is_ring": 0,
            "ring_id": -1,
        }
    )

    # Legit accounts have near-unique devices/addresses; payees include a few
    # popular merchants (hubs the graph layer must learn to ignore).
    n_devices = n_accounts * 2
    n_addresses = n_accounts
    n_payees = n_accounts // 2
    device = rng.integers(0, n_devices, n_accounts)
    address = rng.integers(0, n_addresses, n_accounts)
    payee = rng.integers(0, n_payees, n_accounts)
    popular = rng.integers(0, n_payees, 5)
    hub_mask = rng.random(n_accounts) < 0.2
    payee[hub_mask] = rng.choice(popular, hub_mask.sum())

    # Legit sharing: some households share a device/address (pairs only).
    for _ in range(n_accounts // 25):
        a, b = rng.integers(0, n_accounts, 2)
        device[b] = device[a]
        address[b] = address[a]

    # Plant rings: each ring shares a small pool of devices/addresses/payees.
    ring_members: list[np.ndarray] = []
    used = set()
    for ring in range(n_rings):
        size = int(rng.integers(ring_size_range[0], ring_size_range[1] + 1))
        members = []
        while len(members) < size:
            c = int(rng.integers(0, n_accounts))
            if c not in used:
                used.add(c)
                members.append(c)
        members = np.array(members)
        ring_members.append(members)
        shared_devices = rng.integers(0, n_devices, max(size // 3, 1))
        shared_addresses = rng.integers(0, n_addresses, max(size // 4, 1))
        shared_payees = rng.integers(0, n_payees, max(size // 3, 1))
        for m in members:
            device[m] = rng.choice(shared_devices)
            address[m] = rng.choice(shared_addresses)
            payee[m] = rng.choice(shared_payees)
        accounts.loc[members, "is_ring"] = 1
        accounts.loc[members, "ring_id"] = ring
        # Ring accounts are younger and burstier — but with overlap vs legit,
        # and ~30% are "sleepers" whose behavior looks entirely normal. Only
        # the graph can catch sleepers.
        accounts.loc[members, "age_days"] = rng.integers(1, 240, size)
        accounts.loc[members, "txn_count_30d"] = rng.poisson(28, size)
        sleepers = members[rng.random(size) < 0.3]
        if len(sleepers):
            accounts.loc[sleepers, "age_days"] = rng.integers(200, 2000, len(sleepers))
            accounts.loc[sleepers, "txn_count_30d"] = rng.poisson(16, len(sleepers))

    accounts["device_id"] = device
    accounts["address_id"] = address
    accounts["payee_id"] = payee
    return accounts


def main() -> None:
    cfg = get_config()["data"]
    df = generate(cfg["n_accounts"], cfg["n_rings"], tuple(cfg["ring_size_range"]), cfg["seed"])
    out = resolve_path(cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "accounts.parquet", index=False)
    print(
        f"Wrote {len(df)} accounts, {df['is_ring'].sum()} in {df.loc[df.ring_id >= 0, 'ring_id'].nunique()} rings -> {out}"
    )


if __name__ == "__main__":
    main()
