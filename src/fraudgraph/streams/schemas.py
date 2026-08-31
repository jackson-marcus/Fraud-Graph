"""Records for the live account-opening watch.

The batch pipeline scores a frozen snapshot. Rings do not arrive frozen — they
are assembled one registration at a time, and the question a fraud desk
actually asks is not "is this account suspicious" but "what did this
registration just do to everyone it touches". These are the records in which
that question is asked and answered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Attributes that link accounts together in the shared-infrastructure graph.
LINK_ATTRIBUTES: tuple[str, ...] = ("device_id", "address_id", "payee_id")
# Per-account behavioural fields the classifier needs alongside graph features.
BEHAVIOUR_FIELDS: tuple[str, ...] = ("age_days", "txn_count_30d", "avg_txn_amount")
REQUIRED_FIELDS: tuple[str, ...] = LINK_ATTRIBUTES + BEHAVIOUR_FIELDS


@dataclass(frozen=True)
class AccountOpened:
    """One account registration entering the live graph."""

    seq: int
    account_id: int
    device_id: int
    address_id: int
    payee_id: int
    age_days: int
    txn_count_30d: int
    avg_txn_amount: float

    @classmethod
    def create(cls, seq: int, account_id: int, payload: dict[str, Any]) -> AccountOpened:
        missing = [name for name in REQUIRED_FIELDS if payload.get(name) is None]
        if missing:
            raise ValueError(f"account opening is missing {missing}")
        try:
            values = {
                "device_id": int(payload["device_id"]),
                "address_id": int(payload["address_id"]),
                "payee_id": int(payload["payee_id"]),
                "age_days": int(payload["age_days"]),
                "txn_count_30d": int(payload["txn_count_30d"]),
                "avg_txn_amount": float(payload["avg_txn_amount"]),
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(f"account opening has non-numeric fields: {exc}") from exc
        if values["age_days"] < 0 or values["txn_count_30d"] < 0:
            raise ValueError("age_days and txn_count_30d must be non-negative")
        return cls(seq=seq, account_id=int(account_id), **values)

    def attributes(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in LINK_ATTRIBUTES}

    def behaviour(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in BEHAVIOUR_FIELDS}

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "account_id": self.account_id,
            **self.attributes(),
            **self.behaviour(),
        }


@dataclass(frozen=True)
class ScoreShift:
    """An already-registered account whose ring probability moved on arrival."""

    account_id: int
    before: float
    after: float

    @property
    def delta(self) -> float:
        return round(self.after - self.before, 6)

    def as_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "before": round(self.before, 4),
            "after": round(self.after, 4),
            "delta": round(self.delta, 4),
        }


@dataclass(frozen=True)
class HubEviction:
    """An attribute that just grew past the hub guard and stopped linking.

    ``configs/config.yaml -> graph.max_group_size`` drops attributes shared by
    too many accounts, on the entity-resolution argument that a value held by
    hundreds of accounts is a default or a popular merchant, not evidence. The
    guard is applied per snapshot, so under arrivals it is *non-monotone*: the
    registration that pushes a group over the line deletes every edge that
    group was contributing, and accounts that had nothing to do with the
    arrival get quieter.
    """

    attribute: str
    value: int
    group_size: int
    accounts: list[int]
    edges_dropped: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "value": self.value,
            "group_size": self.group_size,
            "accounts": sorted(self.accounts),
            "edges_dropped": self.edges_dropped,
        }


@dataclass(frozen=True)
class ArrivalVerdict:
    """What one registration did to the account graph."""

    event: AccountOpened
    ring_probability: float
    component_size: int
    reasons: list[str]
    shifts: list[ScoreShift]
    evictions: list[HubEviction]
    n_rescored: int

    def raised(self) -> list[ScoreShift]:
        return sorted([s for s in self.shifts if s.delta > 0], key=lambda s: -s.delta)

    def lowered(self) -> list[ScoreShift]:
        return sorted([s for s in self.shifts if s.delta < 0], key=lambda s: s.delta)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": self.event.as_dict(),
            "ring_probability": round(self.ring_probability, 4),
            "component_size": self.component_size,
            "reasons": list(self.reasons),
            "n_rescored": self.n_rescored,
            "raised": [s.as_dict() for s in self.raised()],
            "lowered": [s.as_dict() for s in self.lowered()],
            "hub_evictions": [e.as_dict() for e in self.evictions],
        }
