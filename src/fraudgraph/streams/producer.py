"""Append-only journal of the account openings folded into the live graph.

The classifier was fitted on a snapshot. Every arrival admitted afterwards
moves the live graph away from that snapshot, so there has to be an ordered,
replayable record of exactly how far it has moved — otherwise nobody can say
whether a live score came from the trained distribution or from an hour of
undocumented drift. Replaying the journal over a fresh
:class:`~fraudgraph.workers.processor.LiveGraph` reproduces the current state
exactly, which is also how the arrivals get folded back into the next training
snapshot.
"""

from __future__ import annotations

from fraudgraph.streams.schemas import AccountOpened


class ArrivalJournal:
    """Ordered log of admitted registrations, with account-id allocation."""

    def __init__(self, next_account_id: int = 1) -> None:
        self._next_account_id = int(next_account_id)
        self._entries: list[AccountOpened] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def next_account_id(self) -> int:
        return self._next_account_id

    def record(self, payload: dict, account_id: int | None = None) -> AccountOpened:
        """Validate a registration and append it. Raises ValueError if malformed.

        Nothing is appended when validation fails, so a rejected payload does
        not consume a sequence number or an account id.
        """
        assigned = self._next_account_id if account_id is None else int(account_id)
        event = AccountOpened.create(seq=len(self._entries) + 1, account_id=assigned, payload=payload)
        self._entries.append(event)
        self._next_account_id = max(self._next_account_id, assigned + 1)
        return event

    def replay(self, since: int = 0) -> list[AccountOpened]:
        """Entries with ``seq > since``, in arrival order."""
        return [e for e in self._entries if e.seq > since]
