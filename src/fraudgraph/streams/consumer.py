"""The live account-opening watch: admit a registration, report the fallout.

Ties the three halves of the system together. The journal orders arrivals, the
:class:`~fraudgraph.workers.processor.LiveGraph` folds each one into the
shared-attribute graph, the trained classifier re-scores the accounts that
moved, and the Blackboard Knowledge Sources say *why* in words an analyst can
act on. Scoring one account is the easy half; naming the already-registered
accounts that a stranger's registration just implicated is the half the batch
pipeline cannot do at all.
"""

from __future__ import annotations

import logging

import networkx as nx
import pandas as pd

from fraudgraph.blackboard.core import BlackboardController, FraudBlackboard
from fraudgraph.streams.producer import ArrivalJournal
from fraudgraph.streams.schemas import AccountOpened, ArrivalVerdict, ScoreShift
from fraudgraph.workers.processor import LiveGraph

logger = logging.getLogger(__name__)


class ArrivalWatch:
    """Stateful view of the account graph as registrations come in."""

    def __init__(
        self,
        accounts: pd.DataFrame,
        model,
        feature_columns: list[str],
        max_group_size: int,
        controller: BlackboardController | None = None,
        min_shift: float = 0.01,
    ) -> None:
        self.graph = LiveGraph(accounts, max_group_size=max_group_size)
        self.model = model
        self.feature_columns = list(feature_columns)
        self.controller = controller or BlackboardController()
        self.min_shift = float(min_shift)
        self.journal = ArrivalJournal(next_account_id=int(accounts["account_id"].max()) + 1)

    # -- scoring -----------------------------------------------------------

    def score(self, account_ids) -> dict[int, float]:
        features = self.graph.features_for(account_ids)
        if features.empty:
            return {}
        probs = self.model.predict_proba(features[self.feature_columns])[:, 1]
        return {int(a): float(p) for a, p in zip(features["account_id"], probs, strict=True)}

    def reasons_for(self, account_id: int) -> list[str]:
        """Per-detector explanations from the Blackboard Knowledge Sources."""
        frame = self.graph.neighbourhood_frame(account_id)
        if frame.empty:
            return []
        nodes = frame["account_id"].tolist()
        board = FraudBlackboard(accounts=frame, graph=nx.Graph(self.graph.graph.subgraph(nodes)))
        self.controller.run(board)
        return [v["reason"] for v in board.verdicts.get(int(account_id), {}).values()]

    # -- arrivals ----------------------------------------------------------

    def admit(self, payload: dict, account_id: int | None = None) -> ArrivalVerdict:
        """Fold one registration into the live graph and report what it changed."""
        event = self.journal.record(payload, account_id=account_id)
        scope = self.graph.rescore_scope(event)
        before = self.score(scope - {event.account_id})
        evictions = self.graph.open_account(event)
        after = self.score(scope)

        moved = (
            ScoreShift(account_id=aid, before=prob, after=after[aid])
            for aid, prob in before.items()
            if aid in after
        )
        shifts = [s for s in moved if abs(s.delta) >= self.min_shift]
        component = self.graph.corroborated_component(event.account_id)
        verdict = ArrivalVerdict(
            event=event,
            ring_probability=after.get(event.account_id, 0.0),
            component_size=len(component),
            reasons=self.reasons_for(event.account_id),
            shifts=shifts,
            evictions=evictions,
            n_rescored=len(before),
        )
        logger.info(
            "admitted account %s: p=%.3f, %d/%d neighbours moved, %d hub evictions",
            event.account_id,
            verdict.ring_probability,
            len(shifts),
            verdict.n_rescored,
            len(evictions),
        )
        return verdict

    def history(self, since: int = 0) -> list[AccountOpened]:
        return self.journal.replay(since)
