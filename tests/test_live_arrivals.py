"""The live account-opening watch.

Two things have to hold for this to be worth running at all: the incrementally
maintained graph must stay bit-identical to a from-scratch rebuild (otherwise
live scores drift away from the distribution the classifier was fitted on), and
the re-score scope must contain every account the arrival actually moved
(otherwise the watch quietly under-reports). Both are asserted against a full
recompute rather than against themselves.
"""

from __future__ import annotations

import pytest
from lightgbm import LGBMClassifier

from fraudgraph.graph.build import build_graph, graph_features
from fraudgraph.models.train import BEHAVIOR, GRAPH
from fraudgraph.streams.consumer import ArrivalWatch
from fraudgraph.streams.producer import ArrivalJournal
from fraudgraph.streams.schemas import LINK_ATTRIBUTES, AccountOpened
from fraudgraph.workers.processor import FEATURE_COLUMNS, LiveGraph

MAX_GROUP = 8
COLS = list(FEATURE_COLUMNS)


def _registration(**overrides) -> dict:
    payload = {
        "device_id": 900_001,
        "address_id": 900_002,
        "payee_id": 900_003,
        "age_days": 12,
        "txn_count_30d": 25,
        "avg_txn_amount": 61.5,
    }
    payload.update(overrides)
    return payload


@pytest.fixture(scope="module")
def snapshot(accounts):
    """A trained model plus the batch features it was fitted on."""
    features = graph_features(accounts, build_graph(accounts))
    cols = BEHAVIOR + GRAPH
    model = LGBMClassifier(n_estimators=60, verbose=-1, random_state=0)
    model.fit(features[cols], features["is_ring"])
    return model, features, cols


@pytest.fixture()
def watch(snapshot):
    model, features, cols = snapshot
    return ArrivalWatch(
        accounts=features, model=model, feature_columns=cols, max_group_size=MAX_GROUP
    )


def _rebuild(frame):
    return graph_features(frame, build_graph(frame)).set_index("account_id")[COLS]


def test_seeded_live_graph_matches_the_batch_builder(accounts, watch):
    live = watch.graph.features_for(accounts["account_id"]).set_index("account_id")[COLS]
    batch = _rebuild(accounts)
    assert (batch - live.loc[batch.index]).abs().max().max() == pytest.approx(0.0)


def test_arrivals_keep_the_live_graph_identical_to_a_rebuild(accounts, watch):
    """Twenty arrivals; after each one every account still matches a rebuild."""
    import pandas as pd

    frame = accounts.copy()
    next_id = int(accounts["account_id"].max()) + 1
    for i in range(20):
        payload = _registration(
            device_id=int(accounts["device_id"].iloc[i * 7]),
            address_id=int(accounts["address_id"].iloc[i * 11]),
            payee_id=int(accounts["payee_id"].iloc[i * 3]),
        )
        watch.admit(payload)
        frame = pd.concat(
            [frame, pd.DataFrame([{"account_id": next_id + i, "is_ring": 0, "ring_id": -1, **payload}])],
            ignore_index=True,
        )
        live = watch.graph.features_for(frame["account_id"]).set_index("account_id")[COLS]
        batch = _rebuild(frame)
        drift = (batch - live.loc[batch.index]).abs().max().max()
        assert drift == pytest.approx(0.0), f"live graph drifted from rebuild at arrival {i}"


def test_rescore_scope_covers_every_account_that_moved(accounts, watch):
    """Nothing outside the reported scope may change — that is the whole claim."""
    everyone = accounts["account_id"].tolist()
    for i in range(10):
        payload = _registration(
            device_id=int(accounts["device_id"].iloc[i * 13]),
            address_id=int(accounts["address_id"].iloc[i * 5]),
        )
        event = AccountOpened.create(seq=i + 1, account_id=10_000 + i, payload=payload)
        before = watch.graph.features_for(everyone).set_index("account_id")[COLS]
        scope = watch.graph.rescore_scope(event)
        watch.graph.open_account(event)
        after = watch.graph.features_for(everyone).set_index("account_id")[COLS]
        moved = set(before.index[(before != after).any(axis=1)])
        assert not moved - scope, f"accounts moved outside the re-score scope: {moved - scope}"


def test_scope_is_a_small_fraction_of_the_population(accounts, watch):
    """A scope of 'everyone' would be correct and useless."""
    event = AccountOpened.create(
        seq=1,
        account_id=10_000,
        payload=_registration(device_id=int(accounts["device_id"].iloc[0])),
    )
    assert len(watch.graph.rescore_scope(event)) < 0.05 * len(accounts)


# ---------------------------------------------------------------------------
# The hub guard is non-monotone under arrivals
# ---------------------------------------------------------------------------


def _cluster(n: int):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "account_id": i,
                "device_id": 1,
                "address_id": 1,
                "payee_id": 100 + i,
                "age_days": 30,
                "txn_count_30d": 20,
                "avg_txn_amount": 50.0,
            }
            for i in range(1, n + 1)
        ]
    )


def test_one_arrival_dissolves_a_corroborated_component():
    """Eight accounts sharing a device *and* an address are one corroborated
    component. A ninth account sharing only the device pushes that device past
    the guard, the device edge is deleted from all 28 pairs, every remaining
    link drops to a single shared attribute, and the component of 8 becomes
    eight components of 1 — without any of the eight doing anything."""
    live = LiveGraph(_cluster(8), max_group_size=MAX_GROUP)
    before = live.features_for(range(1, 9)).set_index("account_id")
    assert (before["component_size"] == 8).all()
    assert (before["multi_attr_edges"] == 7).all()

    event = AccountOpened.create(
        seq=1, account_id=9, payload=_registration(device_id=1, address_id=777, payee_id=777)
    )
    evictions = live.open_account(event)

    assert [e.attribute for e in evictions] == ["device_id"]
    assert evictions[0].group_size == 9
    assert evictions[0].edges_dropped == 28  # every pair among the original eight

    after = live.features_for(range(1, 9)).set_index("account_id")
    assert (after["component_size"] == 1).all()
    assert (after["multi_attr_edges"] == 0).all()
    assert (after["degree"] == 7).all()  # the address still links them, weakly
    assert live.graph.degree(9) == 0  # and the arrival links to nobody at all


def test_flooding_a_ring_device_lowers_already_registered_scores(accounts, snapshot):
    """The scored version of the same effect: somewhere in this population an
    arrival makes an existing account *less* suspicious. If that ever stops
    being true the guard has become monotone and the probe is meaningless."""
    model, features, cols = snapshot
    rings = features[features["ring_id"] >= 0].groupby("ring_id")["account_id"].apply(list)
    lowered_anywhere = []
    evicted_anywhere = 0

    for members in rings.head(8):
        watch = ArrivalWatch(
            accounts=features, model=model, feature_columns=cols, max_group_size=MAX_GROUP
        )
        device = features.set_index("account_id").loc[members, "device_id"].mode().iloc[0]
        size = len(watch.graph.attr_members[("device_id", int(device))])
        if size > MAX_GROUP:
            continue
        for k in range(MAX_GROUP + 1 - size):
            verdict = watch.admit(_registration(device_id=int(device), address_id=800_000 + k))
            evicted_anywhere += len(verdict.evictions)
            lowered_anywhere.extend(verdict.lowered())

    assert evicted_anywhere > 0, "no flood ever tripped the guard"
    assert lowered_anywhere, "no existing account was ever made less suspicious by an arrival"
    assert min(s.delta for s in lowered_anywhere) < -0.1


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


def test_journal_rejects_malformed_payloads_without_consuming_a_slot():
    journal = ArrivalJournal(next_account_id=500)
    for bad in (
        _registration() | {"device_id": None},
        _registration() | {"avg_txn_amount": "not a number"},
        _registration() | {"age_days": -1},
        {k: v for k, v in _registration().items() if k != "payee_id"},
    ):
        with pytest.raises(ValueError):
            journal.record(bad)
    assert len(journal) == 0
    assert journal.next_account_id == 500  # nothing burned an id
    first = journal.record(_registration())
    assert (first.seq, first.account_id) == (1, 500)


def test_journal_replay_reproduces_the_live_graph(accounts, watch):
    """Replaying the journal onto a fresh graph must land in the same state —
    otherwise the audit trail does not actually explain the live scores."""
    for i in range(15):
        watch.admit(
            _registration(
                device_id=int(accounts["device_id"].iloc[i * 9]),
                address_id=int(accounts["address_id"].iloc[i * 4]),
            )
        )
    replayed = LiveGraph(accounts, max_group_size=MAX_GROUP)
    for event in watch.history():
        replayed.open_account(event)

    ids = sorted(watch.graph.graph.nodes)
    assert ids == sorted(replayed.graph.nodes)
    original = watch.graph.features_for(ids).set_index("account_id")[COLS]
    again = replayed.features_for(ids).set_index("account_id")[COLS]
    assert (original - again).abs().max().max() == pytest.approx(0.0)


def test_replay_since_a_sequence_number_is_the_tail(watch):
    for _ in range(4):
        watch.admit(_registration())
    assert [e.seq for e in watch.history(since=2)] == [3, 4]
    assert [e.seq for e in watch.history()] == [1, 2, 3, 4]


def test_reopening_an_account_is_refused(watch, accounts):
    existing = int(accounts["account_id"].iloc[0])
    with pytest.raises(ValueError, match="already registered"):
        watch.admit(_registration(), account_id=existing)


# ---------------------------------------------------------------------------
# Detector reasons
# ---------------------------------------------------------------------------


def test_arrival_into_a_ring_is_explained_by_the_knowledge_sources(snapshot):
    """An arrival that lands inside a planted ring must come back with a
    detector reason naming the attribute it shares, not an empty list."""
    model, features, cols = snapshot
    watch = ArrivalWatch(
        accounts=features, model=model, feature_columns=cols, max_group_size=MAX_GROUP
    )
    ring = features[features["ring_id"] == features["ring_id"].max()]
    seed = ring.iloc[0]
    verdict = watch.admit(
        _registration(
            device_id=int(seed["device_id"]),
            address_id=int(seed["address_id"]),
            payee_id=int(seed["payee_id"]),
        )
    )
    assert verdict.reasons, "a KS should have something to say about a ring join"
    assert any(str(int(seed[a])) in r for a in LINK_ATTRIBUTES for r in verdict.reasons)
    assert verdict.component_size > 1


def test_isolated_arrival_implicates_nobody(watch):
    verdict = watch.admit(_registration(device_id=7_000_001, address_id=7_000_002, payee_id=7_000_003))
    assert verdict.shifts == []
    assert verdict.evictions == []
    assert verdict.component_size == 1
    assert verdict.reasons == []
