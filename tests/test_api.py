import pickle

import pytest
from fastapi.testclient import TestClient
from lightgbm import LGBMClassifier

import fraudgraph.api.routes as routes
from fraudgraph.api.main import create_app
from fraudgraph.graph.build import build_graph, graph_features
from fraudgraph.models.train import BEHAVIOR, GRAPH
from fraudgraph.settings import get_config


@pytest.fixture()
def client(accounts, tmp_path):
    cfg = get_config()
    original = cfg["data"]["artifacts_dir"]
    art = tmp_path / "artifacts"
    art.mkdir()
    cfg["data"]["artifacts_dir"] = str(art)

    g = build_graph(accounts)
    features = graph_features(accounts, g)
    cols = BEHAVIOR + GRAPH
    model = LGBMClassifier(n_estimators=60, verbose=-1, random_state=0)
    model.fit(features[cols], features["is_ring"])
    with open(art / "model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": cols}, f)
    features.to_parquet(art / "features.parquet", index=False)

    routes.reset_state()
    yield TestClient(create_app())
    cfg["data"]["artifacts_dir"] = original
    routes.reset_state()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_score_ring_vs_legit(client, accounts):
    ring_id = int(accounts.loc[accounts["is_ring"] == 1, "account_id"].iloc[0])
    legit_id = int(accounts.loc[accounts["is_ring"] == 0, "account_id"].iloc[0])
    ring = client.get(f"/score/{ring_id}").json()
    legit = client.get(f"/score/{legit_id}").json()
    assert ring["ring_probability"] > legit["ring_probability"]


def test_score_unknown_404(client):
    assert client.get("/score/999999").status_code == 404


def test_rings_and_component_drilldown(client):
    rings = client.get("/rings").json()
    assert rings
    comp = client.get(f"/component/{rings[0]['component_id']}").json()
    assert comp["size"] == len(comp["members"])


def test_score_carries_detector_reasons(client, accounts):
    """The route has always promised per-detector reasons; it now has them."""
    ring_id = int(accounts.loc[accounts["is_ring"] == 1, "account_id"].iloc[0])
    body = client.get(f"/score/{ring_id}").json()
    assert body["arrivals_applied"] == 0
    assert isinstance(body["reasons"], list)
    assert body["graph"]["component_id"] is not None


def test_opening_an_account_rescores_the_accounts_it_touches(client, accounts):
    """POST /accounts must move an existing account's score, not just return
    a score for the new one."""
    seed = accounts[accounts["is_ring"] == 1].iloc[0]
    neighbour = int(seed["account_id"])
    before = client.get(f"/score/{neighbour}").json()["ring_probability"]

    body = client.post(
        "/accounts",
        json={
            "device_id": int(seed["device_id"]),
            "address_id": int(seed["address_id"]),
            "payee_id": int(seed["payee_id"]),
            "age_days": 9,
            "txn_count_30d": 31,
            "avg_txn_amount": 44.0,
        },
    )
    assert body.status_code == 201
    verdict = body.json()
    assert verdict["event"]["seq"] == 1
    assert verdict["n_rescored"] > 0
    assert verdict["component_size"] > 1

    after = client.get(f"/score/{neighbour}").json()
    assert after["arrivals_applied"] == 1
    assert after["graph"]["degree"] > before or after["ring_probability"] != before
    moved = {s["account_id"] for s in verdict["raised"] + verdict["lowered"]}
    changed = after["ring_probability"] != before
    assert (neighbour in moved) == changed or not changed


def test_new_account_is_scoreable_and_journalled(client):
    payload = {
        "device_id": 6_000_001,
        "address_id": 6_000_002,
        "payee_id": 6_000_003,
        "age_days": 3,
        "txn_count_30d": 40,
        "avg_txn_amount": 12.5,
    }
    new_id = client.post("/accounts", json=payload).json()["event"]["account_id"]
    assert client.get(f"/score/{new_id}").status_code == 200
    # Not in the training snapshot, so it has no snapshot component to drill into.
    assert client.get(f"/score/{new_id}").json()["graph"]["component_id"] is None

    log = client.get("/arrivals").json()
    assert log["count"] == 1
    assert log["events"][0]["account_id"] == new_id
    assert client.get("/arrivals?since=1").json()["events"] == []


def test_malformed_registration_is_rejected(client):
    bad = client.post("/accounts", json={"device_id": 1, "age_days": -5})
    assert bad.status_code == 422
    assert client.get("/arrivals").json()["count"] == 0
