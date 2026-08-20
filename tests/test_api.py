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

    routes._artifacts.cache_clear()
    yield TestClient(create_app())
    cfg["data"]["artifacts_dir"] = original
    routes._artifacts.cache_clear()


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
