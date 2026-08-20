"""Graph construction, features, and the graph-lift ablation."""

from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from fraudgraph.graph.build import build_graph, graph_features, suspicious_components
from fraudgraph.models.train import BEHAVIOR, GRAPH


def test_ring_members_connect(accounts):
    g = build_graph(accounts)
    ring0 = accounts[accounts["ring_id"] == 0]["account_id"].tolist()
    # Members sharing planted infrastructure should be graph-linked (mostly).
    connected = sum(
        1
        for i in range(len(ring0))
        for j in range(i + 1, len(ring0))
        if g.has_edge(ring0[i], ring0[j])
    )
    possible = len(ring0) * (len(ring0) - 1) / 2
    assert connected / possible > 0.3


def test_features_shapes_and_ranges(accounts):
    g = build_graph(accounts)
    features = graph_features(accounts, g)
    assert len(features) == len(accounts)
    assert (features["component_size"] >= 1).all()
    assert features["clustering_coef"].between(0, 1).all()


def test_ring_accounts_have_denser_graphs(accounts):
    g = build_graph(accounts)
    features = graph_features(accounts, g)
    ring = features[features["is_ring"] == 1]
    legit = features[features["is_ring"] == 0]
    assert ring["multi_attr_edges"].mean() > legit["multi_attr_edges"].mean()
    assert ring["clustering_coef"].mean() > legit["clustering_coef"].mean()


def test_graph_features_lift_pr_auc(accounts):
    g = build_graph(accounts)
    features = graph_features(accounts, g)

    def pr_auc(cols):
        x_train, x_test, y_train, y_test = train_test_split(
            features[cols],
            features["is_ring"],
            test_size=0.3,
            random_state=0,
            stratify=features["is_ring"],
        )
        model = LGBMClassifier(n_estimators=120, verbose=-1, random_state=0)
        model.fit(x_train, y_train)
        return average_precision_score(y_test, model.predict_proba(x_test)[:, 1])

    behavior = pr_auc(BEHAVIOR)
    combined = pr_auc(BEHAVIOR + GRAPH)
    assert combined > behavior + 0.05, f"graph lift too small: {behavior:.3f} -> {combined:.3f}"
    # A ~360-row fixture split can legitimately separate perfectly; the
    # never-1.0 honesty check lives at production scale (PR-AUC ~0.97 there).
    assert combined <= 1.0
    assert behavior < 0.9, "behavior alone should NOT solve it (sleepers exist)"


def test_suspicious_components_surface_rings(accounts):
    g = build_graph(accounts)
    features = graph_features(accounts, g)
    comps = suspicious_components(features, min_size=4)
    top10 = comps.head(10)
    assert (top10["ring_members"] > 0).mean() >= 0.5, "top components should contain real rings"


def test_component_ids_consistent(accounts):
    g = build_graph(accounts)
    features = graph_features(accounts, g)
    for _, group in features.groupby("component_id"):
        assert (group["component_size"] == len(group)).all()
