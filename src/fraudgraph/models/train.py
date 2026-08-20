"""Train the ring classifier: behavioral-only vs behavioral+graph features.

The headline result IS the ablation — how much the graph adds over per-account
behavior. Logged to MLflow.

Usage:
    python -m fraudgraph.models.train
"""

from __future__ import annotations

import logging
import pickle

import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from fraudgraph.graph.build import build_graph, graph_features
from fraudgraph.settings import get_config, get_settings, resolve_path

logger = logging.getLogger(__name__)

BEHAVIOR = ["age_days", "txn_count_30d", "avg_txn_amount"]
GRAPH = ["degree", "component_size", "clustering_coef", "multi_attr_edges"]


def evaluate(
    features: pd.DataFrame, cols: list[str], name: str, cfg
) -> tuple[dict, LGBMClassifier]:
    x_train, x_test, y_train, y_test = train_test_split(
        features[cols],
        features["is_ring"],
        test_size=cfg["model"]["test_frac"],
        random_state=42,
        stratify=features["is_ring"],
    )
    model = LGBMClassifier(**cfg["model"]["lgbm"], random_state=42)
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]
    return (
        {
            f"{name}_roc_auc": float(roc_auc_score(y_test, prob)),
            f"{name}_pr_auc": float(average_precision_score(y_test, prob)),
        },
        model,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = get_config()
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(cfg["eval"]["experiment_name"])

    accounts = pd.read_parquet(resolve_path(cfg["data"]["processed_dir"]) / "accounts.parquet")
    g = build_graph(accounts)
    features = graph_features(accounts, g)

    with mlflow.start_run(run_name="graph-ablation"):
        behavior_metrics, _ = evaluate(features, BEHAVIOR, "behavior_only", cfg)
        full_metrics, model = evaluate(features, BEHAVIOR + GRAPH, "behavior_plus_graph", cfg)
        metrics = {**behavior_metrics, **full_metrics}
        metrics["graph_pr_auc_lift"] = (
            metrics["behavior_plus_graph_pr_auc"] - metrics["behavior_only_pr_auc"]
        )
        mlflow.log_params(
            {"n_accounts": len(accounts), "base_rate": float(features["is_ring"].mean())}
        )
        mlflow.log_metrics(metrics)
        logger.info("%s", {k: round(v, 4) for k, v in metrics.items()})

    artifacts = resolve_path(cfg["data"]["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    with open(artifacts / "model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": BEHAVIOR + GRAPH}, f)
    features.to_parquet(artifacts / "features.parquet", index=False)
    logger.info("Saved model + features -> %s", artifacts)


if __name__ == "__main__":
    main()
