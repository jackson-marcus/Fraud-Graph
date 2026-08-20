"""API routes: /score/{account_id}, /rings, /component/{id}, /health."""

from __future__ import annotations

import functools
import json
import logging
import pickle

import pandas as pd
from fastapi import APIRouter, HTTPException

from fraudgraph.graph.build import suspicious_components
from fraudgraph.settings import get_config, resolve_path

logger = logging.getLogger(__name__)
router = APIRouter()


@functools.lru_cache(maxsize=1)
def _artifacts():
    art = resolve_path(get_config()["data"]["artifacts_dir"])
    model_path, feat_path = art / "model.pkl", art / "features.parquet"
    if not (model_path.exists() and feat_path.exists()):
        raise FileNotFoundError("Artifacts missing; run make_network.py + models.train")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle, pd.read_parquet(feat_path)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/score/{account_id}")
def score(account_id: int) -> dict:
    try:
        bundle, features = _artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    row = features[features["account_id"] == account_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Unknown account {account_id}")
    prob = float(bundle["model"].predict_proba(row[bundle["features"]])[0, 1])
    r = row.iloc[0]
    return {
        "account_id": account_id,
        "ring_probability": round(prob, 4),
        "graph": {
            "degree": int(r["degree"]),
            "component_size": int(r["component_size"]),
            "component_id": int(r["component_id"]),
            "clustering_coef": float(r["clustering_coef"]),
            "multi_attr_edges": int(r["multi_attr_edges"]),
        },
    }


@router.get("/rings")
def rings() -> list[dict]:
    try:
        _, features = _artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    comps = suspicious_components(features, get_config()["graph"]["min_component_flag_size"])
    return json.loads(comps.head(30).to_json(orient="records"))


@router.get("/component/{component_id}")
def component(component_id: int) -> dict:
    try:
        _, features = _artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    members = features[features["component_id"] == component_id]
    if members.empty:
        raise HTTPException(status_code=404, detail=f"Unknown component {component_id}")
    cols = [
        "account_id",
        "degree",
        "clustering_coef",
        "multi_attr_edges",
        "age_days",
        "txn_count_30d",
        "is_ring",
    ]
    return {
        "component_id": component_id,
        "size": len(members),
        "members": json.loads(members[cols].to_json(orient="records")),
    }
