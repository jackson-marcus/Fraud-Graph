"""API routes: snapshot views (/rings, /component) and the live arrival watch."""

from __future__ import annotations

import functools
import json
import logging
import pickle

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from fraudgraph.graph.build import suspicious_components
from fraudgraph.settings import get_config, resolve_path
from fraudgraph.streams.consumer import ArrivalWatch

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


@functools.lru_cache(maxsize=1)
def _watch() -> ArrivalWatch:
    """The live graph, seeded from the snapshot the classifier was fitted on.

    Process-local and in-memory: arrivals are held in the journal for the life
    of the worker, not persisted. Rebuilding it is a full graph build, so it is
    built once and mutated in place.
    """
    bundle, features = _artifacts()
    return ArrivalWatch(
        accounts=features,
        model=bundle["model"],
        feature_columns=bundle["features"],
        max_group_size=get_config()["graph"]["max_group_size"],
    )


def reset_state() -> None:
    """Drop cached artifacts and the live graph (used when re-pointing config)."""
    _watch.cache_clear()
    _artifacts.cache_clear()


def _require_watch() -> ArrivalWatch:
    try:
        return _watch()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _require_features() -> pd.DataFrame:
    try:
        return _artifacts()[1]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


class Registration(BaseModel):
    """A new account opening, as a registration desk would submit it."""

    device_id: int
    address_id: int
    payee_id: int
    age_days: int = Field(ge=0)
    txn_count_30d: int = Field(ge=0)
    avg_txn_amount: float = Field(ge=0)
    account_id: int | None = None


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/score/{account_id}")
def score(account_id: int) -> dict:
    watch = _require_watch()
    scores = watch.score([account_id])
    if account_id not in scores:
        raise HTTPException(status_code=404, detail=f"Unknown account {account_id}")
    row = watch.graph.features_for([account_id]).iloc[0]
    snapshot = _require_features()
    match = snapshot[snapshot["account_id"] == account_id]
    return {
        "account_id": account_id,
        "ring_probability": round(scores[account_id], 4),
        "reasons": watch.reasons_for(account_id),
        "graph": {
            "degree": int(row["degree"]),
            "component_size": int(row["component_size"]),
            # Snapshot component id, so /component/{id} drill-down still lines
            # up; null for accounts admitted after the snapshot was built.
            "component_id": None if match.empty else int(match.iloc[0]["component_id"]),
            "clustering_coef": float(row["clustering_coef"]),
            "multi_attr_edges": int(row["multi_attr_edges"]),
        },
        "arrivals_applied": len(watch.journal),
    }


@router.post("/accounts", status_code=201)
def open_account(registration: Registration) -> dict:
    """Register an account against the live graph and report what it changed.

    The answer is not only this account's score: opening it can implicate
    accounts that have been sitting quietly for months, and — when the arrival
    pushes a shared attribute past the hub guard — it can just as easily make
    an existing cluster go quiet. Both directions come back.
    """
    watch = _require_watch()
    payload = registration.model_dump()
    account_id = payload.pop("account_id")
    try:
        verdict = watch.admit(payload, account_id=account_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return verdict.as_dict()


@router.get("/arrivals")
def arrivals(since: int = 0) -> dict:
    """Registrations folded into the live graph since the snapshot was built."""
    watch = _require_watch()
    return {
        "count": len(watch.journal),
        "events": [event.as_dict() for event in watch.history(since)],
    }


@router.get("/rings")
def rings() -> list[dict]:
    features = _require_features()
    comps = suspicious_components(features, get_config()["graph"]["min_component_flag_size"])
    return json.loads(comps.head(30).to_json(orient="records"))


@router.get("/component/{component_id}")
def component(component_id: int) -> dict:
    features = _require_features()
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
