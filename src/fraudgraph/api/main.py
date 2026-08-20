"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from fraudgraph import __version__
from fraudgraph.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    app = FastAPI(
        title="fraudgraph",
        description="Fraud-ring detection on transaction graphs: shared-attribute link building, community detection, graph features feeding a LightGBM ring classifier.",
        version=__version__,
    )
    app.include_router(router)
    return app


app = create_app()
