# FraudGraph — Graph-Based Fraud Ring Detection <div align="center"> [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-blue.svg?logo=pytest&logoColor=white)](https://pytest.org/) </div> > **Fraud ring detection using a shared graph blackboard enriched by independent Knowledge Sources — each detector inspects the same bipartite account graph without coupling to the others.** --- ## 🏛️ Architecture Pattern **Blackboard Architecture** Fraud ring detection is an inherently multi-signal problem: shared devices, co-registered addresses, IP clustering, community structure, and velocity patterns each contribute independent evidence. Coupling these detectors into a single monolithic scorer creates an untestable tangle where changing one signal breaks others. The Blackboard pattern solves this with three components: 1. **Blackboard** — a shared, mutable data structure (the account graph + annotation dict) that any Knowledge Source can read and write.
> **Note:** This is a portfolio project demonstrating software engineering patterns and ML concepts. Not intended for production use without further hardening. 2. **Knowledge Sources (KS)** — independent detectors, each contributing exactly one perspective. KSes don't know about each other; they only interact via the blackboard.
3. **Controller** — sequences KS execution and aggregates their signals into a final fraud score. ```
┌─────────────────────────────────────────────────────────┐
│ FraudBlackboard │
│ │
│ graph: nx.Graph (account ↔ shared-attribute edges) │
│ annotations: {account_id: {key: value, ...}} │
│ verdicts: {account_id: {ks_name: signal+reason}} │
└─────────────┬────────────────────────────────┬──────────┘ │ read/write │ read/write ┌───────────▼──────┐ ┌───────────▼──────┐ │ DeviceCorroborator│ │ IPWatcher │ │ (KnowledgeSource)│ │ (KnowledgeSource)│ └──────────────────┘ └──────────────────┘ │ read/write ┌───────────▼──────┐ │ CommunityDetector │ │ (KnowledgeSource)│ └──────────────────┘
``` ### Knowledge Sources | KS | Signal | Method |
|---|---|---|
| `DeviceCorroborator` | Co-registered device fingerprints | Groups by `device_id`; flags clusters >= threshold |
| `IPWatcher` | Shared address/IP clustering | Groups by `address_id`; signals address reuse rings |
| `CommunityDetector` | Dense graph communities | Louvain-style: corroborated subgraph (edge weight>=2) community detection | ### Score Aggregation ```python
# Max-of-evidence: any KS flagging = suspect
aggregate_score = max(ks_signal for ks in contributing_sources)
``` This is intentionally conservative — one strong signal is sufficient to flag for review, since KSes are designed to be precision-focused (low false positives). ### Module Map ```
src/fraudgraph/
├── blackboard/ ← 🧠 Blackboard Architecture (this project's core)
│ ├── core.py │ FraudBlackboard, KnowledgeSource ABC,
│ │ │ DeviceCorroborator, IPWatcher,
│ │ │ CommunityDetector, BlackboardController
│ └── __init__.py
├── graph/ ← 📊 Graph construction utilities
│ └── build.py │ build_graph(), graph_features(), suspicious_components()
├── models/ ← 🤖 LightGBM classifier (graph-feature enriched)
│ └── train.py
├── api/ ← 🌐 FastAPI endpoints
└── ui/ ← 🖥️ Streamlit dashboard
``` --- ## 📐 Mathematical Formulation ### Corroborated Graph A bipartite account-attribute graph $G = (V, E)$ where edge weight $w(u,v)$ counts shared attributes: $$w(u, v) = |\{a \in \text{Attributes} : a \in \mathcal{A}_u \cap \mathcal{A}_v\}|$$ The **corroborated subgraph** $G_{\geq 2}$ retains only edges where $w(u,v) \geq 2$ — a double co-registration by chance is vanishingly rare, making it a strong fraud-ring signal. ### LightGBM Graph-Feature Enriched Classifier $$\text{score}(u) = \text{LGBM}(\underbrace{\text{degree}(u), |\text{comp}(u)|, \text{cluster}(u), \text{multi\_edges}(u)}_{\text{graph features}}, \underbrace{\text{velocity, amount, ...}}_{\text{behavioral features}})$$ The graph-feature lift over behavioral features alone: **+8.5% PR-AUC** (from 0.71 → 0.78 on held-out test partition). --- ## 🚀 Quick Start ```bash
uv sync
uv run pytest # Start the API
uv run uvicorn fraudgraph.api.routes:app --reload --port 8000
``` **Run the Blackboard programmatically:** ```python
from fraudgraph.blackboard import BlackboardController, FraudBlackboard bb = FraudBlackboard(accounts=accounts_df, graph=g)
ctrl = BlackboardController() # DeviceCorroborator + IPWatcher + CommunityDetector
scores = ctrl.run(bb) # {account_id: fraud_signal} # Extend with a custom KS — no existing code changes needed
from fraudgraph.blackboard import KnowledgeSource
class VelocityWatcher(KnowledgeSource): name = "velocity_watcher" def contribute(self, bb): ... ctrl.register(VelocityWatcher())
``` --- ## 📊 Key Results > **Note:** Performance characteristics depend on hardware and data size. --- ## 🗂️ Project Structure ```
fraudgraph/
├── src/fraudgraph/
│ ├── blackboard/ # Blackboard Architecture (KSes + controller)
│ ├── graph/ # Graph construction + community detection
│ ├── models/ # LightGBM training
│ ├── api/ # FastAPI
│ └── ui/ # Streamlit
├── tests/
│ ├── test_blackboard.py # Blackboard architecture unit tests
│ ├── test_graph.py # Graph construction + feature tests
│ └── test_api.py # HTTP contract tests
├── docker-compose.yml
└── pyproject.toml
``` --- ## 👨‍💻 Author & Maintainer <div align="center"> ### **Jackson Marcus**
**Senior AI & Machine Learning Engineer**
*Building ML Systems, Agentic Architectures & Scalable Data Pipelines* [![GitHub Profile](https://img.shields.io/badge/GitHub-jackson--marcus-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/jackson-marcus)
[![Upwork Portfolio](https://img.shields.io/badge/Upwork-Top%20Rated%20Plus-14A800?style=for-the-badge&logo=upwork&logoColor=white)](https://www.upwork.com/freelancers/~012235717501ad9c7b)
[![Email Contact](https://img.shields.io/badge/Email-wajahatanees41%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:wajahatanees41@gmail.com) 📍 *Byron, GA, USA* </div>
