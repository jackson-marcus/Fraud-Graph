"""Streamlit demo: suspicious-component queue + account drill-down."""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("FRAUDGRAPH_API_URL", "http://localhost:8140")

st.set_page_config(page_title="fraudgraph", page_icon="🕸️", layout="wide")
st.title("🕸️ fraudgraph")
st.caption("Fraud rings via shared infrastructure: graph features + LightGBM")


def _ok() -> bool:
    try:
        return httpx.get(f"{API_URL}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


if not _ok():
    st.error(f"API not reachable at {API_URL}. Start it with `make api`.")
    st.stop()

tab_rings, tab_account = st.tabs(["Suspicious components", "Score an account"])

with tab_rings:
    r = httpx.get(f"{API_URL}/rings", timeout=60)
    if r.status_code != 200:
        st.warning(r.json().get("detail", r.text))
    else:
        df = pd.DataFrame(r.json())
        st.markdown(
            "Connected components ranked by shared-infrastructure density "
            "(`ring_members` is ground truth, shown for validation)"
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
        if not df.empty:
            comp = st.selectbox("Inspect component", df["component_id"].tolist())
            rc = httpx.get(f"{API_URL}/component/{comp}", timeout=30)
            if rc.status_code == 200:
                st.dataframe(
                    pd.DataFrame(rc.json()["members"]), use_container_width=True, hide_index=True
                )

with tab_account:
    account_id = st.number_input("Account ID", 1, 10000, 17)
    if st.button("Score", type="primary"):
        r = httpx.get(f"{API_URL}/score/{int(account_id)}", timeout=30)
        if r.status_code != 200:
            st.error(r.json().get("detail", r.text))
        else:
            body = r.json()
            c1, c2, c3 = st.columns(3)
            c1.metric("Ring probability", f"{body['ring_probability']:.1%}")
            c2.metric("Component size", body["graph"]["component_size"])
            c3.metric("Multi-attribute edges", body["graph"]["multi_attr_edges"])
            st.json(body["graph"])
