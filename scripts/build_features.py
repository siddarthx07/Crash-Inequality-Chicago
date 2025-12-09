"""
Build intersection-level features and hotspot labels.

Assumptions:
- Clean crash data with intersection_id exists at data/processed/crashes_with_nodes.parquet
- OSM nodes/edges at data/processed/osm_nodes.parquet / osm_edges.parquet

Outputs:
- data/processed/intersection_features.parquet
"""

from __future__ import annotations

import os
from datetime import timedelta

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd

CRS_PROJECTED = "EPSG:26971"
HISTORY_DAYS = 365
PREDICT_DAYS = 180
HOTSPOT_TOP_PCT = 0.10

SEVERITY_WEIGHTS = {
    "FATAL": 5,
    "INCAPACITATING INJURY": 4,
    "NONINCAPACITATING INJURY": 3,
    "REPORTED, NOT EVIDENT": 2,
    "NO INDICATION OF INJURY": 1,
}


def ensure_dirs() -> None:
    os.makedirs("data/processed", exist_ok=True)


def load_crashes() -> pd.DataFrame:
    crashes = pd.read_parquet(
        "data/processed/crashes_with_nodes.parquet",
        columns=["crash_record_id", "crash_date", "intersection_id", "most_severe_injury"],
    )
    crashes = crashes.dropna(subset=["intersection_id", "crash_date"])
    crashes["crash_dt"] = pd.to_datetime(crashes["crash_date"], errors="coerce")
    crashes = crashes.dropna(subset=["crash_dt"])
    crashes["severity_wt"] = crashes["most_severe_injury"].map(SEVERITY_WEIGHTS).fillna(1)
    return crashes


def compute_centralities() -> pd.DataFrame:
    G = ox.load_graphml("data/raw/osm_chicago.graphml")
    # Degree and closeness are inexpensive; skip full betweenness for speed.
    deg = nx.degree_centrality(G)
    clo = nx.closeness_centrality(G)
    df = pd.DataFrame(
        {
            "node_id": list(deg.keys()),
            "centrality_degree": list(deg.values()),
            "centrality_closeness": [clo[k] for k in deg.keys()],
        }
    )
    return df


def aggregate_window(crashes: pd.DataFrame, start, end) -> pd.DataFrame:
    mask = (crashes["crash_dt"] >= start) & (crashes["crash_dt"] < end)
    sub = crashes.loc[mask]
    agg = (
        sub.groupby("intersection_id")
        .agg(
            crashes_count=("crash_record_id", "count"),
            severity_sum=("severity_wt", "sum"),
        )
        .reset_index()
    )
    return agg


def main() -> None:
    ensure_dirs()
    crashes = load_crashes()

    max_dt = crashes["crash_dt"].max()
    cutoff = max_dt - timedelta(days=PREDICT_DAYS)
    hist_start = cutoff - timedelta(days=HISTORY_DAYS)
    future_end = cutoff + timedelta(days=PREDICT_DAYS)

    # History aggregates
    hist = aggregate_window(crashes, hist_start, cutoff)
    hist = hist.rename(
        columns={"crashes_count": "hist_crashes", "severity_sum": "hist_severity"}
    )

    # Short-term recent 90 days for trend-ish signal
    recent = aggregate_window(crashes, cutoff - timedelta(days=90), cutoff).rename(
        columns={"crashes_count": "recent90_crashes", "severity_sum": "recent90_severity"}
    )

    # Future window for labels
    future = aggregate_window(crashes, cutoff, future_end).rename(
        columns={"crashes_count": "future_crashes", "severity_sum": "future_severity"}
    )

    # Merge aggregates
    feats = hist.merge(recent, on="intersection_id", how="left")
    feats = feats.merge(future, on="intersection_id", how="left")
    feats = feats.fillna(0)

    # Centralities
    centr = compute_centralities()
    feats = feats.merge(centr, left_on="intersection_id", right_on="node_id", how="left")
    feats = feats.drop(columns=["node_id"])

    # Hotspot label: top 10% by future_crashes (ties included)
    threshold = feats["future_crashes"].quantile(1 - HOTSPOT_TOP_PCT)
    feats["label_hotspot"] = (feats["future_crashes"] >= threshold).astype(int)

    # Persist
    feats.to_parquet("data/processed/intersection_features.parquet", index=False)
    print(
        f"Saved features for {len(feats)} intersections -> "
        "data/processed/intersection_features.parquet"
    )
    print(
        f"Hotspot threshold (top {int(HOTSPOT_TOP_PCT*100)}%): "
        f"{threshold}, positives={feats['label_hotspot'].sum()}"
    )


if __name__ == "__main__":
    main()

