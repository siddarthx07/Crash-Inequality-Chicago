"""
Compute (approximate) betweenness centrality and merge into intersection features.

Uses NetworkX approximation with k samples for speed.
Outputs:
- data/processed/intersection_features_enriched.parquet (updated)
"""

from __future__ import annotations

import os
import networkx as nx
import osmnx as ox
import pandas as pd


def compute_betweenness(k: int = 500) -> pd.DataFrame:
    G = ox.load_graphml("data/raw/osm_chicago.graphml")
    # Approximate betweenness for scale
    btw = nx.betweenness_centrality(G, k=k, seed=42, normalized=True, weight=None)
    return pd.DataFrame({"node_id": list(btw.keys()), "centrality_betweenness": list(btw.values())})


def main() -> None:
    if not os.path.exists("data/processed/intersection_features_enriched.parquet"):
        raise SystemExit("Run enrich_demographics.py first.")
    centr = compute_betweenness(k=500)
    feats = pd.read_parquet("data/processed/intersection_features_enriched.parquet")
    feats = feats.merge(centr, left_on="intersection_id", right_on="node_id", how="left").drop(columns=["node_id"])
    feats.to_parquet("data/processed/intersection_features_enriched.parquet", index=False)
    print("Added betweenness centrality to intersection_features_enriched.parquet")


if __name__ == "__main__":
    main()

