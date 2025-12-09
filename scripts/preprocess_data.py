"""
Clean crash data and map to OSM intersections.

Outputs:
- data/processed/crashes_clean.parquet
- data/processed/osm_nodes.parquet
- data/processed/osm_edges.parquet
- data/processed/crashes_with_nodes.parquet
"""

from __future__ import annotations

import os
from typing import Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

CRS_LATLON = "EPSG:4326"
CRS_PROJECTED = "EPSG:26971"  # NAD83 / Illinois East (meters)
SNAP_TOLERANCE_M = 70  # max distance to snap crash to intersection (was 40)


def ensure_dirs() -> None:
    os.makedirs("data/processed", exist_ok=True)


def load_and_clean_crashes(path: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["crash_record_id"])

    # Parse datetime
    df["crash_dt"] = pd.to_datetime(df["crash_date"], errors="coerce")

    # Filter plausible coords
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[
        (df["latitude"].between(41.4, 42.3))
        & (df["longitude"].between(-88.5, -87.3))
    ]

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=CRS_LATLON,
    ).to_crs(CRS_PROJECTED)

    return gdf


def load_osm_graph(path: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    import osmnx as ox

    G = ox.load_graphml(path)
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    nodes = nodes.to_crs(CRS_PROJECTED)
    edges = edges.to_crs(CRS_PROJECTED)
    nodes = nodes.reset_index().rename(columns={"osmid": "node_id"})
    edges = edges.reset_index().rename(columns={"u": "from_node", "v": "to_node"})
    # Normalize list-like object columns to scalars/strings for Parquet
    for col in edges.columns:
        if col == "geometry":
            continue
        if edges[col].dtype == object:
            edges[col] = edges[col].apply(
                lambda v: v
                if not isinstance(v, (list, tuple))
                else (v[0] if len(v) == 1 else ";".join(map(str, v)))
            )
            edges[col] = edges[col].astype(str)
        elif edges[col].dtype == bool:
            edges[col] = edges[col].astype(str)
    if "osmid" in edges.columns:
        edges["osmid"] = edges["osmid"].astype(str)
    return nodes, edges


def snap_crashes_to_nodes(
    crashes: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    # Use sjoin_nearest with max distance tolerance
    nearest = gpd.sjoin_nearest(
        crashes,
        nodes[["node_id", "geometry"]],
        how="left",
        distance_col="dist_to_node_m",
        max_distance=SNAP_TOLERANCE_M,
    )
    nearest = nearest.rename(columns={"node_id": "intersection_id"})
    return nearest


def main() -> None:
    ensure_dirs()

    crashes = load_and_clean_crashes("data/raw/chicago_crashes.csv")
    crashes.to_parquet("data/processed/crashes_clean.parquet", index=False)
    print(f"Saved {len(crashes)} clean crashes -> data/processed/crashes_clean.parquet")

    nodes, edges = load_osm_graph("data/raw/osm_chicago.graphml")
    nodes.to_parquet("data/processed/osm_nodes.parquet", index=False)
    edges.to_parquet("data/processed/osm_edges.parquet", index=False)
    print(
        f"Saved {len(nodes)} nodes and {len(edges)} edges "
        "-> data/processed/osm_nodes.parquet / osm_edges.parquet"
    )

    snapped = snap_crashes_to_nodes(crashes, nodes)
    snapped.to_parquet("data/processed/crashes_with_nodes.parquet", index=False)
    matched = snapped["intersection_id"].notna().sum()
    print(
        f"Saved crash-node mapping -> data/processed/crashes_with_nodes.parquet "
        f"(matched {matched}/{len(snapped)})"
    )


if __name__ == "__main__":
    main()

