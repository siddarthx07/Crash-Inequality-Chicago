"""
Join Chicago Community Areas to intersections and enrich features.

Outputs:
- data/processed/intersection_features_enriched.parquet (updated with community fields)
"""

from __future__ import annotations

import os
import geopandas as gpd
import pandas as pd

CRS_PROJECTED = "EPSG:26971"
COMMUNITY_URL = "https://data.cityofchicago.org/api/geospatial/cauq-8yn6?method=export&format=Shapefile"
COMMUNITY_URL_FALLBACK = "https://data.cityofchicago.org/api/geospatial/igwz-8jzy?method=export&format=Shapefile"
COMMUNITY_ZIP = "data/raw/community_areas.zip"


def ensure_dirs() -> None:
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)


def main() -> None:
    ensure_dirs()
    # Download shapefile locally to avoid GeoJSON parse issues
    import requests

    for url in (COMMUNITY_URL, COMMUNITY_URL_FALLBACK):
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            with open(COMMUNITY_ZIP, "wb") as f:
                f.write(resp.content)
            break
        except Exception as exc:
            print(f"Failed to download {url}: {exc}")
            continue
    else:
        raise SystemExit("Failed to download community areas shapefile from all sources.")

    comm = gpd.read_file(COMMUNITY_ZIP).to_crs(CRS_PROJECTED)
    comm = comm.rename(columns={"community": "community_name", "area_numbe": "community_id"})
    comm = comm[["community_id", "community_name", "geometry"]]

    nodes = gpd.read_parquet("data/processed/osm_nodes.parquet").to_crs(CRS_PROJECTED)
    nodes = gpd.sjoin(nodes, comm, how="left", predicate="within")
    node_comm = nodes[["node_id", "community_id", "community_name"]]

    feats = pd.read_parquet("data/processed/intersection_features_enriched.parquet")
    feats = feats.merge(node_comm, left_on="intersection_id", right_on="node_id", how="left").drop(columns=["node_id"])
    feats.to_parquet("data/processed/intersection_features_enriched.parquet", index=False)
    print("Added community area fields to intersection_features_enriched.parquet")


if __name__ == "__main__":
    main()

