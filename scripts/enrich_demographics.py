"""
Attach ACS demographics to intersections via tract polygons.

Outputs:
- data/processed/intersection_features_enriched.parquet
"""

from __future__ import annotations

import os
import geopandas as gpd
import pandas as pd

CRS_PROJECTED = "EPSG:26971"
TIGER_TRACTS_2022_IL = "https://www2.census.gov/geo/tiger/TIGER2022/TRACT/tl_2022_17_tract.zip"


def ensure_dirs() -> None:
    os.makedirs("data/processed", exist_ok=True)


def load_acs() -> pd.DataFrame:
    acs = pd.read_csv("data/raw/acs_il_tracts.csv")
    # Expect columns: NAME, B01003_001E, B19013_001E, B08201_002E, B17021_001E, state, county, tract
    acs["state"] = acs["state"].astype(str).str.zfill(2)
    acs["county"] = acs["county"].astype(str).str.zfill(3)
    acs["tract"] = acs["tract"].astype(str).str.zfill(6)
    acs["GEOID"] = acs["state"] + acs["county"] + acs["tract"]
    acs = acs.rename(
        columns={
            "B01003_001E": "acs_pop",
            "B19013_001E": "acs_median_income",
            "B08201_002E": "acs_households_with_vehicle",
            "B17021_001E": "acs_poverty_universe",
        }
    )
    # Derive simple rates using available fields (vehicle access per pop as proxy)
    acs["acs_vehicle_access_rate"] = acs["acs_households_with_vehicle"] / acs["acs_pop"].replace(0, pd.NA)
    return acs[
        [
            "GEOID",
            "acs_pop",
            "acs_median_income",
            "acs_households_with_vehicle",
            "acs_poverty_universe",
            "acs_vehicle_access_rate",
        ]
    ]


def load_tracts() -> gpd.GeoDataFrame:
    tracts = gpd.read_file(TIGER_TRACTS_2022_IL)
    tracts = tracts.to_crs(CRS_PROJECTED)
    return tracts[["GEOID", "geometry"]]


def main() -> None:
    ensure_dirs()
    acs = load_acs()
    tracts = load_tracts()
    tracts = tracts.merge(acs, on="GEOID", how="left")

    nodes = gpd.read_parquet("data/processed/osm_nodes.parquet")
    nodes = nodes.to_crs(CRS_PROJECTED)
    nodes = gpd.sjoin(nodes, tracts, how="left", predicate="within")
    node_dem = nodes[
        [
            "node_id",
            "GEOID",
            "acs_pop",
            "acs_median_income",
            "acs_households_with_vehicle",
            "acs_poverty_universe",
            "acs_vehicle_access_rate",
        ]
    ]

    feats = pd.read_parquet("data/processed/intersection_features.parquet")
    feats = feats.merge(
        node_dem,
        left_on="intersection_id",
        right_on="node_id",
        how="left",
    ).drop(columns=["node_id"])

    feats.to_parquet("data/processed/intersection_features_enriched.parquet", index=False)
    print(
        f"Saved enriched features -> data/processed/intersection_features_enriched.parquet "
        f"({len(feats)} intersections)"
    )


if __name__ == "__main__":
    main()

