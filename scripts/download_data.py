"""
Download Chicago crash datasets, OSM network, and ACS demographics.

Outputs:
- data/raw/chicago_crashes.csv
- data/raw/chicago_people.csv
- data/raw/osm_chicago.graphml
- data/raw/acs_il_tracts.csv (if CENSUS_API_KEY is set)
"""

import os
import sys
from typing import Optional

import pandas as pd


def ensure_dirs() -> None:
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


def download_csv(url: str, path: str, limit: Optional[int] = None) -> None:
    if limit is not None and "$limit=" not in url:
        url = f"{url}&$limit={limit}" if "?" in url else f"{url}?$limit={limit}"
    print(f"Downloading {url}")
    df = pd.read_csv(url)
    df.to_csv(path, index=False)
    print(f"Saved {path} ({len(df)} rows)")


def download_crashes(limit: int = 2_000_000) -> None:
    url = "https://data.cityofchicago.org/resource/85ca-t3if.csv"
    download_csv(url, "data/raw/chicago_crashes.csv", limit=limit)


def download_people(chunk: int = 500_000) -> None:
    """
    Download the People table with pagination to avoid the 2M row cap.
    """
    base = "https://data.cityofchicago.org/resource/u6pd-qa9d.csv"
    frames = []
    offset = 0
    while True:
        url = f"{base}?$limit={chunk}&$offset={offset}"
        print(f"Downloading people chunk offset={offset}")
        df = pd.read_csv(url)
        if df.empty:
            break
        frames.append(df)
        offset += len(df)
        if len(df) < chunk:
            break
    if not frames:
        raise SystemExit("No people data downloaded")
    people = pd.concat(frames, ignore_index=True)
    people.to_csv("data/raw/chicago_people.csv", index=False)
    print(f"Saved data/raw/chicago_people.csv ({len(people)} rows)")


def download_osm(place: str = "Chicago, Illinois, USA") -> None:
    try:
        import osmnx as ox
    except ImportError as exc:
        raise SystemExit(
            "Install osmnx to download OSM data: pip install osmnx"
        ) from exc
    print(f"Downloading OSM network for {place}")
    G = ox.graph_from_place(place, network_type="drive")
    ox.save_graphml(G, "data/raw/osm_chicago.graphml")
    print("Saved data/raw/osm_chicago.graphml")


def download_acs() -> None:
    """
    Downloads Illinois tract-level ACS variables. Requires CENSUS_API_KEY.
    """
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        print("Skipping ACS download: CENSUS_API_KEY not set")
        return
    try:
        import censusdata as cd
    except ImportError as exc:
        raise SystemExit(
            "Install censusdata for ACS download: pip install censusdata"
        ) from exc

    vars_ = [
        "B01003_001E",  # total population
        "B19013_001E",  # median household income
        "B08201_002E",  # households with a vehicle
        "B17021_001E",  # poverty universe (used with other vars if needed)
    ]
    print("Downloading ACS 2022 5-year estimates for Illinois tracts")
    geo = cd.censusgeo([("state", "17"), ("county", "*"), ("tract", "*")])
    df = cd.download("acs5", 2022, geo, vars_, key=key)
    df.reset_index().to_csv("data/raw/acs_il_tracts.csv", index=False)
    print("Saved data/raw/acs_il_tracts.csv")


def main() -> None:
    ensure_dirs()
    download_crashes()
    download_people()
    download_osm()
    download_acs()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - download helper
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)

