# Crash Inequality – Chicago (Urban Computing Project)

## Objective
Identify crash inequality and evolving hotspots across Chicago intersections/neighborhoods, and predict future hotspot intersections.

## Quickstart
1) Create env (example):
```
python -m venv .venv
source .venv/bin/activate
pip install -q pandas geopandas osmnx censusdata pyarrow networkx requests
```
2) Download data:
```
python scripts/download_data.py
```
3) Preprocess and build features (snap radius 70m):
```
python scripts/preprocess_data.py
python scripts/build_features.py
python scripts/enrich_demographics.py
python scripts/aggregate_people_features.py
python scripts/add_centrality_betweenness.py
python scripts/join_community_areas.py
```
Outputs land in `data/processed/` (not tracked in git).

## Key scripts
- `scripts/download_data.py`: pulls crashes, people, OSM, ACS.
- `scripts/preprocess_data.py`: cleans crashes, snaps to nearest OSM intersection (70m), exports clean crashes and crash→node mapping.
- `scripts/build_features.py`: temporal aggregates (hist 12mo, recent 90d, future 6mo), hotspot labels (top 10%).
- `scripts/enrich_demographics.py`: ACS demographics join via tracts (+ vehicle access rate).
- `scripts/aggregate_people_features.py`: injury aggregates from People table.
- `scripts/add_centrality_betweenness.py`: approximate betweenness centrality.
- `scripts/join_community_areas.py`: adds community area ID/name.

## Data (not in git)
- Raw: Chicago crashes/people CSVs, OSM graph, ACS, community area shapefile.
- Processed: clean crashes, crash→node mapping, nodes/edges, enriched intersection features, people-linked tables.

## Notes
- Current snap tolerance: 70m (match rate ~88%).
- Hotspot label: top 10% intersections by future crash count (threshold currently 6).
- CRS: EPSG:26971 for spatial operations.
# Crash-Inequality-Chicago
