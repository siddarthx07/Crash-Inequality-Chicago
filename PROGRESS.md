# Project Progress (Crash Inequality – Chicago)

## Scope
- Goal: identify crash inequality and evolving hotspots across Chicago intersections/neighborhoods; predict future hotspot intersections.

## Data collected
- Chicago Traffic Crashes (events) and People (injuries) — full datasets.
- OpenStreetMap (OSM) road network for intersections/roads.
- ACS demographics (population, income, vehicle access), tract polygons.
- Chicago community area boundaries.

## Preprocessing
- Cleaned crashes: dedup, parsed datetimes, filtered bad coords, projected to EPSG:26971.
- Snap crashes to nearest OSM intersection within 70 m (was 40 m); match rate ≈ 88% (880,457 / 1,001,020). Unmatched kept flagged for exclusion.
- Built crash/intersection mappings and clean crash table.
- Normalized OSM nodes/edges to Parquet; computed degree, closeness, betweenness (approx k=500).

## Features (intersection level)
- Temporal aggregates: past 12 months crashes/severity, recent 90 days, future 6 months (for labels).
- Hotspot label: top 10% by future crash count (current threshold = 6; positives ≈ 2,292 of 19,200 intersections).
- People (injury) aggregates: total, fatal, incapacitating, non-incap for history and recent windows.
- Demographics: ACS pop, median income, vehicle access (count + rate), tract GEOID.
- Community area: ID and name.

## Key artifacts (data/processed/)
- `crashes_clean.parquet`
- `crashes_with_nodes.parquet`
- `osm_nodes.parquet`, `osm_edges.parquet`
- `intersection_features.parquet`
- `intersection_features_enriched.parquet` (features + labels + demographics + injuries + centralities + community areas)
- `people_clean.parquet`, `people_with_nodes.parquet`

## Open items / next steps
- Decide if 70 m snap is final; 80 m optional for a bit more coverage vs. false matches.
- Define train/val/test (chronological) and run models (logistic, RF/GBM) with class weighting; report AUC/PR AUC/F1.
- Spatial checks: Moran’s I/LISA on residuals; inequality summaries by community area/tract.
- Figures: hotspot maps (current/predicted), feature importances, PR curves.

