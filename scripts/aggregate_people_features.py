"""
Aggregate people-level injuries per intersection and add to feature table.

Outputs:
- data/processed/intersection_features_enriched.parquet (updated with people aggregates)
"""

from __future__ import annotations

import pandas as pd

HISTORY_DAYS = 365
RECENT_DAYS = 90


def main() -> None:
    people = pd.read_parquet(
        "data/processed/people_with_nodes.parquet",
        columns=[
            "intersection_id",
            "injury_classification",
            "crash_dt",
        ],
    )
    people = people.dropna(subset=["intersection_id", "crash_dt"])
    people["crash_dt"] = pd.to_datetime(people["crash_dt"], errors="coerce")
    people = people.dropna(subset=["crash_dt"])

    max_dt = people["crash_dt"].max()
    hist_start = max_dt - pd.Timedelta(days=HISTORY_DAYS)
    recent_start = max_dt - pd.Timedelta(days=RECENT_DAYS)

    def agg_window(start, end, prefix):
        mask = (people["crash_dt"] >= start) & (people["crash_dt"] < end)
        sub = people.loc[mask]
        out = (
            sub.groupby("intersection_id")
            .agg(
                **{
                    f"{prefix}_injuries_total": ("injury_classification", "count"),
                    f"{prefix}_injuries_fatal": (
                        "injury_classification",
                        lambda s: (s == "FATAL").sum(),
                    ),
                    f"{prefix}_injuries_incapacitating": (
                        "injury_classification",
                        lambda s: (s == "INCAPACITATING INJURY").sum(),
                    ),
                    f"{prefix}_injuries_nonincap": (
                        "injury_classification",
                        lambda s: (s == "NONINCAPACITATING INJURY").sum(),
                    ),
                }
            )
            .reset_index()
        )
        return out

    hist = agg_window(hist_start, max_dt, "hist_people")
    recent = agg_window(recent_start, max_dt, "recent90_people")

    feats = pd.read_parquet("data/processed/intersection_features_enriched.parquet")
    # Ensure GEOID is string for parquet
    if "GEOID" in feats.columns:
        feats["GEOID"] = feats["GEOID"].astype(str)
    feats = feats.merge(hist, on="intersection_id", how="left")
    feats = feats.merge(recent, on="intersection_id", how="left")
    feats = feats.fillna(0)

    feats.to_parquet("data/processed/intersection_features_enriched.parquet", index=False)
    print(
        "Updated data/processed/intersection_features_enriched.parquet with people aggregates"
    )


if __name__ == "__main__":
    main()

