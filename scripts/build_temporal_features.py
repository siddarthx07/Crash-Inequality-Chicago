"""
Build temporal features with proper chronological train/val/test splits.

Strategy:
- Create rolling windows of features for multiple time periods
- Each sample has a cutoff date for temporal ordering
- Train on early periods, validate on middle, test on recent

Outputs:
- data/processed/intersection_features_temporal.parquet (with cutoff_date column)
"""

from __future__ import annotations

import os
from datetime import timedelta, datetime
from typing import Tuple

import pandas as pd

HISTORY_DAYS = 365
RECENT_DAYS = 90
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
    """Load crash data with datetime parsing."""
    crashes = pd.read_parquet(
        "data/processed/crashes_with_nodes.parquet",
        columns=["crash_record_id", "crash_date", "intersection_id", "most_severe_injury"],
    )
    crashes = crashes.dropna(subset=["intersection_id", "crash_date"])
    crashes["crash_dt"] = pd.to_datetime(crashes["crash_date"], errors="coerce")
    crashes = crashes.dropna(subset=["crash_dt"])
    crashes["severity_wt"] = crashes["most_severe_injury"].map(SEVERITY_WEIGHTS).fillna(1)
    return crashes


def load_people() -> pd.DataFrame:
    """Load people/injury data."""
    people = pd.read_parquet(
        "data/processed/people_with_nodes.parquet",
        columns=["intersection_id", "injury_classification", "crash_dt"],
    )
    people = people.dropna(subset=["intersection_id", "crash_dt"])
    people["crash_dt"] = pd.to_datetime(people["crash_dt"], errors="coerce")
    people = people.dropna(subset=["crash_dt"])
    return people


def aggregate_crashes(crashes: pd.DataFrame, start, end) -> pd.DataFrame:
    """Aggregate crash counts and severity for a time window."""
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


def aggregate_injuries(people: pd.DataFrame, start, end) -> pd.DataFrame:
    """Aggregate injury counts by severity for a time window."""
    mask = (people["crash_dt"] >= start) & (people["crash_dt"] < end)
    sub = people.loc[mask]
    
    agg = (
        sub.groupby("intersection_id")
        .agg(
            injuries_total=("injury_classification", "count"),
            injuries_fatal=("injury_classification", lambda s: (s == "FATAL").sum()),
            injuries_incapacitating=(
                "injury_classification",
                lambda s: (s == "INCAPACITATING INJURY").sum(),
            ),
            injuries_nonincap=(
                "injury_classification",
                lambda s: (s == "NONINCAPACITATING INJURY").sum(),
            ),
        )
        .reset_index()
    )
    return agg


def build_features_for_cutoff(
    crashes: pd.DataFrame,
    people: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build features for a single cutoff date."""
    
    hist_start = cutoff_date - timedelta(days=HISTORY_DAYS)
    recent_start = cutoff_date - timedelta(days=RECENT_DAYS)
    future_end = cutoff_date + timedelta(days=PREDICT_DAYS)
    
    # Historical features (12 months before cutoff)
    hist_crashes = aggregate_crashes(crashes, hist_start, cutoff_date)
    hist_crashes = hist_crashes.rename(
        columns={"crashes_count": "hist_crashes", "severity_sum": "hist_severity"}
    )
    
    hist_injuries = aggregate_injuries(people, hist_start, cutoff_date)
    hist_injuries = hist_injuries.rename(
        columns={
            "injuries_total": "hist_injuries_total",
            "injuries_fatal": "hist_injuries_fatal",
            "injuries_incapacitating": "hist_injuries_incapacitating",
            "injuries_nonincap": "hist_injuries_nonincap",
        }
    )
    
    # Recent features (90 days before cutoff)
    recent_crashes = aggregate_crashes(crashes, recent_start, cutoff_date)
    recent_crashes = recent_crashes.rename(
        columns={"crashes_count": "recent90_crashes", "severity_sum": "recent90_severity"}
    )
    
    recent_injuries = aggregate_injuries(people, recent_start, cutoff_date)
    recent_injuries = recent_injuries.rename(
        columns={
            "injuries_total": "recent90_injuries_total",
            "injuries_fatal": "recent90_injuries_fatal",
            "injuries_incapacitating": "recent90_injuries_incapacitating",
            "injuries_nonincap": "recent90_injuries_nonincap",
        }
    )
    
    # Future labels (6 months after cutoff)
    future_crashes = aggregate_crashes(crashes, cutoff_date, future_end)
    future_crashes = future_crashes.rename(
        columns={"crashes_count": "future_crashes", "severity_sum": "future_severity"}
    )
    
    # Merge all features
    feats = hist_crashes.merge(hist_injuries, on="intersection_id", how="outer")
    feats = feats.merge(recent_crashes, on="intersection_id", how="outer")
    feats = feats.merge(recent_injuries, on="intersection_id", how="outer")
    feats = feats.merge(future_crashes, on="intersection_id", how="outer")
    feats = feats.fillna(0)
    
    # Add cutoff date for temporal ordering
    feats["cutoff_date"] = cutoff_date
    
    return feats


def main() -> None:
    ensure_dirs()
    
    print("Loading data...")
    crashes = load_crashes()
    people = load_people()
    
    # Determine date range
    min_dt = crashes["crash_dt"].min()
    max_dt = crashes["crash_dt"].max()
    print(f"Crash data range: {min_dt.date()} to {max_dt.date()}")
    
    # Define cutoff dates for temporal splits
    # We'll create features every 6 months (PREDICT_DAYS)
    # Start from a date that has enough history (HISTORY_DAYS before)
    first_cutoff = min_dt + timedelta(days=HISTORY_DAYS + 90)  # Need history + some data
    last_cutoff = max_dt - timedelta(days=PREDICT_DAYS)  # Need future window
    
    # Generate cutoff dates every 6 months
    cutoff_dates = pd.date_range(
        start=first_cutoff,
        end=last_cutoff,
        freq=f"{PREDICT_DAYS}D"  # Every 6 months
    )
    
    print(f"\nGenerating features for {len(cutoff_dates)} time periods:")
    print(f"  First cutoff: {cutoff_dates[0].date()}")
    print(f"  Last cutoff: {cutoff_dates[-1].date()}")
    
    # Build features for each cutoff date
    all_features = []
    for i, cutoff in enumerate(cutoff_dates):
        print(f"  Processing cutoff {i+1}/{len(cutoff_dates)}: {cutoff.date()}")
        feats = build_features_for_cutoff(crashes, people, cutoff)
        all_features.append(feats)
    
    # Combine all periods
    combined = pd.concat(all_features, ignore_index=True)
    
    # Add static features (centrality, demographics, community areas)
    print("\nAdding static features...")
    
    # Load existing enriched features for static data
    static_feats = pd.read_parquet("data/processed/intersection_features_enriched.parquet")
    static_cols = [
        "intersection_id",
        "centrality_degree",
        "centrality_closeness",
        "centrality_betweenness",
        "GEOID",
        "acs_pop",
        "acs_median_income",
        "acs_households_with_vehicle",
        "acs_poverty_universe",
        "acs_vehicle_access_rate",
        "community_id",
        "community_name",
    ]
    static_data = static_feats[static_cols].drop_duplicates(subset=["intersection_id"])
    
    combined = combined.merge(static_data, on="intersection_id", how="left")
    
    # Compute hotspot labels per cutoff period
    # Top 10% by future crashes within each time period
    combined["label_hotspot"] = 0
    for cutoff in combined["cutoff_date"].unique():
        mask = combined["cutoff_date"] == cutoff
        threshold = combined.loc[mask, "future_crashes"].quantile(1 - HOTSPOT_TOP_PCT)
        combined.loc[mask, "label_hotspot"] = (
            combined.loc[mask, "future_crashes"] >= threshold
        ).astype(int)
    
    # Save
    combined.to_parquet("data/processed/intersection_features_temporal.parquet", index=False)
    
    print(f"\n✅ Saved {len(combined)} samples across {len(cutoff_dates)} time periods")
    print(f"   -> data/processed/intersection_features_temporal.parquet")
    print(f"\nSamples per period: {len(combined) // len(cutoff_dates)}")
    print(f"Total hotspot positives: {combined['label_hotspot'].sum()} ({combined['label_hotspot'].mean()*100:.1f}%)")
    
    # Show temporal split suggestion
    print("\n📅 Suggested temporal splits:")
    sorted_dates = sorted(combined["cutoff_date"].unique())
    n_periods = len(sorted_dates)
    train_end_idx = int(0.7 * n_periods)
    val_end_idx = int(0.85 * n_periods)
    
    print(f"  Train: {sorted_dates[0].date()} to {sorted_dates[train_end_idx-1].date()} ({train_end_idx} periods)")
    print(f"  Val:   {sorted_dates[train_end_idx].date()} to {sorted_dates[val_end_idx-1].date()} ({val_end_idx - train_end_idx} periods)")
    print(f"  Test:  {sorted_dates[val_end_idx].date()} to {sorted_dates[-1].date()} ({n_periods - val_end_idx} periods)")


if __name__ == "__main__":
    main()
