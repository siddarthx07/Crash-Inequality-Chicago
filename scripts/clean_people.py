"""
Clean and link People table to crash and intersection data.

Outputs:
- data/processed/people_clean.parquet
- data/processed/people_with_nodes.parquet (includes crash_dt, intersection_id)
"""

from __future__ import annotations

import os
import pandas as pd


USECOLS = [
    "crash_record_id",
    "person_id",
    "person_type",
    "injury_classification",
    "age",
    "sex",
    "safety_equipment",
]


def ensure_dirs() -> None:
    os.makedirs("data/processed", exist_ok=True)


def load_people() -> pd.DataFrame:
    df = pd.read_csv("data/raw/chicago_people.csv", usecols=USECOLS)
    df = df.drop_duplicates(subset=["person_id"])
    df["injury_classification"] = df["injury_classification"].str.upper()
    df["person_type"] = df["person_type"].str.upper()
    return df


def link_to_intersections(people: pd.DataFrame) -> pd.DataFrame:
    crashes = pd.read_parquet(
        "data/processed/crashes_with_nodes.parquet",
        columns=["crash_record_id", "crash_date", "intersection_id"],
    )
    crashes["crash_dt"] = pd.to_datetime(crashes["crash_date"], errors="coerce")
    merged = people.merge(crashes, on="crash_record_id", how="left")
    return merged


def main() -> None:
    ensure_dirs()
    people = load_people()
    people.to_parquet("data/processed/people_clean.parquet", index=False)
    linked = link_to_intersections(people)
    linked.to_parquet("data/processed/people_with_nodes.parquet", index=False)
    print(
        f"Saved {len(people)} people -> data/processed/people_clean.parquet; "
        f"linked {linked['intersection_id'].notna().sum()} to intersections"
    )


if __name__ == "__main__":
    main()

