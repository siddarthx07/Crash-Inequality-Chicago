"""
Handle missing values with domain-appropriate imputation strategies.

This script should be run AFTER all enrichment steps and BEFORE modeling.

Imputation strategies:
- Crash/injury counts: Fill with 0 (absence of crashes)
- Centrality measures: Fill with median (network-dependent)
- Demographics: Fill with median + create missing indicators
- Categorical: Create "UNKNOWN" category

Outputs:
- data/processed/intersection_features_enriched.parquet (updated with imputed values)
- data/processed/intersection_features_temporal.parquet (updated if exists)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def ensure_dirs() -> None:
    os.makedirs("data/processed", exist_ok=True)


def identify_feature_types(feats: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize features by their semantic meaning for appropriate imputation.
    
    Returns:
        Dictionary with feature categories as keys and column lists as values.
    """
    feature_categories = {
        # Crash/injury counts - should be 0 if missing (no crashes recorded)
        "crash_counts": [
            "hist_crashes",
            "recent90_crashes",
            "future_crashes",
            "hist_severity",
            "recent90_severity",
            "future_severity",
        ],
        
        # Injury aggregates - should be 0 if missing
        "injury_counts": [
            "hist_people_injuries_total",
            "hist_people_injuries_fatal",
            "hist_people_injuries_incapacitating",
            "hist_people_injuries_nonincap",
            "recent90_people_injuries_total",
            "recent90_people_injuries_fatal",
            "recent90_people_injuries_incapacitating",
            "recent90_people_injuries_nonincap",
            "hist_injuries_total",
            "hist_injuries_fatal",
            "hist_injuries_incapacitating",
            "hist_injuries_nonincap",
            "recent90_injuries_total",
            "recent90_injuries_fatal",
            "recent90_injuries_incapacitating",
            "recent90_injuries_nonincap",
        ],
        
        # Network centrality - use median (network structure dependent)
        "centrality": [
            "centrality_degree",
            "centrality_closeness",
            "centrality_betweenness",
        ],
        
        # Demographics - use median + missing indicator
        "demographics": [
            "acs_pop",
            "acs_median_income",
            "acs_households_with_vehicle",
            "acs_poverty_universe",
            "acs_vehicle_access_rate",
        ],
        
        # Categorical - create "UNKNOWN" category
        "categorical": [
            "community_id",
            "community_name",
            "GEOID",
        ],
    }
    
    # Filter to only columns that exist in the dataframe
    filtered_categories = {}
    for category, columns in feature_categories.items():
        existing_cols = [col for col in columns if col in feats.columns]
        if existing_cols:
            filtered_categories[category] = existing_cols
    
    return filtered_categories


def impute_crash_counts(feats: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Impute crash/injury counts with 0 (absence of crashes).
    
    Missing values here mean no crashes were recorded, so 0 is appropriate.
    """
    for col in cols:
        if col in feats.columns:
            missing_before = feats[col].isna().sum()
            feats[col] = feats[col].fillna(0)
            if missing_before > 0:
                print(f"  • {col}: {missing_before:,} → 0 (no crashes)")
    return feats


def impute_centrality(feats: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Impute centrality measures with median.
    
    Missing values may indicate disconnected nodes or computation issues.
    Median is more robust than mean for skewed distributions.
    """
    for col in cols:
        if col in feats.columns:
            missing_before = feats[col].isna().sum()
            if missing_before > 0:
                median_val = feats[col].median()
                feats[col] = feats[col].fillna(median_val)
                print(f"  • {col}: {missing_before:,} → median ({median_val:.6f})")
    return feats


def impute_demographics(feats: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Impute demographics with median and create missing indicators.
    
    Missing demographics may be informative (e.g., unpopulated areas),
    so we create indicator variables before imputing.
    """
    for col in cols:
        if col in feats.columns:
            missing_before = feats[col].isna().sum()
            if missing_before > 0:
                # Create missing indicator
                indicator_col = f"{col}_missing"
                feats[indicator_col] = feats[col].isna().astype(int)
                
                # Impute with median
                median_val = feats[col].median()
                
                # Handle case where all values are missing
                if pd.isna(median_val):
                    median_val = 0
                    print(f"  ⚠ {col}: All values missing, using 0")
                else:
                    feats[col] = feats[col].fillna(median_val)
                    print(f"  • {col}: {missing_before:,} → median ({median_val:.2f}) + indicator")
    return feats


def impute_categorical(feats: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Impute categorical variables with "UNKNOWN" or appropriate sentinel.
    """
    for col in cols:
        if col in feats.columns:
            missing_before = feats[col].isna().sum()
            if missing_before > 0:
                # Convert to string type if needed
                feats[col] = feats[col].astype(str)
                
                # Fill missing values
                if "GEOID" in col:
                    feats[col] = feats[col].replace("nan", "UNKNOWN")
                elif "community" in col.lower():
                    feats[col] = feats[col].replace("nan", "UNKNOWN")
                else:
                    feats[col] = feats[col].fillna("UNKNOWN")
                
                print(f"  • {col}: {missing_before:,} → 'UNKNOWN'")
    return feats


def impute_features(feats: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-appropriate imputation to all feature categories.
    """
    print("\n🔧 Applying domain-appropriate imputation strategies...")
    
    # Identify feature categories
    feature_cats = identify_feature_types(feats)
    
    # Report missing values before imputation
    missing_before = feats.isna().sum()
    missing_before = missing_before[missing_before > 0].sort_values(ascending=False)
    
    if len(missing_before) == 0:
        print("  ✓ No missing values found - skipping imputation")
        return feats
    
    print(f"\n  Found {len(missing_before)} features with missing values:")
    for col, count in missing_before.head(10).items():
        pct = count / len(feats) * 100
        print(f"    - {col}: {count:,} ({pct:.1f}%)")
    
    if len(missing_before) > 10:
        print(f"    ... and {len(missing_before) - 10} more")
    
    print("\n  Imputing by category:")
    
    # Apply imputation strategies
    if "crash_counts" in feature_cats:
        print("\n  Crash/injury counts (→ 0):")
        feats = impute_crash_counts(feats, feature_cats["crash_counts"])
    
    if "injury_counts" in feature_cats:
        print("\n  Injury counts (→ 0):")
        feats = impute_crash_counts(feats, feature_cats["injury_counts"])
    
    if "centrality" in feature_cats:
        print("\n  Centrality measures (→ median):")
        feats = impute_centrality(feats, feature_cats["centrality"])
    
    if "demographics" in feature_cats:
        print("\n  Demographics (→ median + indicator):")
        feats = impute_demographics(feats, feature_cats["demographics"])
    
    if "categorical" in feature_cats:
        print("\n  Categorical variables (→ UNKNOWN):")
        feats = impute_categorical(feats, feature_cats["categorical"])
    
    # Report remaining missing values
    missing_after = feats.isna().sum()
    missing_after = missing_after[missing_after > 0].sort_values(ascending=False)
    
    if len(missing_after) > 0:
        print(f"\n  ⚠ Still have {len(missing_after)} features with missing values:")
        for col, count in missing_after.head(5).items():
            pct = count / len(feats) * 100
            print(f"    - {col}: {count:,} ({pct:.1f}%)")
        print("\n  These may need manual inspection.")
    else:
        print("\n  ✓ All missing values imputed successfully!")
    
    return feats


def main() -> None:
    ensure_dirs()
    
    print("="*70)
    print("MISSING VALUE IMPUTATION")
    print("="*70)
    print("\nThis script applies domain-appropriate imputation strategies to")
    print("handle missing values in preprocessed intersection features.\n")
    
    # Process main enriched features
    feats_path = Path("data/processed/intersection_features_enriched.parquet")
    
    if not feats_path.exists():
        print(f"❌ ERROR: {feats_path} not found")
        print("\nRun preprocessing and enrichment scripts first:")
        print("  1. python scripts/preprocess_data.py")
        print("  2. python scripts/build_features.py")
        print("  3. python scripts/enrich_demographics.py")
        print("  4. python scripts/aggregate_people_features.py")
        print("  5. python scripts/add_centrality_betweenness.py")
        print("  6. python scripts/join_community_areas.py")
        return
    
    print(f"📂 Loading: {feats_path.name}")
    feats = pd.read_parquet(feats_path)
    print(f"  • {len(feats):,} intersections")
    print(f"  • {len(feats.columns)} features")
    
    # Apply imputation
    feats = impute_features(feats)
    
    # Ensure GEOID is string for parquet compatibility
    if "GEOID" in feats.columns:
        feats["GEOID"] = feats["GEOID"].astype(str)
    
    # Save updated features
    feats.to_parquet(feats_path, index=False)
    print(f"\n✅ Saved updated features → {feats_path}")
    
    # Process temporal features if they exist
    temp_path = Path("data/processed/intersection_features_temporal.parquet")
    
    if temp_path.exists():
        print(f"\n📂 Loading: {temp_path.name}")
        temp_feats = pd.read_parquet(temp_path)
        print(f"  • {len(temp_feats):,} temporal samples")
        print(f"  • {len(temp_feats.columns)} features")
        
        # Apply same imputation
        temp_feats = impute_features(temp_feats)
        
        # Ensure GEOID is string
        if "GEOID" in temp_feats.columns:
            temp_feats["GEOID"] = temp_feats["GEOID"].astype(str)
        
        # Save updated temporal features
        temp_feats.to_parquet(temp_path, index=False)
        print(f"\n✅ Saved updated temporal features → {temp_path}")
    else:
        print(f"\n⚠ {temp_path.name} not found (skipping)")
    
    print("\n" + "="*70)
    print("✅ IMPUTATION COMPLETE")
    print("="*70)
    print("\nYour features are now ready for modeling!")
    print("\nNext steps:")
    print("  1. Run validation: python scripts/validate_preprocessing.py")
    print("  2. Start modeling in notebooks/")


if __name__ == "__main__":
    main()

