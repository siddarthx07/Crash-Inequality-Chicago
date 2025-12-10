"""
Validate preprocessing pipeline outputs for data quality and consistency.

Run this after completing all preprocessing steps to ensure:
- Match rates are acceptable
- No unexpected missing values
- Feature distributions are reasonable
- Temporal integrity is maintained

Usage:
    python scripts/validate_preprocessing.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import geopandas as gpd

# Color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}{text}{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")


def print_pass(text: str) -> None:
    """Print a passing check."""
    print(f"{GREEN}✓{RESET} {text}")


def print_warn(text: str) -> None:
    """Print a warning."""
    print(f"{YELLOW}⚠{RESET} {text}")


def print_fail(text: str) -> None:
    """Print a failure."""
    print(f"{RED}✗{RESET} {text}")


def validate_file_exists(path: Path) -> bool:
    """Check if a file exists."""
    if path.exists():
        size_mb = path.stat().st_size / 1024 / 1024
        print_pass(f"{path.name} exists ({size_mb:.2f} MB)")
        return True
    else:
        print_fail(f"{path.name} NOT FOUND")
        return False


def validate_crashes() -> Tuple[bool, dict]:
    """Validate crash preprocessing."""
    print_header("1. CRASH DATA VALIDATION")
    
    results = {}
    all_pass = True
    
    # Check files exist
    clean_path = Path("data/processed/crashes_clean.parquet")
    snapped_path = Path("data/processed/crashes_with_nodes.parquet")
    
    if not validate_file_exists(clean_path) or not validate_file_exists(snapped_path):
        return False, results
    
    # Load data
    crashes_clean = pd.read_parquet(clean_path)
    crashes_snapped = pd.read_parquet(snapped_path)
    
    # Validate counts
    results["crashes_clean"] = len(crashes_clean)
    results["crashes_snapped"] = len(crashes_snapped)
    
    if len(crashes_clean) == len(crashes_snapped):
        print_pass(f"Row counts match: {len(crashes_clean):,} crashes")
    else:
        print_fail(f"Row count mismatch: clean={len(crashes_clean):,}, snapped={len(crashes_snapped):,}")
        all_pass = False
    
    # Validate match rate
    if "intersection_id" in crashes_snapped.columns:
        match_rate = crashes_snapped["intersection_id"].notna().mean()
        results["match_rate"] = match_rate
        
        if match_rate >= 0.85:
            print_pass(f"Match rate: {match_rate*100:.2f}% (≥85% threshold)")
        elif match_rate >= 0.75:
            print_warn(f"Match rate: {match_rate*100:.2f}% (below 85% but acceptable)")
        else:
            print_fail(f"Match rate: {match_rate*100:.2f}% (below 75% threshold)")
            all_pass = False
        
        # Distance statistics
        if "dist_to_node_m" in crashes_snapped.columns:
            matched = crashes_snapped[crashes_snapped["dist_to_node_m"].notna()]
            if len(matched) > 0:
                mean_dist = matched["dist_to_node_m"].mean()
                median_dist = matched["dist_to_node_m"].median()
                p95_dist = matched["dist_to_node_m"].quantile(0.95)
                
                print_pass(f"Snap distances: mean={mean_dist:.1f}m, median={median_dist:.1f}m, 95th={p95_dist:.1f}m")
                
                if p95_dist > 70:
                    print_warn(f"95th percentile distance ({p95_dist:.1f}m) exceeds tolerance (70m)")
    else:
        print_fail("intersection_id column not found in snapped crashes")
        all_pass = False
    
    # Validate date range
    if "crash_dt" in crashes_snapped.columns:
        crashes_snapped["crash_dt"] = pd.to_datetime(crashes_snapped["crash_dt"], errors="coerce")
        min_dt = crashes_snapped["crash_dt"].min()
        max_dt = crashes_snapped["crash_dt"].max()
        results["date_range"] = (min_dt, max_dt)
        
        print_pass(f"Date range: {min_dt.date()} to {max_dt.date()}")
        
        # Check for invalid dates
        invalid_dates = crashes_snapped["crash_dt"].isna().sum()
        if invalid_dates > 0:
            print_warn(f"{invalid_dates:,} crashes with invalid dates")
    
    return all_pass, results


def validate_people() -> Tuple[bool, dict]:
    """Validate people/injury data."""
    print_header("2. PEOPLE/INJURY DATA VALIDATION")
    
    results = {}
    all_pass = True
    
    # Check files exist
    clean_path = Path("data/processed/people_clean.parquet")
    linked_path = Path("data/processed/people_with_nodes.parquet")
    
    if not validate_file_exists(clean_path) or not validate_file_exists(linked_path):
        return False, results
    
    # Load data
    people_clean = pd.read_parquet(clean_path)
    people_linked = pd.read_parquet(linked_path)
    
    results["people_count"] = len(people_clean)
    print_pass(f"{len(people_clean):,} people records")
    
    # Validate linking
    if "intersection_id" in people_linked.columns:
        link_rate = people_linked["intersection_id"].notna().mean()
        results["link_rate"] = link_rate
        
        if link_rate >= 0.85:
            print_pass(f"Link rate to intersections: {link_rate*100:.2f}%")
        else:
            print_warn(f"Link rate to intersections: {link_rate*100:.2f}% (expected ≥85%)")
    
    # Validate injury classifications
    if "injury_classification" in people_clean.columns:
        injury_counts = people_clean["injury_classification"].value_counts()
        print_pass(f"Injury breakdown:")
        for injury_type, count in injury_counts.head(5).items():
            print(f"    {injury_type}: {count:,} ({count/len(people_clean)*100:.1f}%)")
    
    return all_pass, results


def validate_network() -> Tuple[bool, dict]:
    """Validate OSM network data."""
    print_header("3. NETWORK DATA VALIDATION")
    
    results = {}
    all_pass = True
    
    # Check files exist
    nodes_path = Path("data/processed/osm_nodes.parquet")
    edges_path = Path("data/processed/osm_edges.parquet")
    
    if not validate_file_exists(nodes_path) or not validate_file_exists(edges_path):
        return False, results
    
    # Load data
    nodes = gpd.read_parquet(nodes_path)
    edges = gpd.read_parquet(edges_path)
    
    results["nodes_count"] = len(nodes)
    results["edges_count"] = len(edges)
    
    print_pass(f"{len(nodes):,} nodes (intersections)")
    print_pass(f"{len(edges):,} edges (road segments)")
    
    # Validate required columns
    required_node_cols = ["node_id", "geometry"]
    missing_cols = [col for col in required_node_cols if col not in nodes.columns]
    
    if missing_cols:
        print_fail(f"Missing node columns: {missing_cols}")
        all_pass = False
    else:
        print_pass("Required node columns present")
    
    return all_pass, results


def validate_features() -> Tuple[bool, dict]:
    """Validate intersection features."""
    print_header("4. INTERSECTION FEATURES VALIDATION")
    
    results = {}
    all_pass = True
    
    # Check main features file
    feats_path = Path("data/processed/intersection_features_enriched.parquet")
    
    if not validate_file_exists(feats_path):
        return False, results
    
    # Load data
    feats = pd.read_parquet(feats_path)
    
    results["intersections_count"] = len(feats)
    print_pass(f"{len(feats):,} intersections with features")
    
    # Validate hotspot labels
    if "label_hotspot" in feats.columns:
        hotspot_rate = feats["label_hotspot"].mean()
        hotspot_count = feats["label_hotspot"].sum()
        results["hotspot_rate"] = hotspot_rate
        results["hotspot_count"] = hotspot_count
        
        if 0.08 <= hotspot_rate <= 0.15:
            print_pass(f"Hotspot rate: {hotspot_rate*100:.1f}% ({int(hotspot_count):,} hotspots)")
        else:
            print_warn(f"Hotspot rate: {hotspot_rate*100:.1f}% (expected 8-15%)")
    else:
        print_fail("label_hotspot column not found")
        all_pass = False
    
    # Check for expected feature categories
    feature_groups = {
        "Crash history": ["hist_crashes", "recent90_crashes", "future_crashes"],
        "Severity": ["hist_severity", "recent90_severity"],
        "Injuries": ["hist_people_injuries_total", "recent90_people_injuries_total"],
        "Centrality": ["centrality_degree", "centrality_closeness", "centrality_betweenness"],
        "Demographics": ["acs_pop", "acs_median_income"],
        "Geographic": ["community_id", "community_name"],
    }
    
    for group_name, cols in feature_groups.items():
        present = [col for col in cols if col in feats.columns]
        missing = [col for col in cols if col not in feats.columns]
        
        if len(present) == len(cols):
            print_pass(f"{group_name}: {len(present)}/{len(cols)} features present")
        elif len(present) > 0:
            print_warn(f"{group_name}: {len(present)}/{len(cols)} features present (missing: {missing})")
        else:
            print_fail(f"{group_name}: 0/{len(cols)} features present")
            all_pass = False
    
    # Missing value analysis
    missing_pct = feats.isna().mean() * 100
    high_missing = missing_pct[missing_pct > 20].sort_values(ascending=False)
    
    if len(high_missing) > 0:
        print_warn(f"{len(high_missing)} features with >20% missing values:")
        for col, pct in high_missing.head(10).items():
            print(f"    {col}: {pct:.1f}% missing")
        results["high_missing_features"] = len(high_missing)
    else:
        print_pass("No features with >20% missing values")
        results["high_missing_features"] = 0
    
    # Feature distribution checks (detect anomalies)
    numeric_cols = feats.select_dtypes(include=["int64", "float64"]).columns
    
    print_pass(f"{len(numeric_cols)} numeric features")
    
    # Check for suspicious zeros
    zero_heavy = []
    for col in numeric_cols:
        if col != "label_hotspot":  # Skip label
            zero_rate = (feats[col] == 0).mean()
            if zero_rate > 0.95:
                zero_heavy.append((col, zero_rate))
    
    if zero_heavy:
        print_warn(f"{len(zero_heavy)} features with >95% zeros (may indicate missing data):")
        for col, rate in sorted(zero_heavy, key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {col}: {rate*100:.1f}% zeros")
    
    return all_pass, results


def validate_temporal_features() -> Tuple[bool, dict]:
    """Validate temporal features for proper time-series modeling."""
    print_header("5. TEMPORAL FEATURES VALIDATION")
    
    results = {}
    all_pass = True
    
    # Check temporal features file
    temp_path = Path("data/processed/intersection_features_temporal.parquet")
    
    if not validate_file_exists(temp_path):
        print_warn("Temporal features not found (may not be needed for random split models)")
        return True, results
    
    # Load data
    temp_feats = pd.read_parquet(temp_path)
    
    results["temporal_samples"] = len(temp_feats)
    print_pass(f"{len(temp_feats):,} temporal samples")
    
    # Validate cutoff_date column
    if "cutoff_date" not in temp_feats.columns:
        print_fail("cutoff_date column not found")
        all_pass = False
        return all_pass, results
    
    temp_feats["cutoff_date"] = pd.to_datetime(temp_feats["cutoff_date"])
    
    # Check number of periods
    periods = temp_feats["cutoff_date"].nunique()
    results["periods"] = periods
    print_pass(f"{periods} unique time periods")
    
    if periods < 3:
        print_warn(f"Only {periods} time periods (recommend ≥3 for train/val/test)")
    
    # Check temporal ordering
    sorted_dates = sorted(temp_feats["cutoff_date"].unique())
    print_pass(f"Date range: {sorted_dates[0].date()} to {sorted_dates[-1].date()}")
    
    # Check samples per period
    samples_per_period = temp_feats.groupby("cutoff_date").size()
    results["samples_per_period"] = samples_per_period.mean()
    
    if samples_per_period.std() / samples_per_period.mean() < 0.1:
        print_pass(f"Consistent samples per period: ~{samples_per_period.mean():.0f} intersections")
    else:
        print_warn(f"Variable samples per period: {samples_per_period.min()}-{samples_per_period.max()}")
    
    # Check for data leakage (future features should exist for all samples)
    if "future_crashes" in temp_feats.columns:
        missing_future = temp_feats["future_crashes"].isna().sum()
        if missing_future == 0:
            print_pass("All samples have future labels (no leakage from missing data)")
        else:
            print_warn(f"{missing_future} samples missing future labels")
    
    # Validate temporal split suggestions
    n_periods = len(sorted_dates)
    train_end_idx = int(0.7 * n_periods)
    val_end_idx = int(0.85 * n_periods)
    
    print_pass("Suggested temporal splits:")
    print(f"    Train: {sorted_dates[0].date()} to {sorted_dates[train_end_idx-1].date()} ({train_end_idx} periods)")
    print(f"    Val:   {sorted_dates[train_end_idx].date()} to {sorted_dates[val_end_idx-1].date()} ({val_end_idx - train_end_idx} periods)")
    print(f"    Test:  {sorted_dates[val_end_idx].date()} to {sorted_dates[-1].date()} ({n_periods - val_end_idx} periods)")
    
    return all_pass, results


def validate_consistency() -> Tuple[bool, dict]:
    """Cross-validate consistency between files."""
    print_header("6. CROSS-FILE CONSISTENCY CHECKS")
    
    results = {}
    all_pass = True
    
    # Load key files
    try:
        crashes = pd.read_parquet("data/processed/crashes_with_nodes.parquet")
        feats = pd.read_parquet("data/processed/intersection_features_enriched.parquet")
        nodes = pd.read_parquet("data/processed/osm_nodes.parquet")
    except Exception as e:
        print_fail(f"Error loading files: {e}")
        return False, results
    
    # Check intersection IDs consistency
    crash_nodes = set(crashes["intersection_id"].dropna().unique())
    feat_nodes = set(feats["intersection_id"].unique())
    osm_nodes = set(nodes["node_id"].unique())
    
    results["crash_unique_nodes"] = len(crash_nodes)
    results["feat_unique_nodes"] = len(feat_nodes)
    results["osm_unique_nodes"] = len(osm_nodes)
    
    # Features should be a superset of crash nodes
    if crash_nodes.issubset(feat_nodes):
        print_pass(f"All crash nodes ({len(crash_nodes):,}) present in features")
    else:
        missing = len(crash_nodes - feat_nodes)
        print_warn(f"{missing:,} crash nodes missing from features")
    
    # Feature nodes should be subset of OSM nodes
    if feat_nodes.issubset(osm_nodes):
        print_pass(f"All feature nodes ({len(feat_nodes):,}) exist in OSM network")
    else:
        orphaned = len(feat_nodes - osm_nodes)
        print_fail(f"{orphaned:,} feature nodes not found in OSM network")
        all_pass = False
    
    # Check date consistency
    if "crash_dt" in crashes.columns and "future_crashes" in feats.columns:
        crashes["crash_dt"] = pd.to_datetime(crashes["crash_dt"], errors="coerce")
        crash_max = crashes["crash_dt"].max()
        
        # Features should account for crashes up to max date
        print_pass(f"Crash data extends to: {crash_max.date()}")
    
    return all_pass, results


def generate_summary(all_results: dict) -> None:
    """Generate summary report."""
    print_header("VALIDATION SUMMARY")
    
    total_checks = len(all_results)
    passed_checks = sum(1 for passed, _ in all_results.values() if passed)
    
    if passed_checks == total_checks:
        print(f"\n{GREEN}{BOLD}✓ ALL {total_checks} VALIDATION CHECKS PASSED{RESET}\n")
        print("Your preprocessing pipeline is ready for modeling!")
    else:
        failed_checks = total_checks - passed_checks
        print(f"\n{YELLOW}{BOLD}⚠ {passed_checks}/{total_checks} CHECKS PASSED{RESET}")
        print(f"{RED}{failed_checks} CHECKS FAILED OR HAD WARNINGS{RESET}\n")
        print("Review the issues above before proceeding to modeling.")
    
    # Key statistics
    print(f"\n{BOLD}KEY STATISTICS:{RESET}")
    
    if "crashes" in all_results and all_results["crashes"][1]:
        crash_stats = all_results["crashes"][1]
        if "crashes_clean" in crash_stats:
            print(f"  • Total crashes: {crash_stats['crashes_clean']:,}")
        if "match_rate" in crash_stats:
            print(f"  • Crash-to-intersection match rate: {crash_stats['match_rate']*100:.1f}%")
    
    if "features" in all_results and all_results["features"][1]:
        feat_stats = all_results["features"][1]
        if "intersections_count" in feat_stats:
            print(f"  • Intersections with features: {feat_stats['intersections_count']:,}")
        if "hotspot_count" in feat_stats:
            print(f"  • Hotspot intersections: {feat_stats['hotspot_count']:,} ({feat_stats['hotspot_rate']*100:.1f}%)")
        if "high_missing_features" in feat_stats:
            print(f"  • Features with >20% missing: {feat_stats['high_missing_features']}")
    
    if "temporal" in all_results and all_results["temporal"][1]:
        temp_stats = all_results["temporal"][1]
        if "periods" in temp_stats:
            print(f"  • Temporal periods: {temp_stats['periods']}")
        if "temporal_samples" in temp_stats:
            print(f"  • Total temporal samples: {temp_stats['temporal_samples']:,}")


def main() -> None:
    """Run all validation checks."""
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}PREPROCESSING PIPELINE VALIDATION{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")
    print("\nThis script validates the quality and consistency of preprocessed data.")
    print("Any failures or warnings should be investigated before modeling.\n")
    
    # Check if processed directory exists
    if not Path("data/processed").exists():
        print_fail("data/processed directory not found!")
        print("\nRun preprocessing scripts first:")
        print("  1. python scripts/preprocess_data.py")
        print("  2. python scripts/build_features.py")
        print("  3. python scripts/enrich_demographics.py")
        print("  4. python scripts/aggregate_people_features.py")
        print("  5. python scripts/add_centrality_betweenness.py")
        print("  6. python scripts/join_community_areas.py")
        sys.exit(1)
    
    # Run all validations
    all_results = {}
    
    all_results["crashes"] = validate_crashes()
    all_results["people"] = validate_people()
    all_results["network"] = validate_network()
    all_results["features"] = validate_features()
    all_results["temporal"] = validate_temporal_features()
    all_results["consistency"] = validate_consistency()
    
    # Generate summary
    generate_summary(all_results)
    
    # Exit with error if any checks failed
    if not all(passed for passed, _ in all_results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

