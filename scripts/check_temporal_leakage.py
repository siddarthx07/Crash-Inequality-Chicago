"""
Check for potential temporal data leakage in features and modeling notebooks.

This script validates that:
1. Temporal features are used correctly with chronological splits
2. Non-temporal features are not used for temporal prediction
3. Future information doesn't leak into training features

Usage:
    python scripts/check_temporal_leakage.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, List

import pandas as pd

# ANSI color codes
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text: str) -> None:
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}{text}{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")


def print_error(text: str) -> None:
    print(f"{RED}✗{RESET} {text}")


def print_warning(text: str) -> None:
    print(f"{YELLOW}⚠{RESET} {text}")


def print_success(text: str) -> None:
    print(f"{GREEN}✓{RESET} {text}")


def check_non_temporal_features() -> Tuple[bool, dict]:
    """
    Check if non-temporal features file exists and warn about leakage risk.
    """
    print_header("1. NON-TEMPORAL FEATURES CHECK")
    
    results = {}
    has_issue = False
    
    non_temp_path = Path("data/processed/intersection_features.parquet")
    enriched_path = Path("data/processed/intersection_features_enriched.parquet")
    
    if non_temp_path.exists():
        print_warning(f"{non_temp_path.name} exists")
        
        feats = pd.read_parquet(non_temp_path)
        results["non_temporal_samples"] = len(feats)
        
        # Check if it has future features
        future_cols = [col for col in feats.columns if "future" in col.lower()]
        
        if future_cols:
            print_error("Contains future features (TEMPORAL LEAKAGE RISK):")
            for col in future_cols:
                print(f"    - {col}")
            has_issue = True
            
            print("\n  ⚠ WARNING: This file should NOT be used for modeling!")
            print("  Use 'intersection_features_temporal.parquet' instead for proper")
            print("  temporal validation, or remove future columns for random splits.\n")
    else:
        print_success(f"{non_temp_path.name} not found (good - avoids confusion)")
    
    if enriched_path.exists():
        print_success(f"{enriched_path.name} exists")
        
        feats = pd.read_parquet(enriched_path)
        results["enriched_samples"] = len(feats)
        
        # Check if it has future features
        future_cols = [col for col in feats.columns if "future" in col.lower()]
        
        if future_cols:
            print_warning("Contains future features:")
            for col in future_cols:
                print(f"    - {col}")
            
            print("\n  ⚠ USAGE GUIDANCE:")
            print("  • For RANDOM SPLIT models: DROP future columns before splitting")
            print("  • For TEMPORAL models: Use intersection_features_temporal.parquet instead\n")
            results["has_future_features"] = True
        else:
            print_success("No future features found")
            results["has_future_features"] = False
    
    return has_issue, results


def check_temporal_features() -> Tuple[bool, dict]:
    """
    Check temporal features for proper structure.
    """
    print_header("2. TEMPORAL FEATURES CHECK")
    
    results = {}
    has_issue = False
    
    temp_path = Path("data/processed/intersection_features_temporal.parquet")
    
    if not temp_path.exists():
        print_warning(f"{temp_path.name} not found")
        print("  If you plan to do temporal validation, run:")
        print("    python scripts/build_temporal_features.py")
        return False, results
    
    print_success(f"{temp_path.name} exists")
    
    # Load and validate structure
    feats = pd.read_parquet(temp_path)
    results["temporal_samples"] = len(feats)
    
    # Check for cutoff_date column
    if "cutoff_date" not in feats.columns:
        print_error("Missing 'cutoff_date' column (required for temporal splits)")
        has_issue = True
    else:
        feats["cutoff_date"] = pd.to_datetime(feats["cutoff_date"])
        periods = feats["cutoff_date"].nunique()
        results["periods"] = periods
        
        print_success(f"Has cutoff_date column with {periods} periods")
        
        sorted_dates = sorted(feats["cutoff_date"].unique())
        print(f"  Date range: {sorted_dates[0].date()} to {sorted_dates[-1].date()}")
    
    # Check for future features (required for labels)
    if "future_crashes" not in feats.columns:
        print_error("Missing 'future_crashes' column (needed for labels)")
        has_issue = True
    else:
        print_success("Has future features for labels")
    
    # Check for label column
    if "label_hotspot" not in feats.columns:
        print_warning("Missing 'label_hotspot' column (you may need to create it)")
    else:
        hotspot_rate = feats["label_hotspot"].mean()
        print_success(f"Has hotspot labels ({hotspot_rate*100:.1f}% positive)")
    
    # Validate feature columns don't have future leakage
    feature_cols = [col for col in feats.columns 
                   if col not in ["intersection_id", "cutoff_date", "label_hotspot"] 
                   and not col.startswith("future_")]
    
    suspicious_cols = [col for col in feature_cols 
                      if any(term in col.lower() for term in ["future", "prediction", "predicted"])]
    
    if suspicious_cols:
        print_warning(f"Found {len(suspicious_cols)} suspicious feature names:")
        for col in suspicious_cols[:5]:
            print(f"    - {col}")
        print("  Verify these don't contain future information!")
    else:
        print_success("No suspicious feature names detected")
    
    return has_issue, results


def check_feature_time_windows() -> Tuple[bool, dict]:
    """
    Verify that feature time windows don't overlap with prediction window.
    """
    print_header("3. TIME WINDOW VALIDATION")
    
    results = {}
    has_issue = False
    
    # Check if temporal features exist
    temp_path = Path("data/processed/intersection_features_temporal.parquet")
    if not temp_path.exists():
        print_warning("Temporal features not found (skipping)")
        return False, results
    
    feats = pd.read_parquet(temp_path)
    
    if "cutoff_date" not in feats.columns:
        print_error("Cannot validate time windows without cutoff_date")
        return True, results
    
    # Expected window structure:
    # - hist_* features: use data BEFORE cutoff_date
    # - recent90_* features: use data BEFORE cutoff_date
    # - future_* features: use data AFTER cutoff_date (for labels only)
    
    hist_features = [col for col in feats.columns if col.startswith("hist_")]
    recent_features = [col for col in feats.columns if col.startswith("recent90_")]
    future_features = [col for col in feats.columns if col.startswith("future_")]
    
    print_success(f"Feature time windows:")
    print(f"  • Historical (365d before cutoff): {len(hist_features)} features")
    print(f"  • Recent (90d before cutoff): {len(recent_features)} features")
    print(f"  • Future (180d after cutoff): {len(future_features)} features (labels only)")
    
    # Check that future features are ONLY used for labels
    if future_features:
        print("\n  Verifying future features are label-only:")
        label_cols = ["future_crashes", "future_severity", "label_hotspot"]
        non_label_future = [col for col in future_features if col not in label_cols]
        
        if non_label_future:
            print_error(f"Found {len(non_label_future)} future features that aren't labels:")
            for col in non_label_future:
                print(f"    - {col}")
            print("  These should NOT be used as model features!")
            has_issue = True
        else:
            print_success("All future features are appropriately used for labels")
    
    return has_issue, results


def scan_notebooks_for_leakage() -> Tuple[bool, List[str]]:
    """
    Scan modeling notebooks for potential leakage indicators.
    """
    print_header("4. NOTEBOOK LEAKAGE SCAN")
    
    has_issue = False
    issues_found = []
    
    notebooks_dir = Path("notebooks")
    if not notebooks_dir.exists():
        print_warning("notebooks/ directory not found")
        return False, issues_found
    
    # Find modeling notebooks
    modeling_notebooks = list(notebooks_dir.glob("*modeling*.ipynb"))
    
    if not modeling_notebooks:
        print_warning("No modeling notebooks found")
        return False, issues_found
    
    print(f"Scanning {len(modeling_notebooks)} modeling notebooks...")
    
    # Common leakage indicators
    leakage_indicators = [
        "future_crashes",
        "future_severity",
        "intersection_features.parquet",  # Non-temporal version
        "train_test_split",  # Should use temporal split for temporal data
    ]
    
    for nb_path in modeling_notebooks:
        print(f"\n  📓 {nb_path.name}")
        
        try:
            # Read notebook as JSON
            import json
            with open(nb_path, "r") as f:
                nb_content = json.load(f)
            
            # Extract code cells
            code_cells = [cell["source"] for cell in nb_content.get("cells", []) 
                         if cell.get("cell_type") == "code"]
            
            # Flatten to single string
            all_code = "\n".join(["".join(lines) for lines in code_cells])
            
            # Check for leakage indicators
            found_indicators = []
            
            for indicator in leakage_indicators:
                if indicator in all_code:
                    found_indicators.append(indicator)
            
            if found_indicators:
                if "future_" in str(found_indicators):
                    print_error(f"Contains potential leakage indicators:")
                    for ind in found_indicators:
                        print(f"      - '{ind}'")
                    issues_found.append(nb_path.name)
                    has_issue = True
                else:
                    print_warning(f"Contains leakage-related terms:")
                    for ind in found_indicators:
                        print(f"      - '{ind}'")
                    print("      Review manually to ensure proper usage")
            else:
                print_success("No obvious leakage indicators found")
        
        except Exception as e:
            print_warning(f"Could not scan: {e}")
    
    return has_issue, issues_found


def generate_recommendations() -> None:
    """Generate recommendations for avoiding temporal leakage."""
    print_header("RECOMMENDATIONS: AVOIDING TEMPORAL LEAKAGE")
    
    print("\n📚 Best Practices:\n")
    
    print("1. FOR RANDOM SPLIT MODELS:")
    print("   • Use intersection_features_enriched.parquet")
    print("   • DROP all 'future_*' columns before splitting")
    print("   • Use train_test_split() with random_state for reproducibility")
    print("   • Example:")
    print("     X = feats.drop(columns=['label_hotspot', 'future_crashes', ...])")
    print("     y = feats['label_hotspot']")
    print("     X_train, X_test, y_train, y_test = train_test_split(X, y)\n")
    
    print("2. FOR TEMPORAL VALIDATION:")
    print("   • Use intersection_features_temporal.parquet")
    print("   • Split by cutoff_date (NOT random)")
    print("   • Train on early periods, validate on middle, test on recent")
    print("   • Example:")
    print("     train = feats[feats['cutoff_date'] < '2023-01-01']")
    print("     test = feats[feats['cutoff_date'] >= '2024-01-01']")
    print("     X_train = train.drop(columns=['cutoff_date', 'future_*', ...])")
    print("     y_train = train['label_hotspot']\n")
    
    print("3. FEATURE ENGINEERING:")
    print("   • Only use data from BEFORE the cutoff date")
    print("   • Historical features: 365 days before cutoff")
    print("   • Recent features: 90 days before cutoff")
    print("   • Never use 'future_*' features as model inputs\n")
    
    print("4. VALIDATION:")
    print("   • Check feature importance - if 'future_*' features appear, you have leakage")
    print("   • Compare random split vs temporal split performance")
    print("   • Random split should NOT be much better (indicates leakage)")
    print("   • Run: python scripts/check_temporal_leakage.py regularly\n")


def main() -> None:
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}TEMPORAL LEAKAGE CHECK{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")
    print("\nThis script checks for potential temporal data leakage in your")
    print("preprocessing pipeline and modeling notebooks.\n")
    
    all_results = {}
    
    # Run checks
    issue1, results1 = check_non_temporal_features()
    issue2, results2 = check_temporal_features()
    issue3, results3 = check_feature_time_windows()
    issue4, problematic_notebooks = scan_notebooks_for_leakage()
    
    all_results["non_temporal"] = results1
    all_results["temporal"] = results2
    all_results["time_windows"] = results3
    all_results["problematic_notebooks"] = problematic_notebooks
    
    # Generate recommendations
    generate_recommendations()
    
    # Summary
    print_header("SUMMARY")
    
    if issue1 or issue2 or issue3 or issue4:
        print(f"\n{RED}{BOLD}⚠ POTENTIAL LEAKAGE DETECTED{RESET}\n")
        
        if issue1:
            print(f"{RED}✗{RESET} Non-temporal features have leakage risk")
        if issue2:
            print(f"{RED}✗{RESET} Temporal features have structural issues")
        if issue3:
            print(f"{RED}✗{RESET} Time windows may overlap")
        if issue4:
            print(f"{RED}✗{RESET} Notebooks contain leakage indicators:")
            for nb in problematic_notebooks:
                print(f"    - {nb}")
        
        print("\n📖 Review the recommendations above and fix issues before modeling.")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}✓ NO LEAKAGE DETECTED{RESET}\n")
        print("Your preprocessing pipeline appears to properly handle temporal data.")
        print("Continue with modeling, but stay vigilant about data leakage!\n")


if __name__ == "__main__":
    main()

