"""
Fix the prediction issue in geographic visualizations.

The problem: Only 20 predictions were saved, causing all intersections 
to be marked as hotspots (threshold = 0).

Solution: Generate predictions for ALL intersections using the trained model.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Paths
DATA_DIR = Path('data/processed')
RESULTS_DIR = Path('results')
MODEL_DIR = Path('models')

print("Loading data...")
feats = pd.read_parquet(DATA_DIR / 'intersection_features_enriched.parquet')
print(f"Loaded {len(feats):,} intersections")

# Load the trained model
try:
    with open(MODEL_DIR / 'temporal_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Loaded trained model: {type(model).__name__}")
except FileNotFoundError:
    print("❌ Model file not found at models/temporal_model.pkl")
    print("   You need to run the temporal modeling notebook first!")
    exit(1)

# Prepare features for prediction
feature_cols = [c for c in feats.columns if c not in [
    'intersection_id', 'label_hotspot', 'future_crashes', 'future_severity',
    'community_name', 'node_id'
]]

X = feats[feature_cols].fillna(0)

print(f"\nGenerating predictions for {len(feats):,} intersections...")
print(f"Using {len(feature_cols)} features")

# Get predictions
pred_proba = model.predict_proba(X)[:, 1]  # Probability of being a hotspot

# Add to dataframe
feats['predicted_probability'] = pred_proba

# Use the same threshold as the model (top 10% or best F1 threshold)
# Let's use top 10% to match the current hotspot definition
threshold = np.percentile(pred_proba, 90)
feats['predicted_hotspot'] = (pred_proba >= threshold).astype(int)

print(f"\n✅ Predictions generated!")
print(f"   Threshold: {threshold:.4f}")
print(f"   Predicted hotspots: {feats['predicted_hotspot'].sum():,} ({feats['predicted_hotspot'].mean()*100:.1f}%)")
print(f"   Current hotspots: {feats['label_hotspot'].sum():,} ({feats['label_hotspot'].mean()*100:.1f}%)")

# Save full predictions
output = feats[['intersection_id', 'predicted_probability', 'predicted_hotspot']].copy()
output.to_csv(RESULTS_DIR / 'all_predictions.csv', index=False)
print(f"\n✅ Saved: results/all_predictions.csv")

# Also save top predictions for reference
top_predictions = feats.nlargest(100, 'predicted_probability')[[
    'intersection_id', 'predicted_probability', 'hist_crashes', 
    'label_hotspot', 'community_name'
]]
top_predictions.to_csv(RESULTS_DIR / 'top_100_predicted_hotspots.csv', index=False)
print(f"✅ Saved: results/top_100_predicted_hotspots.csv")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Re-run the geographic visualization notebook")
print("2. It will now load predictions from results/all_predictions.csv")
print("3. The maps should show realistic predictions (not all intersections)")
