# Chicago Traffic Crash Prediction & Inequality Analysis

A comprehensive urban computing project analyzing traffic crash inequality across Chicago's 77 community areas using machine learning, network science, and spatial statistics.

## Project Overview

This project predicts future crash hotspots at 19,200 Chicago intersections using 1.0 million historical crash records (2013-2025) and analyzes socioeconomic disparities in crash exposure.

### Key Findings

- **Predictive Model**: Gradient Boosting achieved PR-AUC of 0.772 and ROC-AUC of 0.946
- **Crash Inequality**: Low-income communities experience 80% higher severe injury rates (1.8% vs 1.0%)
- **Spatial Clustering**: Crashes exhibit significant positive autocorrelation (Moran's I = 0.161, p<0.001)
- **Actionable Insights**: Identified 1,875 persistent hotspots and 457 emerging hotspots for intervention

## Project Structure

```
project/
├── data/                          # Raw and processed datasets
│   ├── raw/                       # Original data files
│   └── processed/                 # Cleaned and processed data
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_exploratory_analysis.ipynb
│   ├── 03_modeling_temporal.ipynb
│   ├── 04_inequality_analysis.ipynb
│   ├── 05_spatial_analysis.ipynb
│   └── 06_geographic_visualizations.ipynb
├── scripts/                       # Python scripts for data processing
│   ├── build_features.py          # Feature engineering
│   └── build_temporal_features.py # Temporal feature generation
├── models/                        # Trained model files
├── results/                       # Analysis results and visualizations
└── report/                        # Final research paper (LaTeX)
```

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- LaTeX (for compiling the report)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/siddarthx07/chicago-crash-inequality.git
cd chicago-crash-inequality
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

### Required Python Packages

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
geopandas>=0.10.0
matplotlib>=3.4.0
seaborn>=0.11.0
networkx>=2.6.0
osmnx>=1.1.0
pysal>=2.5.0
esda>=2.4.0
libpysal>=4.5.0
```

## Data Sources

1. **Chicago Traffic Crashes**: [Chicago Data Portal](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/)
2. **Chicago Community Areas**: [Boundaries Data](https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas/)
3. **OpenStreetMap**: Downloaded via OSMnx Python library
4. **U.S. Census ACS**: [Census Data](https://data.census.gov/)

## Methodology

### 1. Data Preprocessing
- Spatial matching of crashes to intersections (88% match rate)
- Temporal feature engineering with rolling windows
- Network centrality computation (degree, betweenness, closeness)

### 2. Machine Learning Models
- **Logistic Regression** (baseline)
- **Random Forest** (ensemble)
- **Gradient Boosting** (best performer)

All models use:
- Chronological train/validation/test splits (70/15/15)
- Temporal validation to prevent data leakage
- Class balancing for imbalanced hotspot labels (11.9% positive class)

### 3. Spatial Analysis
- **Global Moran's I**: Tests for overall spatial clustering
- **Local Moran's I (LISA)**: Identifies specific cluster types (HH, LL, HL, LH)
- 964 High-High cluster intersections identified

### 4. Inequality Analysis
- Crash rates aggregated by community area and income quartile
- Statistical testing: ANOVA, Pearson correlation, T-tests
- 80% higher severe injury rates in low-income communities (1.8% vs 1.0%)

## Running the Analysis

### Step 1: Data Preprocessing
```bash
python scripts/build_temporal_features.py
```

### Step 2: Run Analysis Notebooks
```bash
jupyter notebook
```

Open and run notebooks in order:
1. `01_exploratory_analysis.ipynb` - Data exploration
2. `03_modeling_temporal.ipynb` - Model training and evaluation
3. `04_inequality_analysis.ipynb` - Socioeconomic disparity analysis
4. `05_spatial_analysis.ipynb` - Moran's I and LISA clustering
5. `06_geographic_visualizations.ipynb` - Map generation

### Step 3: Compile the Report
```bash
cd report/
pdflatex final_report.tex
pdflatex final_report.tex  # Run twice for references
```

Or use your preferred LaTeX editor (Overleaf, TeXShop, etc.)

## Key Results

### Model Performance
| Model | PR-AUC | ROC-AUC | F1-Score |
|-------|--------|---------|----------|
| Gradient Boosting | **0.772** | **0.946** | **0.691** |
| Logistic Regression | 0.762 | 0.945 | 0.684 |
| Random Forest | 0.754 | 0.942 | 0.675 |

### Feature Importance
- **Historical crash count**: 90.5% importance
- **Recent trends**: 3.4%
- **Network centrality**: 2.3%
- **Demographics**: 1.1%

### Inequality by Income Quartile
| Quartile | Median Income | Crash Rate (per 1k) | Severe Injury Rate |
|----------|---------------|---------------------|-------------------|
| Q1 (Lowest) | $38,576 | 423.9 | 1.8% |
| Q4 (Highest) | $116,002 | 407.7 | 1.0% |

**Disparity**: 80% higher severe injury rates in low-income communities

## Visualizations

The project generates multiple visualizations:
- Temporal window diagrams
- ROC and Precision-Recall curves
- Feature importance plots
- Inequality analysis charts
- LISA cluster maps
- Hotspot agreement maps
- Community-level choropleth maps

## Policy Recommendations

1. **Immediate Priorities**: Deploy engineering improvements at 1,875 persistent hotspots
2. **Proactive Measures**: Monitor 457 emerging hotspots with enforcement and temporary calming
3. **Equity Focus**: Prioritize severe-injury reduction in low-income communities

## Author

**Siddarth Bandi**

## Citation

If you use this work, please cite:

```bibtex
@article{chicago-crash-2025,
  title={Traffic Safety: Analyzing Crash Inequality Across Chicago Neighborhoods Using Machine Learning and Network Science},
  author={Bandi, Siddarth},
  year={2025},
  institution={Virginia Tech}
}
```

## License

This project is open source and available for academic and non-commercial use.

## Acknowledgements

- City of Chicago for providing open crash data
- OpenStreetMap contributors for road network data
- U.S. Census Bureau for demographic data
- Virginia Tech for computational resources
- Professor Naren Ramakrishnan for guidance

## Contact

For questions or collaboration opportunities:
- Siddarth Bandi: siddarth24@vt.edu


---

**Project Repository**: [https://github.com/siddarthx07/Crash-Inequality-Chicago](https://github.com/siddarthx07/Crash-Inequality-Chicago)


