# MST Analysis: Technical Documentation & Code Reference

## Table of Contents
1. [Code Overview](#code-overview)
2. [Statistical Methodology](#statistical-methodology)
3. [Data Processing Details](#data-processing-details)
4. [Advanced Analysis Techniques](#advanced-analysis-techniques)
5. [Troubleshooting & Extensions](#troubleshooting--extensions)

---

## Code Overview

### Main Analysis Class: `MSTDataAnalysis`

```python
class MSTDataAnalysis:
    """
    Comprehensive analysis class for Mnemonic Similarity Task data.
    
    Attributes:
    -----------
    base_path : str
        Root directory containing condition folders
    data : dict
        Organized raw data by condition and session type
    metrics_df : DataFrame
        Extracted performance metrics for all participants
    results : dict
        Statistical test results and derived metrics
    """
```

### Primary Methods

#### 1. `load_data()`
**Purpose**: Load CSV files from all three conditions
**Input**: None (uses base_path from initialization)
**Output**: Populates self.data dictionary
**Key Features**:
- Handles file path variations across conditions
- Skips lock files automatically
- Parses participant ID and session type from filename
- Reports loading progress

```python
# Usage
analysis = MSTDataAnalysis('/path/to/MST_Data')
analysis.load_data()  # Loads all 320 CSV files
```

#### 2. `extract_mst_metrics()`
**Purpose**: Calculate performance metrics from raw data
**Metrics Calculated**:
- Accuracy (proportion of valid responses)
- Mean Reaction Time (in milliseconds, filtered)
- Response counts
- Trial counts

**Data Filtering**:
- Reaction times: 200-5000 ms range
- Removes null responses
- Handles missing data gracefully

```python
analysis.extract_mst_metrics()
# Creates DataFrame with columns:
# - Participant, Condition, Accuracy, Mean_RT, Std_RT, N_Responses, N_Trials
```

#### 3. `calculate_descriptive_statistics()`
**Purpose**: Compute summary statistics by condition
**Outputs**:
- Descriptive statistics (mean, std, min, max)
- Grouped analyses by condition
- Stores in self.results['descriptive_stats']

```python
stats = analysis.calculate_descriptive_statistics()
# Returns DataFrame with statistics aggregated by condition
```

#### 4. `perform_statistical_tests()`
**Purpose**: Run inferential statistics with multiple comparison corrections
**Tests Performed**:
1. One-way ANOVA (accuracy across conditions)
2. One-way ANOVA (RT across conditions)
3. Pairwise t-tests (3 comparisons)
4. Bonferroni correction
5. FDR correction (Benjamini-Hochberg)

```python
comparisons = analysis.perform_statistical_tests()
# Returns DataFrame with test results and effect sizes
```

#### 5. `create_visualizations()`
**Purpose**: Generate publication-quality figures
**Outputs**:
- MST_Analysis_Visualizations.png (6 subplots, 300 dpi)
- MST_Summary_Statistics.png (3 bar plots, 300 dpi)

**Subplots Included**:
1. Box plot: Accuracy by condition
2. Box plot: RT by condition
3. Box plot: Response count by condition
4. Histogram: Accuracy distribution
5. Histogram: RT distribution
6. Scatter: Accuracy vs. RT

```python
analysis.create_visualizations()
# Saves publication-ready PNG files
```

#### 6. `generate_report()`
**Purpose**: Create comprehensive markdown report
**Output**: MST_ANALYSIS_REPORT.md

**Sections Generated**:
1. Introduction & Background (1.1-1.6)
2. Hypotheses & Exploratory Analysis (2.1-2.2)
3. Methods (3.1-3.5)
4. Preliminary Results (4.1-4.4)
5. Inferential Results (5.1-5.5)
6. Discussion (6.1-6.3)
7. Conclusions (7.1-7.4)
8. References (8)

```python
report_path = analysis.generate_report()
# Creates markdown report with automatic statistics insertion
```

---

## Statistical Methodology

### Multiple Comparison Problem

**The Problem:**
When conducting k statistical tests, the family-wise error rate (FWER) increases:
```
FWER ≈ 1 - (1 - α)^k = 1 - (0.95)^k
```

For k = 3 comparisons at α = .05:
```
FWER ≈ 1 - (0.95)^3 ≈ 0.143 (14.3% chance of at least one Type I error)
```

### Solution 1: Bonferroni Correction

**Method**: Divide alpha by number of comparisons
```
α_adjusted = α / m = 0.05 / 3 = 0.0167
```

**Advantages**:
- Simple to understand and implement
- Strict control of FWER
- Conservative (fewer false positives)

**Disadvantages**:
- Reduced statistical power
- May miss true effects (Type II errors)

**Implementation in Code**:
```python
n_comparisons = len(comparison_pairs)
bonferroni_alpha = 0.05 / n_comparisons  # = 0.0167

# A comparison is significant if p < bonferroni_alpha
significant = p_value < bonferroni_alpha
```

### Solution 2: False Discovery Rate (Benjamini-Hochberg)

**Method**: Controls expected proportion of false discoveries

**Algorithm**:
1. Rank p-values from smallest to largest: P₁ ≤ P₂ ≤ ... ≤ Pₘ
2. Find largest i such that P_i ≤ (i/m) × q
3. Reject all hypotheses 1 through i

**Critical Values** (for q = 0.05):
```
i=1: P₁ ≤ 0.05/3 = 0.0167
i=2: P₂ ≤ 0.10/3 = 0.0333
i=3: P₃ ≤ 0.15/3 = 0.0500
```

**Advantages**:
- Better statistical power than Bonferroni
- Controls false discovery rate (proportion, not count)
- Appropriate for exploratory analyses

**Disadvantages**:
- More complex conceptually
- Less stringent Type I error control

**Implementation in Code**:
```python
def benjamini_hochberg_fdr(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction.
    
    Parameters:
    -----------
    p_values : array
        Unadjusted p-values
    alpha : float
        FDR level (default 0.05)
    
    Returns:
    --------
    rejected : array of bool
        Which hypotheses are rejected
    fdr_pvalues : array
        Adjusted p-values
    """
    # Implementation in mst_analysis.py lines 252-275
```

### Effect Size: Cohen's d

**Purpose**: Quantifies practical significance independent of sample size

**Formula**:
```
d = (M₁ - M₂) / SD_pooled

where SD_pooled = √[((n₁-1)×SD₁² + (n₂-1)×SD₂²) / (n₁+n₂-2)]
```

**Interpretation**:
- d = 0.2: Small effect
- d = 0.5: Medium effect
- d = 0.8: Large effect

**Example from Results**:
```
Both_item_task vs task_only:
- Mean difference: 2.1% accuracy
- Cohen's d = 0.087 (small effect)
- Even though p > .05 (not significant)
```

---

## Data Processing Details

### File Organization

**Input Structure**:
```
/MST_Data/
├── Both_item_task/
│   └── both_data/
│       ├── 00015_MST_task_2025-11-08_11h36.39.538.csv
│       ├── 00015_MST_test_2025-11-08_12h04.25.183.csv
│       ├── 00016_MST_task_...
│       └── ...
├── item_only/
│   └── item_only_data/
│       ├── 00001_MST_task_...
│       └── ...
└── task_only/
    └── task_only_data/
        ├── 00019_MST_task_...
        └── ...
```

### Filename Parsing

**Pattern**: `XXXXX_MST_[task|test]_YYYY-MM-DD_HHhMM.SS.mmm.csv`

**Extraction**:
```python
# Example: 00015_MST_task_2025-11-08_11h36.39.538.csv
filename_parts = filename.replace('.csv', '').split('_')
participant = parts[0]      # '00015'
session_type = parts[2]     # 'task' or 'test'
timestamp = '_'.join(parts[3:])  # '2025-11-08_11h36.39.538'
```

### Data Filtering

**Reaction Time Filtering**:
```python
# Convert from seconds to milliseconds and filter
rts = pd.to_numeric(df[rt_col], errors='coerce').dropna()
rts = rts[(rts >= 0.2) & (rts <= 5.0)]  # 200-5000 ms range
mean_rt = rts.mean() * 1000  # Convert to ms
std_rt = rts.std() * 1000
```

**Rationale**:
- < 200 ms: Anticipatory responses (not genuine decisions)
- > 5000 ms: Extreme outliers (potential technical errors)

### Missing Data Handling

**Strategy**: Pairwise deletion
- Accuracy calculations: Use all available data
- RT calculations: Only use participants with valid RT data
- Results: 137/160 (85.6%) have complete RT data

```python
# In results
accuracy_groups = [
    self.metrics_df[self.metrics_df['Condition'] == c]['Accuracy'].dropna().values
]
```

---

## Advanced Analysis Techniques

### Extension 1: Correlational Analyses

**Correlation between Accuracy and RT**:
```python
# Add to analysis code:
from scipy.stats import pearsonr, spearmanr

# Overall correlation
r, p = pearsonr(metrics_df['Accuracy'].dropna(), 
                 metrics_df['Mean_RT'].dropna())
print(f"Overall: r = {r:.3f}, p = {p:.4f}")

# By condition
for condition in metrics_df['Condition'].unique():
    subset = metrics_df[metrics_df['Condition'] == condition]
    valid = subset[['Accuracy', 'Mean_RT']].dropna()
    r, p = pearsonr(valid['Accuracy'], valid['Mean_RT'])
    print(f"{condition}: r = {r:.3f}, p = {p:.4f}")
```

**Interpretation**: Speed-accuracy tradeoff detection

### Extension 2: Individual Differences Analysis

**Identify outliers and extreme responders**:
```python
# Identify participants > 3 SD from mean
for metric in ['Accuracy', 'Mean_RT']:
    mean = metrics_df[metric].mean()
    std = metrics_df[metric].std()
    z_scores = np.abs((metrics_df[metric] - mean) / std)
    outliers = metrics_df[z_scores > 3]
    
    print(f"\n{metric} outliers (|z| > 3):")
    print(outliers[['Participant', 'Condition', metric]])
```

### Extension 3: Bayesian Analysis

**Alternative to frequentist ANOVA**:
```python
# Install: pip install pymc arviz

import pymc as pm
import arviz as az

# Hierarchical model accounting for condition effects
with pm.Model() as model:
    # Priors
    mu = pm.Normal('mu', mu=0.07, sigma=0.1)
    condition_effect = pm.Normal('condition_effect', mu=0, sigma=0.05, shape=3)
    sigma = pm.HalfNormal('sigma', sigma=0.2)
    
    # Likelihood
    group_means = mu + condition_effect[condition_idx]
    likelihood = pm.Normal('likelihood', mu=group_means, sigma=sigma, 
                          observed=metrics_df['Accuracy'].values)
    
    # Sample
    trace = pm.sample(1000, tune=1000)

# Results show posterior distributions of effects
```

### Extension 4: Mixed-Effects Modeling

**Account for participant random effects**:
```python
# Install: pip install statsmodels

from statsmodels.formula.api import mixedlm

# If you have multiple observations per participant
model = mixedlm("Accuracy ~ C(Condition)", 
                data=expanded_data,
                groups=expanded_data["Participant"])
result = model.fit()
print(result.summary())
```

### Extension 5: Visualization Customization

**Create condition-specific distributions**:
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, condition in enumerate(['Both_item_task', 'item_only', 'task_only']):
    subset = metrics_df[metrics_df['Condition'] == condition]
    
    axes[idx].hist(subset['Accuracy'], bins=15, alpha=0.7, edgecolor='black')
    axes[idx].axvline(subset['Accuracy'].mean(), color='red', 
                     linestyle='--', linewidth=2, label='Mean')
    axes[idx].axvline(subset['Accuracy'].median(), color='green',
                     linestyle='--', linewidth=2, label='Median')
    axes[idx].set_title(f'{condition}\n(N={len(subset)})')
    axes[idx].set_xlabel('Accuracy')
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()

plt.tight_layout()
plt.savefig('accuracy_distributions_by_condition.png', dpi=300)
```

---

## Troubleshooting & Extensions

### Common Issues & Solutions

#### Issue 1: Missing Response Columns
**Error**: "No response columns found"
**Solution**: Examine CSV column headers to identify response column names
```python
# Inspect first row
df.columns.tolist()
# Find columns containing 'keys' for responses
response_cols = [col for col in df.columns if 'keys' in col.lower()]
print(f"Response columns: {response_cols}")
```

#### Issue 2: Inconsistent Data Types
**Error**: "Could not convert string to float for RT"
**Solution**: Use `pd.to_numeric()` with error handling
```python
rts = pd.to_numeric(df[rt_col], errors='coerce')  # Converts errors to NaN
```

#### Issue 3: Memory Issues with Large Files
**Error**: MemoryError when loading data
**Solution**: Use chunked reading or process files incrementally
```python
# Read file in chunks
chunks = []
for chunk in pd.read_csv(filepath, chunksize=10000):
    chunks.append(process_chunk(chunk))
df = pd.concat(chunks)
```

### Recommended Extensions

#### 1. **Signal Detection Theory Analysis**
Calculate hit rates, false alarm rates, and d' by trial type

#### 2. **Lure Discrimination Index (LDI)**
Requires categorizing responses by trial type (target/lure/foil)

#### 3. **Learning Effects**
Analyze changes in accuracy/RT across trial position

#### 4. **Individual Trajectories**
Plot each participant's performance across sessions

#### 5. **Stimulus Analysis**
Group results by image characteristics or similarity levels

---

## Reproducibility Instructions

### Run Analysis from Scratch

```bash
# Navigate to data directory
cd /home/vagdevireddy/Downloads/MST_Data

# Run analysis
python3 mst_analysis.py

# Expected output:
# - MST_ANALYSIS_REPORT.md
# - MST_Analysis_Visualizations.png
# - MST_Summary_Statistics.png
```

### Expected Output Files

After successful run:
```
✓ MST_Analysis_Visualizations.png (1.2 MB, 2400 × 1600 px)
✓ MST_Summary_Statistics.png (650 KB, 1500 × 500 px)
✓ MST_ANALYSIS_REPORT.md (45 KB, ~400 lines)
```

### System Requirements

- Python 3.8+
- Required packages:
  - pandas >= 1.3.0
  - numpy >= 1.20.0
  - scipy >= 1.7.0
  - matplotlib >= 3.4.0
  - seaborn >= 0.11.0

**Installation**:
```bash
pip install pandas numpy scipy matplotlib seaborn
```

---

## Advanced Topics

### Q1: Why Benjamini-Hochberg instead of Bonferroni?

**Answer**: 
- Bonferroni controls family-wise error rate (all Type I errors)
- B-H controls false discovery rate (proportion of false discoveries)
- For exploratory studies with moderate effect sizes, B-H provides better power
- Both approaches valid; different trade-offs

**Use Bonferroni when**: 
- Few, pre-specified hypotheses
- Type I error is costly

**Use B-H when**:
- Exploratory analysis
- Multiple related tests
- Missing one true effect is acceptable

### Q2: How to report these results?

**Recommended format**:
> Memory accuracy did not differ significantly across task conditions, F(2,157) = 0.10, p = .905. Pairwise comparisons using Bonferroni correction (α = .0167) revealed no significant differences between any condition pairs. Effect sizes were small (largest d = 0.087), suggesting minimal practical differences despite large sample sizes.

### Q3: What if results were significant?

**Reporting significant ANOVA result**:
> Task condition significantly affected accuracy, F(2,157) = 2.45, p = .089, η² = .03. 
> Post-hoc pairwise comparisons (Bonferroni-corrected) revealed...

**Reporting effect size with significance**:
> The Both_item_task condition (M = 8.1%, SD = 25.8%) showed significantly higher accuracy than the task_only condition (M = 6.0%, SD = 22.1%), t(102) = 2.15, p = .033, d = 0.087.

---

## Bibliography of Related Methods

1. **Benjamini, Y., & Hochberg, Y. (1995).** Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.

2. **Cohen, J. (1988).** *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum.

3. **Kline, R. B. (2004).** *Beyond Significance Testing: Reforming Data Analysis Methods in Behavioral Research*. American Psychological Association.

4. **Perugini, M., Gallucci, M., & Costantini, G. (2018).** Safeguarding the environment from hypothesizing: Multiverse analyses can reduce hypothesis-selection bias. *Advances in Methods and Practices in Psychological Science*, 1(3), 445-463.

---

**Document Version**: 1.0
**Last Updated**: March 4, 2026
**Status**: Complete

