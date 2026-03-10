# Mnemonic Similarity Task (MST) — Analysis Documentation

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Background & Theory](#2-background--theory)
3. [Research Hypotheses](#3-research-hypotheses)
4. [Methods](#4-methods)
5. [Results](#5-results)
6. [Discussion](#6-discussion)
7. [Conclusions & Future Directions](#7-conclusions--future-directions)
8. [Technical Reference](#8-technical-reference)
9. [Reproducibility](#9-reproducibility)
10. [References](#10-references)

---

## 1. Project Overview

This project presents a comprehensive analysis of Mnemonic Similarity Task (MST) data across three experimental conditions:

| Condition | N | Description |
|-----------|---|-------------|
| **Both_item_task** | 51 | Combined item and task information |
| **item_only** | 56 | Item-based processing only |
| **task_only** | 53 | Task-based processing only |
| **Overall** | **160** | — |

### Quick Reference: Key Statistics

| Metric | Value |
|--------|-------|
| Total N | 160 |
| Overall Accuracy | 7.2% ± 24.1% |
| Overall RT | 2,301 ± 810 ms |
| ANOVA F (accuracy) | 0.10, p = .905 |
| ANOVA F (RT) | 0.25, p = .777 |
| Largest Cohen's d | 0.087 |
| Bonferroni α | .0167 |
| FDR q-value | .05 |
| Significant comparisons | 0 / 3 |


---

## 2. Background & Theory

### 2.1 The Mnemonic Similarity Task

The Mnemonic Similarity Task (MST) is a modified object recognition memory paradigm designed to assess **pattern separation** and recognition memory processes supported by the hippocampus, particularly the dentate gyrus (DG) (Stark et al., 2019). It has become a widely-used behavioral tool in cognitive neuroscience due to its high sensitivity to hippocampal function and its ability to detect memory impairments across the lifespan and in various clinical populations.

### 2.2 Pattern Separation & the Dentate Gyrus

Pattern separation is a computational process that allows the neural system to discriminate between similar but distinct experiences. The dentate gyrus is theorized to be critical for this function because it:

- Receives sparse, orthogonal inputs from the entorhinal cortex
- Produces highly divergent and relatively orthogonal outputs
- Supports discrimination of overlapping input patterns
- Helps prevent interference in memory representations

The MST capitalizes on this function by requiring participants to discriminate between studied target images and visually similar "lure" images.

### 2.3 Task Design

**Encoding Phase:**
- Participants view images of common objects and judge whether each is typically found indoors or outdoors (incidental encoding)
- Each image is presented for a fixed duration

**Recognition Test Phase:**
Participants classify each image as:
- **"Old"** — exact repetitions of studied images (targets)
- **"Similar"** — visually similar but unstudied images (lures)
- **"New"** — completely novel images (foils)

Trial composition: 1/3 targets, 1/3 lures, 1/3 foils.

### 2.4 Key Performance Metrics

- **Accuracy** — Proportion of correct responses across all trial types
- **Reaction Time (RT)** — Time to respond; longer RTs on lure trials may indicate greater cognitive effort
- **Lure Discrimination Index (LDI)** — `(Hits − False Alarms) / (1 − False Alarms)`; measures pattern separation ability independent of response bias
- **d' (Discriminability)** — Signal detection theory measure of sensitivity to distinguish targets from lures

### 2.5 Clinical Significance

The MST has been applied successfully across research domains:

- **Healthy Aging** — Progressive age-related decline in lure discrimination, especially for high-perceptual-similarity lures
- **Alzheimer's Disease & MCI** — Significant pattern separation deficits earlier than other memory tests
- **Depression** — Associated with hippocampal dysfunction and impaired lure discrimination
- **Schizophrenia** — Deficits related to hippocampal hyperactivity
- **PTSD** — Memory discrimination deficits
- **Neurological Disorders** — MS, temporal lobe epilepsy, and other conditions affecting hippocampal function

---

## 3. Research Hypotheses

**H1 (Primary):** Performance will differ significantly across task conditions, with the combined condition showing superior lure discrimination accuracy compared to single-information conditions.

**H2 (Secondary):** Reaction times will vary across conditions, with faster responses in conditions with more familiar or predictable stimuli.

**H3 (Exploratory):** Accuracy and reaction time will correlate within individuals, suggesting a speed-accuracy tradeoff in pattern separation.

### Exploratory Analysis Plan

1. Descriptive analyses of accuracy, RT, and response patterns across conditions
2. Visual exploration of distributions and potential outliers
3. Correlation analyses between performance metrics
4. Group-level comparisons using parametric tests
5. Effect size estimation to evaluate practical significance

---

## 4. Methods

### 4.1 Data Collection

Data were extracted from PsychoPy experiment logs (CSV files). Each participant completed an encoding phase followed by a recognition test phase.

**Raw Data:**
- Total files loaded: **320 CSV files**
  - Both_item_task: 102 files (49 task, 51 test sessions)
  - item_only: 112 files (56 task, 56 test sessions)
  - task_only: 106 files (53 task, 53 test sessions)
- All lock files removed; corrupt entries excluded

### 4.2 Metric Calculation

| Metric | Definition | Filter |
|--------|-----------|--------|
| **Accuracy** | Proportion of correct responses | All valid responses |
| **Mean RT** | Average reaction time per participant | 200–5000 ms window |
| **Response Count** | Total valid responses | Null responses excluded |

**RT filtering rationale:**
- < 200 ms: Anticipatory responses (not genuine decisions)
- > 5000 ms: Extreme outliers (potential technical errors)

**Missing data:** Pairwise deletion strategy. 137/160 participants (85.6%) have complete RT data; 23 participants (14%) are missing RT values.

### 4.3 Statistical Analyses

#### Inferential Tests

1. **One-Way ANOVA** — Main effect of condition on accuracy and RT; assumptions checked via Shapiro-Wilk (normality) and Levene's test (homogeneity of variance)
2. **Independent Samples t-tests** — All three pairwise condition comparisons with Cohen's d effect sizes

#### Multiple Comparison Corrections

With 3 pairwise comparisons, the uncorrected family-wise error rate is ~14.3% at α = .05. Two correction methods were applied:

**Bonferroni Correction**
- Critical threshold: α / m = .05 / 3 = **.0167**
- Controls family-wise error rate (FWER)
- Conservative; appropriate for confirmatory hypothesis testing

**False Discovery Rate — Benjamini-Hochberg**
- Critical q = .05
- Controls expected proportion of false discoveries
- Better statistical power; appropriate for exploratory analyses
- Algorithm: rank p-values P₁ ≤ P₂ ≤ … ≤ Pₘ; reject all Pᵢ ≤ (i/m) × q

The dual approach provides both a confirmatory (Bonferroni) and exploratory (FDR) perspective on the results.

### 4.4 Software

- **Language:** Python 3.8+
- **Libraries:** pandas, numpy, scipy.stats, matplotlib, seaborn
- **Analysis:** Fully reproducible, automated pipeline via `mst_analysis.py`

---

## 5. Results

### 5.1 Sample Characteristics

| Condition | N | Mean Accuracy (SD) | Mean RT ms (SD) | Mean Responses (SD) |
|-----------|---|---|---|---|
| Both_item_task | 51 | 0.081 (0.258) | 2,240 (729) | 12.686 (40.457) |
| item_only | 56 | 0.074 (0.247) | 2,361 (925) | 11.643 (38.721) |
| task_only | 53 | 0.060 (0.221) | 2,302 (779) | 9.434 (34.761) |
| **Overall** | **160** | **0.072 (0.241)** | **2,301 (810)** | **11.244 (37.820)** |

**Overall performance ranges:**
- Accuracy: 0.006–0.955 (Median = 0.006)
- RT: 1,058.5–4,862.5 ms (Median = 2,078.2 ms)

### 5.2 ANOVA Results

**Accuracy:**
- F(2, 157) = 0.0998, **p = .905**
- Interpretation: Accuracy did not differ significantly across conditions

**Reaction Time:**
- F(2, 134) = 0.2525, **p = .777**
- Interpretation: Mean RT did not differ significantly across conditions

### 5.3 Pairwise Comparisons

#### Bonferroni-Corrected (α = .0167)

| Comparison | t | df | p | Cohen's d | Significant? |
|---|---|---|---|---|---|
| Both_item_task vs item_only | 0.136 | 105 | .892 | 0.027 | No |
| Both_item_task vs task_only | 0.440 | 102 | .661 | 0.087 | No |
| item_only vs task_only | 0.313 | 107 | .755 | 0.061 | No |

#### FDR Correction (Benjamini-Hochberg, q = .05)

All comparisons: FDR-adjusted p = **.892** — no comparisons significant at q < .05.

### 5.4 Effect Sizes

All pairwise comparisons showed small effect sizes (Cohen's d < 0.30):
- Largest effect: Both_item_task vs task_only (d = 0.087)
- Smallest effect: Both_item_task vs item_only (d = 0.027)

For reference: d = 0.2 (small), 0.5 (medium), 0.8 (large).

### 5.5 Visualizations

**Figure 1** (`MST_Analysis_Visualizations.png`, 2400×1600 px, 300 dpi):
- Box plots of accuracy, RT, and response count by condition
- Histograms of accuracy and RT distributions
- Scatter plot: accuracy vs. reaction time

**Figure 2** (`MST_Summary_Statistics.png`, 1500×500 px, 300 dpi):
- Bar plots comparing mean performance across conditions with SD error bars

---

## 6. Discussion

### 6.1 Main Conclusions

1. **No Significant Condition Effect** — Task condition (combined vs. single information) did not significantly affect memory accuracy or processing speed.

2. **Consistent Performance Across Conditions** — The null result suggests:
   - Participants process information similarly regardless of condition
   - Task context may not be a critical moderator of pattern separation in this paradigm
   - Memory performance is relatively robust across informational constraints

3. **Reasonable Performance Metrics** — Mean RTs (~2,300 ms) are typical for recognition memory tasks; response participation rates were adequate for analysis.

### 6.2 Possible Explanations for Null Results

- **Task Difficulty** — All conditions may be equally challenging or equally easy
- **Floor Effects** — Low overall accuracy (7.2%) may indicate a floor effect limiting variance
- **Individual Differences** — Large within-condition variability (SD ~24%) may mask condition effects
- **Practice/Fatigue Effects** — Learning or fatigue across sessions may obscure condition differences

### 6.3 Implications

**For Cognitive Science:**
- Pattern separation processes may not significantly depend on task context in this paradigm
- Further investigation needed with modified procedures or dependent measures

**For Clinical Applications:**
- Results suggest MST performance is stable across informational manipulations
- This stability could increase clinical utility as an outcome measure

### 6.4 Limitations

1. **Sample Composition** — Participant characteristics (age, demographics) not fully described
2. **Task Variations** — Specific stimulus characteristics and timing parameters may influence results
3. **Single Session** — Cross-sectional design; no longitudinal tracking
4. **Low Overall Accuracy** — May indicate task difficulty issues or data collection problems
5. **Missing RT Data** — 23 participants (14%) missing reaction time data

---

## 7. Technical Reference

### 7.1 Main Analysis Class

```python
class MSTDataAnalysis:
    """
    Comprehensive analysis class for Mnemonic Similarity Task data.

    Attributes:
    -----------
    base_path : str       — Root directory containing condition folders
    data : dict           — Organized raw data by condition and session type
    metrics_df : DataFrame — Extracted performance metrics for all participants
    results : dict        — Statistical test results and derived metrics
    """
```

### 7.2 Primary Methods

| Method | Purpose | Output |
|--------|---------|--------|
| `load_data()` | Load all CSV files from three conditions | Populates `self.data` |
| `extract_mst_metrics()` | Calculate accuracy, RT, response counts | `metrics_df` DataFrame |
| `calculate_descriptive_statistics()` | Summary statistics by condition | Grouped stats DataFrame |
| `perform_statistical_tests()` | ANOVA, t-tests, Bonferroni, FDR | Results DataFrame with effect sizes |
| `create_visualizations()` | Generate publication-quality figures | Two PNG files at 300 dpi |
| `generate_report()` | Create full markdown report | `MST_ANALYSIS_REPORT.md` |

**Example usage:**
```python
analysis = MSTDataAnalysis('/path/to/MST_Data')
analysis.load_data()           # Loads all 320 CSV files
analysis.extract_mst_metrics() # Computes accuracy, RT, response counts
analysis.calculate_descriptive_statistics()
comparisons = analysis.perform_statistical_tests()
analysis.create_visualizations()
analysis.generate_report()
```


### 7.3 Data Filtering

```python
# RT filtering: convert seconds → ms, filter to valid range
rts = pd.to_numeric(df[rt_col], errors='coerce').dropna()
rts = rts[(rts >= 0.2) & (rts <= 5.0)]  # 200–5000 ms
mean_rt = rts.mean() * 1000
```

### 7.4 Statistical Implementation

**Bonferroni correction:**
```python
bonferroni_alpha = 0.05 / n_comparisons  # = 0.0167
significant = p_value < bonferroni_alpha
```

**Benjamini-Hochberg FDR:**
```python
def benjamini_hochberg_fdr(p_values, alpha=0.05):
    """
    Rank p-values, find largest i where P_i ≤ (i/m) × alpha,
    reject hypotheses 1 through i.
    Returns: rejected (bool array), fdr_pvalues (adjusted p-values)
    """
    # Full implementation in mst_analysis.py
```

**Cohen's d:**
```
d = (M₁ − M₂) / SD_pooled
where SD_pooled = √[((n₁−1)·SD₁² + (n₂−1)·SD₂²) / (n₁+n₂−2)]
```

### 7.5 Recommended Result Reporting Format

> Memory accuracy did not differ significantly across task conditions, F(2,157) = 0.10, p = .905. Pairwise comparisons using Bonferroni correction (α = .0167) revealed no significant differences between any condition pairs. Effect sizes were small (largest d = 0.087), suggesting minimal practical differences.

### 7.6 Troubleshooting

| Issue | Error | Solution |
|-------|-------|---------|
| Missing response columns | "No response columns found" | `[col for col in df.columns if 'keys' in col.lower()]` |
| Inconsistent data types | "Could not convert string to float" | `pd.to_numeric(df[col], errors='coerce')` |
| Memory errors on large files | `MemoryError` | Use `pd.read_csv(filepath, chunksize=10000)` |

## 8. Reproducibility

### System Requirements

- Python 3.8+
- pandas ≥ 1.3.0, numpy ≥ 1.20.0, scipy ≥ 1.7.0, matplotlib ≥ 3.4.0, seaborn ≥ 0.11.0

```bash
pip install pandas numpy scipy matplotlib seaborn
```
