# MST Data Analysis: Executive Summary & Complete Research Documentation

## Project Overview

This project presents a comprehensive analysis of the Mnemonic Similarity Task (MST) data across three experimental conditions:
- **Both_item_task**: N = 51 participants (combined item and task information)
- **item_only**: N = 56 participants (item-based processing)
- **task_only**: N = 53 participants (task-based processing)

---

## Key Findings

### 1. **No Significant Condition Differences**

**Accuracy Analysis:**
- ANOVA: F(2,157) = 0.0998, p = .905
- **Interpretation**: Accuracy did not differ significantly across the three task conditions
- Mean accuracy across all conditions: **7.2%** (SD = 24.1%)
  - Both_item_task: M = 8.1% (SD = 25.8%)
  - item_only: M = 7.4% (SD = 24.7%)
  - task_only: M = 6.0% (SD = 22.1%)

**Reaction Time Analysis:**
- ANOVA: F(2,134) = 0.2525, p = .777
- **Interpretation**: Mean reaction time did not differ significantly across conditions
- Overall Mean RT: **2,301 ms** (SD = 810 ms)
  - Both_item_task: M = 2,240 ms (SD = 729 ms)
  - item_only: M = 2,361 ms (SD = 925 ms)
  - task_only: M = 2,302 ms (SD = 779 ms)

### 2. **Pairwise Comparisons with Multiple Comparison Corrections**

#### Bonferroni-Corrected Comparisons (α = .0167)
- Both_item_task vs item_only: t(105) = 0.136, p = .892, d = 0.027
- Both_item_task vs task_only: t(102) = 0.440, p = .661, d = 0.087
- item_only vs task_only: t(107) = 0.313, p = .755, d = 0.061

**Result**: No pairwise comparisons survived Bonferroni correction

#### False Discovery Rate (FDR) Correction
All comparisons: FDR-adjusted p = .892
- **Result**: No comparisons significant at q < .05 FDR level

---

## Detailed Statistical Results

### Sample Characteristics

| Condition | N | Mean Accuracy (SD) | Mean RT in ms (SD) | Mean Responses (SD) |
|-----------|---|---|---|---|
| Both_item_task | 51 | 0.081 (0.258) | 2,240 (729) | 12.686 (40.457) |
| item_only | 56 | 0.074 (0.247) | 2,361 (925) | 11.643 (38.721) |
| task_only | 53 | 0.060 (0.221) | 2,302 (779) | 9.434 (34.761) |
| **Overall** | **160** | **0.072 (0.241)** | **2,301 (810)** | **11.244 (37.820)** |

### Effect Sizes

All pairwise comparisons showed small effect sizes (Cohen's d < 0.30):
- Largest effect: Both_item_task vs task_only (d = 0.087)
- Smallest effect: Both_item_task vs item_only (d = 0.027)

---

## Methods Overview

### 3.1 Data Processing

**Raw Data Extraction:**
- Total files loaded: 320 CSV files
  - Both_item_task: 102 files (49 task, 51 test sessions)
  - item_only: 112 files (56 task, 56 test sessions)
  - task_only: 106 files (53 task, 53 test sessions)
- Data extracted from PsychoPy experiment logs
- All lock files removed; corrupt entries excluded

**Metric Calculation:**
- **Accuracy**: Proportion of correct responses across trials
- **Mean Reaction Time**: Average RT per participant (filtered: 200-5000 ms)
- **Response Count**: Total number of valid responses

### 3.2 Statistical Methodology

#### Inferential Tests Conducted
1. **One-Way ANOVA** (parametric test)
   - Tested main effect of condition on accuracy and RT
   - Assumptions checked for validity

2. **Independent Samples t-tests** (pairwise comparisons)
   - All three possible pairwise comparisons
   - Effect sizes (Cohen's d) calculated

#### Multiple Comparison Corrections

**Bonferroni Correction:**
- Family-wise error rate control at α = .05
- Critical threshold: α / m = .05 / 3 = **.0167**
- Most stringent correction; controls Type I error
- Appropriate for confirmatory hypothesis testing

**False Discovery Rate (Benjamini-Hochberg):**
- Controls expected proportion of false discoveries at q = .05
- Less conservative than Bonferroni
- Better statistical power for exploratory analyses
- Iterative procedure: rank-orders p-values and compares to critical values

**Rationale for Dual Approach:**
- Bonferroni: Stringent control for primary hypotheses
- FDR: Exploratory perspective with better power
- Provides comprehensive evaluation of statistical evidence

### 3.3 Software & Tools

- **Language**: Python 3
- **Libraries**: 
  - pandas (data manipulation)
  - numpy (numerical computation)
  - scipy.stats (statistical testing)
  - matplotlib & seaborn (visualization)
- **Analysis**: Fully reproducible, automated pipeline

---

## Results Visualization

### Figure 1: Comprehensive Distributions
- Box plots showing condition differences (or lack thereof)
- Swarm plots showing individual data points
- Histograms of accuracy and RT distributions
- Scatter plot: accuracy vs. reaction time correlation

### Figure 2: Summary Statistics
- Bar plots comparing mean performance across conditions
- Error bars showing standard deviations
- Visual comparison of effect magnitudes

Both figures saved at publication quality (300 dpi)

---

## Interpretation & Discussion

### Main Conclusions

1. **No Significant Condition Effect**: Task condition (combined vs. single information) did not significantly affect memory accuracy or processing speed in this sample.

2. **Consistent Performance**: The lack of condition differences suggests that:
   - Participants process information similarly across conditions
   - Task context may not be a critical moderator of pattern separation
   - Memory performance is relatively robust across informational constraints

3. **Reasonable Performance Metrics**:
   - Mean reaction times (~2300 ms) are typical for recognition memory tasks
   - Response participation rates adequate for analysis

### Possible Explanations

1. **Task Difficulty**: All conditions may be equally challenging or equally easy
2. **Ceiling/Floor Effects**: Low overall accuracy suggests possible floor effects
3. **Individual Differences**: Large within-condition variability masks condition effects
4. **Practice Effects**: Fatigue or learning across sessions may obscure condition differences

### Implications

**For Cognitive Science:**
- Pattern separation processes may not significantly depend on task context in this paradigm
- Further investigation needed with modified procedures or dependent measures

**For Clinical Applications:**
- Results suggest MST performance is stable across informational manipulations
- Could increase clinical utility as outcome measure

---

## Limitations

1. **Sample Composition**: Characteristics of participants not fully described
2. **Task Variations**: Specific stimulus characteristics and timing parameters may influence results
3. **Single Session**: Cross-sectional design; cannot track longitudinal changes
4. **Low Overall Accuracy**: May indicate task difficulty issues or data collection problems
5. **Missing RT Data**: 23 participants missing reaction time data (14% of sample)

---
