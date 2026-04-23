"""
MST Analysis Pipeline — Complete Final Version
===============================================

WHAT THIS VERSION ADDS OVER mst_analysis_final.py:

Effect sizes (previously missing):
  - H3: Cohen's d now printed for every bin t-test
  - H5: eta-squared for Kruskal-Wallis (eta2 = (H - k + 1) / (n - k))
         r = U / sqrt(n1 * n2) for each Mann-Whitney pairwise test
         partial eta-squared and Cohen's f2 for OLS model
  - H6: Power analysis for each Pearson r (via statsmodels NormalIndPower)
  - H7: Cohen's f2 for the interaction (f2 = delta_R2 / (1 - R2_full))

Multiple comparison corrections (previously missing):
  - H5: Holm-Bonferroni applied to the 3 pairwise Mann-Whitney tests
         within each condition (3 tests share one omnibus, so form a family)
  - H6: Holm applied across 3 conditions (3 correlations share one hypothesis)
  - H7: Holm applied across 3 conditions (3 interaction F-tests, same hypothesis)

Post-hoc power:
  - H5: Power estimate for detecting the observed Kruskal-Wallis effect at N
  - H6: Power for observed r at N per condition
  - H7: Power for observed f2 (interaction) at N per condition

Everything else (pipeline, figures, H1-H4) is unchanged from mst_analysis_final.py.

Run from the folder containing:
  Both_item_task/   item_only/   task_only/
"""

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
from itertools import combinations
from scipy.stats import (pearsonr, spearmanr, kruskal, mannwhitneyu,
                         norm as scipy_norm)
import statsmodels.formula.api as smf
from statsmodels.stats.power import NormalIndPower, TTestIndPower, FTestAnovaPower

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linestyle": "--", "figure.dpi": 150
})

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

CONDITIONS = {
    "item_only":       {"folder": "item_only",      "data_subdir": "item_only_data"},
    "task_only":       {"folder": "task_only",       "data_subdir": "task_only_data"},
    "both_item_task":  {"folder": "Both_item_task",  "data_subdir": "both_data"},
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD BINS
# ─────────────────────────────────────────────────────────────────────────────
def load_bins(folder):
    def _read(fname):
        out = {}
        fpath = Path(folder) / fname
        if not fpath.exists():
            return out
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = re.split(r"[\t,\s]+", line)
                if len(parts) >= 2:
                    try:
                        out[parts[0].lstrip("0") or "0"] = int(parts[1])
                    except ValueError:
                        pass
        return out

    obj_bins   = _read("Set6 bins.txt") or _read("Set6 bins_ob.txt")
    scene_bins = _read("SetScC bins.txt")
    return obj_bins, scene_bins


def stem_to_bin_key(image_stem):
    s = re.sub(r"[ab]$", "", image_stem, flags=re.IGNORECASE)
    try:
        return str(int(s))
    except ValueError:
        return s


def is_scene(image_path):
    return re.search(r"[Ss]cenes", str(image_path)) is not None


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRIAL CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def classify_trial(image_path):
    p = str(image_path).replace("\\", "/")
    lower = p.lower()
    if "foil" in lower:
        return "foil"
    fname = lower.split("/")[-1]
    if fname.endswith("b.jpg"):
        return "lure"
    if fname.endswith("a.jpg"):
        return "target"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD TASK CSV
# ─────────────────────────────────────────────────────────────────────────────
def load_task_csv(fpath):
    df = pd.read_csv(fpath)
    df.columns = df.columns.str.strip()
    return df


def get_encoding_rt(task_df):
    rt_map = {}
    task_rows = task_df[
        task_df["image_path"].notna() &
        (task_df["image_path"].str.len() > 2) &
        (~task_df["image_path"].str.startswith("practice"))
    ]
    for _, row in task_rows.iterrows():
        stem = Path(str(row["image_path"]).replace("\\", "/")).stem
        rt9 = row.get("key_resp_9.rt", np.nan)
        rt8 = row.get("key_resp_8.rt", np.nan)
        if pd.notna(rt9) and float(rt9) > 0:
            rt_map[stem] = float(rt9)
        elif pd.notna(rt8) and float(rt8) > 0:
            rt_map[stem] = 3.0 + float(rt8)
    return rt_map


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOAD TEST CSV
# ─────────────────────────────────────────────────────────────────────────────
RESPONSE_MAP = {
    "o": "old", "s": "similar", "n": "new",
    "f": "old", "j": "old",
    "g": "similar", "k": "similar",
    "h": "new",  "l": "new",
    "old": "old", "similar": "similar", "new": "new",
}

def norm_resp(r):
    if pd.isna(r) or str(r).lower() in ("nan", "none", ""):
        return np.nan
    return RESPONSE_MAP.get(str(r).strip().lower(), np.nan)


def load_test_csv(fpath):
    df = pd.read_csv(fpath)
    df.columns = df.columns.str.strip()
    df = df[df["image_path"].notna() & (df["image_path"].str.len() > 2)].copy()
    df["trial_type"] = df["image_path"].apply(classify_trial)
    df["image_stem"] = df["image_path"].apply(
        lambda p: Path(str(p).replace("\\", "/")).stem
    )
    df["is_scene"] = df["image_path"].apply(is_scene)

    resp_col = next((c for c in ["key_resp_3.keys", "key_resp_3.key", "response"]
                     if c in df.columns), None)
    if resp_col:
        df["response"] = df[resp_col].astype(str).str.strip().str.lower()
    else:
        df["response"] = np.nan

    rt_col = next((c for c in ["key_resp_3.rt", "rt"] if c in df.columns), None)
    df["rt"] = df[rt_col].astype(float) if rt_col else np.nan

    if "position_of_stimuli" in df.columns:
        df["position"] = df["position_of_stimuli"].str.strip().str.lower()
        df["position"] = df["position"].replace({"none": np.nan, "": np.nan})
    else:
        df["position"] = np.nan

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. COMPUTE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(test_df, obj_bins, scene_bins):
    test_df = test_df.copy()
    test_df["resp_norm"] = test_df["response"].apply(norm_resp)

    def get_bin(row):
        key  = stem_to_bin_key(row["image_stem"])
        bins = scene_bins if row["is_scene"] else obj_bins
        return bins.get(key, np.nan)

    test_df["lure_bin"] = test_df.apply(get_bin, axis=1)
    test_df = test_df[test_df["resp_norm"].notna()]

    foils = test_df[test_df["trial_type"] == "foil"]
    foil_old_rate = (foils["resp_norm"] == "old").mean()     if len(foils) > 0 else 0
    foil_sim_rate = (foils["resp_norm"] == "similar").mean() if len(foils) > 0 else 0
    metrics = {"foil_old_rate": foil_old_rate, "foil_sim_rate": foil_sim_rate}

    for pos in ["pre", "mid", "post"]:
        tgt = test_df[(test_df["trial_type"] == "target") & (test_df["position"] == pos)]
        lur = test_df[(test_df["trial_type"] == "lure")   & (test_df["position"] == pos)]
        hit = (tgt["resp_norm"] == "old").mean()     if len(tgt) > 0 else np.nan
        sim = (lur["resp_norm"] == "similar").mean() if len(lur) > 0 else np.nan
        metrics[f"REC_{pos}"] = hit - foil_old_rate if pd.notna(hit) else np.nan
        metrics[f"LDI_{pos}"] = sim - foil_sim_rate if pd.notna(sim) else np.nan
        rt_vals = test_df[
            test_df["trial_type"].isin(["target", "lure"]) & (test_df["position"] == pos)
        ]["rt"]
        metrics[f"RT_{pos}"] = rt_vals.median() if len(rt_vals) > 0 else np.nan

    for b in range(1, 6):
        lur = test_df[(test_df["trial_type"] == "lure") & (test_df["lure_bin"] == b)]
        sim = (lur["resp_norm"] == "similar").mean() if len(lur) > 0 else np.nan
        metrics[f"LDI_bin{b}"] = sim - foil_sim_rate if pd.notna(sim) else np.nan

    for pos in ["pre", "mid", "post"]:
        for b in range(1, 6):
            lur = test_df[
                (test_df["trial_type"] == "lure") &
                (test_df["position"]   == pos) &
                (test_df["lure_bin"]   == b)
            ]
            sim = (lur["resp_norm"] == "similar").mean() if len(lur) > 0 else np.nan
            metrics[f"LDI_{pos}_bin{b}"] = sim - foil_sim_rate if pd.notna(sim) else np.nan

    tgt_all = test_df[test_df["trial_type"] == "target"]
    lur_all = test_df[test_df["trial_type"] == "lure"]
    hit_all = (tgt_all["resp_norm"] == "old").mean()     if len(tgt_all) > 0 else np.nan
    sim_all = (lur_all["resp_norm"] == "similar").mean() if len(lur_all) > 0 else np.nan
    metrics["REC_overall"] = hit_all - foil_old_rate if pd.notna(hit_all) else np.nan
    metrics["LDI_overall"] = sim_all - foil_sim_rate if pd.notna(sim_all) else np.nan

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 6. PARTICIPANT LOADING
# ─────────────────────────────────────────────────────────────────────────────
def find_participant_pairs(data_dir):
    csvs = sorted(Path(data_dir).glob("*.csv"))
    task_files, test_files = {}, {}
    for f in csvs:
        m = re.match(r"(\d{5})", f.name)
        if not m:
            continue
        pid = m.group(1)
        if "_task_" in f.name.lower():
            task_files[pid] = f
        elif "_test_" in f.name.lower():
            test_files[pid] = f
    return [(pid, task_files.get(pid), test_files[pid]) for pid in test_files]


def process_condition(cond_name, cond_info):
    folder   = cond_info["folder"]
    data_dir = Path(folder) / cond_info["data_subdir"]
    if not data_dir.exists():
        print(f"  [SKIP] {data_dir} not found")
        return pd.DataFrame()

    obj_bins, scene_bins = load_bins(folder)
    pairs = find_participant_pairs(data_dir)
    print(f"  Found {len(pairs)} participants in {cond_name}")

    records = []
    for pid, task_path, test_path in pairs:
        try:
            test_df = load_test_csv(test_path)
            m = compute_metrics(test_df, obj_bins, scene_bins)
            m["participant_id"] = pid
            m["condition"]      = cond_name
            records.append(m)
        except Exception as e:
            print(f"    [ERROR] pid={pid}: {e}")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 7. STAT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def one_sample_t(values, label=""):
    v = values.dropna()
    if len(v) < 3:
        return None
    t, p = stats.ttest_1samp(v, 0)
    d = v.mean() / v.std()          # Cohen's d for one-sample = mean / SD
    return {"label": label, "n": len(v), "mean": v.mean(), "sem": v.sem(),
            "t": t, "p": p, "d": d}


def paired_t(a, b, label=""):
    df_ = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df_) < 3:
        return None
    t, p = stats.ttest_rel(df_["a"], df_["b"])
    diff = df_["a"] - df_["b"]
    d = diff.mean() / diff.std()
    return {"label": label, "n": len(df_), "mean_diff": diff.mean(),
            "sem_diff": diff.sem(), "t": t, "p": p, "d": d}


def holm_correct(p_values):
    """Holm-Bonferroni correction. Returns corrected p-values in original order."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    corrected = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = p_values[idx] * (n - rank)
        running_max = max(running_max, adj)
        corrected[idx] = min(running_max, 1.0)
    return corrected


def bh_fdr(p_values, q=0.05):
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    reject     = [False] * n
    thresholds = [0.0]   * n
    for rank, idx in enumerate(order):
        thr = (rank + 1) / n * q
        thresholds[idx] = thr
        if p_values[idx] <= thr:
            reject[idx] = True
    return reject, thresholds


def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def power_for_r(r, n, alpha=0.05):
    """Post-hoc power for a Pearson correlation using Fisher z transformation."""
    if abs(r) < 1e-9:
        return 0.0
    z   = np.arctanh(abs(r))
    se  = 1.0 / np.sqrt(n - 3)
    z_alpha = scipy_norm.ppf(1 - alpha / 2)
    power = 1 - scipy_norm.cdf(z_alpha - z / se) + scipy_norm.cdf(-z_alpha - z / se)
    return float(power)


def power_for_f2(f2, n_obs, n_predictors, alpha=0.05):
    """Post-hoc power for OLS/regression using Cohen's f2."""
    # Uses noncentral F approximation
    # df1 = number of predictors tested (interaction terms = 2)
    # df2 = n - n_predictors - 1
    # lambda = f2 * n
    try:
        df1 = n_predictors
        df2 = n_obs - n_predictors - 1
        lam = f2 * n_obs
        crit = stats.f.ppf(1 - alpha, df1, df2)
        power = 1 - stats.ncf.cdf(crit, df1, df2, lam)
        return float(power)
    except Exception:
        return np.nan


def power_for_kruskal(eta2, n_per_group, k=3, alpha=0.05):
    """
    Approximate power for Kruskal-Wallis using the ARE (Asymptotic Relative
    Efficiency) approximation: treat as one-way ANOVA with f = sqrt(eta2/(1-eta2))
    and apply the ANOVA power formula. This is an approximation.
    """
    if eta2 <= 0 or eta2 >= 1:
        return np.nan
    f = np.sqrt(eta2 / (1 - eta2))
    try:
        analysis = FTestAnovaPower()
        power = analysis.power(effect_size=f, nobs=n_per_group * k,
                               alpha=alpha, k_groups=k)
        return float(power)
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# 8. FIGURES (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "pre":  "#2563EB", "mid": "#6B7280", "post": "#DC2626",
    "item_only": "#0EA5E9", "task_only": "#F59E0B", "both_item_task": "#10B981",
}
POSITIONS  = ["pre", "mid", "post"]
POS_LABELS = {"pre": "Pre-boundary", "mid": "Mid-event", "post": "Post-boundary"}


def bar_with_points(ax, data_by_group, ylabel, title, reference_line=True, colors=None):
    xs = np.arange(len(data_by_group))
    for i, (label, vals) in enumerate(data_by_group.items()):
        v = pd.Series(vals).dropna()
        if len(v) == 0:
            continue
        col = (colors or COLORS).get(label, "#666")
        ax.bar(i, v.mean(), color=col, alpha=0.8, width=0.5, zorder=3)
        ax.errorbar(i, v.mean(), yerr=v.sem(), color="black",
                    fmt="none", capsize=4, linewidth=1.5, zorder=4)
        jitter = np.random.uniform(-0.18, 0.18, len(v))
        ax.scatter(i + jitter, v.values, color=col, alpha=0.3, s=14, zorder=5)
    if reference_line:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([POS_LABELS.get(k, k) for k in data_by_group.keys()], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)


def fig1_rec_by_position(df, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle("Figure 1 — Target Recognition (REC) by Event Position",
                 fontsize=13, fontweight="bold")
    for ax, cond in zip(axes, list(CONDITIONS.keys())):
        sub  = df[df["condition"] == cond]
        data = {pos: sub[f"REC_{pos}"].dropna().values for pos in POSITIONS}
        bar_with_points(ax, data, "REC (corrected hit rate)",
                        cond.replace("_", " ").title(), colors=COLORS)
        for i, pos in enumerate(POSITIONS):
            vals = sub[f"REC_{pos}"].dropna()
            if len(vals) >= 3:
                t, p = stats.ttest_1samp(vals, 0)
                ymax = vals.mean() + vals.sem() + 0.04
                ax.text(i, ymax, sig_label(p), ha="center", fontsize=10,
                        color="darkred" if p < 0.05 else "gray")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def fig2_ldi_by_position(df, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle("Figure 2 — Lure Discrimination Index (LDI) by Event Position",
                 fontsize=13, fontweight="bold")
    for ax, cond in zip(axes, list(CONDITIONS.keys())):
        sub  = df[df["condition"] == cond]
        data = {pos: sub[f"LDI_{pos}"].dropna().values for pos in POSITIONS}
        bar_with_points(ax, data, "LDI (corrected similar rate)",
                        cond.replace("_", " ").title(), colors=COLORS)
        for i, pos in enumerate(POSITIONS):
            vals = sub[f"LDI_{pos}"].dropna()
            if len(vals) >= 3:
                t, p = stats.ttest_1samp(vals, 0)
                ymax = vals.mean() + vals.sem() + 0.03
                ax.text(i, ymax, sig_label(p), ha="center", fontsize=10,
                        color="darkblue" if p < 0.05 else "gray")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def fig3_lure_bins(df, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in CONDITIONS:
        color = COLORS[cond]
        sub   = df[df["condition"] == cond]
        means, sems = [], []
        for b in range(1, 6):
            col = f"LDI_bin{b}"
            v   = sub[col].dropna() if col in sub.columns else pd.Series(dtype=float)
            means.append(v.mean() if len(v) > 0 else np.nan)
            sems.append( v.sem()  if len(v) > 0 else 0)
        ax.errorbar(range(1, 6), means, yerr=sems, marker="o",
                    label=cond.replace("_", " ").title(),
                    color=color, linewidth=2, capsize=4, markersize=6)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lure Similarity Bin (1=Most Similar → 5=Least Similar)", fontsize=10)
    ax.set_ylabel("LDI", fontsize=10)
    ax.set_title("Figure 3 — LDI by Lure Similarity Bin", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def fig4_rt(df, save_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs, w = np.arange(3), 0.25
    for i, cond in enumerate(CONDITIONS):
        color = COLORS[cond]
        sub   = df[df["condition"] == cond]
        means = [sub[f"RT_{pos}"].dropna().mean() for pos in POSITIONS]
        sems  = [sub[f"RT_{pos}"].dropna().sem()  for pos in POSITIONS]
        ax.bar(xs + i*w, means, w, label=cond.replace("_", " ").title(),
               color=color, alpha=0.8)
        ax.errorbar(xs + i*w, means, yerr=sems, fmt="none",
                    color="black", capsize=3, linewidth=1)
    ax.set_xticks(xs + w)
    ax.set_xticklabels([POS_LABELS[p] for p in POSITIONS])
    ax.set_ylabel("Median RT (s)", fontsize=10)
    ax.set_title("Figure 4 — Response Time by Event Position",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def fig5_condition_comparison(df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, metric in zip(axes, ["REC", "LDI"]):
        diffs, cond_labels = [], []
        for cond in CONDITIONS:
            sub  = df[df["condition"] == cond]
            pre  = sub[f"{metric}_pre"].dropna()
            post = sub[f"{metric}_post"].dropna()
            common = min(len(pre), len(post))
            if common < 3:
                continue
            diff = pre.values[:common] - post.values[:common]
            diffs.append(diff)
            cond_labels.append(cond.replace("_", " ").title())
        for i, (d, lbl) in enumerate(zip(diffs, cond_labels)):
            color = list(COLORS.values())[i + 3]
            ax.bar(i, d.mean(), color=color, alpha=0.8, width=0.5)
            ax.errorbar(i, d.mean(), yerr=d.std()/np.sqrt(len(d)),
                        fmt="none", color="black", capsize=4)
            jitter = np.random.uniform(-0.15, 0.15, len(d))
            ax.scatter(i + jitter, d, color=color, alpha=0.3, s=12)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(cond_labels)))
        ax.set_xticklabels(cond_labels, fontsize=9)
        ax.set_ylabel(f"{metric} Pre - Post", fontsize=10)
        ax.set_title(f"{metric}: Pre minus Post Difference",
                     fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def fig6_rec_ldi_scatter(df, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Figure 6 — REC-LDI Correlation by Condition (Exploratory H6)",
                 fontsize=12, fontweight="bold")
    for ax, cond in zip(axes, list(CONDITIONS.keys())):
        sub = df[df["condition"] == cond][["REC_overall", "LDI_overall"]].dropna()
        if len(sub) < 5:
            ax.set_title(f"{cond}\n(insufficient data)")
            continue
        x = sub["REC_overall"].values
        y = sub["LDI_overall"].values
        color = COLORS[cond]
        ax.scatter(x, y, color=color, alpha=0.6, s=30, zorder=3)
        model = smf.ols("LDI_overall ~ REC_overall", data=sub).fit()
        x_range = np.linspace(x.min(), x.max(), 100)
        y_pred  = model.params["Intercept"] + model.params["REC_overall"] * x_range
        ax.plot(x_range, y_pred, color="black", linewidth=1.5, linestyle="--")
        r, p = pearsonr(x, y)
        r2   = model.rsquared
        p_str = f"p = {p:.3f}" if p >= 0.001 else "p < .001"
        ax.set_title(
            f"{cond.replace('_', ' ').title()}\nr = {r:.3f}, {p_str}, R2 = {r2:.3f}",
            fontsize=9
        )
        ax.set_xlabel("REC (overall)", fontsize=9)
        ax.set_ylabel("LDI (overall)", fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def fig7_ldi_heatmap(df, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        "Figure 7 — LDI by Position x Lure Bin (Exploratory H7)\n"
        "Green = high LDI (better discrimination); Red = low LDI",
        fontsize=11, fontweight="bold"
    )
    for ax, cond in zip(axes, list(CONDITIONS.keys())):
        sub = df[df["condition"] == cond]
        matrix = np.full((3, 5), np.nan)
        for r, pos in enumerate(POSITIONS):
            for c, b in enumerate(range(1, 6)):
                col = f"LDI_{pos}_bin{b}"
                if col in sub.columns:
                    matrix[r, c] = sub[col].dropna().mean()
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.6)
        for r in range(3):
            for c in range(5):
                val = matrix[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="black" if 0.1 < val < 0.5 else "white")
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"Bin {b}" for b in range(1, 6)], fontsize=8)
        ax.set_yticks(range(3))
        ax.set_yticklabels(["Pre", "Mid", "Post"], fontsize=9)
        ax.set_title(cond.replace("_", " ").title(), fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. FULL STATS REPORT
# ─────────────────────────────────────────────────────────────────────────────
def run_all_stats(df, out_lines):
    W = out_lines.append

    W("=" * 70)
    W("MST ANALYSIS — COMPLETE STATISTICAL RESULTS")
    W("=" * 70)
    W("")
    W("CORRECTION AND EFFECT SIZE FRAMEWORK:")
    W("  Confirmatory H1-H4:")
    W("    H1: one-sample t, uncorrected (decisive effects)")
    W("    H2: paired t, Holm-Bonferroni within each condition's 6-comparison family")
    W("    H3: one-sample t per bin, BH-FDR (q=0.05) per condition")
    W("        Effect size: Cohen's d = mean / SD")
    W("    H4: independent t, Holm-Bonferroni across 6 between-group comparisons")
    W("  Exploratory H5-H7: uncorrected alpha=0.05 per test family,")
    W("    but Holm applied WITHIN each condition's pairwise family:")
    W("    H5: 3 Mann-Whitney pairs per condition -> Holm within condition")
    W("        Effect sizes: eta2 (Kruskal-Wallis), r=U/sqrt(n1*n2) (Mann-Whitney)")
    W("    H6: 3 correlations (one per condition) -> Holm across conditions")
    W("        Effect size: r (Pearson). Power reported post-hoc.")
    W("    H7: 3 interaction F-tests (one per condition) -> Holm across conditions")
    W("        Effect size: Cohen's f2 = delta_R2 / (1 - R2_full). Power post-hoc.")

    # ── H1, H2, H3 per condition ─────────────────────────────────────────────
    for cond in list(CONDITIONS.keys()) + ["ALL"]:
        sub = df if cond == "ALL" else df[df["condition"] == cond]
        n   = len(sub)
        if n < 3:
            continue
        W(f"\n{'─'*60}")
        W(f"CONDITION: {cond}  (N={n})")
        W(f"{'─'*60}")

        # H1
        W("\n[H1] One-sample t-tests vs 0  (uncorrected; effects decisive)")
        W(f"{'Metric':<18} {'N':>4} {'Mean':>7} {'SEM':>7} {'t':>8} {'p':>9} {'d':>7}")
        for metric in ["REC_pre","REC_mid","REC_post","LDI_pre","LDI_mid","LDI_post"]:
            if metric in sub.columns:
                r = one_sample_t(sub[metric], label=metric)
                if r:
                    W(f"{r['label']:<18} {r['n']:>4} {r['mean']:>7.3f} {r['sem']:>7.3f} "
                      f"{r['t']:>8.3f} {r['p']:>9.4f} {r['d']:>7.3f}")

        # H2
        W("\n[H2] Paired t-tests  (Holm corrected within condition, 6-comparison family)")
        W(f"{'Comparison':<25} {'N':>4} {'Dmean':>7} {'DSEM':>7} {'t':>8} {'p_raw':>9} {'p_holm':>8} {'d':>7}")
        raw_results = []
        for metric in ["REC", "LDI"]:
            for (a, b_pos) in [("pre","post"), ("pre","mid"), ("mid","post")]:
                ca, cb = f"{metric}_{a}", f"{metric}_{b_pos}"
                if ca in sub.columns and cb in sub.columns:
                    r = paired_t(sub[ca], sub[cb], label=f"{metric}: {a}-{b_pos}")
                    if r:
                        raw_results.append(r)
        if raw_results:
            p_holms = holm_correct([r["p"] for r in raw_results])
            for r, ph in zip(raw_results, p_holms):
                W(f"{r['label']:<25} {r['n']:>4} {r['mean_diff']:>7.3f} {r['sem_diff']:>7.3f} "
                  f"{r['t']:>8.3f} {r['p']:>9.4f} {ph:>8.4f} {r['d']:>7.3f}")

        # H3 (per condition only)
        if cond != "ALL":
            W("\n[H3] LDI by Lure Bin — one-sample t  (BH-FDR q=0.05; d = mean/SD)")
            W(f"{'Bin':<6} {'N':>4} {'Mean':>7} {'SEM':>7} {'t':>8} {'p_raw':>9} {'BH_thr':>8} {'Reject':>7} {'d':>7}")
            bin_ps, bin_rs = [], []
            for b in range(1, 6):
                col = f"LDI_bin{b}"
                if col in sub.columns:
                    r = one_sample_t(sub[col], label=f"Bin {b}")
                    if r:
                        bin_rs.append(r)
                        bin_ps.append(r["p"])
            if bin_rs:
                reject, thresholds = bh_fdr(bin_ps)
                for r, rej, thr in zip(bin_rs, reject, thresholds):
                    W(f"{r['label']:<6} {r['n']:>4} {r['mean']:>7.3f} {r['sem']:>7.3f} "
                      f"{r['t']:>8.3f} {r['p']:>9.4f} {thr:>8.4f} {'Yes' if rej else 'No':>7} {r['d']:>7.3f}")

    # H4
    W(f"\n{'='*70}")
    W("[H4] BETWEEN-GROUP COMPARISONS  (Holm-Bonferroni, 6-comparison family)")
    W(f"{'='*70}")
    W(f"{'Comparison':<35} {'Metric':<13} {'t':>7} {'df':>4} {'p_raw':>9} {'p_holm':>8} {'d':>7}  Result")
    all_bt = []
    for metric in ["REC_overall", "LDI_overall"]:
        for (c1, c2) in combinations(list(CONDITIONS.keys()), 2):
            g1 = df[df["condition"] == c1][metric].dropna()
            g2 = df[df["condition"] == c2][metric].dropna()
            if len(g1) < 3 or len(g2) < 3:
                continue
            t, p = stats.ttest_ind(g1, g2)
            pool = np.sqrt(((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2) /
                           (len(g1)+len(g2)-2))
            d = (g1.mean() - g2.mean()) / pool if pool > 0 else 0
            all_bt.append({"label": f"{c1} vs {c2}", "metric": metric,
                            "t": t, "df": len(g1)+len(g2)-2, "p": p, "d": d})
    if all_bt:
        p_holms = holm_correct([r["p"] for r in all_bt])
        for r, ph in zip(all_bt, p_holms):
            result = "**SIG**" if ph < 0.05 else ("trend" if r["p"] < 0.05 else "ns")
            W(f"{r['label']:<35} {r['metric']:<13} {r['t']:>7.3f} {r['df']:>4} "
              f"{r['p']:>9.4f} {ph:>8.4f} {r['d']:>7.3f}  {result}")

    # ── EXPLORATORY ───────────────────────────────────────────────────────────
    W(f"\n{'='*70}")
    W("EXPLORATORY HYPOTHESES  (uncorrected alpha=0.05 per test;")
    W("Holm applied within pairwise families as noted)")
    W(f"{'='*70}")

    # ── H5 ────────────────────────────────────────────────────────────────────
    W(f"\n{'─'*60}")
    W("[H5 EXPLORATORY] Response Time by Event Position")
    W("Omnibus: Kruskal-Wallis. Effect: eta2 = (H - k + 1) / (n - k)")
    W("Pairwise: Mann-Whitney + r = U/sqrt(n1*n2). Holm within each condition.")
    W("Model: OLS RT ~ C(position). Effect: partial eta2, Cohen's f2.")
    W("Power: post-hoc for Kruskal-Wallis eta2 at observed N.")
    W(f"{'─'*60}")

    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        n_cond = len(sub)
        W(f"\n  Condition: {cond}  (N={n_cond})")
        W(f"  {'Position':<10} {'N':>4} {'Median':>9} {'Mean':>9} {'SD':>8}")

        rt_groups = {}
        for pos in POSITIONS:
            vals = sub[f"RT_{pos}"].dropna()
            rt_groups[pos] = vals.values
            W(f"  {pos:<10} {len(vals):>4} {vals.median():>9.4f} {vals.mean():>9.4f} {vals.std():>8.4f}")

        grps = [v for v in rt_groups.values() if len(v) >= 3]
        if len(grps) >= 2:
            h_stat, p_kw = kruskal(*grps)
            k = len(grps)
            n_total = sum(len(g) for g in grps)
            # eta-squared for Kruskal-Wallis
            eta2_kw = (h_stat - k + 1) / (n_total - k) if n_total > k else np.nan
            eta2_kw = max(0.0, eta2_kw)  # floor at 0
            n_per = n_total / k
            pwr_kw = power_for_kruskal(eta2_kw, n_per_group=n_per, k=k)
            W(f"\n  Kruskal-Wallis: H({k-1}) = {h_stat:.3f}, p = {p_kw:.4f}  {sig_label(p_kw)}")
            W(f"  Effect size: eta2 = {eta2_kw:.4f}  "
              f"({'small' if eta2_kw<0.06 else 'medium' if eta2_kw<0.14 else 'large'})")
            W(f"  Post-hoc power (approx, ANOVA ARE): {pwr_kw:.3f}")

            # Pairwise Mann-Whitney with Holm within condition
            W(f"\n  Pairwise Mann-Whitney (Holm corrected within condition):")
            W(f"  {'Pair':<15} {'U':>9} {'p_raw':>9} {'p_holm':>9} {'r':>7}  sig")
            mw_results = []
            for (p1, p2) in combinations(POSITIONS, 2):
                g1, g2 = rt_groups[p1], rt_groups[p2]
                if len(g1) < 3 or len(g2) < 3:
                    continue
                u, p_mw = mannwhitneyu(g1, g2, alternative="two-sided")
                r_mw = u / np.sqrt(len(g1) * len(g2))   # effect size r
                mw_results.append({"pair": f"{p1} vs {p2}", "u": u,
                                   "p": p_mw, "r": r_mw})
            if mw_results:
                p_holms_mw = holm_correct([r["p"] for r in mw_results])
                for r, ph in zip(mw_results, p_holms_mw):
                    s = sig_label(ph)
                    W(f"  {r['pair']:<15} {r['u']:>9.1f} {r['p']:>9.4f} {ph:>9.4f} "
                      f"{r['r']:>7.4f}  {s}")

        # OLS
        rt_long = []
        for pos in POSITIONS:
            for val in sub[f"RT_{pos}"].dropna():
                rt_long.append({"rt": val, "position": pos})
        if len(rt_long) >= 9:
            rt_df = pd.DataFrame(rt_long)
            rt_df["position"] = pd.Categorical(rt_df["position"],
                                               categories=["pre","mid","post"])
            try:
                model = smf.ols("rt ~ C(position, Treatment('pre'))", data=rt_df).fit()
                # partial eta2 = SS_model / (SS_model + SS_resid)
                ss_model = model.ess
                ss_resid = model.ssr
                partial_eta2 = ss_model / (ss_model + ss_resid)
                f2 = partial_eta2 / (1 - partial_eta2) if partial_eta2 < 1 else np.nan
                pwr_ols = power_for_f2(f2, n_obs=len(rt_df), n_predictors=2)
                W(f"\n  OLS: RT ~ C(position)  [reference = pre]")
                W(f"  R2 = {model.rsquared:.4f}, partial_eta2 = {partial_eta2:.4f}, "
                  f"Cohen's f2 = {f2:.4f}")
                W(f"  F({int(model.df_model)},{int(model.df_resid)}) = {model.fvalue:.3f}, "
                  f"p = {model.f_pvalue:.4f}  {sig_label(model.f_pvalue)}")
                W(f"  Post-hoc power (f2): {pwr_ols:.3f}")
                W(f"  {'Coefficient':<40} {'beta':>8} {'SE':>8} {'t':>7} {'p':>9}")
                for name in model.params.index:
                    W(f"  {name:<40} {model.params[name]:>8.4f} {model.bse[name]:>8.4f} "
                      f"{model.tvalues[name]:>7.3f} {model.pvalues[name]:>9.4f}")
            except Exception as e:
                W(f"  OLS failed: {e}")

    # ── H6 ────────────────────────────────────────────────────────────────────
    W(f"\n{'─'*60}")
    W("[H6 EXPLORATORY] REC-LDI Within-Participant Correlation")
    W("Test: Pearson r + OLS LDI ~ REC per condition.")
    W("Correction: Holm across 3 conditions (same hypothesis tested 3 times).")
    W("Effect size: r (Pearson). Post-hoc power via Fisher z transformation.")
    W(f"{'─'*60}")

    h6_results = []
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond][["REC_overall", "LDI_overall"]].dropna()
        if len(sub) < 5:
            h6_results.append({"cond": cond, "r": np.nan, "p": 1.0,
                               "rho": np.nan, "n": len(sub)})
            continue
        x, y = sub["REC_overall"].values, sub["LDI_overall"].values
        r_val, p_r = pearsonr(x, y)
        rho, p_rho = spearmanr(x, y)
        h6_results.append({"cond": cond, "r": r_val, "p": p_r,
                           "rho": rho, "p_rho": p_rho, "n": len(sub),
                           "x": x, "y": y, "sub": sub})

    p_holms_h6 = holm_correct([r["p"] for r in h6_results])

    for res, ph in zip(h6_results, p_holms_h6):
        cond = res["cond"]
        W(f"\n  Condition: {cond}  (N={res['n']})")
        if np.isnan(res["r"]):
            W("  Insufficient data.")
            continue
        pwr = power_for_r(res["r"], res["n"])
        W(f"  Pearson r = {res['r']:.4f},  p_raw = {res['p']:.4f},  "
          f"p_holm = {ph:.4f}  {sig_label(ph)}")
        W(f"  Spearman rho = {res['rho']:.4f},  p = {res['p_rho']:.4f}  (robustness)")
        W(f"  Effect size: r = {res['r']:.4f}  "
          f"({'small' if abs(res['r'])<0.3 else 'medium' if abs(res['r'])<0.5 else 'large'})")
        W(f"  Post-hoc power (alpha=0.05, N={res['n']}): {pwr:.3f}")
        try:
            model = smf.ols("LDI_overall ~ REC_overall", data=res["sub"]).fit()
            W(f"  OLS LDI ~ REC:  beta = {model.params['REC_overall']:.4f}  "
              f"(SE={model.bse['REC_overall']:.4f}, p={model.pvalues['REC_overall']:.4f}),  "
              f"R2 = {model.rsquared:.4f}")
        except Exception as e:
            W(f"  OLS failed: {e}")

    # ── H7 ────────────────────────────────────────────────────────────────────
    W(f"\n{'─'*60}")
    W("[H7 EXPLORATORY] LDI ~ Position x Lure Bin Interaction")
    W("Test: OLS LDI ~ bin_num * C(position) per condition.")
    W("Correction: Holm across 3 conditions (3 interaction F-tests, same hypothesis).")
    W("Effect size: Cohen's f2 = delta_R2 / (1 - R2_full).")
    W("Post-hoc power for interaction at observed N.")
    W(f"{'─'*60}")

    h7_results = []
    h7_data    = {}

    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        rows = []
        for _, prow in sub.iterrows():
            for pos in POSITIONS:
                for b in range(1, 6):
                    col = f"LDI_{pos}_bin{b}"
                    val = prow.get(col, np.nan)
                    if pd.notna(val):
                        rows.append({"ldi": val, "bin_num": b,
                                     "position": pos, "pid": prow["participant_id"]})
        if len(rows) < 10:
            h7_results.append({"cond": cond, "p_int": 1.0, "f2": np.nan, "n": 0})
            continue

        long_df = pd.DataFrame(rows)
        long_df["position"] = pd.Categorical(long_df["position"],
                                             categories=["pre","mid","post"])
        try:
            model_main = smf.ols(
                "ldi ~ bin_num + C(position, Treatment('pre'))",
                data=long_df
            ).fit()
            model_int = smf.ols(
                "ldi ~ bin_num * C(position, Treatment('pre'))",
                data=long_df
            ).fit()
            from statsmodels.stats.anova import anova_lm
            aov = anova_lm(model_main, model_int)
            p_int = float(aov["Pr(>F)"].iloc[1])
            delta_r2 = model_int.rsquared - model_main.rsquared
            f2 = delta_r2 / (1 - model_int.rsquared) if model_int.rsquared < 1 else np.nan
            h7_results.append({"cond": cond, "p_int": p_int, "f2": f2,
                               "n": len(long_df), "delta_r2": delta_r2,
                               "r2_main": model_main.rsquared,
                               "r2_full": model_int.rsquared})
            h7_data[cond] = {"model_main": model_main, "model_int": model_int,
                             "aov": aov, "long_df": long_df, "sub": sub}
        except Exception as e:
            h7_results.append({"cond": cond, "p_int": 1.0, "f2": np.nan,
                               "n": 0, "error": str(e)})

    p_holms_h7 = holm_correct([r["p_int"] for r in h7_results])

    for res, ph in zip(h7_results, p_holms_h7):
        cond = res["cond"]
        W(f"\n  Condition: {cond}  (N observations = {res['n']})")
        if res.get("error"):
            W(f"  ERROR: {res['error']}")
            continue
        if res["n"] == 0:
            W("  Insufficient data.")
            continue

        pwr = power_for_f2(res["f2"], n_obs=res["n"], n_predictors=2) \
              if not np.isnan(res.get("f2", np.nan)) else np.nan

        W(f"  R2 main effects = {res['r2_main']:.4f},  "
          f"R2 with interaction = {res['r2_full']:.4f}")
        W(f"  Delta R2 = {res['delta_r2']:.4f},  "
          f"Cohen's f2 = {res['f2']:.4f}  "
          f"({'small' if res['f2']<0.15 else 'medium' if res['f2']<0.35 else 'large'})")
        W(f"  F-test for interaction: p_raw = {res['p_int']:.4f},  "
          f"p_holm = {ph:.4f}  {sig_label(ph)}")
        W(f"  Post-hoc power (f2, 2 interaction df): {pwr:.3f}")

        if cond in h7_data:
            d = h7_data[cond]
            W(f"\n  Interaction model coefficients:")
            W(f"  {'Coefficient':<50} {'beta':>8} {'SE':>8} {'t':>7} {'p':>9} sig")
            for name in d["model_int"].params.index:
                coef = d["model_int"].params[name]
                se   = d["model_int"].bse[name]
                t    = d["model_int"].tvalues[name]
                p    = d["model_int"].pvalues[name]
                W(f"  {name:<50} {coef:>8.4f} {se:>8.4f} {t:>7.3f} {p:>9.4f} "
                  f"{'*' if p<0.05 else 'ns'}")

            W(f"\n  Bin gradient by position (mean LDI):")
            W(f"  {'Position':<10}  Bin1    Bin2    Bin3    Bin4    Bin5")
            for pos in POSITIONS:
                means = []
                for b in range(1, 6):
                    col = f"LDI_{pos}_bin{b}"
                    v   = d["sub"][col].dropna() if col in d["sub"].columns else pd.Series(dtype=float)
                    means.append(f"{v.mean():>6.3f}" if len(v) > 0 else "   NaN")
                W(f"  {pos:<10}  {'  '.join(means)}")

        int_sig = ph < 0.05
        W(f"\n  CONCLUSION: {'Significant' if int_sig else 'No significant'} "
          f"Position x Bin interaction in {cond} "
          f"(p_holm = {ph:.4f}). "
          f"{'Bin gradient differs by event position.' if int_sig else 'Bin gradient is consistent across positions.'}")

    W(f"\n{'='*70}")
    W("END OF REPORT")
    W(f"{'='*70}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\nMST Analysis Pipeline — Complete Final Version\n")
    all_dfs = []
    for cond_name, cond_info in CONDITIONS.items():
        print(f"Processing: {cond_name}")
        df_cond = process_condition(cond_name, cond_info)
        if not df_cond.empty:
            all_dfs.append(df_cond)

    if not all_dfs:
        print("\n[ERROR] No data found. Check folder structure.")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv("mst_results.csv", index=False)
    print(f"\nSaved: mst_results.csv ({len(df)} rows)")

    print("\nGenerating figures...")
    np.random.seed(42)
    fig1_rec_by_position(df,      FIGURES_DIR / "fig1_REC_by_position.png")
    fig2_ldi_by_position(df,      FIGURES_DIR / "fig2_LDI_by_position.png")
    fig3_lure_bins(df,            FIGURES_DIR / "fig3_LDI_lure_bins.png")
    fig4_rt(df,                   FIGURES_DIR / "fig4_RT_by_position.png")
    fig5_condition_comparison(df, FIGURES_DIR / "fig5_condition_comparison.png")
    fig6_rec_ldi_scatter(df,      FIGURES_DIR / "fig6_REC_LDI_scatter.png")
    fig7_ldi_heatmap(df,          FIGURES_DIR / "fig7_LDI_heatmap.png")

    print("\nRunning statistics...")
    out_lines = []
    run_all_stats(df, out_lines)
    report = "\n".join(out_lines)
    print(report)
    with open("mst_summary.txt", "w") as f:
        f.write(report)
    print("\nSaved: mst_summary.txt")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()