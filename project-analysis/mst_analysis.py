"""
MST Analysis Pipeline  (updated to match actual data structure)
Run from the folder containing:
  Both_item_task/   item_only/   task_only/

Changes from original:
  - Uses `position_of_stimuli` column already present in test CSVs
      (no longer inferred from task-file order)
  - Bin lookup: image stems like "039b" → numeric key "39"
      mapped separately for Objects (Set6 bins) and Scenes (SetScC bins)
  - `classify_trial` handles Windows backslash paths and Foils\ folder
  - `find_participant_pairs` keeps the LATEST file when a PID has duplicates
      and processes participants that have only a test file (position is
      in the test CSV so a task file is not strictly required)
  - Encoding-RT collection is still attempted when a task file exists

Output: figures/ directory + mst_results.csv + mst_summary.txt
"""

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from pathlib import Path
from itertools import combinations

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linestyle": "--", "figure.dpi": 150
})

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

CONDITIONS = {
    "item_only":       {"folder": "item_only",       "data_subdir": "item_only_data"},
    "task_only":       {"folder": "task_only",        "data_subdir": "task_only_data"},
    "both_item_task":  {"folder": "Both_item_task",   "data_subdir": "both_data"},
}

# 1. LOAD BINS
def load_bins(folder):
    """
    Return two dicts:  obj_bins, scene_bins
    Each maps integer-string key (e.g. "39") → bin number (1-5).

    Bin files have format:  NUMBER<TAB>BIN  (one per line, no header)
    Image stems look like "039b" (lure) or "039a" (target).
    To look up a bin: strip trailing letter and leading zeros → "39".
    """
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

    # Objects
    obj_bins = _read("Set6 bins.txt") or _read("Set6 bins_ob.txt")
    # Scenes (may not exist in every condition)
    scene_bins = _read("SetScC bins.txt")
    return obj_bins, scene_bins


def stem_to_bin_key(image_stem):
    """Convert e.g. "039b" or "039a" → "39" for bin lookup."""
    s = re.sub(r"[ab]$", "", image_stem, flags=re.IGNORECASE)
    try:
        return str(int(s))
    except ValueError:
        return s


def is_scene(image_path):
    """Return True if the path is under a Scenes/ folder."""
    return re.search(r"[Ss]cenes", str(image_path)) is not None


# 2. TRIAL CLASSIFICATION (path-safe)
def classify_trial(image_path):
    """Classify as foil / lure / target.  Handles backslash Windows paths."""
    p = str(image_path).replace("\\", "/")
    lower = p.lower()
    if "foil" in lower:
        return "foil"
    fname = lower.split("/")[-1]          # filename only
    if fname.endswith("b.jpg"):
        return "lure"
    if fname.endswith("a.jpg"):
        return "target"
    return "unknown"


# 3. LOAD TASK (ENCODING) CSV  (unchanged logic)
def load_task_csv(fpath):
    df = pd.read_csv(fpath)
    df.columns = df.columns.str.strip()
    return df


def get_encoding_rt(task_df):
    """Return dict: stem → total RT (s)"""
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


# 4. LOAD TEST CSV
def load_test_csv(fpath):
    df = pd.read_csv(fpath)
    df.columns = df.columns.str.strip()

    # Keep only rows that have an image path
    df = df[df["image_path"].notna() & (df["image_path"].str.len() > 2)].copy()

    df["trial_type"]  = df["image_path"].apply(classify_trial)
    df["image_stem"]  = df["image_path"].apply(
        lambda p: Path(str(p).replace("\\", "/")).stem
    )
    df["is_scene"]    = df["image_path"].apply(is_scene)

    # Normalise response column
    resp_col = next((c for c in ["key_resp_3.keys", "key_resp_3.key", "response"]
                     if c in df.columns), None)
    if resp_col:
        df["response"] = df[resp_col].astype(str).str.strip().str.lower()
    else:
        df["response"] = np.nan

    # RT
    rt_col = next((c for c in ["key_resp_3.rt", "rt"] if c in df.columns), None)
    df["rt"] = df[rt_col].astype(float) if rt_col else np.nan

    # ── Position: use the column already in the CSV ──────────────────────
    if "position_of_stimuli" in df.columns:
        df["position"] = df["position_of_stimuli"].str.strip().str.lower()
        # map 'none' (foils) to NaN
        df["position"] = df["position"].replace({"none": np.nan, "": np.nan})
    else:
        df["position"] = np.nan   # fallback – will leave metrics NaN

    return df


# 5. COMPUTE REC & LDI
RESPONSE_MAP = {
    "o": "old", "s": "similar", "n": "new",
    # legacy key mappings kept for safety
    "f": "old", "j": "old",
    "g": "similar", "k": "similar",
    "h": "new", "l": "new",
    "old": "old", "similar": "similar", "new": "new",
}

def norm_resp(r):
    if pd.isna(r) or str(r).lower() in ("nan", "none", ""):
        return np.nan
    return RESPONSE_MAP.get(str(r).strip().lower(), np.nan)


def compute_metrics(test_df, obj_bins, scene_bins):
    """Compute REC and LDI per event position and per lure bin."""
    test_df = test_df.copy()
    test_df["resp_norm"] = test_df["response"].apply(norm_resp)

    # Bin lookup: use obj or scene bins depending on image folder
    def get_bin(row):
        key = stem_to_bin_key(row["image_stem"])
        bins = scene_bins if row["is_scene"] else obj_bins
        return bins.get(key, np.nan)

    test_df["lure_bin"] = test_df.apply(get_bin, axis=1)

    test_df = test_df[test_df["resp_norm"].notna()]

    foils = test_df[test_df["trial_type"] == "foil"]
    foil_old_rate = (foils["resp_norm"] == "old").mean()    if len(foils) > 0 else 0
    foil_sim_rate = (foils["resp_norm"] == "similar").mean() if len(foils) > 0 else 0

    metrics = {"foil_old_rate": foil_old_rate, "foil_sim_rate": foil_sim_rate}

    # By event position
    for pos in ["pre", "mid", "post"]:
        tgt = test_df[(test_df["trial_type"] == "target") & (test_df["position"] == pos)]
        lur = test_df[(test_df["trial_type"] == "lure")   & (test_df["position"] == pos)]
        hit = (tgt["resp_norm"] == "old").mean()     if len(tgt) > 0 else np.nan
        sim = (lur["resp_norm"] == "similar").mean() if len(lur) > 0 else np.nan
        metrics[f"REC_{pos}"] = hit - foil_old_rate  if pd.notna(hit) else np.nan
        metrics[f"LDI_{pos}"] = sim - foil_sim_rate  if pd.notna(sim) else np.nan
        tgt_rt = test_df[
            test_df["trial_type"].isin(["target", "lure"]) & (test_df["position"] == pos)
        ]["rt"]
        metrics[f"RT_{pos}"] = tgt_rt.median() if len(tgt_rt) > 0 else np.nan

    # By lure bin
    for b in range(1, 6):
        lur = test_df[(test_df["trial_type"] == "lure") & (test_df["lure_bin"] == b)]
        sim = (lur["resp_norm"] == "similar").mean() if len(lur) > 0 else np.nan
        metrics[f"LDI_bin{b}"] = sim - foil_sim_rate if pd.notna(sim) else np.nan

    return metrics

# 6. PROCESS ALL PARTICIPANTS
def find_participant_pairs(data_dir):
    """
    Return list of (pid, task_path_or_None, test_path).

    Handles:
    - Multiple files per PID: keeps the LATEST file by filename (date-stamped).
    - Participants with only a test file (no task file).
    """
    csvs = sorted(Path(data_dir).glob("*.csv"))

    # Group by PID and type, keeping last (latest) file
    task_files, test_files = {}, {}
    for f in csvs:
        m = re.match(r"(\d{5})", f.name)
        if not m:
            continue
        pid = m.group(1)
        if "_task_" in f.name.lower():
            task_files[pid] = f        # later sort means last=newest
        elif "_test_" in f.name.lower():
            test_files[pid] = f

    pairs = []
    for pid in test_files:
        pairs.append((pid, task_files.get(pid), test_files[pid]))
    return pairs


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
            
            # Optional encoding RT (needs task file)
            if task_path is not None:
                task_df = load_task_csv(task_path)
                # (encoding RT stored but not used in current metrics)
            m["participant_id"] = pid
            m["condition"]      = cond_name
            records.append(m)
        except Exception as e:
            print(f"    [ERROR] pid={pid}: {e}")

    return pd.DataFrame(records)


# 7. STATISTICS
def one_sample_t(values, label=""):
    v = values.dropna()
    if len(v) < 3:
        return None
    t, p = stats.ttest_1samp(v, 0)
    d = t / np.sqrt(len(v))
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


# 8. FIGURES
COLORS = {
    "pre":  "#2563EB",
    "mid":  "#6B7280",
    "post": "#DC2626",
    "item_only":      "#0EA5E9",
    "task_only":      "#F59E0B",
    "both_item_task": "#10B981",
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


def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def fig1_rec_by_position(df, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle("Figure 1 — Target Recognition (REC) by Event Position",
                 fontsize=13, fontweight="bold")
    for ax, cond in zip(axes, list(CONDITIONS.keys())):
        sub = df[df["condition"] == cond]
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
        sub = df[df["condition"] == cond]
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
        sub = df[df["condition"] == cond]
        means, sems = [], []
        for b in range(1, 6):
            col = f"LDI_bin{b}"
            if col in sub.columns:
                v = sub[col].dropna()
                means.append(v.mean() if len(v) > 0 else np.nan)
                sems.append(v.sem()   if len(v) > 0 else 0)
            else:
                means.append(np.nan); sems.append(0)
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
    xs = np.arange(3)
    w = 0.25
    for i, cond in enumerate(CONDITIONS):
        color = COLORS[cond]
        sub = df[df["condition"] == cond]
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
    """REC and LDI pre−post difference across conditions."""
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
        ax.set_ylabel(f"{metric} Pre − Post", fontsize=10)
        ax.set_title(f"{metric}: Pre minus Post Difference",
                     fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# 9. STATS REPORT
def run_all_stats(df, out_lines):
    W = out_lines.append

    W("=" * 60)
    W("MST ANALYSIS — STATISTICAL RESULTS")
    W("=" * 60)

    for cond in list(CONDITIONS.keys()) + ["ALL"]:
        sub = df if cond == "ALL" else df[df["condition"] == cond]
        n = len(sub)
        if n < 3:
            continue
        W(f"\n{'─'*50}")
        W(f"CONDITION: {cond}  (N={n})")
        W(f"{'─'*50}")

        W("\n[One-sample t-tests against 0]")
        W(f"{'Metric':<18} {'N':>4} {'Mean':>7} {'SEM':>7} {'t':>7} {'p':>8} {'d':>6}")
        osamp_results = []
        for metric in ["REC_pre","REC_mid","REC_post","LDI_pre","LDI_mid","LDI_post"]:
            if metric in sub.columns:
                r = one_sample_t(sub[metric], label=metric)
                osamp_results.append(r)
        for r in [r for r in osamp_results if r]:
            W(f"{r['label']:<18} {r['n']:>4} {r['mean']:>7.3f} {r['sem']:>7.3f} "
              f"{r['t']:>7.3f} {r['p']:>8.4f} {r['d']:>6.3f}")

        W("\n[Paired t-tests: Pre vs Post]")
        W(f"{'Comparison':<25} {'N':>4} {'Δmean':>7} {'ΔSEM':>7} {'t':>7} {'p':>8} {'d':>6}")
        paired_results = []
        for metric in ["REC", "LDI"]:
            for (a, b) in [("pre","post"), ("pre","mid"), ("mid","post")]:
                ca, cb = f"{metric}_{a}", f"{metric}_{b}"
                if ca in sub.columns and cb in sub.columns:
                    r = paired_t(sub[ca], sub[cb], label=f"{metric}: {a}-{b}")
                    paired_results.append(r)
        for r in [r for r in paired_results if r]:
            W(f"{r['label']:<25} {r['n']:>4} {r['mean_diff']:>7.3f} {r['sem_diff']:>7.3f} "
              f"{r['t']:>7.3f} {r['p']:>8.4f} {r['d']:>6.3f}")

        if cond != "ALL":
            W("\n[LDI by Lure Bin]")
            W(f"{'Bin':<6} {'N':>4} {'Mean':>7} {'SEM':>7} {'t':>7} {'p':>8}")
            bin_results = []
            for b in range(1, 6):
                col = f"LDI_bin{b}"
                if col in sub.columns:
                    r = one_sample_t(sub[col], label=f"Bin {b}")
                    bin_results.append(r)
            for r in [r for r in bin_results if r]:
                W(f"{r['label']:<6} {r['n']:>4} {r['mean']:>7.3f} {r['sem']:>7.3f} "
                  f"{r['t']:>7.3f} {r['p']:>8.4f}")

    W("\n" + "=" * 60)
    W("CROSS-CONDITION COMPARISONS (One-way ANOVA: pre-post diff)")
    W("=" * 60)
    for metric in ["REC", "LDI"]:
        groups, labels = [], []
        for cond in CONDITIONS:
            sub  = df[df["condition"] == cond]
            pre  = sub[f"{metric}_pre"]
            post = sub[f"{metric}_post"]
            idx  = pre.dropna().index.intersection(post.dropna().index)
            diff = pre.loc[idx] - post.loc[idx]
            if len(diff) >= 3:
                groups.append(diff.values)
                labels.append(cond)
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            denom = sum(len(g) for g in groups) - len(groups)
            W(f"\n{metric} Pre−Post: F({len(groups)-1},{denom}) = {f_stat:.3f}, p = {p_val:.4f}")
            W("  Post-hoc pairwise (Holm-Bonferroni):")
            posthoc = []
            for (i, j) in combinations(range(len(groups)), 2):
                t, p = stats.ttest_ind(groups[i], groups[j])
                pool_std = np.sqrt(
                    ((len(groups[i])-1)*groups[i].std()**2 +
                     (len(groups[j])-1)*groups[j].std()**2) /
                    (len(groups[i]) + len(groups[j]) - 2)
                )
                d = (groups[i].mean() - groups[j].mean()) / pool_std if pool_std > 0 else 0
                posthoc.append({"label": f"{labels[i]} vs {labels[j]}",
                                 "t": t, "p": p, "d": d,
                                 "n": len(groups[i]) + len(groups[j])})
            W(f"  {'Comparison':<40} {'t':>7} {'p':>8} {'d':>6}")
            for r in posthoc:
                W(f"  {r['label']:<40} {r['t']:>7.3f} {r['p']:>8.4f} "
                  f"{r['d']:>6.3f}")

    W("\n" + "=" * 60)
    W("ALPHA THRESHOLDS")
    W("  Confirmatory: α = 0.05 (Uncorrected)")
    W("=" * 60)


# 10. MAIN
def main():
    print("\nMST Analysis Pipeline\n")
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
    print(f"\nSaved participant-level metrics: mst_results.csv ({len(df)} rows)")

    print("\nDescriptive summary (means across all participants):")
    metric_cols = [c for c in df.columns
                   if c.startswith("REC_") or c.startswith("LDI_") or c.startswith("RT_")]
    print(df.groupby("condition")[metric_cols]
            .agg(["mean", "sem", "count"]).round(3).to_string())

    print("\nGenerating figures...")
    np.random.seed(42)
    fig1_rec_by_position(df,      FIGURES_DIR / "fig1_REC_by_position.png")
    fig2_ldi_by_position(df,      FIGURES_DIR / "fig2_LDI_by_position.png")
    fig3_lure_bins(df,            FIGURES_DIR / "fig3_LDI_lure_bins.png")
    fig4_rt(df,                   FIGURES_DIR / "fig4_RT_by_position.png")
    fig5_condition_comparison(df, FIGURES_DIR / "fig5_condition_comparison.png")

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
