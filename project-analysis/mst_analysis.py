"""
MST (Mnemonic Similarity Task) — Complete Behavioral Analysis DATA ANALYSIS

TASK phase (encoding):
  - 280 images (Objects & Scenes), all 'a' versions (originals)
  - Responses: f / j  (indoor/outdoor or living/non-living depending on condition)
  - Key metric: encoding_task_accuracy, reaction time

TEST phase (recognition memory):
  - 150 trials: targets (a), lures (b), foils (new)
  - Responses: o=old, s=similar, n=new
  - Positions: pre / mid / post (temporal encoding position) / none (foils)
  - Key metrics:
      * Target HR  (hits)     : responded 'o' to targets
      * Lure CR    (lure discrimination): responded 's' to lures  <- MST core metric
      * Foil CR    (correct rejection): responded 'n' to foils
      * Lure FA    (false alarms): responded 'o' to lures
      * LDI        (Lure Discrimination Index) = Lure CR - Lure FA
      * REC Index  (Recognition) = Target HR - Foil FA

Datasets: Both_item_task, item_only, task_only

Usage:
    python3 mst_data_analysis.py --base_dir .
"""

import os, glob, warnings, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# Palette
C = {
    "both":      "#2196F3",
    "item_only": "#E91E63",
    "task_only": "#4CAF50",
    "target":    "#3F51B5",
    "lure":      "#FF5722",
    "foil":      "#607D8B",
    "Objects":   "#FF9800",
    "Scenes":    "#9C27B0",
    "pre":       "#26C6DA",
    "mid":       "#AB47BC",
    "post":      "#EF5350",
}
DATASET_COLORS = [C["both"], C["item_only"], C["task_only"]]
FOLDER_ALIASES = {
    "both":      ["Both_item_task", "both_data", "both"],
    "item_only": ["item_only", "item_only_data"],
    "task_only": ["task_only", "task_only_data"],
}
DATASETS  = ['both', 'item_only', 'task_only']
DS_LABELS = ['Both', 'Item Only', 'Task Only']

# DATA LOADING

def find_folder(base, aliases):
    for a in aliases:
        p = os.path.join(base, a)
        if os.path.isdir(p):
            return p
    return None


def parse_task_file(path, dataset, subject_id):
    df = pd.read_csv(path)
    trials = df[df['trials.thisN'].notna()].copy()
    if trials.empty:
        return None

    resp9 = trials['trials.key_resp_9.keys'] if 'trials.key_resp_9.keys' in trials.columns else pd.Series(np.nan, index=trials.index)
    resp8 = trials['trials.key_resp_8.keys'] if 'trials.key_resp_8.keys' in trials.columns else pd.Series(np.nan, index=trials.index)
    rt9   = trials['trials.key_resp_9.rt']   if 'trials.key_resp_9.rt'   in trials.columns else pd.Series(np.nan, index=trials.index)
    rt8   = trials['trials.key_resp_8.rt']   if 'trials.key_resp_8.rt'   in trials.columns else pd.Series(np.nan, index=trials.index)
    trials['response'] = resp9.fillna(resp8)
    trials['rt']       = rt9.fillna(rt8)

    trials['image_folder'] = trials['image_path'].apply(
        lambda x: str(x).replace('\\', '/').split('/')[0])
    trials['image_file'] = trials['image_path'].apply(
        lambda x: str(x).replace('\\', '/').split('/')[-1])

    if 'encoding_task_accuracy' in df.columns:
        enc_acc = df['encoding_task_accuracy'].dropna().values
        encoding_accuracy = float(enc_acc[0]) if len(enc_acc) > 0 else np.nan
    else:
        encoding_accuracy = np.nan

    trials['dataset']           = dataset
    trials['subject_id']        = subject_id
    trials['encoding_accuracy'] = encoding_accuracy
    return trials


def parse_test_file(path, dataset, subject_id):
    df = pd.read_csv(path)
    trials = df[df['trials.thisN'].notna()].copy()
    if trials.empty:
        return None

    trials['image_folder'] = trials['image_path'].apply(
        lambda x: 'Objects' if 'Object' in str(x) else ('Scenes' if 'Scene' in str(x) else 'Foils'))
    trials['image_file'] = trials['image_path'].apply(
        lambda x: str(x).replace('\\\\', '/').split('/')[-1])
    trials['image_version'] = trials['image_file'].str[-5]

    trials['stimulus_type'] = trials.apply(lambda r:
        'target' if r['image_folder'] != 'Foils' and r['image_version'] == 'a'
        else ('lure' if r['image_folder'] != 'Foils' and r['image_version'] == 'b'
        else 'foil'), axis=1)

    trials['response'] = trials['trials.key_resp_3.keys']
    trials['rt']       = trials['trials.key_resp_3.rt']

    trials['correct'] = trials.apply(lambda r:
        (r['stimulus_type'] == 'target' and r['response'] == 'o') or
        (r['stimulus_type'] == 'lure'   and r['response'] == 's') or
        (r['stimulus_type'] == 'foil'   and r['response'] == 'n'), axis=1)

    trials['dataset']    = dataset
    trials['subject_id'] = subject_id
    return trials


def load_all(base_dir):
    task_frames, test_frames = [], []
    for ds_key, aliases in FOLDER_ALIASES.items():
        folder = find_folder(base_dir, aliases)
        if not folder:
            print(f"  [SKIP] {ds_key} not found")
            continue
        task_csvs = sorted(glob.glob(os.path.join(folder, '**', '*_MST_task_*.csv'), recursive=True))
        test_csvs = sorted(glob.glob(os.path.join(folder, '**', '*_MST_test_*.csv'), recursive=True))
        print(f"  {ds_key}: {len(task_csvs)} task + {len(test_csvs)} test files")
        for path in task_csvs:
            sid = os.path.basename(path).split('_MST_task_')[0]
            df = parse_task_file(path, ds_key, sid)
            if df is not None:
                task_frames.append(df)
        for path in test_csvs:
            sid = os.path.basename(path).split('_MST_test_')[0]
            df = parse_test_file(path, ds_key, sid)
            if df is not None:
                test_frames.append(df)

    task_df = pd.concat(task_frames, ignore_index=True) if task_frames else pd.DataFrame()
    test_df = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
    return task_df, test_df


# METRICS

def compute_task_metrics(task_df):
    rows = []
    for (ds, sid), g in task_df.groupby(['dataset', 'subject_id']):
        total   = len(g)
        n_resp  = g['response'].notna().sum()
        miss_rt = (g['rt'].isna() | (g['response'].isna())).sum()

        row = dict(dataset=ds, subject_id=sid,
                   n_trials=total, n_responded=n_resp, n_missed=miss_rt,
                   miss_rate=miss_rt / total,
                   encoding_accuracy=g['encoding_accuracy'].iloc[0])

        rt_vals = g.loc[g['rt'].notna(), 'rt']
        row.update(rt_mean=rt_vals.mean(), rt_median=rt_vals.median(),
                   rt_std=rt_vals.std(),
                   rt_cv=rt_vals.std() / rt_vals.mean() if rt_vals.mean() > 0 else np.nan)

        for folder in ['Objects', 'Scenes']:
            fg   = g[g['image_folder'] == folder]
            rt_f = fg.loc[fg['rt'].notna(), 'rt']
            row[f'{folder}_n']       = len(fg)
            row[f'{folder}_rt_mean'] = rt_f.mean() if len(rt_f) > 0 else np.nan
            row[f'{folder}_rt_std']  = rt_f.std()  if len(rt_f) > 1 else np.nan
            row[f'{folder}_miss']    = fg['rt'].isna().sum()

        rows.append(row)
    return pd.DataFrame(rows)


def compute_test_metrics(test_df):
    rows = []
    for (ds, sid), g in test_df.groupby(['dataset', 'subject_id']):
        row = dict(dataset=ds, subject_id=sid, n_trials=len(g))

        for stype in ['target', 'lure', 'foil']:
            sg = g[g['stimulus_type'] == stype]
            if sg.empty:
                continue
            row[f'{stype}_n']   = len(sg)
            row[f'{stype}_acc'] = sg['correct'].mean()
            rt_v = sg.loc[sg['rt'].notna(), 'rt']
            row[f'{stype}_rt_mean']   = rt_v.mean()
            row[f'{stype}_rt_median'] = rt_v.median()
            row[f'{stype}_rt_std']    = rt_v.std()
            for resp in ['o', 's', 'n']:
                row[f'{stype}_resp_{resp}'] = (sg['response'] == resp).sum() / max(len(sg), 1)

        target_hr    = row.get('target_resp_o', np.nan)
        lure_cr      = row.get('lure_resp_s',   np.nan)
        foil_sim_rate = row.get('foil_resp_s',  np.nan)
        foil_fa      = row.get('foil_resp_o',   np.nan)

        # LDI = lure_similar_rate - foil_similar_rate (foil-corrected; Yassa et al.)
        # Matches the formula used in mst_analysis_final.py
        row['LDI']         = lure_cr - foil_sim_rate
        row['REC']         = target_hr - foil_fa
        row['overall_acc'] = g['correct'].mean()

        for pos in ['pre', 'mid', 'post']:
            pg = g[g['position_of_stimuli'] == pos]
            if pg.empty:
                continue
            for stype in ['target', 'lure']:
                sg = pg[pg['stimulus_type'] == stype]
                if not sg.empty:
                    row[f'{pos}_{stype}_acc'] = sg['correct'].mean()
                    row[f'{pos}_{stype}_rt']  = sg.loc[sg['rt'].notna(), 'rt'].mean()

        for folder in ['Objects', 'Scenes']:
            fg = g[g['image_folder'] == folder]
            if fg.empty:
                continue
            for stype in ['target', 'lure']:
                sg = fg[fg['stimulus_type'] == stype]
                if not sg.empty:
                    row[f'{folder}_{stype}_acc'] = sg['correct'].mean()
                    cr_col = 's' if stype == 'lure' else 'o'
                    row[f'{folder}_{stype}_cr'] = (sg['response'] == cr_col).mean()

        rows.append(row)
    return pd.DataFrame(rows)


# PLOTTING HELPERS

def save(pdf, fig):
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def violin_with_strip(ax, data, x, y, palette, order, title, ylabel):
    sns.violinplot(data=data, x=x, y=y, palette=palette, order=order,
                   ax=ax, inner=None, alpha=0.5, cut=0)
    sns.stripplot(data=data, x=x, y=y, palette=palette, order=order,
                  ax=ax, size=4, jitter=True, alpha=0.7, zorder=2)
    for i, grp in enumerate(order):
        val = data[data[x] == grp][y].mean()
        ax.hlines(val, i - 0.3, i + 0.3, colors='black', linewidths=2.5, zorder=3)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')


def grouped_bar(ax, means, sems, groups, xlabel_list, colors, title, ylabel, ylim=None):
    n_cats = len(groups)
    x      = np.arange(len(xlabel_list))
    width  = 0.8 / n_cats
    for i, (grp, col) in enumerate(zip(groups, colors)):
        offset = (i - n_cats / 2 + 0.5) * width
        ax.bar(x + offset, means[i], width * 0.9, yerr=sems[i],
               label=grp, color=col, alpha=0.85, capsize=4,
               error_kw=dict(elinewidth=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels(xlabel_list, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    if ylim:
        ax.set_ylim(ylim)


# PLOT SECTIONS  (data analysis only — no p-values / sig bars)

# 1. TASK: Encoding Accuracy
def plot_task_encoding_accuracy(tm, pdf):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('TASK — Encoding Accuracy', fontsize=14, fontweight='bold')

    ax = axes[0]
    violin_with_strip(ax, tm, 'dataset', 'encoding_accuracy',
                      {d: c for d, c in zip(DATASETS, DATASET_COLORS)},
                      DATASETS, 'Encoding Accuracy by Dataset', 'Accuracy')
    ax.set_xticklabels(DS_LABELS)
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    for ds, col in zip(DATASETS, DATASET_COLORS):
        vals = tm[tm['dataset'] == ds]['encoding_accuracy'].dropna()
        ax.hist(vals, bins=15, alpha=0.6, color=col, label=ds, edgecolor='white')
    ax.set_title('Distribution of Encoding Accuracy', fontsize=11, fontweight='bold')
    ax.set_xlabel('Accuracy'); ax.set_ylabel('Count')
    ax.legend(labels=DS_LABELS, fontsize=9)

    ax = axes[2]
    means = [tm[tm['dataset'] == d]['encoding_accuracy'].mean() for d in DATASETS]
    sems  = [tm[tm['dataset'] == d]['encoding_accuracy'].sem()  for d in DATASETS]
    bars  = ax.bar(DS_LABELS, means, yerr=sems, color=DATASET_COLORS, alpha=0.85,
                   capsize=6, edgecolor='white', error_kw=dict(elinewidth=1.5))
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.01, f'{m:.3f}',
                ha='center', fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_title('Mean Encoding Accuracy ± SEM', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy')
    save(pdf, fig)


# 2. TASK: Reaction Time
def plot_task_rt(tm, task_df, pdf):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('TASK — Reaction Time Analysis', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    for ds, col in zip(DATASETS, DATASET_COLORS):
        vals = task_df[task_df['dataset'] == ds]['rt'].dropna()
        if vals.std() > 0:
            vals.plot.kde(ax=ax, color=col, linewidth=2.5, label=ds)
    ax.set_title('RT Distribution by Dataset', fontsize=11, fontweight='bold')
    ax.set_xlabel('RT (s)'); ax.set_ylabel('Density')
    ax.legend(labels=DS_LABELS, fontsize=9); ax.set_xlim(0, 6)

    ax = axes[0, 1]
    violin_with_strip(ax, tm, 'dataset', 'rt_mean',
                      {d: c for d, c in zip(DATASETS, DATASET_COLORS)},
                      DATASETS, 'Mean RT per Subject', 'RT (s)')
    ax.set_xticklabels(DS_LABELS)

    ax = axes[0, 2]
    obj_means = [tm[tm['dataset'] == d]['Objects_rt_mean'].mean() for d in DATASETS]
    sce_means = [tm[tm['dataset'] == d]['Scenes_rt_mean'].mean()  for d in DATASETS]
    obj_sems  = [tm[tm['dataset'] == d]['Objects_rt_mean'].sem()  for d in DATASETS]
    sce_sems  = [tm[tm['dataset'] == d]['Scenes_rt_mean'].sem()   for d in DATASETS]
    x = np.arange(3); w = 0.35
    ax.bar(x - w/2, obj_means, w, yerr=obj_sems, label='Objects', color=C['Objects'], alpha=0.85, capsize=5)
    ax.bar(x + w/2, sce_means, w, yerr=sce_sems, label='Scenes',  color=C['Scenes'],  alpha=0.85, capsize=5)
    ax.set_xticks(x); ax.set_xticklabels(DS_LABELS)
    ax.set_title('Mean RT: Objects vs Scenes', fontsize=11, fontweight='bold')
    ax.set_ylabel('RT (s)'); ax.legend(fontsize=9)

    ax = axes[1, 0]
    for ds, col in zip(DATASETS, DATASET_COLORS):
        vals = task_df[task_df['dataset'] == ds]['rt'].dropna()
        ax.hist(vals, bins=40, alpha=0.5, color=col, label=ds, edgecolor='none', range=(0, 6))
    ax.set_title('RT Histogram by Dataset', fontsize=11, fontweight='bold')
    ax.set_xlabel('RT (s)'); ax.set_ylabel('Count')
    ax.legend(labels=DS_LABELS, fontsize=9)

    ax = axes[1, 1]
    violin_with_strip(ax, tm, 'dataset', 'rt_cv',
                      {d: c for d, c in zip(DATASETS, DATASET_COLORS)},
                      DATASETS, 'RT Coefficient of Variation', 'CoV')
    ax.set_xticklabels(DS_LABELS)

    ax = axes[1, 2]
    means = [tm[tm['dataset'] == d]['miss_rate'].mean() for d in DATASETS]
    sems  = [tm[tm['dataset'] == d]['miss_rate'].sem()  for d in DATASETS]
    ax.bar(DS_LABELS, means, yerr=sems, color=DATASET_COLORS, alpha=0.85,
           capsize=6, edgecolor='white')
    ax.set_title('Miss Rate (No Response)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Miss Rate')
    save(pdf, fig)


# 3. TASK: RT by Stimulus Category
def plot_task_rt_by_category(task_df, pdf):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('TASK — RT by Stimulus Category per Dataset', fontsize=14, fontweight='bold')

    for ax, (ds, dslabel, col) in zip(axes, zip(DATASETS, DS_LABELS, DATASET_COLORS)):
        sub     = task_df[task_df['dataset'] == ds].copy()
        folders = [f for f in ['Objects', 'Scenes'] if f in sub['image_folder'].unique()]
        if not folders:
            ax.set_visible(False); continue
        data_list = [sub[sub['image_folder'] == f]['rt'].dropna() for f in folders]
        bp = ax.boxplot(data_list, patch_artist=True, labels=folders,
                        medianprops=dict(color='white', linewidth=2))
        for patch, fc in zip(bp['boxes'], [C.get(f, col) for f in folders]):
            patch.set_facecolor(fc); patch.set_alpha(0.75)
        ax.set_title(dslabel, fontsize=11, fontweight='bold')
        ax.set_ylabel('RT (s)'); ax.set_ylim(0, 7)
    save(pdf, fig)


# 4. TEST: Overall Accuracy & Response Distributions
def plot_test_overall(sm, pdf):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('TEST — Overall Accuracy & Response Distributions', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    violin_with_strip(ax, sm, 'dataset', 'overall_acc',
                      {d: c for d, c in zip(DATASETS, DATASET_COLORS)},
                      DATASETS, 'Overall Accuracy', 'Accuracy')
    ax.set_xticklabels(DS_LABELS); ax.set_ylim(0, 1)

    ax = axes[0, 1]
    x = np.arange(3); w = 0.25
    for i, (st, stcol) in enumerate(zip(['target', 'lure', 'foil'],
                                         [C['target'], C['lure'], C['foil']])):
        means = [sm[sm['dataset'] == d][f'{st}_acc'].mean() for d in DATASETS]
        sems  = [sm[sm['dataset'] == d][f'{st}_acc'].sem()  for d in DATASETS]
        ax.bar(x + (i - 1) * w, means, w, yerr=sems, label=st.capitalize(),
               color=stcol, alpha=0.85, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(DS_LABELS)
    ax.set_title('Accuracy by Stimulus Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy'); ax.legend(fontsize=9); ax.set_ylim(0, 1)

    ax = axes[0, 2]
    violin_with_strip(ax, sm, 'dataset', 'LDI',
                      {d: c for d, c in zip(DATASETS, DATASET_COLORS)},
                      DATASETS, 'Lure Discrimination Index (LDI)', 'LDI')
    ax.set_xticklabels(DS_LABELS)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)

    ax = axes[1, 0]
    violin_with_strip(ax, sm, 'dataset', 'REC',
                      {d: c for d, c in zip(DATASETS, DATASET_COLORS)},
                      DATASETS, 'Recognition Index (REC)', 'REC')
    ax.set_xticklabels(DS_LABELS)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)

    ax = axes[1, 1]
    for ds, col in zip(DATASETS, DATASET_COLORS):
        sm[sm['dataset'] == ds]['LDI'].dropna().pipe(
            lambda v: ax.hist(v, bins=12, alpha=0.6, color=col, edgecolor='white'))
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title('LDI Distribution by Dataset', fontsize=11, fontweight='bold')
    ax.set_xlabel('LDI'); ax.set_ylabel('Count')

    ax = axes[1, 2]
    for ds, col in zip(DATASETS, DATASET_COLORS):
        sm[sm['dataset'] == ds]['REC'].dropna().pipe(
            lambda v: ax.hist(v, bins=12, alpha=0.6, color=col, edgecolor='white'))
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title('REC Distribution by Dataset', fontsize=11, fontweight='bold')
    ax.set_xlabel('REC'); ax.set_ylabel('Count')
    save(pdf, fig)


# 5. TEST: Response Proportions
def plot_test_response_proportions(sm, pdf):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('TEST — Response Proportions by Stimulus Type & Dataset',
                 fontsize=14, fontweight='bold')

    resp_labels = {'o': 'Old', 's': 'Similar', 'n': 'New'}
    resp_colors = {'o': '#1976D2', 's': '#F57C00', 'n': '#388E3C'}

    for ax, st in zip(axes, ['target', 'lure', 'foil']):
        resp_means = {r: [sm[sm['dataset'] == d][f'{st}_resp_{r}'].mean() for d in DATASETS]
                      for r in ['o', 's', 'n']}
        x = np.arange(3); bottom = np.zeros(3)
        for resp, col in resp_colors.items():
            vals = np.array(resp_means[resp])
            ax.bar(x, vals, bottom=bottom, color=col, alpha=0.85, label=resp_labels[resp])
            for xi, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0.05:
                    ax.text(xi, b + v / 2, f'{v:.2f}', ha='center', va='center',
                            fontsize=8, fontweight='bold', color='white')
            bottom += vals
        ax.set_xticks(x); ax.set_xticklabels(DS_LABELS)
        ax.set_title(f'{st.capitalize()} Stimuli', fontsize=11, fontweight='bold')
        ax.set_ylabel('Proportion'); ax.set_ylim(0, 1.02)
        if st == 'foil':
            ax.legend(fontsize=9, loc='upper right')
    save(pdf, fig)


# 6. TEST: By Temporal Position
def plot_test_by_position(sm, pdf):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TEST — Performance by Temporal Encoding Position',
                 fontsize=14, fontweight='bold')

    positions  = ['pre', 'mid', 'post']
    pos_colors = [C['pre'], C['mid'], C['post']]

    grouped_bar(axes[0, 0],
        means=[[sm[sm['dataset'] == d][f'{p}_target_acc'].mean() for d in DATASETS] for p in positions],
        sems= [[sm[sm['dataset'] == d][f'{p}_target_acc'].sem()  for d in DATASETS] for p in positions],
        groups=positions, xlabel_list=DS_LABELS, colors=pos_colors,
        title='Target Accuracy by Position', ylabel='Accuracy', ylim=(0, 1.1))

    grouped_bar(axes[0, 1],
        means=[[sm[sm['dataset'] == d][f'{p}_lure_acc'].mean() for d in DATASETS] for p in positions],
        sems= [[sm[sm['dataset'] == d][f'{p}_lure_acc'].sem()  for d in DATASETS] for p in positions],
        groups=positions, xlabel_list=DS_LABELS, colors=pos_colors,
        title='Lure Correct Rejection by Position', ylabel='Accuracy', ylim=(0, 1.1))

    grouped_bar(axes[1, 0],
        means=[[sm[sm['dataset'] == d][f'{p}_target_rt'].mean() for d in DATASETS] for p in positions],
        sems= [[sm[sm['dataset'] == d][f'{p}_target_rt'].sem()  for d in DATASETS] for p in positions],
        groups=positions, xlabel_list=DS_LABELS, colors=pos_colors,
        title='Target RT by Position', ylabel='RT (s)')

    grouped_bar(axes[1, 1],
        means=[[sm[sm['dataset'] == d][f'{p}_lure_rt'].mean() for d in DATASETS] for p in positions],
        sems= [[sm[sm['dataset'] == d][f'{p}_lure_rt'].sem()  for d in DATASETS] for p in positions],
        groups=positions, xlabel_list=DS_LABELS, colors=pos_colors,
        title='Lure RT by Position', ylabel='RT (s)')
    save(pdf, fig)


# 7. TEST: Objects vs Scenes
def plot_test_by_category(sm, pdf):
    ds_with_scenes = [d for d in DATASETS
                      if sm[sm['dataset'] == d]['Scenes_target_acc'].notna().any()]
    if not ds_with_scenes:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TEST — Objects vs Scenes Performance', fontsize=14, fontweight='bold')

    for ax_row, stype in zip([0, 1], ['target', 'lure']):
        col_label = 'Target' if stype == 'target' else 'Lure'
        x         = np.arange(len(ds_with_scenes)); w = 0.35
        x_labels  = [DS_LABELS[DATASETS.index(d)] for d in ds_with_scenes]

        for col_idx, (cat, cat_col) in enumerate(zip(['Objects', 'Scenes'],
                                                      [C['Objects'], C['Scenes']])):
            m = [sm[sm['dataset'] == d][f'{cat}_{stype}_acc'].mean() for d in ds_with_scenes]
            e = [sm[sm['dataset'] == d][f'{cat}_{stype}_acc'].sem()  for d in ds_with_scenes]
            axes[ax_row, 0].bar(x + (col_idx - 0.5) * w, m, w, yerr=e,
                                label=cat, color=cat_col, alpha=0.85, capsize=5)

            cr = [sm[sm['dataset'] == d][f'{cat}_{stype}_cr'].mean() for d in ds_with_scenes]
            er = [sm[sm['dataset'] == d][f'{cat}_{stype}_cr'].sem()  for d in ds_with_scenes]
            axes[ax_row, 1].bar(x + (col_idx - 0.5) * w, cr, w, yerr=er,
                                label=cat, color=cat_col, alpha=0.85, capsize=5)

        for c in [0, 1]:
            axes[ax_row, c].set_xticks(x)
            axes[ax_row, c].set_xticklabels(x_labels)
            axes[ax_row, c].legend(fontsize=9)
            axes[ax_row, c].set_ylim(0, 1.1)

        axes[ax_row, 0].set_title(f'{col_label} Accuracy: Objects vs Scenes',
                                   fontsize=11, fontweight='bold')
        axes[ax_row, 0].set_ylabel('Accuracy')
        axes[ax_row, 1].set_title(f'{col_label} Correct Response Rate: Objects vs Scenes',
                                   fontsize=11, fontweight='bold')
        axes[ax_row, 1].set_ylabel('Proportion')
    save(pdf, fig)


# 8. TEST: RT Analysis
def plot_test_rt(sm, test_df, pdf):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('TEST — Reaction Time Analysis', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    for ds, col in zip(DATASETS, DATASET_COLORS):
        vals = test_df[test_df['dataset'] == ds]['rt'].dropna()
        if vals.std() > 0:
            vals.plot.kde(ax=ax, color=col, linewidth=2.5)
    ax.set_title('RT Distribution by Dataset', fontsize=11, fontweight='bold')
    ax.set_xlabel('RT (s)'); ax.set_ylabel('Density'); ax.set_xlim(0, 35)

    ax = axes[0, 1]
    x = np.arange(3); w = 0.25
    for i, (st, stcol) in enumerate(zip(['target', 'lure', 'foil'],
                                         [C['target'], C['lure'], C['foil']])):
        means = [sm[sm['dataset'] == d][f'{st}_rt_mean'].mean() for d in DATASETS]
        sems  = [sm[sm['dataset'] == d][f'{st}_rt_mean'].sem()  for d in DATASETS]
        ax.bar(x + (i - 1) * w, means, w, yerr=sems, label=st.capitalize(),
               color=stcol, alpha=0.85, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(DS_LABELS)
    ax.set_title('Mean RT by Stimulus Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('RT (s)'); ax.legend(fontsize=9)

    ax = axes[0, 2]
    rt_all = pd.concat([
        test_df[test_df['stimulus_type'] == st][['rt', 'dataset']].assign(stype=st)
        for st in ['target', 'lure', 'foil']
    ])
    sns.violinplot(data=rt_all, x='stype', y='rt',
                   palette={'target': C['target'], 'lure': C['lure'], 'foil': C['foil']},
                   ax=ax, inner='box', cut=0)
    ax.set_title('RT by Stimulus Type (All)', fontsize=11, fontweight='bold')
    ax.set_xlabel(''); ax.set_ylabel('RT (s)')

    positions  = ['pre', 'mid', 'post']
    pos_colors = [C['pre'], C['mid'], C['post']]

    grouped_bar(axes[1, 0],
        means=[[sm[sm['dataset'] == d][f'{p}_target_rt'].mean() for d in DATASETS] for p in positions],
        sems= [[sm[sm['dataset'] == d][f'{p}_target_rt'].sem()  for d in DATASETS] for p in positions],
        groups=positions, xlabel_list=DS_LABELS, colors=pos_colors,
        title='Target RT by Temporal Position', ylabel='RT (s)')

    grouped_bar(axes[1, 1],
        means=[[sm[sm['dataset'] == d][f'{p}_lure_rt'].mean() for d in DATASETS] for p in positions],
        sems= [[sm[sm['dataset'] == d][f'{p}_lure_rt'].sem()  for d in DATASETS] for p in positions],
        groups=positions, xlabel_list=DS_LABELS, colors=pos_colors,
        title='Lure RT by Temporal Position', ylabel='RT (s)')

    ax = axes[1, 2]
    correct_rt   = [test_df[(test_df['dataset'] == d) & (test_df['correct'] == True)]['rt'].mean()
                    for d in DATASETS]
    incorrect_rt = [test_df[(test_df['dataset'] == d) & (test_df['correct'] == False)]['rt'].mean()
                    for d in DATASETS]
    x = np.arange(3); w = 0.35
    ax.bar(x - w/2, correct_rt,   w, label='Correct',   color='#43A047', alpha=0.85)
    ax.bar(x + w/2, incorrect_rt, w, label='Incorrect',  color='#E53935', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(DS_LABELS)
    ax.set_title('RT: Correct vs Incorrect', fontsize=11, fontweight='bold')
    ax.set_ylabel('RT (s)'); ax.legend(fontsize=9)
    save(pdf, fig)


# 9. SUMMARY STATISTICS TABLE
def plot_stats_summary(tm, sm, pdf):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Summary Statistics Table', fontsize=14, fontweight='bold')

    task_rows = []
    for ds, label in zip(DATASETS, DS_LABELS):
        sub = tm[tm['dataset'] == ds]
        task_rows.append([
            label,
            f"{sub['encoding_accuracy'].mean():.3f} ± {sub['encoding_accuracy'].std():.3f}",
            f"{sub['rt_mean'].mean():.2f} ± {sub['rt_mean'].std():.2f}",
            f"{sub['rt_median'].mean():.2f}",
            f"{sub['miss_rate'].mean() * 100:.1f}%",
            str(len(sub)),
        ])
    t1 = axes[0].table(
        cellText=task_rows,
        colLabels=['Dataset', 'Enc. Acc (M±SD)', 'Mean RT (M±SD)', 'Median RT', 'Miss Rate', 'N'],
        loc='center', cellLoc='center')
    t1.auto_set_font_size(False); t1.set_fontsize(9.5); t1.scale(1, 2.2)
    axes[0].set_title('TASK Phase Summary', fontsize=12, fontweight='bold', pad=20)
    axes[0].axis('off')

    test_rows = []
    for ds, label in zip(DATASETS, DS_LABELS):
        sub = sm[sm['dataset'] == ds]
        test_rows.append([
            label,
            f"{sub['overall_acc'].mean():.3f} ± {sub['overall_acc'].std():.3f}",
            f"{sub['target_acc'].mean():.3f}",
            f"{sub['lure_acc'].mean():.3f}",
            f"{sub['foil_acc'].mean():.3f}",
            f"{sub['LDI'].mean():.3f} ± {sub['LDI'].std():.3f}",
            f"{sub['REC'].mean():.3f} ± {sub['REC'].std():.3f}",
            str(len(sub)),
        ])
    t2 = axes[1].table(
        cellText=test_rows,
        colLabels=['Dataset', 'Overall Acc', 'Target HR', 'Lure CR', 'Foil CR',
                   'LDI (M±SD)', 'REC (M±SD)', 'N'],
        loc='center', cellLoc='center')
    t2.auto_set_font_size(False); t2.set_fontsize(9); t2.scale(1, 2.2)
    axes[1].set_title('TEST Phase Summary', fontsize=12, fontweight='bold', pad=20)
    axes[1].axis('off')
    save(pdf, fig)


# MAIN
def main(base_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  MST Analysis  |  base: {base_dir}\n")

    task_df, test_df = load_all(base_dir)
    print(f"\nTotal task rows: {len(task_df)}  |  test rows: {len(test_df)}")

    print("Computing per-subject metrics …")
    tm = compute_task_metrics(task_df)
    sm = compute_test_metrics(test_df)

    tm.to_csv(os.path.join(out_dir, 'task_subject_metrics.csv'), index=False)
    sm.to_csv(os.path.join(out_dir, 'test_subject_metrics.csv'), index=False)
    print("  Saved task_subject_metrics.csv & test_subject_metrics.csv")

    pdf_path = os.path.join(out_dir, 'MST_analysis_report.pdf')
    print(f"Generating PDF: {pdf_path} …")

    with PdfPages(pdf_path) as pdf:
        # Cover page
        fig = plt.figure(figsize=(12, 7))
        fig.patch.set_facecolor('#1a1a2e')
        fig.text(0.5, 0.6, 'MST Behavioral Analysis', ha='center', fontsize=26,
                 fontweight='bold', color='white')
        fig.text(0.5, 0.5, 'Both · Item-Only · Task-Only', ha='center', fontsize=16, color='#aaa')
        fig.text(0.5, 0.4,
                 f'N: both={len(tm[tm.dataset=="both"])}, '
                 f'item_only={len(tm[tm.dataset=="item_only"])}, '
                 f'task_only={len(tm[tm.dataset=="task_only"])}',
                 ha='center', fontsize=12, color='#ccc')
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

        print("  1/8  Task: Encoding accuracy …")
        plot_task_encoding_accuracy(tm, pdf)

        print("  2/8  Task: Reaction time …")
        plot_task_rt(tm, task_df, pdf)

        print("  3/8  Task: RT by stimulus category …")
        plot_task_rt_by_category(task_df, pdf)

        print("  4/8  Test: Overall accuracy & indices …")
        plot_test_overall(sm, pdf)

        print("  5/8  Test: Response proportions …")
        plot_test_response_proportions(sm, pdf)

        print("  6/8  Test: Performance by temporal position …")
        plot_test_by_position(sm, pdf)

        print("  7/8  Test: Objects vs Scenes …")
        plot_test_by_category(sm, pdf)

        print("  8/8  Test: Reaction time analysis …")
        plot_test_rt(sm, test_df, pdf)

        print("  Summary statistics table …")
        plot_stats_summary(tm, sm, pdf)

    print(f"\n  Done! -> {pdf_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MST Behavioral Data Analysis — data analysis only')
    parser.add_argument('--base_dir', required=True,
                        help='Root folder containing Both_item_task/, item_only/, task_only/')
    parser.add_argument('--out_dir', default='./mst_analysis_output',
                        help='Output directory for PDF and CSVs')
    args = parser.parse_args()
    main(args.base_dir, args.out_dir)
