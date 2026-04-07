"""
Redraw all SF-P3O paper figures from raw experiment CSV data.
Each figure: one environment, large & clear, proper axis labels.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict
import os, glob, json

# ==================== Configuration ====================
BASE = r'C:\Users\33277\Desktop'
RUNS    = os.path.join(BASE, 'sf_p3o_runs')
HIGHDIM = os.path.join(BASE, 'sf_p3o_highdim')
LONGRUN = os.path.join(BASE, 'sf_p3o_longrun')
PROBE   = os.path.join(BASE, 'sf_p3o_probe')
ORIG_FIG = os.path.join(BASE, 'sf_p3o_figures')
OUT     = os.path.join(BASE, 'sf_p3o_figures', 'redrawn')
os.makedirs(OUT, exist_ok=True)

# Algorithms
MAIN_ALGOS = ['SF-P3O', 'PPO', 'CReLU', 'LayerNorm', 'L2-Init',
              'PLASTIC', 'ReDo', 'Shrink-Perturb']
ABLATION_ALGOS = ['SF-P3O', 'P3O-static', 'SF-P3O-NoDual',
                  'SF-P3O-NoFisher', 'SF-P3O-NoSpectral', 'SF-P3O-v1']

# Colors — SF-P3O in bold red, others distinguishable
COLORS = {
    'SF-P3O':            '#d62728',
    'PPO':               '#1f77b4',
    'CReLU':             '#9467bd',
    'LayerNorm':         '#7f7f7f',
    'L2-Init':           '#e377c2',
    'PLASTIC':           '#2ca02c',
    'ReDo':              '#ff7f0e',
    'Shrink-Perturb':    '#8c564b',
    'P3O-static':        '#17becf',
    'SF-P3O-NoDual':     '#bcbd22',
    'SF-P3O-NoFisher':   '#9467bd',
    'SF-P3O-NoSpectral': '#ff7f0e',
    'SF-P3O-v1':         '#8c564b',
}

# Display-friendly names
ALGO_DISPLAY = {
    'SF-P3O':            'SF-P3O (Ours)',
    'PPO':               'PPO',
    'CReLU':             'CReLU',
    'LayerNorm':         'LayerNorm',
    'L2-Init':           'L2-Init',
    'PLASTIC':           'PLASTIC',
    'ReDo':              'ReDo',
    'Shrink-Perturb':    'Shrink & Perturb',
    'P3O-static':        'P3O-static',
    'SF-P3O-NoDual':     'w/o Dual Pathway',
    'SF-P3O-NoFisher':   'w/o Fisher',
    'SF-P3O-NoSpectral': 'w/o Spectral',
    'SF-P3O-v1':         'SF-P3O-v1',
}

ENV_DISPLAY = {
    'HalfCheetah-v4': 'HalfCheetah',
    'Hopper-v4':      'Hopper',
    'Walker2d-v4':    'Walker2d',
    'Ant-v4':         'Ant',
    'Humanoid-v4':    'Humanoid',
}

# ==================== Data Loading ====================

def parse_reward(csv_path):
    """Return (steps, running_mean_return) arrays."""
    steps, vals = [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        f.readline()  # skip "step,tag" header
        for line in f:
            p = line.strip().split(',')
            if len(p) >= 5 and p[1] == 'reward':
                try:
                    steps.append(int(p[0]))
                    vals.append(float(p[4]))   # running_mean_return
                except (ValueError, IndexError):
                    pass
    return np.array(steps), np.array(vals)

def parse_diagnostics(csv_path):
    """Return dict with step, dead_neurons, spectral_norm, grad_norm,
    weight_change, explained_var, field7 (gate), field8."""
    out = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split(',')
            if len(p) >= 10 and p[1] == 'diagnostics':
                try:
                    out['step'].append(int(p[0]))
                    out['dead_neurons'].append(float(p[2]))
                    out['spectral_norm'].append(float(p[3]))
                    out['grad_norm'].append(float(p[4]))
                    out['weight_change'].append(float(p[5]))
                    out['explained_var'].append(float(p[6]))
                    out['gate'].append(float(p[7]))
                    out['field8'].append(float(p[8]))
                except (ValueError, IndexError):
                    pass
    return {k: np.array(v) for k, v in out.items()}

def find_runs(base_dirs, env, algo, steps_filter=None, probe=False):
    """Find all seed directories for env/algo."""
    results = []
    for d in (base_dirs if isinstance(base_dirs, list) else [base_dirs]):
        if probe:
            pat = os.path.join(d, f'probe_{env}_{algo}_seed*')
        elif steps_filter:
            pat = os.path.join(d, f'{env}_{algo}_seed*_steps{steps_filter}')
        else:
            pat = os.path.join(d, f'{env}_{algo}_seed*')
        results.extend(glob.glob(pat))
    return sorted(results)

def load_seeds(base_dirs, env, algo, steps_filter=None, probe=False, min_steps=None):
    """Load running-mean reward for all seeds. Returns list of (steps, vals).
    If min_steps is set, skip seeds whose last step < min_steps."""
    runs = find_runs(base_dirs, env, algo, steps_filter, probe)
    data = []
    for run in runs:
        csv = os.path.join(run, 'metrics.csv')
        if os.path.exists(csv):
            s, v = parse_reward(csv)
            if len(s) > 1:
                if min_steps and s[-1] < min_steps:
                    print(f'    [SKIP] {os.path.basename(run)}: only {s[-1]} steps')
                    continue
                data.append((s, v))
    return data

def load_diag_seeds(base_dirs, env, algo, steps_filter=None):
    """Load diagnostics for all seeds. Returns list of dicts."""
    runs = find_runs(base_dirs, env, algo, steps_filter)
    data = []
    for run in runs:
        csv = os.path.join(run, 'metrics.csv')
        if os.path.exists(csv):
            d = parse_diagnostics(csv)
            if len(d.get('step', [])) > 1:
                data.append(d)
    return data

def aggregate(seed_data, n_pts=500):
    """Interpolate all seeds to common grid → (x, mean, std)."""
    if not seed_data:
        return None, None, None
    max_step = min(d[0][-1] for d in seed_data)
    min_step = max(d[0][0] for d in seed_data)
    if min_step >= max_step:
        return None, None, None
    x = np.linspace(min_step, max_step, n_pts)
    interps = [np.interp(x, s, v) for s, v in seed_data]
    arr = np.array(interps)
    return x, arr.mean(0), arr.std(0)

def aggregate_diag(seed_data, key, n_pts=200):
    """Aggregate a diagnostics field across seeds."""
    if not seed_data:
        return None, None, None
    pairs = [(d['step'], d[key]) for d in seed_data if key in d and len(d[key]) > 1]
    if not pairs:
        return None, None, None
    return aggregate(pairs, n_pts)

# ==================== Style Setup ====================

def setup_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 13,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 11,
        'legend.title_fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def fmt_steps(x, _):
    """Format step values: 200000 → '200K', 1000000 → '1.0M'."""
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    return f'{x:.0f}'

# ==================== Core Plot Function ====================

def plot_env(algos, colors, env, base_dirs, steps_filter,
             title_suffix='', xlabel='Training Steps',
             ylabel='Average Return', figsize=(8, 6),
             n_pts=500, probe=False, reversal_step=None, min_steps=None):
    """Create one figure for one environment."""
    fig, ax = plt.subplots(figsize=figsize)

    # Draw other algos first, SF-P3O last so it's on top
    ordered = [a for a in algos if a != 'SF-P3O'] + [a for a in algos if a == 'SF-P3O']
    for algo in ordered:
        sd = load_seeds(base_dirs, env, algo, steps_filter, probe, min_steps)
        if not sd:
            print(f'    [SKIP] {algo} on {env}: no data')
            continue
        x, m, s = aggregate(sd, n_pts)
        if x is None:
            print(f'    [SKIP] {algo} on {env}: aggregate failed')
            continue
        lw = 2.8 if algo == 'SF-P3O' else 1.6
        zo = 20 if algo == 'SF-P3O' else 5
        label = ALGO_DISPLAY.get(algo, algo)
        ax.plot(x, m, label=label, color=colors.get(algo, '#333'),
                linewidth=lw, zorder=zo)
        ax.fill_between(x, m - s, m + s, alpha=0.15,
                        color=colors.get(algo, '#333'), zorder=zo - 1)

    if reversal_step is not None:
        ax.axvline(reversal_step, color='black', ls='--', lw=1.5, alpha=0.7)
        ylo, yhi = ax.get_ylim()
        ax.text(reversal_step + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01,
                yhi - (yhi - ylo) * 0.05, 'Reward\nReversal',
                fontsize=10, va='top', ha='left', alpha=0.7,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7))

    disp = ENV_DISPLAY.get(env, env)
    ax.set_title(f'{disp}{title_suffix}', fontweight='bold', pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_steps))
    ax.legend(loc='best', framealpha=0.9, edgecolor='none', fancybox=True)
    fig.tight_layout()
    return fig

# ==================== Figure Generation ====================

setup_style()

# ===== Fig 1: Training curves (5 envs × 8 main algos) =====
print('=' * 50)
print('Fig 1: Training Curves')
print('=' * 50)
for env in ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4']:
    dirs = [HIGHDIM] if env in ('Ant-v4', 'Humanoid-v4') else [RUNS]
    fig = plot_env(MAIN_ALGOS, COLORS, env, dirs, '1000000')
    name = ENV_DISPLAY[env]
    path = os.path.join(OUT, f'fig1_training_{name}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  [OK] fig1_training_{name}.png')

# ===== Fig 2: Reward reversal (3 envs × 8 main algos) =====
print('=' * 50)
print('Fig 2: Reward Reversal')
print('=' * 50)
for env in ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']:
    # Get reversal step from config
    sample = glob.glob(os.path.join(PROBE, f'probe_{env}_PPO_seed0'))
    rev_step = None
    if sample:
        cfg_path = os.path.join(sample[0], 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                rev_step = json.load(f).get('hp', {}).get('reversal_step')
    fig = plot_env(MAIN_ALGOS, COLORS, env, [PROBE], None,
                   title_suffix=' (Reward Reversal)', probe=True,
                   reversal_step=rev_step, min_steps=1400000)
    name = ENV_DISPLAY[env]
    path = os.path.join(OUT, f'fig2_reversal_{name}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  [OK] fig2_reversal_{name}.png')

# ===== Fig 3: Ablation (3 envs × SF-P3O variants) =====
print('=' * 50)
print('Fig 3: Ablation Study')
print('=' * 50)
for env in ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']:
    fig = plot_env(ABLATION_ALGOS, COLORS, env, [RUNS], '1000000',
                   title_suffix=' (Ablation)')
    name = ENV_DISPLAY[env]
    path = os.path.join(OUT, f'fig3_ablation_{name}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  [OK] fig3_ablation_{name}.png')

# ===== Fig 4: Long-run (2 envs × 8 main algos, 3M steps) =====
print('=' * 50)
print('Fig 4: Long-Run (3M Steps)')
print('=' * 50)
for env in ['HalfCheetah-v4', 'Walker2d-v4']:
    fig = plot_env(MAIN_ALGOS, COLORS, env, [LONGRUN], '3000000',
                   title_suffix=' (3M Steps)')
    name = ENV_DISPLAY[env]
    path = os.path.join(OUT, f'fig4_longrun_{name}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  [OK] fig4_longrun_{name}.png')

# ===== Fig 5: Diagnostics (3 panels from SF-P3O diagnostic data) =====
print('=' * 50)
print('Fig 5: Diagnostics')
print('=' * 50)

# --- Panel 1: Gate value (α) over time, SF-P3O across 3 envs ---
fig, ax = plt.subplots(figsize=(8, 6))
env_colors = {'HalfCheetah-v4': '#d62728', 'Hopper-v4': '#1f77b4', 'Walker2d-v4': '#2ca02c'}
for env in ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']:
    sd = load_diag_seeds([RUNS], env, 'SF-P3O', '1000000')
    x, m, s = aggregate_diag(sd, 'gate', 200)
    if x is not None:
        name = ENV_DISPLAY[env]
        ax.plot(x, m, label=name, color=env_colors[env], linewidth=2)
        ax.fill_between(x, m - s, m + s, alpha=0.15, color=env_colors[env])
ax.set_title('SF-P3O Gate Value (α) Over Training', fontweight='bold', pad=10)
ax.set_xlabel('Training Steps')
ax.set_ylabel('Gate Value (α)')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_steps))
ax.legend(loc='best', framealpha=0.9, edgecolor='none')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig5_diagnostics_gate.png'))
plt.close(fig)
print('  [OK] fig5_diagnostics_gate.png')

# --- Panel 2: Spectral norm over time, comparing algos in HalfCheetah ---
fig, ax = plt.subplots(figsize=(8, 6))
for algo in ['SF-P3O', 'PPO', 'CReLU', 'LayerNorm']:
    sd = load_diag_seeds([RUNS], 'HalfCheetah-v4', algo, '1000000')
    x, m, s = aggregate_diag(sd, 'spectral_norm', 200)
    if x is not None:
        lw = 2.5 if algo == 'SF-P3O' else 1.5
        label = ALGO_DISPLAY.get(algo, algo)
        ax.plot(x, m, label=label, color=COLORS.get(algo, '#333'), linewidth=lw)
        ax.fill_between(x, m - s, m + s, alpha=0.12, color=COLORS.get(algo, '#333'))
ax.set_title('Spectral Norm (HalfCheetah)', fontweight='bold', pad=10)
ax.set_xlabel('Training Steps')
ax.set_ylabel('Spectral Norm')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_steps))
ax.legend(loc='best', framealpha=0.9, edgecolor='none')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig5_diagnostics_erank.png'))
plt.close(fig)
print('  [OK] fig5_diagnostics_erank.png')

# --- Panel 3: Weight change over time, comparing SF-P3O across envs ---
fig, ax = plt.subplots(figsize=(8, 6))
for env in ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']:
    sd = load_diag_seeds([RUNS], env, 'SF-P3O', '1000000')
    x, m, s = aggregate_diag(sd, 'weight_change', 200)
    if x is not None:
        name = ENV_DISPLAY[env]
        ax.plot(x, m, label=name, color=env_colors[env], linewidth=2)
        ax.fill_between(x, m - s, m + s, alpha=0.15, color=env_colors[env])
ax.set_title('SF-P3O Weight Change Over Training', fontweight='bold', pad=10)
ax.set_xlabel('Training Steps')
ax.set_ylabel('Weight Change')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_steps))
ax.legend(loc='best', framealpha=0.9, edgecolor='none')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig5_diagnostics_perturb_freq.png'))
plt.close(fig)
print('  [OK] fig5_diagnostics_perturb_freq.png')

# ===== Fig 6: Bar chart (final performance) =====
print('=' * 50)
print('Fig 6: Bar Chart')
print('=' * 50)

table_mean = {
    'SF-P3O':         [1736.8, 603.2, 577.3, 589.1, 364.8],
    'PPO':            [1419.4, 561.2, 513.4, 719.8, 353.5],
    'PLASTIC':        [  47.0, 241.1, 160.1, 839.4, 297.1],
    'CReLU':          [1350.8, 539.3, 565.6, 716.2, 355.2],
    'ReDo':           [1336.0, 651.9, 540.0, 476.6, 347.9],
    'Shrink-Perturb': [1173.1, 594.1, 453.9, 554.6, 323.2],
    'L2-Init':        [1067.1, 478.6, 559.4, 743.1, 329.2],
    'LayerNorm':      [1931.2, 548.8, 590.0, 498.4, 402.8],
}
table_std = {
    'SF-P3O':         [322.7, 48.9, 146.2, 85.9,  6.9],
    'PPO':            [183.2, 89.5,  92.2, 165.4, 19.4],
    'PLASTIC':        [ 10.8, 29.9,  25.8,  19.7, 11.7],
    'CReLU':          [215.2, 145.5, 85.2, 131.5, 11.1],
    'ReDo':           [247.5, 61.1,  67.0,  67.5,  8.8],
    'Shrink-Perturb': [374.9, 37.8,  24.7, 147.8, 20.9],
    'L2-Init':        [121.9, 191.9, 97.9, 112.2, 16.0],
    'LayerNorm':      [357.5, 194.1, 95.9,  74.6, 19.4],
}

envs_bar = ['HalfCheetah', 'Hopper', 'Walker2d', 'Ant', 'Humanoid']

fig, ax = plt.subplots(figsize=(12, 6))
n_a = len(MAIN_ALGOS)
n_e = len(envs_bar)
bar_w = 0.8 / n_a
x_base = np.arange(n_e)

for i, algo in enumerate(MAIN_ALGOS):
    offset = (i - n_a / 2 + 0.5) * bar_w
    vals = table_mean[algo]
    errs = table_std[algo]
    label = ALGO_DISPLAY.get(algo, algo)
    ax.bar(x_base + offset, vals, bar_w, yerr=errs,
           label=label, color=COLORS[algo], capsize=2, alpha=0.88,
           edgecolor='white', linewidth=0.5,
           error_kw={'linewidth': 0.8})

ax.set_xticks(x_base)
ax.set_xticklabels(envs_bar, fontsize=14)
ax.set_ylabel('Average Return (Last 100K Steps)', fontsize=14)
ax.set_title('Final Performance Comparison Across Environments',
             fontweight='bold', fontsize=16, pad=12)
ax.legend(loc='upper right', ncol=2, framealpha=0.9, edgecolor='none',
          fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig6_bar_chart.png'))
plt.close(fig)
print('  [OK] fig6_bar_chart.png')

print()
print('=' * 50)
print(f'All figures saved to: {OUT}')
print('=' * 50)
