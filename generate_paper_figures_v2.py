"""
generate_paper_figures.py

Computes J* via hard value iteration at final zeta configurations from
EnvPG-AC and the model-based oracle, and generates all figures for the
L-CSS paper:

  1. J* table (printed to console and saved as CSV)
  2. Gradient norm vs iteration — all 4 kappa values on one figure
  3. Network design plots — final UAV placements for each kappa and method

USAGE (final placements are now read from the result folders -- nothing to fill in by hand):
  python generate_paper_figures.py --envpg results_envpg --oracle results_oracle [--sf results_sf] --pick median
  --pick  : which run is drawn for each kappa:  median (default; the run whose EnvPG J* is closest to the median over runs),
            best, or an initialization tag such as P1.   The SAME initialization is used for all methods.

All figures saved as PDF (vector) for LaTeX inclusion.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
import os
import glob
import json
import argparse

from env2_paraRLModel import (
    build_env_info, precompute_env, EnvStructural,
    get_state_coords
)
from exact_eval import build_kernel_fast      # exact kernel incl. the hop cost (same as used in training / evaluation)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = 'paper_figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
CFG = dict(eta_tr=100.0, kappa=0.95)   # kappa overridden per-case below

st_d, ac_d, dest, zeta_init, fsl, vac = build_env_info(seed=1)

KAPPAS = [0.85, 0.90, 0.95, 1.00]

# ---------------------------------------------------------------------------
# Final zeta of every run, read from the summary files written by the training / oracle scripts
#   EnvPG, SF : summary_k<kk>_P<n>.json  -> 'zeta_final'
#   Oracle    : summary_oracle_k<kk>_P<n>.json -> 'zeta_star_gd'
# ---------------------------------------------------------------------------
def load_runs(folder, pattern, zkey):
    out = {}                                   # tag -> dict(zeta=(5,2), J=exact hard J)
    for f in glob.glob(os.path.join(folder, '**', pattern), recursive=True):
        s = json.load(open(f))
        out[s['tag']] = dict(zeta=np.array(s[zkey], dtype=float).reshape(5, 2), J=float(s['J_exact_hard_final']))
    return out


def pick_tags(envpg_runs, how):
    """one initialization tag (e.g. 'P3') per kappa; the same one is used for all methods."""
    chosen = {}
    for kappa in KAPPAS:
        pre = f'k{int(round(kappa * 100)):03d}_'
        tags = sorted(t for t in envpg_runs if t.startswith(pre))
        if not tags:
            continue
        Js = np.array([envpg_runs[t]['J'] for t in tags])
        if how == 'median':
            t = tags[int(np.argmin(np.abs(Js - np.median(Js))))]
        elif how == 'best':
            t = tags[int(np.argmin(Js))]
        else:
            t = pre + how
        chosen[kappa] = t
    return chosen

ZETA_ENVPG, ZETA_ORACLE, ZETA_SF = {}, {}, {}
CHOSEN = {}



# ---------------------------------------------------------------------------
# Hard value iteration — exact J* at fixed zeta
# ---------------------------------------------------------------------------

def build_transition_kernel(zeta, env):
    """Exact P[s,a,s'] and C[s,a] (analytic kernel, including the constant hop cost) -- same function as used for evaluation."""
    return build_kernel_fast(zeta, env)


def compute_J_hard_vi(zeta, env, eps=1e-6, max_iter=100000):
    """
    Hard value iteration J* = rho0 @ V,
    rho0 uniform over [s_low, s_high].
    """
    _, V = run_hard_vi(zeta, env, eps=eps, max_iter=max_iter)
    nt  = np.arange(env.s_low, env.s_high + 1)
    rho = np.zeros(env.st_count)
    rho[nt] = 1.0 / len(nt)
    print(f'    J* = {float(rho @ V):.4f}')
    return float(rho @ V)



# ---------------------------------------------------------------------------
# Spatial color matching across methods
# ---------------------------------------------------------------------------

def match_uav_colors(zeta_ref, zeta_other):
    """
    Returns a color_map list for zeta_other such that UAV j in zeta_other
    gets the color of the spatially closest UAV in zeta_ref.
    Uses the Hungarian algorithm (linear_sum_assignment) on pairwise distances.
    color_map[j] = UAV_COLORS index to use for UAV j in zeta_other.
    """
    from scipy.optimize import linear_sum_assignment
    nF   = len(zeta_ref)
    dist = np.zeros((nF, nF))
    for i in range(nF):
        for j in range(nF):
            dist[i, j] = np.linalg.norm(zeta_ref[i] - zeta_other[j])
    row_ind, col_ind = linear_sum_assignment(dist)
    # col_ind[i] = which UAV in zeta_other is matched to UAV i in zeta_ref
    color_map = [0] * nF
    for ref_idx, other_idx in zip(row_ind, col_ind):
        color_map[other_idx] = ref_idx   # other UAV gets ref UAV's color
    return color_map


# ---------------------------------------------------------------------------
# Figure: Network design plots with greedy-policy color coding
# ---------------------------------------------------------------------------

# 5 UAV colors — one per facility
UAV_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def run_hard_vi(zeta, env, eps=1e-6, max_iter=100000):
    """Run hard VI and return greedy action array (S,)."""
    P, C    = build_transition_kernel(zeta, env)
    S       = env.st_count
    delta_s = env.idx_delta_state
    valid   = env.valid_ac
    V = np.zeros(S)
    for _ in range(max_iter):
        Q        = C + P @ V
        Q_masked = np.where(valid, Q, np.inf)
        V_new    = np.min(Q_masked, axis=1)
        V_new[delta_s] = 0.0
        if np.max(np.abs(V_new - V)) < eps:
            break
        V = V_new
    Q        = C + P @ V
    Q_masked = np.where(valid, Q, np.inf)
    return np.argmin(Q_masked, axis=1), V


def get_greedy_uav_assignment(greedy_a, env):
    """
    Given greedy action array, return for each user node
    the 0-based UAV index it routes to.
    """
    nt = np.arange(env.s_low, env.s_high + 1)
    assignment = {}
    for s in nt:
        a = greedy_a[s]
        assignment[s] = int(env.fac_id_action[a]) if env.is_fac_action[a] else 0
    return assignment


def plot_network(zeta, env, kappa, method_name, color_map=None):
    """
    Plot user nodes colored by first-hop UAV under greedy policy,
    UAV positions, destination, and inter-UAV routing arrows.
    color_map: list of length nF mapping UAV index -> UAV_COLORS index.
               If None, identity mapping is used (UAV j -> color j).
    """
    if color_map is None:
        color_map = list(range(env.n_fac))

    coords     = get_state_coords(zeta, env)
    nt         = np.arange(env.s_low, env.s_high + 1)

    # Run hard VI once — used for both user-node coloring and UAV arrows
    greedy_a, _ = run_hard_vi(zeta, env)
    assignment  = get_greedy_uav_assignment(greedy_a, env)

    fig, ax = plt.subplots(figsize=(5, 4.5))

    # User nodes — colored by assigned UAV (via color_map)
    for s in nt:
        uav_idx   = assignment.get(s, 0)
        color_idx = color_map[uav_idx]
        color     = UAV_COLORS[color_idx % len(UAV_COLORS)]
        x, y      = coords[s]
        ax.scatter(x, y, c=color, s=100, zorder=1, alpha=0.7)

    # ── Inter-UAV and UAV->delta routing arrows ───────────────────────────
    # For each UAV state, find the greedy action and draw an arrow.
    # Offset the arrow endpoints slightly toward the midpoint to avoid
    # overlapping with marker centres and labels.
    ARROW_OFFSET = 0.18   # fraction of the vector to trim from each end

    drawn_pairs = set()   # avoid duplicate arrows between same pair
    for j in range(env.n_fac):
        s_uav = int(env.fac_state_idx[j])
        a     = greedy_a[s_uav]
        if a == env.idx_delta_action:
            s_next = env.idx_delta_state
        else:
            s_next = int(env.fac_state_idx[env.fac_id_action[a]])

        pair = (min(s_uav, s_next), max(s_uav, s_next))
        if pair in drawn_pairs:
            continue
        drawn_pairs.add(pair)

        x0, y0 = coords[s_uav]
        x1, y1 = coords[s_next]
        dx, dy  = x1 - x0, y1 - y0

        # Trim both ends so arrow doesn't overlap markers/labels
        xs = x0 + ARROW_OFFSET * dx
        ys = y0 + ARROW_OFFSET * dy
        xe = x1 - ARROW_OFFSET * dx
        ye = y1 - ARROW_OFFSET * dy

        ax.annotate('', xy=(xe, ye), xytext=(xs, ys),
                    arrowprops=dict(
                        arrowstyle='->', color='black',
                        lw=1.4, mutation_scale=14))

    # UAV positions — marker only, no inline label
    for j in range(env.n_fac):
        x, y      = zeta[j]
        color_idx = color_map[j]
        color     = UAV_COLORS[color_idx % len(UAV_COLORS)]
        ax.scatter(x, y, c=color, s=400, marker='^', zorder=3,
                   edgecolors='black', linewidths=0.8)

    # UAV positions — marker with small in-place label
    # UAV positions — marker with side-aware in-place label
    x_center = np.mean(zeta[:, 0])   # config centroid x, splits left/right
    for j in range(env.n_fac):
        x, y      = zeta[j]
        color_idx = color_map[j]
        color     = UAV_COLORS[color_idx % len(UAV_COLORS)]
        ax.scatter(x, y, c=color, s=400, marker='^', zorder=3,
                   edgecolors='black', linewidths=0.8)

        # Facility left of centroid -> label to its left (right-aligned);
        # facility right of centroid -> label to its right (left-aligned).
        if x < x_center:
            dx_off, ha = -8, 'right'
        else:
            dx_off, ha =  8, 'left'
        ax.annotate(f'$f_{{{j+1}}}$', (x, y),
                    textcoords='offset points',
                    xytext=(dx_off, 6), fontsize=25, ha=ha, va='center',
                    zorder=5, color='black')

    # Destination
    dest_coords = coords[env.idx_delta_state]
    ax.scatter(dest_coords[0], dest_coords[1],
               c='black', s=400, marker='*', zorder=4)
    ax.annotate(r'$\delta$', dest_coords,
                textcoords='offset points', xytext=(-25, -9),
                fontsize=22, color='black', fontweight='bold')

    # Legend 1 (top-left): UAV facilities
    handles_fac = [
        plt.Line2D([0], [0], marker='^', color='w',
                   markerfacecolor=UAV_COLORS[color_map[j]],
                   markeredgecolor='black', markeredgewidth=0.8,
                   markersize=20, label=f'$f_{{{j+1}}}$')
        for j in range(env.n_fac)
    ]
    '''leg1 = ax.legend(handles=handles_fac, fontsize=22,
                     loc='upper left', framealpha=0.9,
                     handletextpad=0.3, borderpad=0.4)
    ax.add_artist(leg1)   # keep leg1 when leg2 is added'''

    # Legend 2 (top-right): destination and user nodes
    handles_other = [
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='lightgrey', markeredgecolor='black',
               markersize=13, label='UAVs'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=12,
               label=r'User nodes $\{n_i\}$')
    ]
    ax.legend(handles=handles_other, fontsize=20,
              loc='upper right', frameon=True, handlelength = 0.8,
              handletextpad=0.1, borderaxespad=0.1, borderpad=0.2)

    # Kappa annotation
    '''ax.text(0.03, 0.82, f'$\\kappa = {kappa}$',
            transform=ax.transAxes,
            fontsize=20, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white', edgecolor='grey',
                      alpha=0.8))'''

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.set_aspect(1.2)
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_ylim(bottom=0.2)
    ax.set_ylim(top=7.6)
    ax.set_yticks([1, 3, 5, 7])
    #ax.set_box_aspect(1)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=20)
    fig.tight_layout()

    kappa_str  = f'{kappa:.2f}'.replace('.', '')
    method_tag = method_name.lower().replace('-', '_').replace(' ', '_')
    fname = f'network_{method_tag}_kappa{kappa_str}.pdf'
    path  = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    fig.savefig(path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    print(f'  Saved: {path}')


def plot_all_networks(method_name, zeta_dict, ref_zeta_dict=None, kappas=None):
    """
    ref_zeta_dict: if provided, UAV colors are spatially matched to
    this reference (e.g. ZETA_ENVPG) so both methods use consistent colors.
    """
    for kappa in (kappas if kappas is not None else KAPPAS):
        env  = precompute_env(st_d, ac_d, dest, fsl, vac,
                              eta_tr=CFG['eta_tr'], kappa=kappa)
        zeta = zeta_dict[kappa]
        if ref_zeta_dict is not None:
            color_map = match_uav_colors(ref_zeta_dict[kappa], zeta)
        else:
            color_map = list(range(env.n_fac))
        plot_network(zeta, env, kappa, method_name, color_map=color_map)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--envpg', required=True, help='folder with the EnvPG runs (summary_k*_P*.json inside)')
    ap.add_argument('--oracle', required=True, help='folder with the oracle runs (summary_oracle_k*_P*.json inside)')
    ap.add_argument('--sf', default=None, help='optional folder with the score-function (SF) runs')
    ap.add_argument('--kappa', type=float, nargs='+', default=None, help='only these kappas (default: all of 0.85 0.90 0.95 1.00)')
    ap.add_argument('--pick', default='median', help="median | best | P1, P2, ... : which initialization to draw for every kappa")
    a = ap.parse_args()

    env_runs = load_runs(a.envpg, 'summary_k*.json', 'zeta_final')
    orc_runs = load_runs(a.oracle, 'summary_oracle_k*.json', 'zeta_star_gd')
    sf_runs = load_runs(a.sf, 'summary_k*.json', 'zeta_final') if a.sf else {}
    if a.kappa:
        KAPPAS[:] = [k for k in KAPPAS if any(abs(k - q) < 1e-9 for q in a.kappa)]
    print(f'found {len(env_runs)} EnvPG runs in {a.envpg}: {sorted(env_runs)[:30]}')
    print(f'found {len(orc_runs)} oracle runs in {a.oracle}: {sorted(orc_runs)[:30]}')
    CHOSEN.update(pick_tags(env_runs, a.pick))
    for kappa, tag in list(CHOSEN.items()):
        if tag not in env_runs or tag not in orc_runs:
            where = 'EnvPG' if tag not in env_runs else 'oracle'
            print(f'kappa={kappa}: run {tag} not found in the {where} folder -- skipped')
            del CHOSEN[kappa]; continue
        ZETA_ENVPG[kappa] = env_runs[tag]['zeta']
        if tag in orc_runs: ZETA_ORACLE[kappa] = orc_runs[tag]['zeta']
        if tag in sf_runs: ZETA_SF[kappa] = sf_runs[tag]['zeta']
    print('Runs drawn:', {k: v for k, v in CHOSEN.items()})
    methods = [('EnvPG-AC', ZETA_ENVPG), ('Oracle', ZETA_ORACLE)] + ([('SF', ZETA_SF)] if ZETA_SF else [])
    kappas = [k for k in KAPPAS if k in ZETA_ENVPG and k in ZETA_ORACLE]
    if not kappas:
        print('\nNothing to draw: no run with the requested kappa / --pick exists in BOTH folders.\n'
              'Check the file names above: EnvPG summaries must be named summary_k<kkk>_P<n>.json (e.g. summary_k095_P6.json) '
              'and oracle summaries summary_oracle_k<kkk>_P<n>.json.')
        return

    print('=' * 60)
    print('Computing J* via hard value iteration')
    print('=' * 60)
    results = {}
    for kappa in kappas:
        results[kappa] = {}
        env = precompute_env(st_d, ac_d, dest, fsl, vac, eta_tr=CFG['eta_tr'], kappa=kappa)
        for method, zeta_dict in methods:
            if kappa not in zeta_dict:
                continue
            print(f'  kappa={kappa}  method={method}')
            results[kappa][method] = compute_J_hard_vi(zeta_dict[kappa], env)

    print()
    print('J* of the drawn runs')
    print(f"{'kappa':>6} | " + ' | '.join(f'{m:>10}' for m, _ in methods))
    for kappa in kappas:
        print(f'{kappa:6.2f} | ' + ' | '.join(f"{results[kappa].get(m, float('nan')):10.4f}" for m, _ in methods))
    csv_path = os.path.join(OUT_DIR, 'J_star_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['kappa', 'run'] + [m for m, _ in methods])
        for kappa in kappas:
            w.writerow([kappa, CHOSEN[kappa]] + [f"{results[kappa].get(m, float('nan')):.4f}" for m, _ in methods])
    print(f'\n  Saved: {csv_path}')

    print('\nGenerating network design plots...')
    plot_all_networks('EnvPG-AC', ZETA_ENVPG, kappas=kappas)                                 # UAV j -> color j
    plot_all_networks('Oracle', ZETA_ORACLE, ref_zeta_dict=ZETA_ENVPG, kappas=kappas)        # colors matched to EnvPG
    if ZETA_SF:
        plot_all_networks('SF', ZETA_SF, ref_zeta_dict=ZETA_ENVPG, kappas=kappas)
    print('\nDone. All outputs in:', OUT_DIR)


if __name__ == '__main__':
    main()