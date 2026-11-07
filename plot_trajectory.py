"""
plot_trajectory.py -- how the UAV positions f_1..f_5 move during training of ONE run, and the exact cost of the current placement.

    python plot_trajectory.py --root new_EnvPG_Runs_1 --tag k085_P3 --oracle results_oracle_h0.2
    options: --max_it 100 (last outer iteration drawn)  --every 10 (the paths are joined through the placements every this many outer iterations)  --fs 1.0
             --panels both|paths|cost   --out paper_figures

Reads <root>/<tag>/hist_<tag>.npz ('zeta' = placement after every update row, 50 rows per outer iteration; 'zeta_init').
(a) paths: one coloured line per relay from the initial placement (circle) to the final one (triangle), joined through the placements every --every outer iterations;
    user nodes in grey, destination star; with --oracle also the oracle's final positions as hollow triangles (matched colours).
(b) exact hard-min cost J*(zeta_t) of the placement at the end of every outer iteration (policy iteration on the exact kernel),
    with the oracle's J* as a dashed line.
Needs env2_paraRLModel.py and exact_eval.py in the same folder.
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from env2_paraRLModel import build_env_info, precompute_env, get_state_coords
import exact_eval as ee

COL = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, help='folder with the EnvPG runs (<root>/<tag>/hist_<tag>.npz)')
    ap.add_argument('--tag', required=True, help='run, e.g. k085_P3')
    ap.add_argument('--oracle', default=None, help='folder with the oracle runs (optional)')
    ap.add_argument('--max_it', type=int, default=100)
    ap.add_argument('--every', type=int, default=10)
    ap.add_argument('--path', choices=['iters', 'rows'], default='iters', help="draw the paths through the placement at the end of every outer iteration (smooth, default) or through every update row (noisy)")
    ap.add_argument('--fs', type=float, default=1.0)
    ap.add_argument('--panels', choices=['both', 'paths', 'cost'], default='both')
    ap.add_argument('--out', default='paper_figures')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    kappa = int(a.tag[1:4]) / 100.0
    h = np.load(os.path.join(a.root, a.tag, f'hist_{a.tag}.npz'))
    Z = h['zeta'].reshape(-1, 5, 2); Z0 = h['zeta_init'].reshape(5, 2)
    nit = min(a.max_it, len(Z) // 50)
    P = np.concatenate([Z0[None], Z[:50 * nit]], axis=0)                       # every row, starting with the initial placement
    Zit = np.concatenate([Z0[None], np.array([Z[50 * i - 1] for i in range(1, nit + 1)])], axis=0)   # end of outer iteration 0..nit
    st, ac, dest, zi, fsl, vac = build_env_info(seed=1)
    env = precompute_env(st, ac, dest, fsl, vac, eta_tr=100.0, kappa=kappa)
    zo = Jo = None
    if a.oracle:
        for f in glob.glob(os.path.join(a.oracle, '**', f'summary_oracle_{a.tag}.json'), recursive=True):
            s = json.load(open(f)); zo = np.array(s['zeta_star_gd'], float).reshape(5, 2); Jo = float(s['J_exact_hard_final'])
        if zo is None:
            print('oracle summary for', a.tag, 'not found -- drawn without the oracle')
    F = a.fs
    ncol = 2 if a.panels == 'both' else 1
    fig, axs = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.5), squeeze=False)        # one panel = 5 x 4.5 in, like the network-design figures
    axs = axs[0]
    k = 0
    if a.panels in ('both', 'paths'):
        ax = axs[k]; k += 1
        coords = get_state_coords(Z[50 * nit - 1], env); nt = np.arange(env.s_low, env.s_high + 1)
        ax.scatter(coords[nt, 0], coords[nt, 1], c='#bbbbbb', s=100, alpha=0.7, zorder=1)
        d = coords[env.idx_delta_state]; ax.scatter(*d, c='black', s=400, marker='*', zorder=4)
        ax.annotate(r'$\delta$', d, textcoords='offset points', xytext=(-25, -9), fontsize=22 * F, fontweight='bold')
        for j in range(5):
            pts = sorted(set(list(range(0, nit + 1, a.every)) + [nit]))          # only the marked iterations: 0, every, 2*every, ..., nit
            ax.plot(Zit[pts, j, 0], Zit[pts, j, 1], color=COL[j], lw=2.0, zorder=2)
            ax.scatter(*Z0[j], facecolors='white', edgecolors=COL[j], s=150, linewidths=2.6, zorder=6)        # initial (above the final marker)
            ax.scatter(*Zit[-1, j], c=COL[j], marker='^', s=400, edgecolors='black', linewidths=0.8, zorder=5)   # final
            lab_off = {0: (-9, 11, 'right'), 1: (11, 8, 'left'), 2: (10, 0, 'left'), 3: (-9, -12, 'right'), 4: (21, 27, 'left')}   # (dx, dy, alignment) in points
            dx, dy, ha_ = lab_off[j]
            ax.annotate(f'$f_{{{j + 1}}}$', Zit[-1, j], textcoords='offset points', xytext=(dx, dy), ha=ha_, va='center',
                        fontsize=25 * F, zorder=7,
                        arrowprops=dict(arrowstyle='-', color='#555555', lw=0.8, shrinkA=2, shrinkB=11) if j == 4 else None)
        if zo is not None:
            r, c = linear_sum_assignment(np.linalg.norm(Zit[-1][:, None] - zo[None], axis=2))
            for i, jj in zip(r, c):
                ax.scatter(*zo[jj], facecolors='none', edgecolors=COL[i], marker='^', s=520, linewidths=2.6, zorder=4)
        h_ = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='#555555', markeredgewidth=2, markersize=12, label='initial'),
              plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#bbbbbb', markeredgecolor='black', markersize=15, label='final')]
        if zo is not None:
            h_.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='none', markeredgecolor='#555555', markeredgewidth=2, markersize=15, label='oracle'))
        ax.legend(handles=h_, fontsize=20 * F, loc='upper right', framealpha=0.9, handletextpad=0.2, borderpad=0.3)
        ax.set_aspect(1.2); ax.set_xticks([2, 4, 6, 8, 10]); ax.set_yticks([1, 3, 5, 7]); ax.set_ylim(0.2, 7.6)
        ax.grid(True, alpha=0.2); ax.tick_params(labelsize=20 * F)
    if a.panels in ('both', 'cost'):
        ax = axs[k]
        J = np.array([ee.hard_J(Zit[i], env) for i in range(nit + 1)])
        ax.plot(range(nit + 1), J, color='#0d366b', lw=2.4)
        if Jo is not None:
            ax.axhline(Jo, color='#eb6834', ls='--', lw=2, label='oracle')
            #ax.legend(fontsize=15 * F, frameon=False)
            ax.legend(handles=h_, fontsize=32 * F, loc='upper right', frameon=True,
                handlelength=0.8, handletextpad=0.1, borderaxespad=0.1, borderpad=0.2)
        ax.set_xlabel('outer iteration', fontsize=20 * F); ax.set_ylabel(r'$J^*(\zeta_t)$', fontsize=20 * F)
        ax.set_xlim(0, nit); ax.grid(alpha=0.3); ax.tick_params(labelsize=20 * F)
        print(f'exact cost: initial {J[0]:.3f} -> final {J[-1]:.3f}' + (f'   (oracle {Jo:.3f})' if Jo is not None else ''))
    fig.tight_layout()
    path = os.path.join(a.out, f'trajectory_{a.tag}.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.02); fig.savefig(path[:-4] + '.png', dpi=130, bbox_inches='tight', pad_inches=0.02)
    print('saved', path)


if __name__ == '__main__':
    main()
