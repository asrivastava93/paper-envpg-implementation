"""
plot_jstar.py -- exact cost J*(zeta_t) of the EnvPG placement versus outer iteration, median over placements per kappa, with a 25-75 percentile band.

    python plot_jstar.py --root new_EnvPG_Runs_1 --csv jstar_curves.csv --max_it 100
J* = exact hard-min cost of the placement at the end of every outer iteration (policy iteration on the exact kernel; no oracle line).
If --csv exists it is read (fast); otherwise J* is computed from <root>/<tag>/hist_<tag>.npz and written to --csv (about 4 s per run).
Needs env2_paraRLModel.py and exact_eval.py in the same folder (only when the csv has to be computed).
options: --band iqr|none  --fs 2.0  --out jstar.pdf  --no_text
"""
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL = {'100': '#1f77b4', '095': '#2ca02c', '090': '#ff7f0e', '085': '#d62728'}      # same colours as the old gradient-norm figure


def compute_csv(root, csv):
    from env2_paraRLModel import build_env_info, precompute_env
    import exact_eval as ee
    st, ac, dest, zi, fsl, vac = build_env_info(seed=1)
    rows = []
    for f in sorted(glob.glob(os.path.join(root, '*', 'hist_*.npz'))):
        tag = os.path.basename(f)[5:-4]; kappa = int(tag[1:4]) / 100.0
        env = precompute_env(st, ac, dest, fsl, vac, eta_tr=100.0, kappa=kappa)
        h = np.load(f); Z = h['zeta'].reshape(-1, 5, 2); Z0 = h['zeta_init'].reshape(5, 2); n = len(Z) // 50
        Zit = [Z0] + [Z[50 * i - 1] for i in range(1, n + 1)]
        rows += [(tag, i, ee.hard_J(z, env)) for i, z in enumerate(Zit)]
        print(tag, 'done', flush=True)
    with open(csv, 'w') as fh:
        fh.write('run,iteration,J_star\n' + ''.join(f'{t},{i},{j:.6f}\n' for t, i, j in rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=None); ap.add_argument('--csv', default='jstar_curves.csv')
    ap.add_argument('--max_it', type=int, default=100); ap.add_argument('--band', default='iqr', choices=['iqr', 'none'])
    ap.add_argument('--fs', type=float, default=2.0); ap.add_argument('--out', default='jstar.pdf')
    ap.add_argument('--no_text', action='store_true'); ap.add_argument('--xticks', type=float, nargs='+', default=None, help='x tick positions in x-axis units, e.g. --xticks 0 1500 3000 4500'); ap.add_argument('--xticklabels', nargs='+', default=None, help='optional labels for --xticks'); ap.add_argument('--rows_per_it', type=int, default=50, help='x axis in update rows = iteration x this (50); 1 gives outer iterations')
    a = ap.parse_args()
    if not os.path.exists(a.csv):
        assert a.root, '--root is needed to compute J*'; compute_csv(a.root, a.csv)
    import csv as _csv
    data = {}
    for r in _csv.DictReader(open(a.csv)):
        data.setdefault(r['run'], {})[int(r['iteration'])] = float(r['J_star'])
    F = a.fs
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    for k in ['100', '095', '090', '085']:
        runs = [t for t in data if t[1:4] == k]
        if not runs:
            continue
        its = np.arange(0, a.max_it + 1); xs = its * a.rows_per_it
        M = np.array([[data[t].get(i, np.nan) for i in its] for t in runs])
        c = COL[k]
        ax.plot(xs, np.nanmedian(M, axis=0), color=c, lw=3.5, label=rf'$\kappa={int(k) / 100:g}$')
        if a.band == 'iqr':
            ax.fill_between(xs, np.nanpercentile(M, 25, axis=0), np.nanpercentile(M, 75, axis=0), color=c, alpha=0.2, lw=0)
    ax.set_ylim(13, 21)          # lower and upper y limit
    ax.set_xlim(0, a.max_it * a.rows_per_it); ax.set_xticks(a.xticks if a.xticks else np.linspace(0, a.max_it * a.rows_per_it, 3));
    if a.xticks and a.xticklabels: ax.set_xticklabels(a.xticklabels)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=9 * F, colors='black'); ax.set_box_aspect(0.862)
    for s in ax.spines.values():
        s.set_color('black')
    ax.legend(fontsize=11 * F * 0.65, loc='upper right', ncol=2, frameon=False, handlelength=1.2, labelspacing=0.25, columnspacing=0.8, borderaxespad=0.2)
    if not a.no_text:
        ax.text(0.6, 0.62, r'$J^*$ vs iteration', transform=ax.transAxes, ha='center', va='center', fontsize=11 * F, color='black')
    fig.savefig(a.out, dpi=300, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.splitext(a.out)[0] + '.png', dpi=130, bbox_inches='tight', pad_inches=0.02)
    print('written:', a.out)


if __name__ == '__main__':
    main()