"""
plot_grad_norm_old.py -- norm of the stored stochastic zeta-gradient vs iteration.

Two modes (--agg):
  pick   (default) : first-submission style: faint raw norm of every update row + thick moving average, one placement per kappa (--pick P4).
  median           : median over ALL placements of the moving average, per kappa (thick line), on top of the faint raw norms of every
                     placement (as in the first submission); optional 25-75 percentile band of the averages with --band iqr. No placement has to be chosen.

    python plot_grad_norm_old.py --root new_EnvPG_Runs_1 --agg median --window 200 --min_row 10
    python plot_grad_norm_old.py --root new_EnvPG_Runs_1 --agg pick --pick P4 --window 200 --min_row 10
Output goes to paper_figures/gradnorm_old.pdf (+ .png) by default.
options:
  --kappas 1.0 0.95 0.9 0.85     --window 200 (moving-average length, rows)      --min_row 10 (first row drawn; the average is still computed from row 0)
  --xlim 0 5000                  --ylim 0.08 10                                 --xticks 0 1500 3000 4500    --yticks 0.1 1 10  [--yticklabels 0.1 1 10]
  --fs 2.0   --raw_alpha 0.15    --rescale (exact-gradient scale)               --band iqr|minmax|none (median mode, default none; minmax = full range over placements)  --no_raw (median mode)  --out paper_figures/gradnorm_old.pdf
"""
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL = {1.0: '#1f77b4', 0.95: '#2ca02c', 0.9: '#ff7f0e', 0.85: '#d62728'}


def moving_average(y, w):
    c = np.cumsum(np.insert(y, 0, 0.0)); i = np.arange(1, len(y) + 1); lo = np.maximum(0, i - w)
    return (c[i] - c[lo]) / (i - lo)                                    # causal; the window shrinks at the start


def load_g(f, rescale, max_row):
    h = np.load(f); g = np.asarray(h['gz_norm'], float)
    if rescale:
        g = g * np.asarray(h['mean_len'], float)
    return g[:max_row]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True); ap.add_argument('--agg', choices=['pick', 'median'], default='pick')
    ap.add_argument('--pick', default='P4'); ap.add_argument('--band', choices=['iqr', 'minmax', 'none'], default='none')
    ap.add_argument('--kappas', type=float, nargs='+', default=[1.0, 0.95, 0.9, 0.85])
    ap.add_argument('--window', type=int, default=200); ap.add_argument('--min_row', type=int, default=10)
    ap.add_argument('--max_row', type=int, default=5000); ap.add_argument('--fs', type=float, default=2.0)
    ap.add_argument('--xlim', type=float, nargs=2, default=None); ap.add_argument('--ylim', type=float, nargs=2, default=[0.08, 10])
    ap.add_argument('--xticks', type=float, nargs='+', default=[0, 1500, 3000, 4500])
    ap.add_argument('--yticks', type=float, nargs='+', default=[0.1, 1, 10]); ap.add_argument('--yticklabels', nargs='+', default=None)
    ap.add_argument('--rescale', action='store_true'); ap.add_argument('--raw_alpha', type=float, default=0.15, help='opacity of the faint raw lines (in median mode: of each placement)')
    ap.add_argument('--no_raw', action='store_true', help='median mode: do not draw the faint raw lines of the individual placements')
    ap.add_argument('--out', default=os.path.join('paper_figures', 'gradnorm_old.pdf'))
    a = ap.parse_args(); F = a.fs
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    lab = lambda k: rf'$\kappa={k:g}$' if k != 1.0 else r'$\kappa=1.0$'
    if a.agg == 'pick':
        runs = []
        for k in a.kappas:
            tag = f'k{int(round(k * 100)):03d}_{a.pick}'; f = os.path.join(a.root, tag, f'hist_{tag}.npz')
            if not os.path.exists(f):
                print('missing:', f); continue
            runs.append((k, load_g(f, a.rescale, a.max_row)))
        for k, g in runs:                                                     # faint raw lines first, so the averages stay on top
            x = np.arange(len(g)); sl = x >= a.min_row
            ax.plot(x[sl], g[sl], color=COL.get(round(k, 2)), lw=1.0, alpha=a.raw_alpha)
        for k, g in runs:
            x = np.arange(len(g)); sl = x >= a.min_row
            ax.plot(x[sl], moving_average(g, a.window)[sl], color=COL.get(round(k, 2)), lw=3.5, label=lab(k))
    else:
        data = []
        for k in a.kappas:
            files = sorted(glob.glob(os.path.join(a.root, f'k{int(round(k * 100)):03d}_P*', 'hist_*.npz')))
            if not files:
                print('no runs found for kappa', k); continue
            L = min(len(load_g(f, a.rescale, a.max_row)) for f in files)
            G = np.array([load_g(f, a.rescale, a.max_row)[:L] for f in files])                               # raw norms (placements, rows)
            M = np.array([moving_average(g, a.window) for g in G])                                            # moving averages
            print(f'kappa {k:g}: {len(files)} placements')
            data.append((k, G, M))
        if not a.no_raw:                                                                                      # faint raw lines of every placement first, so the medians stay on top
            for k, G, M in data:
                x = np.arange(G.shape[1]); sl = x >= a.min_row
                for g in G:
                    ax.plot(x[sl], g[sl], color=COL.get(round(k, 2)), lw=0.8, alpha=a.raw_alpha / np.sqrt(len(G)) * 1.5)
        for k, G, M in data:
            x = np.arange(G.shape[1]); sl = x >= a.min_row; c = COL.get(round(k, 2))
            if a.band != 'none':                                                                              # band over placements: 25-75 percentile, or the full range (min-max)
                lo, hi = (25, 75) if a.band == 'iqr' else (0, 100)
                ax.fill_between(x[sl], np.percentile(M, lo, axis=0)[sl], np.percentile(M, hi, axis=0)[sl], color=c, alpha=0.2, lw=0)
            ax.plot(x[sl], np.median(M, axis=0)[sl], color=c, lw=3.5, label=lab(k), zorder=3)
    ax.set_yscale('log')
    ax.set_xticks(a.xticks); ax.set_yticks(a.yticks)
    ax.set_yticklabels(a.yticklabels if a.yticklabels else [f'{v:g}' for v in a.yticks]); ax.minorticks_off()
    ax.set_ylim(*a.ylim); ax.set_xlim(*(a.xlim if a.xlim else (a.min_row, a.max_row)))   # limits last, so ticks cannot widen them
    ax.grid(alpha=0.3); ax.set_box_aspect(0.862); ax.tick_params(labelsize=9 * F, colors='black')
    for s in ax.spines.values():
        s.set_color('black')
    ax.legend(fontsize=13 * F * 0.65, loc='upper right', frameon=True, handlelength=1.2, labelspacing=0.25, borderpad=0.3, borderaxespad=0.3)
    fig.savefig(a.out, dpi=300, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.splitext(a.out)[0] + '.png', dpi=130, bbox_inches='tight', pad_inches=0.02)
    print('written:', a.out)


if __name__ == '__main__':
    main()