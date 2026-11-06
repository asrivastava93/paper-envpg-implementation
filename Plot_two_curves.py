"""
plot_two_curves.py -- cosine with the exact gradient versus checkpoint (outer iteration), two curves only:
   (1) fresh estimate with N trajectories (default N = 5000), ONE draw per run (default: the first draw), at random-action probability eps
   (2) the stochastic gradient stored during training (cosine_stored.csv; 'single' = the checkpoint's last row, or 'ema30')
Each curve: statistic over ALL runs (all kappas pooled; median by default, band = inter-quartile range; --stat mean gives mean +- 1 s.e.).

    python plot_two_curves.py                                   # cosine_out/per_run_*.json + cosine_out/cosine_stored.csv
    python plot_two_curves.py --dir cosine_out --stored_csv cosine_stored_out/cosine_stored.csv --max_it 45 --smooth 3
    python plot_two_curves.py --N 5000 --draw 0 --eps 0 --stat median --stored_variant single
    python plot_two_curves.py --common                          # use only (run, checkpoint) pairs present in BOTH curves (same runs behind every point)
Missing values are never filled: a missing (run, checkpoint) is just left out; the number of runs behind each point is in two_curves.csv (n_runs).
Output: two_curves.png (and two_curves.csv with the plotted numbers) in --dir.
"""
import argparse, csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def smooth(y, k):
    if k <= 1:
        return y
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode='edge')
    return np.convolve(pad, np.ones(k) / k, mode='valid')


def stat(v, kind):
    v = np.asarray(v, float)
    if kind == 'mean':
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        return v.mean(), v.mean() - se, v.mean() + se
    return np.median(v), np.percentile(v, 25), np.percentile(v, 75)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default='cosine_out', help='folder with per_run_<tag>.json (from cosine_vs_N.py)')
    ap.add_argument('--stored_csv', default=None, help='cosine_stored.csv (default: <dir>/cosine_stored.csv)')
    ap.add_argument('--N', type=int, default=5000)
    ap.add_argument('--eps', type=float, default=0.0)
    ap.add_argument('--draw', type=int, default=0, help='which draw to use for each run (0 = first)')
    ap.add_argument('--stat', choices=['median', 'mean'], default='median')
    ap.add_argument('--stored_variant', choices=['single', 'ema30'], default='single')
    ap.add_argument('--min_it', type=int, default=0)
    ap.add_argument('--max_it', type=int, default=10 ** 9)
    ap.add_argument('--smooth', type=int, default=1, help='moving-average window over checkpoints (1 = none)')
    ap.add_argument('--no_band', action='store_true', help='draw no spread band (the band is the spread ACROSS RUNS, not across draws)')
    ap.add_argument('--common', action='store_true', help='use only the (run, checkpoint) pairs available for both curves')
    ap.add_argument('--fs', type=float, default=2.8, help='font-size multiplier for all texts (labels, ticks, legend, title, in-plot text); 1 = matplotlib defaults, 2 = doubled')
    ap.add_argument('--figsize', type=float, nargs=2, default=[8.0, 6.5], help='figure width and height in inches')
    ap.add_argument('--title', action='store_true', help='also print eps / band information as a title (long at large font sizes; better put it in the caption)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    stored_csv = a.stored_csv or os.path.join(a.dir, 'cosine_stored.csv')

    fresh = {}; fresh_k = {}                                     # iteration -> list of cosines (one per run); (run, iteration) -> cosine
    key = f'{a.eps}|{a.N}'
    for f in sorted(glob.glob(os.path.join(a.dir, 'per_run_*.json'))):
        for it, ent in json.load(open(f)).items():
            d = ent.get('draws', {}).get(key)
            if ent.get('skipped') or not d or len(d['cos']) <= a.draw:
                continue
            if a.min_it <= int(it) <= a.max_it:
                fresh_k[(os.path.basename(f)[len('per_run_'):-5], int(it))] = d['cos'][a.draw]
    sto = {}; sto_k = {}
    if os.path.exists(stored_csv):
        for r in csv.DictReader(open(stored_csv)):
            it = int(r['iteration'])
            if a.min_it <= it <= a.max_it:
                sto_k[(r['run'], it)] = float(r['cos_' + a.stored_variant])
    else:
        print('no stored-gradient file found:', stored_csv)

    if a.common:
        both = set(fresh_k) & set(sto_k)
        fresh_k = {k: v for k, v in fresh_k.items() if k in both}; sto_k = {k: v for k, v in sto_k.items() if k in both}
        print(f'--common: {len(both)} (run, checkpoint) pairs present in both curves')
    for src, dst in ((fresh_k, fresh), (sto_k, sto)):
        for (run, it), v in src.items():
            dst.setdefault(it, []).append(v)
    print(f'fresh curve: {len(fresh)} checkpoints, {sum(len(v) for v in fresh.values())} values;  stored curve: {len(sto)} checkpoints, {sum(len(v) for v in sto.values())} values')
    if not sto:
        print('WARNING: no stored-gradient values to plot (file missing or no overlap with the selected checkpoints / runs)')
    fig, ax = plt.subplots(figsize=tuple(a.figsize))
    F = a.fs
    ax.set_box_aspect(0.862)
    band = 'IQR' if a.stat == 'median' else '+-1 s.e.'
    rows = []
    for name, data, col, ls, mk in ((f'N = {a.N:,}', fresh, '#0d366b', '-', 'o'),
                                    ('training gradient' + (' (EMA)' if a.stored_variant == 'ema30' else ''), sto, '#eb6834', '--', 's')):
        if not data:
            continue
        its = sorted(data)
        s = np.array([stat(data[i], a.stat) for i in its])
        ax.plot(its, smooth(s[:, 0], a.smooth), color=col, ls=ls, lw=4, marker=mk if len(its) <= 25 else None, ms=4, label=name)
        if not a.no_band:
            ax.fill_between(its, smooth(s[:, 1], a.smooth), smooth(s[:, 2], a.smooth), color=col, alpha=0.12, lw=0)
        for i, v in zip(its, s):
            rows.append((name, i, len(data[i]), v[0], v[1], v[2]))
    #ax.axhline(0, color='gray', lw=0.8); 
    for sp in ax.spines.values():
        sp.set_color('black'); sp.set_linewidth(1.5)
    #ax.set_ylim(-0.4, 1.05); ax.grid(alpha=0.3)
    ax.set_xlim(0, 46)
    ax.set_xticks([0, 20, 40])
    ax.set_xticklabels(['0', '1000', '2000'])
    #ax.set_xticks([0, 30, 60, 90])
    #ax.set_xticklabels(['0', '1500', '3000', '4500'])     # checkpoint x 50 = number of updates
    ax.set_ylim(-0.4, 1.05)
    ax.set_yticks([-0.2, 0.1, 0.4, 0.7, 1.0])
    #ax.set_xlabel('checkpoint', fontsize=10 * F); ax.set_ylabel('cosine', fontsize=10 * F)
    ax.tick_params(axis='both', labelsize=11 * F)
    ax.text(0.02, 0.02, 'cosine vs iteration', transform=ax.get_yaxis_transform(),
        ha='left', va='bottom', fontsize=11 * F, color="#000000")
    #ax.text(0.98, 0.98, 'cosine vs iteration', transform=ax.transAxes,
    #    ha='right', va='top', fontsize=11 * F, color="#000000")
    #ax.text(0.5, 0.5 * (ax.get_ylim()[0] + 0), 'cosine vs iteration', transform=ax.get_yaxis_transform(), ha='center', va='center',
    #        fontsize=10 * F, color='#333333')                    # written inside the plot, between the bottom edge and the 0 line
    ax.legend(frameon=False, fontsize=9 * F, loc='lower left')
    if a.title:
        ax.set_title(f'eps = {a.eps:g}' + ('' if a.no_band else f'; band: {band} across runs'), loc='left', fontsize=9 * F, color='#52514e')
    fig.tight_layout()
    out = a.out or os.path.join(a.dir, 'two_curves.png')
    fig.savefig(out, dpi=160); print('written:', out)
    with open(os.path.splitext(out)[0] + '.csv', 'w') as fh:
        fh.write('curve,iteration,n_runs,center,low,high\n')
        for r in rows:
            fh.write(f'{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f}\n')


if __name__ == '__main__':
    main()