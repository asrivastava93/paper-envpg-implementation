"""
run_illustration.py -- ONE run of EnvPG (kappa = 0.90, relays initialised close to (6,3)) and the two illustrative panels of the figure:
  (a) the learned network design (relay placement, routing, user nodes coloured by their relay)
  (b) the relay trajectories from the initial (circles) to the final (triangles) positions.

    python run_illustration.py                       # train (100 outer iterations, as in the paper) and then plot (a) and (b)
    python run_illustration.py --skip_train          # only plot, from an existing results/k090_P7/
    options: --total_iters N (pilots only)  --seed S  --out paper_figures  --results_dir <folder with the run>  --tag k090_P7

This run is an additional illustration. It is NOT one of the 24 runs (4 kappas x 6 placements) behind the table and panels (c)-(e).
Needs, in the same folder: train_mf_envpg_resample_runs.py, env2_paraRLModel.py, exact_eval.py,
generate_paper_figures_v2.py and plot_trajectory.py. None of them is modified.
"""
import argparse, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_mf_envpg_resample_runs as T

KAPPA = 0.90
PLACEMENT = 7                                    # tag k090_P7
# (x1, y1, x2, y2, ...) for f1..f5: an irregular cluster of five relays around (6, 3); all pairwise distances >= 0.36
# (the training script pushes apart relays that are closer than CFG['kick_sep'] = 0.3, so none is kicked at the start)
P7 = [6.35, 3.15, 5.85, 3.40, 6.15, 2.70, 5.65, 2.95, 6.00, 3.05]


def register_p7():
    """Add placement 7 for kappa = 0.90 to the table of initial placements (P1..P5 and P6 = their mean are already there)."""
    lst = T.INIT[KAPPA]
    if len(lst) == 6:
        lst.append(list(P7))
    assert len(lst) == 7 and lst[6] == list(P7), "unexpected content of the initial-placement table"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skip_train', action='store_true', help='do not train, only plot from an existing run')
    ap.add_argument('--total_iters', type=int, default=None, help='override the number of outer iterations (pilots only)')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--out', default='paper_figures')
    ap.add_argument('--results_dir', default=os.path.join(HERE, 'results'), help='folder that contains <tag>/summary_<tag>.json')
    ap.add_argument('--tag', default=f'k{int(round(KAPPA * 100)):03d}_P{PLACEMENT}')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- 1. EnvPG training (Algorithm 1), exactly as in train_mf_envpg_resample_runs.py -------------------------------------
    register_p7()
    if not a.skip_train:
        T.main(KAPPA, PLACEMENT, a.seed, a.total_iters)            # writes results/<tag>/ next to the training script
        sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__    # the training script tees stdout into its log file

    summ_fn = os.path.join(a.results_dir, a.tag, f'summary_{a.tag}.json')
    if not os.path.exists(summ_fn):
        sys.exit(f'{summ_fn} not found -- run without --skip_train first, or give --results_dir / --tag')
    s = json.load(open(summ_fn))
    zeta = np.array(s['zeta_final'], dtype=float).reshape(-1, 2)
    print(f"{a.tag}: exact J* at the final placement = {s['J_exact_hard_final']:.3f}   (initial: {s['J_exact_hard_init']:.3f})")

    # ---- 2. panel (a): network design -------------------------------------------------------------------------------------
    import generate_paper_figures_v2 as G
    env = G.precompute_env(G.st_d, G.ac_d, G.dest, G.fsl, G.vac, eta_tr=G.CFG['eta_tr'], kappa=KAPPA)
    G.OUT_DIR = a.out
    G.plot_network(zeta, env, KAPPA, 'EnvPG')
    base = os.path.join(a.out, f'network_envpg_kappa{int(round(KAPPA * 100)):03d}')
    for ext in ('.pdf', '.png'):
        if os.path.exists(base + ext):
            os.replace(base + ext, os.path.join(a.out, f'network_envpg_{a.tag}{ext}'))
    print('saved', os.path.join(a.out, f'network_envpg_{a.tag}.pdf'))

    # ---- 3. panel (b): relay trajectories ---------------------------------------------------------------------------------
    subprocess.run([sys.executable, os.path.join(HERE, 'plot_trajectory.py'), '--root', a.results_dir, '--tag', a.tag,
                    '--panels', 'paths', '--out', a.out], check=True, cwd=HERE)


if __name__ == '__main__':
    main()
