"""
cosine_vs_N.py -- cosine between the stochastic EnvPG zeta-gradient (zeta_gradient_fd) and the EXACT zeta-gradient,
at saved checkpoints, for several numbers of trajectories N and several exploration levels epsilon,
averaged over repeated draws, then over all runs.

Run it from the folder that contains the EnvPG training script, env2_paraRLModel.py and exact_eval.py:

    python cosine_vs_N.py --root results --module train_mf_envpg_resample_runs
    python cosine_vs_N.py --root results --its 10 20 30 40 --eps 0 0.2 --Ns 51 500 5000 --reps 10 3 2 --workers 4

What is computed, per run (tag, e.g. k085_P1) and per checkpoint ckpt_<tag>_p<it>.pt  [actor, critic, zeta_k, beta]:
  exact gradient : d/dzeta of the exact entropy-regularised cost J(pi(.|.,zeta), zeta) of the checkpoint's actor, at its beta
                   (linear solves on the analytic kernel, central finite differences with step 0.02).  Needs only actor, zeta, beta.
                   The policy's own dependence on zeta (it is an input of the actor) is included.
  estimate       : zeta_gradient_fd(actor, critic, N fresh trajectories collected at zeta_k with random-action probability eps)
                   -- the same function as in training (rescaled by the mean episode length, which does not change the cosine).
  cosine         : cos(estimate, exact), averaged over `reps` independent draws for each (N, eps)  -> one number per run.
Then, over all runs: mean, median, inter-quartile range and the share of runs with cosine > 0.5, for every (checkpoint, eps, N).

NOTE on eps > 0: the exact gradient is always that of the actor's own policy pi (no epsilon mixing). At eps > 0 the sampled
w(s) is the visitation of the eps-mixed policy, so the comparison measures the exploration bias together with the sampling noise.
NOTE on the episode cap: the cap / forced jump to delta is whatever the imported training module does (CFG['max_ep_steps']);
use --max_ep_steps to override it.

--stored : ALSO compute the cosine of the stochastic gradient that was actually used during training (hist_<tag>.npz, key 'gz') with the exact gradient.
  The last row of outer iteration `it` (row r = rows_per_iter*it - 1) used exactly the actor/critic saved in checkpoint p<it>, at the placement
  BEFORE that row's update, z0 = hist['zeta'][r-1]; so the exact gradient for this comparison is evaluated at z0 (not at the checkpoint's zeta).
  Two variants: 'single' = the gradient of that one row (51 trajectories, as run); 'ema30' = the Adam-style first moment (beta1 = 0.9) over the last 30 rows.
  hist files are looked for in <hist_root>/<tag>/hist_<tag>.npz (default hist_root = --root).  Written to cosine_stored.csv.
  (The logged gz is divided by the total weight of its trajectories; it is rescaled by the logged mean episode length -- cosines do not depend on this.)

Outputs (in --out, default cosine_out/): per_run_<tag>.json (all draws), cosine_by_run.csv, cosine_summary.csv, cosine_stored.csv (with --stored), and a printed table.
Checkpoints whose exact gradient norm is >= --skip_norm (improper policy: the exact soft cost blows up) are skipped.
"""
import argparse, glob, importlib, json, os, sys, time
import numpy as np

sys.path.insert(0, os.getcwd())


# ----------------------------------------------------------------------------------------------------------------------
# exact quantities
# ----------------------------------------------------------------------------------------------------------------------
class Ctx:
    """Imports the training module once per process and holds the environment of a run."""
    def __init__(self, module):
        import torch
        torch.set_num_threads(1)
        self.torch = torch
        self.T = importlib.import_module(module)
        import exact_eval as ee
        from env2_paraRLModel import build_env_info, precompute_env
        self.ee, self.build_env_info, self.precompute_env = ee, build_env_info, precompute_env

    def make_env(self, kappa, eta_tr):
        st, ac, dest, zi, fsl, vac = self.build_env_info(seed=1)
        return self.precompute_env(st, ac, dest, fsl, vac, eta_tr=eta_tr, kappa=kappa)

    def load(self, path, env):
        ck = self.torch.load(path, weights_only=False, map_location='cpu')
        S, A, nF, d = env.st_count, env.ac_count, env.n_fac, env.dim
        hid = ck.get('cfg', {}).get('hidden_dim', self.T.CFG['hidden_dim'])
        actor = self.T.MLP(d + S + d + A + nF * d, hid)
        critic = self.T.MLP(d + S + d + A + nF * d + nF * d, hid)
        actor.load_state_dict(ck['actor']); critic.load_state_dict(ck['critic'])
        return ck, actor, critic, np.array(ck['zeta'], dtype=float).reshape(nF, d)


def beta_of(ck, T):
    """beta used during the last outer iteration before the checkpoint (the checkpoint stores beta AFTER the anneal)."""
    cfg = ck.get('cfg', {})
    anneal = cfg.get('beta_anneal', T.CFG['beta_anneal']); bmin = cfg.get('beta_min', T.CFG['beta_min'])
    b = float(ck['beta'])
    return bmin if b <= bmin * (1 + 1e-9) else b / anneal


def solve_pi(pi, P, C, beta, env):
    S, dl = env.st_count, env.idx_delta_state
    lp = np.log(np.clip(pi, 1e-12, None))
    r = (pi * (C + beta * lp)).sum(1)
    Pp = np.einsum('sa,sap->sp', pi, P)
    M = np.eye(S) - Pp; M[dl, :] = 0; M[dl, dl] = 1
    b = r.copy(); b[dl] = 0
    return np.linalg.solve(M, b)


def J_soft(ctx, z, env, actor, beta):
    pi = ctx.T.get_pi(actor, z, env)
    P, C = ctx.ee.build_kernel_fast(z, env)
    V = solve_pi(pi, P, C, beta, env)
    return float(V[env.s_low:env.s_high + 1].mean())


def exact_gradient(ctx, zk, env, actor, beta, h=0.02):
    g = np.zeros_like(zk)
    for i in range(zk.shape[0]):
        for j in range(zk.shape[1]):
            a = zk.copy(); a[i, j] += h
            b = zk.copy(); b[i, j] -= h
            g[i, j] = (J_soft(ctx, a, env, actor, beta) - J_soft(ctx, b, env, actor, beta)) / (2 * h)
    return g


def cosine(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ----------------------------------------------------------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------------------------------------------------------
_CTX = None


def ema(G, b=0.9):
    m = np.zeros(G.shape[1])
    for g in G:
        m = b * m + (1 - b) * g
    return m


def stored_cosines(ctx, a, tag, it, env, actor, beta):
    """cosine of the stored training gradient (hist['gz']) with the exact gradient, see the docstring (--stored)."""
    fn = os.path.join(a['hist_root'], tag, f'hist_{tag}.npz')
    if not os.path.exists(fn):
        return None
    h = np.load(fn)
    GZ, ZH, ITR = h['gz'], h['zeta'], h['iteration']
    rows = np.where(ITR == it - 1)[0]                                   # rows of outer iteration it-1 (0-based in the history)
    if len(rows) == 0 or rows[-1] < 30:
        return None
    r = int(rows[-1])
    z0 = ZH[r - 1].reshape(ZH.shape[1], ZH.shape[2]).astype(float)       # placement at which row r's gradient was taken
    gx0 = exact_gradient(ctx, z0, env, actor, beta)
    Lr = float(h['mean_len'][r])
    out = dict(row=r, exact_norm=float(np.linalg.norm(gx0)))
    for name, g in (('single', GZ[r]), ('ema30', ema(GZ[r - 29:r + 1]))):
        g = np.asarray(g, dtype=float) * Lr
        out[name] = cosine(g, gx0)
    out['norm_ratio'] = float(np.linalg.norm(GZ[r] * Lr) / (np.linalg.norm(gx0) + 1e-12))
    return out


def process_run(job):
    global _CTX
    a = job['args']; tag = job['tag']
    out_fn = os.path.join(a['out'], f'per_run_{tag}.json')
    res = json.load(open(out_fn)) if os.path.exists(out_fn) else {}
    if _CTX is None:
        _CTX = Ctx(a['module'])
    ctx = _CTX; T = ctx.T
    if a['max_ep_steps'] is not None:
        T.CFG['max_ep_steps'] = a['max_ep_steps']
    kappa = int(tag[1:4]) / 100.0
    its = a['its'] if a['its'] else sorted(int(os.path.basename(f).split('_p')[-1][:-3])
                                           for f in glob.glob(os.path.join(a['root'], tag, f'ckpt_{tag}_p*.pt')))
    env = None
    for it in its:
        key = str(it)
        want = [(eps, N) for eps in a['eps'] for N in a['Ns'] if f'{eps}|{N}' not in res.get(key, {}).get('draws', {})]
        need_stored = (a['stored'] or a['stored_only']) and 'stored' not in res.get(key, {})
        if a['stored_only']:
            want = []
            if not need_stored:
                continue
        if key in res and not want and not need_stored:
            continue
        path = os.path.join(a['root'], tag, f'ckpt_{tag}_p{it:03d}.pt')
        if not os.path.exists(path):
            continue
        t0 = time.time()
        try:
            ck0 = ctx.torch.load(path, weights_only=False, map_location='cpu')
            if env is None:
                env = ctx.make_env(kappa, ck0.get('cfg', {}).get('eta_tr', T.CFG['eta_tr']))
            ck, actor, critic, zk = ctx.load(path, env)
        except (OSError, EOFError, RuntimeError) as e:      # unreadable / truncated checkpoint (e.g. a cloud-sync placeholder): skip it
            print(f'{tag} it={it:3d}: cannot read {path} ({type(e).__name__}: {e}) -- skipped', flush=True)
            continue
        beta = beta_of(ck, T)
        if a['stored_only']:                                 # only the stored-gradient cosine: one exact gradient per checkpoint, no fresh draws
            st = stored_cosines(ctx, a, tag, it, env, actor, beta)
            if st is not None and st['exact_norm'] < a['skip_norm']:
                res.setdefault(key, dict(exact_norm=st['exact_norm'], beta=beta, draws={}))['stored'] = st
                json.dump(res, open(out_fn, 'w'))
            continue
        gx = exact_gradient(ctx, zk, env, actor, beta)
        ent = dict(exact_norm=float(np.linalg.norm(gx)), beta=beta, draws=res.get(key, {}).get('draws', {}))
        if 'stored' in res.get(key, {}):
            ent['stored'] = res[key]['stored']
        if ent['exact_norm'] >= a['skip_norm']:
            ent['skipped'] = True; res[key] = ent
            json.dump(res, open(out_fn, 'w')); continue
        if need_stored:
            st = stored_cosines(ctx, a, tag, it, env, actor, beta)
            if st is not None:
                ent['stored'] = st
            else:
                print(f'{tag} it={it:3d}: no usable history in {a["hist_root"]}/{tag}/hist_{tag}.npz -- stored gradient skipped', flush=True)
        pi = T.get_pi(actor, zk, env)
        for eps in a['eps']:
            for iN, N in enumerate(a['Ns']):
                if f'{eps}|{N}' in ent['draws']:
                    continue
                R = a['reps'][min(iN, len(a['reps']) - 1)]
                cs, rat = [], []
                for r in range(R):
                    np.random.seed((1000003 * it + 9176 * N + 131 * r + int(1000 * eps) + 7 * kappa_seed(tag)) % (2 ** 31))   # reproducible, distinct per draw
                    trajs = T.collect_episodes(actor, zk, env, M=N, eps=eps, zeta_noise_std=0.0)
                    L = float(np.mean([len(t) for t in trajs]))
                    g = T.zeta_gradient_fd(actor, critic, trajs, zk, pi, env, T.CFG['gamma'], beta, 10 * N) * L   # batch_size = 10 N -> all N used
                    cs.append(cosine(g, gx)); rat.append(float(np.linalg.norm(g) / (np.linalg.norm(gx) + 1e-12)))
                ent['draws'][f'{eps}|{N}'] = dict(cos=cs, norm_ratio=rat)
        res[key] = ent
        json.dump(res, open(out_fn, 'w'))
        print(f"{tag} it={it:3d} |g_exact|={ent['exact_norm']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return tag


def kappa_seed(tag):
    return int(tag[1:4]) * 10 + int(tag.split('_P')[1])


# ----------------------------------------------------------------------------------------------------------------------
# aggregation: mean over draws -> per run;  then median / mean / IQR / share>0.5 over runs
# ----------------------------------------------------------------------------------------------------------------------
def aggregate(out, eps_list, Ns):
    files = sorted(glob.glob(os.path.join(out, 'per_run_*.json')))
    rows = []                                                         # (tag, it, eps, N, mean cos over draws, exact_norm)
    for f in files:
        tag = os.path.basename(f)[len('per_run_'):-5]
        for it, ent in json.load(open(f)).items():
            if ent.get('skipped'):
                continue
            for k, d in ent['draws'].items():
                eps, N = k.split('|')
                rows.append((tag, int(it), float(eps), int(N), float(np.mean(d['cos'])), ent['exact_norm'], float(np.mean(d['norm_ratio']))))
    srows = []
    for f in files:
        tag = os.path.basename(f)[len('per_run_'):-5]
        for it, ent in json.load(open(f)).items():
            if 'stored' in ent and not ent.get('skipped'):
                s_ = ent['stored']; srows.append((tag, int(it), s_['single'], s_['ema30'], s_['exact_norm'], s_['norm_ratio']))
    if srows:
        with open(os.path.join(out, 'cosine_stored.csv'), 'w') as fh:
            fh.write('run,iteration,cos_single,cos_ema30,exact_norm,norm_ratio\n')
            for r in sorted(srows):
                fh.write(f'{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.3f}\n')
        print('written:', os.path.join(out, 'cosine_stored.csv'))
    if not rows:
        print('nothing to aggregate'); return
    with open(os.path.join(out, 'cosine_by_run.csv'), 'w') as fh:
        fh.write('run,iteration,eps,N,cos_mean_over_draws,exact_norm,norm_ratio\n')
        for r in sorted(rows):
            fh.write(f'{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.4f},{r[5]:.4f},{r[6]:.3f}\n')
    keys = sorted({(r[1], r[2], r[3]) for r in rows})
    lines = ['iteration,eps,N,n_runs,median,mean,p25,p75,share_above_0.5']
    print(f"\n{'iter':>5} {'eps':>5} {'N':>7} {'runs':>5} {'median':>7} {'mean':>7} {'p25':>7} {'p75':>7} {'>0.5':>6}")
    for it, eps, N in keys:
        v = np.array([r[4] for r in rows if (r[1], r[2], r[3]) == (it, eps, N)])
        s = (it, eps, N, len(v), np.median(v), v.mean(), np.percentile(v, 25), np.percentile(v, 75), (v > 0.5).mean())
        lines.append('%d,%g,%d,%d,%.4f,%.4f,%.4f,%.4f,%.3f' % s)
        print(f'{it:5d} {eps:5g} {N:7d} {len(v):5d} {s[4]:7.2f} {s[5]:7.2f} {s[6]:7.2f} {s[7]:7.2f} {s[8]:6.2f}')
    open(os.path.join(out, 'cosine_summary.csv'), 'w').write('\n'.join(lines) + '\n')
    print(f"\nwritten: {os.path.join(out, 'cosine_summary.csv')} and cosine_by_run.csv")


def plot(out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import csv
    rows = list(csv.DictReader(open(os.path.join(out, 'cosine_summary.csv'))))
    eps_list = sorted({float(r['eps']) for r in rows}); its = sorted({int(r['iteration']) for r in rows})
    fig, axs = plt.subplots(1, len(eps_list), figsize=(5.5 * len(eps_list), 4.2), squeeze=False, sharey=True)
    cm = plt.cm.viridis(np.linspace(0, 0.9, len(its)))
    for ax, e in zip(axs[0], eps_list):
        for c, it in zip(cm, its):
            pts = sorted((int(r['N']), float(r['median'])) for r in rows if float(r['eps']) == e and int(r['iteration']) == it)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', color=c, label=f'checkpoint {it}')
        ax.set_xscale('log'); ax.set_title(f'eps = {e:g}'); ax.set_xlabel('number of trajectories N'); ax.grid(alpha=0.3); ax.axhline(0, color='gray', lw=0.8)
    axs[0][0].set_ylabel('median cosine with exact gradient (over runs)')
    if len(its) <= 12:
        axs[0][-1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'cosine_vs_N.png'), dpi=150)
    print('written:', os.path.join(out, 'cosine_vs_N.png'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='results', help='folder with one sub-folder per run (<tag>/ckpt_<tag>_pNNN.pt); default: results')
    ap.add_argument('--module', default='train_mf_envpg_resample_runs', help='EnvPG training script (module name, no .py)')
    ap.add_argument('--tags', nargs='+', default=None, help='runs to use; default: all sub-folders of --root named kNNN_P#')
    ap.add_argument('--its', type=int, nargs='+', default=None, help='checkpoint numbers (e.g. 10 20 30 40); default: all found')
    ap.add_argument('--eps', type=float, nargs='+', default=[0.0, 0.2], help='random-action probabilities of the sampled trajectories')
    ap.add_argument('--Ns', type=int, nargs='+', default=[51, 500, 5000], help='numbers of trajectories')
    ap.add_argument('--reps', type=int, nargs='+', default=[10, 3, 2], help='independent draws for each N (list aligned with --Ns; last value is reused)')
    ap.add_argument('--skip_norm', type=float, default=100.0, help='skip checkpoints whose exact gradient norm is at least this (improper policy)')
    ap.add_argument('--max_ep_steps', type=int, default=None, help='override CFG[max_ep_steps] of the training module')
    ap.add_argument('--stored', action='store_true', help='also compute the cosine of the stored training gradient (hist_<tag>.npz) -> cosine_stored.csv')
    ap.add_argument('--stored_only', action='store_true', help='compute ONLY cosine_stored.csv (fast: no fresh draws, one exact gradient per checkpoint)')
    ap.add_argument('--hist_root', default=None, help='folder with <tag>/hist_<tag>.npz (default: --root)')
    ap.add_argument('--workers', type=int, default=2, help='parallel processes (one run each)')
    ap.add_argument('--out', default='cosine_out')
    ap.add_argument('--aggregate_only', action='store_true')
    ap.add_argument('--plot', action='store_true', help='also save cosine_vs_N.png (median over runs)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.aggregate_only:
        tags = a.tags or sorted(d for d in os.listdir(a.root) if len(d) >= 7 and d[0] == 'k' and '_P' in d and os.path.isdir(os.path.join(a.root, d)))
        args = dict(root=a.root, module=a.module, its=a.its, eps=a.eps, Ns=a.Ns, reps=a.reps, skip_norm=a.skip_norm,
                    max_ep_steps=a.max_ep_steps, out=a.out, stored=a.stored, stored_only=a.stored_only, hist_root=a.hist_root or a.root)
        jobs = [dict(tag=t, args=args) for t in tags]
        print(f'{len(tags)} runs, eps={a.eps}, N={a.Ns}, reps={a.reps}, workers={a.workers}', flush=True)
        if a.workers > 1:
            import multiprocessing as mp
            with mp.get_context('spawn').Pool(a.workers) as pool:
                for _ in pool.imap_unordered(process_run, jobs):
                    pass
        else:
            for j in jobs:
                process_run(j)
    aggregate(a.out, a.eps, a.Ns)
    if a.plot:
        plot(a.out)


if __name__ == '__main__':
    main()