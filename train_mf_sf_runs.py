"""
train_mf_sf_runs.py -- score-function (Schaff / Cauz style) co-design baseline, with the SAME
hyper-parameters and the SAME number of environment interactions per row as EnvPG-AC.

Design distribution q_psi = N(mu, diag sigma^2) over the UAV placement zeta; the policy pi_theta(a|s,zeta)
and critic Q_phi(s,a,zeta) take zeta as an input.  Per row (outer step) k, with the EnvPG-AC counts:

  train_at='designs' (default, Schaff/Cauz): the policy and critic are trained OVER the design distribution.
  1. critic data   : 5000 episodes, EVERY episode at its own design zeta_i ~ q_psi (no extra zeta noise), random-action
                     prob. 0.2, actions otherwise SAMPLED from pi(.|.,zeta_i).  100 critic steps, batch 512, Polyak target;
                     target y = c + beta log pi(a|s,zeta_i) + gamma sum_a' pi(a'|s',zeta_i) Q_tgt(s',a',zeta_i).
  2. actor update  : 50 steps; every step draws 140 FRESH episodes, each at its own design zeta_i ~ q_psi, random-action
                     prob. 0.2 (0 from outer iteration 50 on); rows (zeta_i, s_t) weighted gamma^t / total weight;
                     gradient = sum_rows w sum_a pi grad log pi (Q - V + beta).
  train_at='mu': policy / critic trained only at zeta = mu (critic data with per-transition zeta noise 0.5, actor
                     episodes exactly at mu) -- the EnvPG-style inner loop; q_psi is then used only to score the design update.
  train_at='mu_zi': critic data as in 'designs' but with the perturbed designs zeta_i = mu + zi_noise_std * N(0, I) (ONE zeta_i per episode, fixed
                     through the episode; the policy, the transitions and the bootstrap target all use zeta_i, i.e. NOT the policy at mu);
                     the actor update is exactly as in 'mu' (episodes at zeta = mu, Q(s,a,mu), table path -- cheap).  The design update is unchanged.
  3. design update : (all modes)  G = design_traj designs x episodes_per_design (default 51 x 1) episodes (default 51) FRESH episodes, each at its OWN design zeta_i ~ q_psi
                     (start state ~ uniform over user nodes, i.e. the environment's rho), actions from the just-updated
                     pi(.|.,zeta_i) with random-action prob. 0.2 (0 from outer iteration 50 on).  Score of episode i:
                         J_i = sum_t [ c_t + beta * log pi(a_t | s_t, zeta_i) ]            (soft cost, the objective being optimised)
                     REINFORCE:  g_mu     = mean_i  J_i (zeta_i - mu)/sigma^2
                                 g_logsig = mean_i  J_i ((zeta_i-mu)^2/sigma^2 - 1) - alpha      (alpha: entropy bonus of q_psi)
                     No baseline, no score normalisation, independent start states  (= `no_vr=True` of train_mf_sf_v2.py);
                     optional with --loo / --norm.  Adam on (mu, log sigma), lr_zeta = 1e-2.

Interactions per row: 5000 + 50*140 + N, identical to EnvPG-AC (N = 51 there as well).
The first 20 rows of outer iteration 0 are a warm-up (no design update), beta is annealed x0.95 per outer iteration.
Everything that EnvPG-AC saves is saved here too (see main()).
"""
from __future__ import annotations
import argparse, copy, json, logging, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from env2_paraRLModel import build_env_info, precompute_env, plot_env, get_state_coords

HERE = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger()


class Tee:
    def __init__(self, f):
        self.file = open(f, "w", encoding="utf-8", errors="replace")
        self.stdout = sys.stdout

    def write(self, m):
        self.stdout.write(m); self.file.write(m)

    def flush(self):
        self.stdout.flush(); self.file.flush()


def results_root(design_traj, train_at='designs', epd=1):
    return os.path.join(HERE, f"results_sf_{train_at}_n{design_traj}" + (f"x{epd}" if epd != 1 else ""))


def setup_run(tag, design_traj, train_at='designs', epd=1):
    out_dir = os.path.join(results_root(design_traj, train_at, epd), tag)
    os.makedirs(out_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(out_dir, f"log_{tag}.txt")); sys.stderr = sys.stdout
    log.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    log.addHandler(h); log.setLevel(logging.INFO)
    return out_dir


# ---------------------------------------------------------------------------
# Initial placements: the SAME table as the EnvPG-AC runs (R1-R5 per kappa, 6 = mean of R1-R5)
# ---------------------------------------------------------------------------
INIT = {
 0.85: [[6.837,3.346,6.082,3.751,6.128,3.501,5.703,5.033,3.275,3.293],[4.388,4.292,5.275,4.223,4.625,3.266,3.766,4.055,4.668,4.977],[4.442,4.966,3.475,4.436,3.377,4.478,7.208,3.797,5.313,4.701],[7.222,3.177,5.276,5.109,5.904,4.461,5.328,5.268,5.171,3.231],[3.017,4.481,6.295,3.378,3.244,3.287,5.768,4.995,6.067,4.609]],
 0.90: [[6.939,3.700,3.292,4.180,3.114,4.475,6.845,4.041,4.385,4.040],[3.472,3.117,6.207,4.684,7.259,4.009,4.614,4.238,4.137,5.267],[3.989,3.290,7.421,5.465,5.024,5.011,4.428,3.904,3.838,4.121],[5.867,3.237,3.739,5.325,4.541,5.288,4.344,4.179,4.946,4.079],[6.591,4.179,7.209,4.828,3.537,3.977,4.241,3.886,4.403,5.282]],
 0.95: [[6.744,4.760,5.460,5.498,4.731,4.663,4.606,4.344,6.689,4.113],[7.461,3.890,6.425,5.164,4.740,3.073,3.811,4.744,3.941,5.208],[4.434,3.386,7.368,3.955,7.047,5.157,4.413,5.115,6.072,4.172],[5.445,4.797,5.471,4.936,3.049,3.629,4.519,4.539,4.816,5.122],[6.336,5.255,4.402,5.251,6.576,4.114,4.233,3.242,3.970,5.343]],
 1.00: [[5.637,3.190,4.297,3.664,3.859,4.790,5.887,3.486,4.710,3.471],[3.049,5.256,6.425,5.050,3.900,3.919,5.193,3.972,4.923,4.731],[5.631,4.505,3.667,4.745,6.909,3.346,4.767,4.673,6.902,4.332],[3.032,4.055,5.004,3.945,5.678,3.801,4.725,4.740,6.400,3.771],[6.009,3.428,6.583,4.048,5.038,5.141,6.579,4.042,4.571,5.204]],
}
for _k in INIT:
    INIT[_k].append([round(float(v), 3) for v in np.mean(INIT[_k], axis=0)])   # placement 6


def get_init(kappa, placement):
    key = min(INIT, key=lambda k: abs(k - kappa))
    assert abs(key - kappa) < 1e-9, f"no init table for kappa={kappa}"
    return INIT[key][placement - 1]




# ---------------------------------------------------------------------------
# Configuration (identical to EnvPG-AC wherever a counterpart exists)
# ---------------------------------------------------------------------------
CFG = dict(
    seed=0, gamma=1, eta_tr=100.0, kappa=0.9,
    hidden_dim=128, lr_actor=3e-4, lr_critic=1e-4, grad_clip=0.5, nu=0.995,
    total_iters=100, outer_iters=50, warmup_iters=20,
    M_episodes=5000, batch_size=512,           # critic data per row / critic minibatch
    critic_iters=100, actor_steps=50,
    actor_traj=140,                            # fresh episodes per actor step (EnvPG-AC: 140)
    design_traj=51,                            # G: designs zeta_i ~ q per design step (EnvPG-AC uses batch_size//10 = 51 episodes)
    episodes_per_design=1,                     # episodes run at each zeta_i; score J_i = their mean (total episodes = G * episodes_per_design)
    eps_greedy=0.2,                            # random-action prob., critic data (always)
    actor_eps=0.0, resample_eps=0.0,           # random-action prob., actor episodes / design episodes ...
    eps_off_iter=0,                           # ... switched to 0 from this outer iteration on (as in EnvPG-AC)
    zeta_noise_std=0.5,                        # train_at='mu': per-transition zeta noise in the critic data (EnvPG-AC: 0.5)
    sf_zeta_noise=0.0,                         # train_at='designs': extra per-transition zeta noise on top of the sampled design
    zi_noise_std=0.5,                          # train_at='mu_zi': std of the per-EPISODE perturbation zeta_i = mu + noise in the critic data
    train_at='designs',                        # 'designs' (Schaff/Cauz) | 'mu' | 'mu_zi'
    lr_zeta=1e-2,
    beta_max=1.0, beta_min=0.001, beta_anneal=0.95,
    # --- design distribution ---
    sf_sigma0=0.25, sf_sigma_min=0.05, sf_sigma_max=2.0, sf_learn_sigma=True, sf_ent0=0.05,
    sf_box=(0.0, 12.0),
    sf_soft=True,                              # score = sum(c + beta log pi)  (False: sum c only)
    sf_loo=False, sf_norm=False,               # variance reduction OFF (= no_vr=True); --loo / --norm switch on
    sf_kick=True, sf_sep=0.3, sf_kick_patience=1, sf_kick_move=0.5, sf_kick_cooldown=0,
    max_ep_steps=50,                           # time limit: after this many steps the action is FORCED to 'go to delta' until the episode terminates (as in EnvPG-AC)
    exact_every=10,                            # exact hard J*(mu) every this many rows (NaN otherwise); 0 = never
)


# ---------------------------------------------------------------------------
# Features / networks   (same layout as train_mf_sf_v2.py: [x_s | onehot_s | x_a | onehot_a | zeta_flat])
# ---------------------------------------------------------------------------
def feat_rows(zetas: np.ndarray, states: np.ndarray, env) -> np.ndarray:
    R = len(states)
    S, A, nF, d = env.st_count, env.ac_count, env.n_fac, env.dim
    xs = env.state_coords_fixed[states].copy()
    fm = env.is_fac_state[states]
    if fm.any():
        rows = np.where(fm)[0]
        xs[rows] = zetas[rows, env.fac_id_state[states[rows]]]
    ac = np.zeros((R, A, d))
    ac[:, :nF, :] = zetas
    ac[:, env.idx_delta_action, :] = env.dest_loc[0]
    fd = d + S + d + A + nF * d
    F = np.zeros((R, A, fd), dtype=np.float32)
    F[:, :, :d] = xs[:, None, :]
    F[:, :, d:d + S] = np.eye(S, dtype=np.float32)[states][:, None, :]
    F[:, :, d + S:2 * d + S] = ac
    F[:, :, 2 * d + S:2 * d + S + A] = np.eye(A, dtype=np.float32)[None]
    F[:, :, 2 * d + S + A:] = zetas.reshape(R, 1, -1)
    return F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _rows_forward(net, F, grad=False):
    R, A, fd = F.shape
    x = torch.from_numpy(F.reshape(R * A, fd))
    if grad:
        return net(x).reshape(R, A)
    with torch.no_grad():
        return net(x).reshape(R, A)


def fast_forward(net, zetas, states, env, grad=False, acts=None):
    """Same numbers as _rows_forward(net, feat_rows(zetas, states, env)), without building the (R,A,fd) feature tensor.
    The first layer is linear in [x_s | onehot_s | x_a | onehot_a | zeta_flat], so
        W1 x = W_xs x_s + W_ohs[:,s] + W_zeta zeta_flat      (per row)
             + W_xa x_a + W_oha[:,a]                         (per row and action)
    and only layers 2-3 act on (R*A) rows.  Returns (R,A), or (R,) for the actions `acts` if given."""
    S, A, nF, d = env.st_count, env.ac_count, env.n_fac, env.dim
    zetas = np.asarray(zetas, dtype=np.float64)
    R = len(states)
    xs = env.state_coords_fixed[states].copy()
    fm = env.is_fac_state[states]
    if fm.any():
        rows = np.where(fm)[0]
        xs[rows] = zetas[rows, env.fac_id_state[states[rows]]]
    ac = np.zeros((R, A, d))
    ac[:, :nF, :] = zetas
    ac[:, env.idx_delta_action, :] = env.dest_loc[0]
    l1, l2, l3 = net.net[0], net.net[2], net.net[4]
    W = l1.weight
    o1, o2, o3, o4 = d, d + S, 2 * d + S, 2 * d + S + A
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        Ts = (torch.from_numpy(xs.astype(np.float32)) @ W[:, :o1].T + W[:, o1:o2].T[torch.from_numpy(states)]
              + torch.from_numpy(zetas.reshape(R, -1).astype(np.float32)) @ W[:, o4:].T + l1.bias)          # (R,H)
        if acts is None:
            Ta = (torch.from_numpy(ac.astype(np.float32)) @ W[:, o2:o3].T + W[:, o3:o4].T[None])            # (R,A,H)
            h = torch.tanh(Ts[:, None, :] + Ta)
            h = torch.tanh(l2(h))
            return l3(h).squeeze(-1)
        a_t = torch.from_numpy(np.asarray(acts))
        xa = torch.from_numpy(ac[np.arange(R), np.asarray(acts)].astype(np.float32))
        Ta = xa @ W[:, o2:o3].T + W[:, o3:o4].T[a_t]
        h = torch.tanh(l2(torch.tanh(Ts + Ta)))
        return l3(h).squeeze(-1)


class PolicyCache:
    """pi(.|s, zeta_i) for a fixed batch of designs Z (n,nF,d) and a fixed actor, evaluated many times with different (episode, state).
    Everything in the first layer that depends only on the design is computed once; per call only a table lookup
    (state one-hot / fixed coordinates) and layers 2-3 remain.  Numerically the same as pi_rows."""
    def __init__(self, actor, Z, env):
        self.env = env
        S, A, nF, d = env.st_count, env.ac_count, env.n_fac, env.dim
        l1, self.l2, self.l3 = actor.net[0], actor.net[2], actor.net[4]
        W = l1.weight
        o1, o2, o3, o4 = d, d + S, 2 * d + S, 2 * d + S + A
        n = len(Z)
        with torch.no_grad():
            Zt = torch.from_numpy(np.asarray(Z, dtype=np.float32))                                  # (n,nF,d)
            self.Tz = Zt.reshape(n, -1) @ W[:, o4:].T + l1.bias                                     # (n,H)
            ac = torch.zeros(n, A, d)
            ac[:, :nF, :] = Zt
            ac[:, env.idx_delta_action, :] = torch.from_numpy(env.dest_loc[0].astype(np.float32))
            self.Ta = ac @ W[:, o2:o3].T + W[:, o3:o4].T[None]                                      # (n,A,H)
            self.Sfix = torch.from_numpy(env.state_coords_fixed.astype(np.float32)) @ W[:, :o1].T + W[:, o1:o2].T   # (S,H)
            self.U = Zt @ W[:, :o1].T                                                               # (n,nF,H): UAV state uses its own zeta as x_s
        self.uav_of_state = -np.ones(S, dtype=int)
        self.uav_of_state[env.fac_state_idx] = np.arange(nF)

    def probs(self, ids, states):
        with torch.no_grad():
            Ts = self.Tz[ids] + self.Sfix[states]
            u = self.uav_of_state[states]
            m = np.where(u >= 0)[0]
            if len(m):
                Ts[m] += self.U[ids[m], u[m]]
            h = torch.tanh(Ts[:, None, :] + self.Ta[ids])
            f = self.l3(torch.tanh(self.l2(h))).squeeze(-1)
            return _probs(f, torch.from_numpy(self.env.valid_ac[states])).numpy().astype(np.float64)


def _logpi(f, valid):
    lg = (-f).masked_fill(~valid, -1e9)
    return lg - torch.logsumexp(lg, dim=1, keepdim=True)


def _probs(f, valid):
    return torch.softmax((-f).masked_fill(~valid, -1e9), dim=1) * valid.float()


def pi_table(actor, zeta, env):
    """(S,A) float64 policy probabilities at design zeta."""
    S = env.st_count; ss = np.arange(S)
    f = fast_forward(actor, np.repeat(zeta[None], S, 0), ss, env)
    return _probs(f, torch.from_numpy(env.valid_ac)).numpy().astype(np.float64)


def pi_rows(actor, zetas, states, env):
    f = fast_forward(actor, zetas, states, env)
    return _probs(f, torch.from_numpy(env.valid_ac[states])).numpy().astype(np.float64)


def greedy_is_proper(actor, zeta, env) -> bool:
    uav = env.fac_state_idx
    P = pi_rows(actor, np.repeat(zeta[None], len(uav), 0), uav, env)
    a = P.argmax(1)
    nxt = {j: ('delta' if a[j] == env.idx_delta_action else int(a[j])) for j in range(env.n_fac)}
    for j in range(env.n_fac):
        seen, c = set(), j
        while nxt[c] != 'delta':
            if c in seen:
                return False
            seen.add(c); c = nxt[c]
    return True


def sample_designs(mu, logsig, n):
    """n designs zeta_i = clip(mu + sigma * N(0,1), box)  -> (n, nF, d)."""
    lo, hi = CFG['sf_box']
    return np.clip(mu[None] + np.exp(logsig)[None] * np.random.randn(n, *mu.shape), lo, hi)


# ---------------------------------------------------------------------------
# Vectorised episode collection.  All episodes advance together.
#   Zep=None : all M episodes at design mu (policy table, fast path), per-transition noise `znoise` on the environment's zeta
#   Zep=(M,nF,d): episode i runs at design Zep[i] (the network sees Zep[i]); `znoise` added per transition on top
# Same transition kernel / cost as the EnvPG-AC collector.
# Returns flat arrays over all transitions: e (episode), t (step), s, a, sp, c, lp (log pi of the taken action at the
# network's design), zc (the zeta the environment actually used), and per-episode sums J_hard, J_soft, plus length.
# ---------------------------------------------------------------------------
def collect(actor, mu, env, M, eps, znoise, Zep=None, beta=0.0, cap=None, want_zc=False):
    cap = cap or CFG['max_ep_steps']
    S, A = env.st_count, env.ac_count
    nF, d = mu.shape
    valid = env.valid_ac
    delta_s, delta_a = env.idx_delta_state, env.idx_delta_action
    fixed_xy = env.state_coords_fixed
    act_uav = np.where(np.arange(A) == delta_a, -1, env.fac_id_action).astype(int)
    int_state = np.where(act_uav >= 0, env.fac_state_idx[np.clip(act_uav, 0, None)], delta_s).astype(int)
    uav_of_state = -np.ones(S, dtype=int)
    uav_of_state[env.fac_state_idx] = np.arange(nF)
    non_term = np.arange(env.s_low, env.s_high + 1)

    def xy_of_states(st, zc):
        X = fixed_xy[st].copy()
        u = uav_of_state[st]
        m = u >= 0
        if m.any():
            X[m] = zc[np.where(m)[0], u[m]]
        return X

    if Zep is not None:
        pcache = PolicyCache(actor, Zep, env)
    if Zep is None:
        P_tab = pi_table(actor, mu, env)
        C_tab = np.cumsum(P_tab, axis=1)
        L_tab = np.log(np.maximum(P_tab, 1e-12))

    rec = {k: [] for k in ('e', 't', 's', 'a', 'sp', 'c', 'lp')}
    zc_rec = []
    ep_ids = np.arange(M)
    s_cur = np.random.choice(non_term, size=M).astype(int)
    t = 0
    while len(ep_ids):
        n = len(ep_ids)
        if Zep is None:
            cum, lpt = C_tab[s_cur], L_tab[s_cur]
            zbase = np.broadcast_to(mu[None], (n, nF, d))
        else:
            zbase = Zep[ep_ids]
            Pn = pcache.probs(ep_ids, s_cur)
            cum, lpt = np.cumsum(Pn, axis=1), np.log(np.maximum(Pn, 1e-12))
        rnd = np.random.rand(n) < eps
        a_pol = np.minimum((cum < (np.random.rand(n) * cum[:, -1])[:, None]).sum(1), A - 1)
        a_rnd = np.argmax(np.where(valid[s_cur], np.random.rand(n, A), -1.0), axis=1)
        a = np.where(rnd, a_rnd, a_pol).astype(int)
        if t >= cap:                                                    # time limit: from now on go to delta (valid from UAV states)
            a = np.full(n, delta_a, dtype=int)
        zc = zbase + np.random.randn(n, nF, d) * znoise if znoise > 0 else np.array(zbase)
        s_next = int_state[a].copy()
        off = np.random.rand(n) >= env.kappa
        if off.any():
            io = np.where(off)[0]
            alt = valid[s_cur[io]].copy()
            alt[np.arange(len(io)), a[io]] = False
            xs = xy_of_states(s_cur[io], zc[io])
            zio = zc[io]
            Xact = np.where((act_uav >= 0)[None, :, None], zio[:, np.clip(act_uav, 0, None), :],
                            fixed_xy[delta_s][None, None, :])
            d2 = ((Xact - xs[:, None, :]) ** 2).sum(-1)
            lg = np.where(alt, -env.eta_tr * d2, -np.inf)
            has = alt.any(1)
            mx = np.where(has, lg.max(1), 0.0)
            ex = np.exp(lg - mx[:, None])
            ps = ex.sum(1)
            bad = (ps < 1e-300) | ~np.isfinite(ps)
            if bad.any():
                ex[bad] = alt[bad].astype(float)
                ps[bad] = ex[bad].sum(1)
            cp = np.cumsum(ex, axis=1)
            pick = np.minimum((cp < (np.random.rand(len(io)) * cp[:, -1])[:, None]).sum(1), A - 1)
            s_off = int_state[pick]
            s_next[io[has]] = s_off[has]
        cost = ((xy_of_states(s_next, zc) - xy_of_states(s_cur, zc)) ** 2).sum(1)
        cost = cost + env.hop_cost
        if env.kappa <= 0.75:
            cost = np.where(cost <= 0.1, cost + 1.0, cost)
        rec['e'].append(ep_ids.copy()); rec['t'].append(np.full(n, t)); rec['s'].append(s_cur.copy())
        rec['a'].append(a); rec['sp'].append(s_next.copy()); rec['c'].append(cost)
        rec['lp'].append(lpt[np.arange(n), a])
        if want_zc:
            zc_rec.append(zc.astype(np.float32))
        done = s_next == delta_s
        t += 1
        keep = ~done
        ep_ids = ep_ids[keep]
        s_cur = s_next[keep]
    D = {k: np.concatenate(v) for k, v in rec.items()}
    D['c'] = D['c'].astype(np.float64)
    if want_zc:
        D['zc'] = np.concatenate(zc_rec)
    D['M'] = M
    D['len'] = np.bincount(D['e'], minlength=M)
    D['J_hard'] = np.bincount(D['e'], weights=D['c'], minlength=M)
    D['J_soft'] = np.bincount(D['e'], weights=D['c'] + beta * D['lp'], minlength=M)
    return D


# ---------------------------------------------------------------------------
# Critic:  y = c + beta log pi(a|s,mu) + gamma sum_a' pi(a'|s',mu) Q_tgt(s',a',mu);  Q evaluated at the perturbed zeta_curr
# ---------------------------------------------------------------------------
def critic_update(critic, critic_tgt, actor, opt, D, mu, env, gamma, beta, grad_clip, n_iters, batch_size, Zep=None):
    S_, A_, SP, C_, ZQ = D['s'], D['a'], D['sp'], D['c'].astype(np.float32), D['zc']
    N = len(S_)
    delta = env.idx_delta_state
    total = 0.0
    for _ in range(n_iters):
        idx = np.random.choice(N, size=min(batch_size, N), replace=False)
        s, a, sp, c = S_[idx], A_[idx], SP[idx], C_[idx]
        zmu = np.repeat(mu[None], len(idx), 0) if Zep is None else Zep[D['e'][idx]].astype(np.float64)   # design the networks see
        with torch.no_grad():
            lp = _logpi(fast_forward(actor, zmu, s, env), torch.from_numpy(env.valid_ac[s]))
            lp_a = lp[torch.arange(len(idx)), torch.from_numpy(a)]
            pip = _probs(fast_forward(actor, zmu, sp, env), torch.from_numpy(env.valid_ac[sp]))
            Vp = (pip * fast_forward(critic_tgt, zmu, sp, env)).sum(1)
            Vp[torch.from_numpy(sp == delta)] = 0.0
            y = torch.from_numpy(c) + beta * lp_a + gamma * Vp
        critic.train(); opt.zero_grad()
        Qp = fast_forward(critic, ZQ[idx], s, env, grad=True, acts=a)
        loss = ((Qp - y) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
        opt.step()
        total += loss.item()
    return total / n_iters


# ---------------------------------------------------------------------------
# Actor: every step draws FRESH episodes at zeta = mu;  w(s) = sum_t gamma^t 1[s_t=s], normalised
# ---------------------------------------------------------------------------
def actor_loss(f, valid_t, w, Q, beta):
    lg = (-f).masked_fill(~valid_t, -1e9)
    lp = lg - torch.logsumexp(lg, dim=1, keepdim=True)
    pi_t = torch.softmax(lg.detach(), dim=1)
    V = (pi_t * Q).sum(1, keepdim=True)
    return (w[:, None] * pi_t * lp * ((Q - V) + beta) * valid_t.float()).sum()


def actor_update(actor, opt, critic, mu, logsig, env, gamma, beta, grad_clip, n_steps, n_traj, eps):
    S = env.st_count
    total_gn = 0.0
    for _ in range(n_steps):
        if CFG['train_at'] == 'designs':
            Zs = sample_designs(mu, logsig, n_traj)                      # every fresh episode at its own design
            D = collect(actor, mu, env, n_traj, eps, CFG['sf_zeta_noise'], Zep=Zs)
            ss = D['s']                                                   # rows = transitions (zeta_i, s_t)
            wt = gamma ** D['t'].astype(np.float64)
            w = torch.tensor(wt / wt.sum(), dtype=torch.float32)
            ZR = Zs[D['e']]
        else:
            D = collect(actor, mu, env, n_traj, eps, 0.0)
            w_s = np.bincount(D['s'], weights=gamma ** D['t'].astype(np.float64), minlength=S)
            ss = np.nonzero(w_s)[0]
            w = torch.tensor(w_s[ss] / w_s[ss].sum(), dtype=torch.float32)
            ZR = np.repeat(mu[None], len(ss), 0)
        Q = fast_forward(critic, ZR, ss, env)
        valid_t = torch.from_numpy(env.valid_ac[ss])
        actor.train(); opt.zero_grad()
        f = fast_forward(actor, ZR, ss, env, grad=True)
        loss = actor_loss(f, valid_t, w, Q, beta)
        loss.backward()
        total_gn += float(nn.utils.clip_grad_norm_(actor.parameters(), grad_clip))
        opt.step()
    return total_gn / n_steps


# ---------------------------------------------------------------------------
# Design distribution (REINFORCE)
# ---------------------------------------------------------------------------
class _Adam:
    def __init__(self, n, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m = np.zeros(n); self.v = np.zeros(n); self.t = 0
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        return self.lr * (self.m / (1 - self.b1 ** self.t)) / (np.sqrt(self.v / (1 - self.b2 ** self.t)) + self.eps)


def design_gradients(mu, logsig, raw, J, alpha):
    G = len(J)
    sig = np.exp(logsig)
    z = (raw - mu[None]) / sig[None]
    b = (J.sum() - J) / (G - 1) if (CFG['sf_loo'] and G > 1) else np.zeros_like(J)
    adv = J - b
    g_mu = (adv[:, None, None] * z / sig[None]).mean(0)
    g_ls = (adv[:, None, None] * (z ** 2 - 1.0)).mean(0) - alpha
    if not CFG['sf_learn_sigma']:
        g_ls = np.zeros_like(g_ls)
    return g_mu, g_ls


def sf_design_step(actor, mu, logsig, env, opt, beta, eps):
    """G = design_traj designs, episodes_per_design fresh episodes each; returns new (mu, logsig), g_mu, mean score, mean hard cost."""
    N, E = CFG['design_traj'], CFG['episodes_per_design']
    sig = np.exp(logsig)
    lo, hi = CFG['sf_box']
    raw = mu[None] + sig[None] * np.random.randn(N, *mu.shape)
    Z = np.clip(raw, lo, hi)
    D = collect(actor, mu, env, N * E, eps, 0.0, Zep=np.repeat(Z, E, axis=0), beta=beta)
    J = (D['J_soft'] if CFG['sf_soft'] else D['J_hard']).reshape(N, E).mean(1)             # score of design i = mean over its E episodes
    Jn = (J - J.mean()) / (J.std() + 1e-8) if CFG['sf_norm'] else J
    alpha = CFG['sf_ent0'] * beta / CFG['beta_max']
    g_mu, g_ls = design_gradients(mu, logsig, raw, Jn, alpha)
    upd = opt.step(np.concatenate([g_mu.ravel(), g_ls.ravel()]))
    n = mu.size
    mu_new = np.clip(mu - upd[:n].reshape(mu.shape), lo, hi)
    ls_new = np.clip(logsig - upd[n:].reshape(mu.shape), np.log(CFG['sf_sigma_min']), np.log(CFG['sf_sigma_max']))
    return mu_new, ls_new, g_mu, float(J.mean()), float(D['J_hard'].mean())


def evaluate_policy(actor, critic, zeta, env) -> float:
    users = np.arange(env.s_low, env.s_high + 1)
    F = feat_rows(np.repeat(zeta[None], len(users), 0), users, env)
    pi = _probs(_rows_forward(actor, F), torch.from_numpy(env.valid_ac[users]))
    return float((pi * _rows_forward(critic, F)).sum(1).mean())


def log_facility_actions(actor, zeta, env):
    uav = env.fac_state_idx
    pi = pi_rows(actor, np.repeat(zeta[None], len(uav), 0), uav, env)
    log.info("  -- Greedy actions from UAV states --")
    for j in range(env.n_fac):
        valid = np.where(env.valid_ac[uav[j]])[0]
        ag = valid[np.argmax(pi[j][valid])]
        log.info(f"    f{j+1} (s={uav[j]}) | greedy: {env.ac_details[ag][0]} "
                 f"| probs: { {env.ac_details[a][0]: f'{pi[j][a]:.3f}' for a in valid} }")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(kappa, placement, seed=None, total_iters=None, design_traj=None, actor_traj=None, exact_every=None,
         loo=False, norm=False, no_kick=True, sigma0=None, train_at=None, threads=1, episodes_per_design=None):
    torch.set_num_threads(threads)
    CFG['kappa'] = kappa
    if seed is not None: CFG['seed'] = seed
    if total_iters is not None: CFG['total_iters'] = total_iters
    if design_traj is not None: CFG['design_traj'] = design_traj
    if actor_traj is not None: CFG['actor_traj'] = actor_traj
    if episodes_per_design is not None: CFG['episodes_per_design'] = episodes_per_design
    if exact_every is not None: CFG['exact_every'] = exact_every
    if train_at is not None: CFG['train_at'] = train_at
    CFG['sf_sigma0'] = sigma0 if sigma0 is not None else (0.5 if CFG['train_at'] == 'designs' else 0.25)   # as in train_mf_sf_v2.py
    if loo: CFG['sf_loo'] = True
    if norm: CFG['sf_norm'] = True
    if no_kick: CFG['sf_kick'] = False
    zeta_init_list = get_init(kappa, placement)
    tag = f"k{int(round(kappa*100)):03d}_P{placement}"
    out_dir = setup_run(tag, CFG['design_traj'], CFG['train_at'], CFG['episodes_per_design'])
    t_start = time.time()
    log.info(f"RUN {tag}  CFG={json.dumps({k_: v_ for k_, v_ in CFG.items()})}")
    np.random.seed(CFG['seed']); torch.manual_seed(CFG['seed'])

    st_d, ac_d, dest, zeta_init, fsl, vac = build_env_info(seed=1)
    env = precompute_env(st_d, ac_d, dest, fsl, vac, eta_tr=CFG['eta_tr'], kappa=CFG['kappa'])
    zeta_init = np.array(zeta_init_list, dtype=float).reshape(zeta_init.shape)
    np.random.seed(CFG['seed'])        # build_env_info re-seeds numpy; restore the run seed
    S, A = env.st_count, env.ac_count
    nF, d = zeta_init.shape
    gamma, beta = CFG['gamma'], CFG['beta_max']
    fd = d + S + d + A + nF * d
    log.info(f"S={S}  A={A}  nF={nF}  dim={d}  feat_dim={fd}  train_at={CFG['train_at']}  design_traj={CFG['design_traj']}x{CFG['episodes_per_design']}  actor_traj={CFG['actor_traj']}")

    exact_ok = True
    try:
        import exact_eval as ee
    except Exception as e:
        ee, exact_ok = None, False
        log.info(f"exact_eval not available: {e}")

    actor, critic = MLP(fd, CFG['hidden_dim']), MLP(fd, CFG['hidden_dim'])
    critic_tgt = copy.deepcopy(critic)
    for p in critic_tgt.parameters(): p.requires_grad_(False)
    opt_a = optim.Adam(actor.parameters(), lr=CFG['lr_actor'])
    opt_c = optim.Adam(critic.parameters(), lr=CFG['lr_critic'])

    zeta = zeta_init.copy()
    logsig = np.full_like(zeta, np.log(CFG['sf_sigma0']))
    opt_z = _Adam(2 * zeta.size, CFG['lr_zeta'])
    overlap_cnt, last_kick, n_kicks, step_no = {}, -10**9, 0, 0
    history = dict(J=[], loss_c=[], gz_norm=[], zeta=[], gz=[], gn_a=[], mean_len=[], beta=[], iteration=[], k=[], time=[],
                   sigma=[], logsig=[], J_design=[], J_design_hard=[], J_exact=[], proper=[], kicks=[])

    log.info(f"{'iter':>5} | {'k':>4} | {'phase':>7} | {'J_est':>8} | {'loss_c':>8} | {'gn_a':>8} | {'g_mu':>8} | zeta")
    log.info("-" * 120)
    row = 0
    for iteration in range(CFG['total_iters']):
        eps_now = (CFG['actor_eps'] if iteration < CFG['eps_off_iter'] else 0.0)
        eps_des = (CFG['resample_eps'] if iteration < CFG['eps_off_iter'] else 0.0)
        for k in range(CFG['outer_iters']):
            t0 = time.time()
            warmup = (iteration == 0) and (k < CFG['warmup_iters'])

            # 1. critic data at zeta = mu with per-transition zeta noise
            if CFG['train_at'] == 'designs':
                Zc = sample_designs(zeta, logsig, CFG['M_episodes'])      # every episode at its own design
                D = collect(actor, zeta, env, CFG['M_episodes'], CFG['eps_greedy'], CFG['sf_zeta_noise'], Zep=Zc, want_zc=True)
            elif CFG['train_at'] == 'mu_zi':
                lo_, hi_ = CFG['sf_box']                                  # one perturbed design per episode, policy / target at that design
                Zc = np.clip(zeta[None] + CFG['zi_noise_std'] * np.random.randn(CFG['M_episodes'], *zeta.shape), lo_, hi_)
                D = collect(actor, zeta, env, CFG['M_episodes'], CFG['eps_greedy'], 0.0, Zep=Zc, want_zc=True)
            else:
                Zc = None
                D = collect(actor, zeta, env, CFG['M_episodes'], CFG['eps_greedy'], CFG['zeta_noise_std'], want_zc=True)
            mean_len = float(D['len'].mean())
            J_est = evaluate_policy(actor, critic, zeta, env)
            loss_c = critic_update(critic, critic_tgt, actor, opt_c, D, zeta, env, gamma, beta,
                                   CFG['grad_clip'], CFG['critic_iters'], CFG['batch_size'], Zep=Zc)
            with torch.no_grad():
                for p, pt in zip(critic.parameters(), critic_tgt.parameters()):
                    pt.data.mul_(CFG['nu']).add_((1.0 - CFG['nu']) * p.data)
            del D

            # 2. actor: fresh episodes at zeta = mu at every step
            gn_a = actor_update(actor, opt_a, critic, zeta, logsig, env, gamma, beta, CFG['grad_clip'],
                                CFG['actor_steps'], CFG['actor_traj'], eps_now)
            if k % 10 == 0:
                log_facility_actions(actor, zeta, env)
            if any(not torch.isfinite(p).all() for p in actor.parameters()):
                log.warning("NaN detected in actor weights -- reinitialising")
                actor = MLP(fd, CFG['hidden_dim'])
                opt_a = optim.Adam(actor.parameters(), lr=CFG['lr_actor'])
            proper = greedy_is_proper(actor, zeta, env)

            # 3. design update: N fresh episodes, one design zeta_i ~ q each, REINFORCE
            J_des, J_des_hard = float('nan'), float('nan')
            if not warmup:
                zeta, logsig, g_mu, J_des, J_des_hard = sf_design_step(
                    actor, zeta, logsig, env, opt_z, beta, eps_des)
                gz_norm = float(np.linalg.norm(g_mu)); gz = g_mu.flatten().copy()
            else:
                gz_norm = 0.0; gz = np.zeros(zeta.size)

            # overlap perturbation of (mu) -- as in train_mf_sf_v2.py (disable with --no_kick)
            step_no += 1
            if CFG['sf_kick'] and not warmup:
                for i in range(nF):
                    for j in range(i + 1, nF):
                        if np.linalg.norm(zeta[i] - zeta[j]) < CFG['sf_sep']:
                            overlap_cnt[(i, j)] = overlap_cnt.get((i, j), 0) + 1
                        else:
                            overlap_cnt[(i, j)] = 0
                due = [pr for pr, c in overlap_cnt.items() if c >= CFG['sf_kick_patience']]
                if due and step_no - last_kick >= CFG['sf_kick_cooldown']:
                    i, j = max(due, key=lambda pr: overlap_cnt[pr])
                    ang = np.random.uniform(0, 2 * np.pi)
                    u = np.array([np.cos(ang), np.sin(ang)])
                    lo, hi = CFG['sf_box']
                    zeta[i] = np.clip(zeta[i] + CFG['sf_kick_move'] * u, lo, hi)
                    zeta[j] = np.clip(zeta[j] - CFG['sf_kick_move'] * u, lo, hi)
                    for idx in (list(range(2 * i, 2 * i + 2)) + list(range(2 * j, 2 * j + 2))):
                        opt_z.m[idx] = 0.0; opt_z.v[idx] = 0.0
                    overlap_cnt = {}; last_kick = step_no; n_kicks += 1
                    log.info(f"  [kick #{n_kicks}] UAVs f{i+1},f{j+1} overlapped -> moved apart by {2*CFG['sf_kick_move']:.2f}")

            J_ex = float('nan')
            if exact_ok and CFG['exact_every'] and (row % CFG['exact_every'] == 0):
                try:
                    J_ex = float(ee.hard_J(zeta, env))
                except Exception as e:
                    exact_ok = False; log.info(f"exact hard_J disabled: {e}")

            phase = "warmup" if warmup else "joint"
            zs = np.array2string(zeta.flatten(), precision=3, separator=',')
            log.info(f"{iteration:5d} | {k:4d} | {phase:>7} | {J_est:8.4f} | {loss_c:8.4f} | {gn_a:8.3e} | {gz_norm:8.3e} | "
                     f"{zs}  [{time.time()-t0:.1f}s]  len={mean_len:.2f}  sig={np.exp(logsig).mean():.3f}"
                     f"  Jdes={J_des:.3f}  Jex={J_ex:.4f}")
            for key, val in (('J', J_est), ('loss_c', loss_c), ('gz_norm', gz_norm), ('gn_a', float(gn_a)),
                             ('mean_len', mean_len), ('beta', beta), ('iteration', iteration), ('k', k),
                             ('time', time.time() - t0), ('sigma', float(np.exp(logsig).mean())),
                             ('J_design', J_des), ('J_design_hard', J_des_hard), ('J_exact', J_ex),
                             ('proper', int(proper)), ('kicks', n_kicks)):
                history[key].append(val)
            history['zeta'].append(zeta.copy()); history['gz'].append(gz); history['logsig'].append(logsig.copy())
            row += 1

        beta = max(CFG['beta_min'], beta * CFG['beta_anneal'])
        log.info(f"  [anneal] beta -> {beta:.4f}")
        if (iteration + 1) % 10 == 0 or iteration == CFG['total_iters'] - 1:
            torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), critic_tgt=critic_tgt.state_dict(),
                            zeta=zeta, logsig=logsig, beta=beta, iteration=iteration, kappa=CFG['kappa'], cfg=dict(CFG)),
                       os.path.join(out_dir, f"ckpt_{tag}_p{iteration+1:03d}.pt"))

    # ── Final output ──────────────────────────────────────────────────────
    log.info("=" * 120)
    log.info(f"Final zeta:\n{zeta}\nFinal sigma:\n{np.exp(logsig)}")
    log.info(f"Final J_est: {history['J'][-1]:.4f}")
    np.savez(os.path.join(out_dir, f"hist_{tag}.npz"),
             zeta=np.array(history['zeta']), J=np.array(history['J']), loss_c=np.array(history['loss_c']),
             gz_norm=np.array(history['gz_norm']), gz=np.array(history['gz']), gn_a=np.array(history['gn_a']),
             mean_len=np.array(history['mean_len']), beta=np.array(history['beta']),
             iteration=np.array(history['iteration']), k=np.array(history['k']), time=np.array(history['time']),
             sigma=np.array(history['sigma']), logsig=np.array(history['logsig']), J_design=np.array(history['J_design']),
             J_design_hard=np.array(history['J_design_hard']), J_exact=np.array(history['J_exact']),
             proper=np.array(history['proper']), kicks=np.array(history['kicks']),
             zeta_init=zeta_init.flatten(), zeta_final=zeta, logsig_final=logsig)

    pi_final = pi_table(actor, zeta, env)
    Q_final = _rows_forward(critic, feat_rows(np.repeat(zeta[None], S, 0), np.arange(S), env)).numpy()
    np.savez(os.path.join(out_dir, f"final_policy_{tag}.npz"), pi=pi_final, Q=Q_final, zeta=zeta, sigma=np.exp(logsig),
             beta=beta, valid_ac=env.valid_ac, state_coords=get_state_coords(zeta, env))

    R = collect(actor, zeta, env, 200, 0.0, 0.0)
    order = np.lexsort((R['t'], R['e']))
    ro = [[] for _ in range(200)]
    for i in order:
        ro[R['e'][i]].append([int(R['s'][i]), int(R['a'][i]), int(R['sp'][i]), float(R['c'][i])])
    with open(os.path.join(out_dir, f"rollouts_{tag}.json"), "w") as fh:
        json.dump(ro, fh)

    summary = dict(tag=tag, method="score-function baseline", train_at=CFG['train_at'], kappa=kappa, placement=placement, seed=CFG['seed'],
                   zeta_init=list(map(float, zeta_init.flatten())), zeta_final=list(map(float, zeta.flatten())),
                   sigma_final=list(map(float, np.exp(logsig).flatten())), n_kicks=n_kicks,
                   J_est_final=float(history['J'][-1]), mean_len_final=float(history['mean_len'][-1]),
                   beta_final=float(beta), runtime_min=(time.time() - t_start) / 60.0, cfg=dict(CFG))
    try:
        import exact_eval as ee2
        summary['J_exact_hard_init'] = float(ee2.hard_J(zeta_init, env))
        summary['J_exact_hard_final'] = float(ee2.hard_J(zeta, env))
        P, C = ee2.build_kernel_fast(zeta, env)
        d_ = env.idx_delta_state
        Pp = np.einsum('sa,sap->sp', pi_final, P); Cp = (pi_final * C).sum(1)
        M_ = np.eye(S) - Pp; b_ = Cp.copy(); M_[d_, :] = 0; M_[d_, d_] = 1; b_[d_] = 0
        V_ = np.linalg.solve(M_, b_)
        summary['J_exact_policy_final'] = float(V_[np.arange(env.s_low, env.s_high + 1)].mean())
        log.info(f"exact: J*(init)={summary['J_exact_hard_init']:.4f}  J*(final zeta)={summary['J_exact_hard_final']:.4f}  "
                 f"J(learned policy, final zeta)={summary['J_exact_policy_final']:.4f}")
    except Exception as e:
        log.info(f"exact evaluation skipped: {e}")
    with open(os.path.join(out_dir, f"summary_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(history['J']); axes[0].set_title("$J_{est}$ (critic, at $\\mu$)")
    axes[1].plot(history['loss_c']); axes[1].set_title("Critic TD loss")
    axes[2].plot(history['gz_norm']); axes[2].set_title("$\\|g_\\mu\\|$ (REINFORCE)")
    axes[3].plot(history['sigma']); axes[3].set_title("mean $\\sigma$")
    for ax in axes: ax.set_xlabel("row"); ax.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"curves_{tag}.png"), dpi=120); plt.close('all')
    try:
        plot_env(env, zeta, title=f"Final UAV Placement, score-function baseline ({tag})")
        plt.savefig(os.path.join(out_dir, f"placement_{tag}.png"), dpi=120); plt.close('all')
    except Exception as e:
        log.info(f"plot_env skipped: {e}")
    open(os.path.join(out_dir, "DONE"), "w").write("ok")
    return actor, critic, zeta, env, history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--placement", type=int, required=True, choices=range(1, 7))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--total_iters", type=int, default=None, help="override (pilots only)")
    ap.add_argument("--design_traj", type=int, default=None, help="episodes (one design each) per design step; default 51")
    ap.add_argument("--actor_traj", type=int, default=None, help="fresh episodes per actor step; default 140")
    ap.add_argument("--exact_every", type=int, default=None, help="exact hard J*(mu) every n rows (0 = never); default 10")
    ap.add_argument("--sigma0", type=float, default=None)
    ap.add_argument("--loo", action="store_true", help="leave-one-out baseline (default: none)")
    ap.add_argument("--norm", action="store_true", help="standardise the scores (default: no)")
    ap.add_argument("--no_kick", action="store_true", help="disable the overlap perturbation of mu")
    ap.add_argument("--train_at", choices=["designs", "mu", "mu_zi"], default=None, help="default: designs (Schaff/Cauz)")
    ap.add_argument("--episodes_per_design", type=int, default=None, help="episodes per sampled design in the design update; default 1")
    ap.add_argument("--threads", type=int, default=1, help="torch threads for this run (use >1 only when few runs are in flight)")
    a = ap.parse_args()
    main(a.kappa, a.placement, a.seed, a.total_iters, a.design_traj, a.actor_traj, a.exact_every, a.loo, a.norm, a.no_kick, a.sigma0, a.train_at, a.threads, a.episodes_per_design)
