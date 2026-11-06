"""
train_mf.py — EnvPG-AC: Environment Parameter Gradient Actor-Critic.

Networks:
    Actor:  f_theta(s, a, zeta)      -> scalar
            input: [s_coord | s_oh | a_coord | a_oh | zeta_flat]
            policy: pi(a|s,zeta) = softmax_a(-f_theta(s,a,zeta))

    Critic: Q_phi(s, a, zeta, xi)    -> scalar
            input: [s_coord | s_oh | a_coord | a_oh | zeta_flat | xi_flat]
            xi is passed explicitly but held fixed at zeta_k during training
            (operationalizes the two-copy separation in the EnvPG theorem)

Gradient estimators (trajectory-based):

    Zeta gradient — central finite differences on the trajectory-based
    EnvPG objective:
        g(zeta) = E[ sum_t gamma^t * nabla_zeta [
                      log pi(a_t|s_t,zeta) * Q(s_t,a_t,zeta_k,zeta_k)
                    + Q(s_t,a_t,zeta,zeta_k) ] ]

    Actor gradient — trajectory-based policy gradient:
        g(theta) = E[ sum_t gamma^t * nabla_theta log pi(a_t|s_t,zeta) *
                      (Q(s_t,a_t,zeta,zeta) - V(s_t) + beta) ]

    Both estimators use gamma^t discounting along sampled trajectories,
    directly mirroring the theoretical expressions for nabla J.

    Critic is updated DQN-style using transitions flattened from the
    same trajectories, with the target network fixed at xi = zeta_k.
"""

from __future__ import annotations
import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
import logging, sys, time, copy
import argparse, json
import matplotlib
matplotlib.use("Agg")                       # no windows: figures are saved to files
import matplotlib.pyplot as plt
from env2_paraRLModel import (
    build_env_info, precompute_env, EnvStructural,
    env_step, get_state_coords, plot_env, plot_routing
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Tee:
    def __init__(self, f):
        self.file   = open(f, "w", encoding="utf-8", errors="replace")
        self.stdout = sys.stdout
    def write(self, m):
        self.stdout.write(m)
        self.file.write(m)
    def flush(self):
        self.stdout.flush()
        self.file.flush()

log  = logging.getLogger()
HERE = os.path.dirname(os.path.abspath(__file__))


def setup_run(tag):
    """Create results/<tag>/ next to this script; tee stdout/stderr into its log file."""
    out_dir = os.path.join(HERE, "results", tag)
    os.makedirs(out_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(out_dir, f"log_{tag}.txt"))
    sys.stderr = sys.stdout
    log.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    log.addHandler(h); log.setLevel(logging.INFO)
    return out_dir


# ---------------------------------------------------------------------------
# Initial placements (R1-R5 = starts of the original EnvPG logs; 6 = mean of R1-R5)
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
# Configuration
# ---------------------------------------------------------------------------

CFG = dict(
    seed            = 0,
    gamma           = 1,
    eta_tr          = 100.0,
    kappa           = 0.9,  #original 0.85

    # Networks
    hidden_dim      = 128,
    lr_actor        = 3e-4,
    lr_critic       = 1e-4,
    grad_clip       = 0.5,

    # Target network
    nu              = 0.995,          # Polyak rate
    max_ep_steps    = 50,         # after this many steps the action is FORCED to 'go to delta' (valid from every UAV state) until the episode
                                      # terminates. The forced steps are ordinary kernel transitions with their true cost, so every episode ends
                                      # at delta and no cut/truncation occurs. Inactive when the policy reaches delta within max_ep_steps.

    # Training loop
    total_iters     = 100,            # outer temperature annealing iterations
    outer_iters     = 50,             # gradient steps per temperature
    warmup_iters    = 20,             # critic+actor only, zeta fixed

    # Data collection
    M_episodes      = 5000,           # episodes per outer iteration
    batch_size      = 512,

    # Critic
    critic_iters    = 100,            # gradient steps per outer iter
#    grad_clip       = 1.0,

    # Actor
    actor_steps     = 50,             # gradient steps per outer iter

    # Exploration
    eps_greedy      = 0.2,            # prob. of a uniformly random valid action
    vectorized_collect = True,        # True: all episodes advance together (fast);  False: original one-episode-at-a-time loop
    sample_actions  = True,           # True: otherwise SAMPLE a ~ pi(.|s,zeta_k); False: argmax (original EnvPG-AC)
    resample_at_zeta_k = True,        # True: zeta-gradient uses FRESH trajectories generated at zeta_k (no zeta noise)
    resample_eps    = 0.0,            #   random-action prob. used when generating those fresh trajectories
    actor_eps       = 0.0,            # random-action prob. for the fresh trajectories used in the policy (actor) update; collected at FIXED zeta_k
    zeta_at_theta_k = True,           # True: the zeta step (fresh trajectories, pi, finite-difference objective) uses the actor pi_{theta_k} as it was
                                      #   BEFORE this iteration's actor update, i.e. the same policy the critic was trained for, so that w(s), pi and Q of
                                      #   the EnvPG gradient all refer to pi_{theta_k}.  False: previous behaviour (actor already updated to theta_{k+1}).
    zeta_init       = [6.837,3.346,6.082,3.751,6.128,3.501,5.703,5.033,3.275,3.293],   # R1; None -> random as before
    zeta_noise_std  = 0.5,            # std of perturbation added to zeta

    # Zeta optimiser (Adam)
    lr_zeta         = 1e-2,
    eps_fd          = 1e-1,           # finite-difference step size

    # Entropy temperature annealing
    beta_max        = 1.0,
    beta_min        = 0.001,
    beta_anneal     = 0.95,

    # Overlap kick (same rule as in the score-function baseline): if two UAVs get closer than kick_sep (Euclidean),
    # one pair per row is pushed apart, each by kick_move in opposite directions along a random direction,
    # and the Adam moments of those two UAVs are cleared.  Not applied during warm-up.  False -> off.
    kick            = True,
    kick_sep        = 0.3,
    kick_move       = 0.5,
    kick_box        = (0.0, 12.0),    # kicked positions are clipped to this box
)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def make_actor_feat_batch(zeta: np.ndarray,
                           env: EnvStructural) -> np.ndarray:
    """
    Construct actor feature matrix for all (s, a) pairs.

    Feature vector for (s, a):
        [x_s (dim) | s_oh (S) | x_a (dim) | a_oh (A) | zeta_flat (nF*dim)]

    where x_s is the coordinate of state s (UAV coords from zeta),
    and x_a is the coordinate of the intended next state for action a.

    Returns: (S, A, actor_feat_dim) float array.
    """
    S, A  = env.st_count, env.ac_count
    nF, d = zeta.shape

    # State coordinates with current zeta
    coords = get_state_coords(zeta, env)              # (S, dim)

    # Action coordinates: coordinate of intended next state for each action
    # UAV action a -> coordinate of UAV fac_id_action[a]
    # Delta action -> coordinate of destination
    ac_coords = np.zeros((A, d))
    for a in range(A):
        if a == env.idx_delta_action:
            ac_coords[a] = env.dest_loc[0]
        else:
            ac_coords[a] = zeta[env.fac_id_action[a]]

    # One-hot encodings
    s_oh = np.eye(S)                                  # (S, S)
    a_oh = np.eye(A)                                  # (A, A)
    zf   = zeta.flatten()                             # (nF*d,)

    feat_dim = d + S + d + A + nF * d
    F = np.zeros((S, A, feat_dim))
    F[:, :, :d]            = coords[:, None, :]       # x_s
    F[:, :, d:d+S]         = s_oh[:, None, :]         # s_oh
    F[:, :, d+S:2*d+S]     = ac_coords[None, :, :]   # x_a
    F[:, :, 2*d+S:2*d+S+A] = a_oh[None, :, :]        # a_oh
    F[:, :, 2*d+S+A:]      = zf                       # zeta_flat
    return F


def make_critic_feat_batch(zeta: np.ndarray, xi: np.ndarray,
                            env: EnvStructural) -> np.ndarray:
    """
    Construct critic feature matrix for all (s, a) pairs.

    Feature vector for (s, a):
        [x_s (dim) | s_oh (S) | x_a (dim) | a_oh (A) |
         zeta_flat (nF*dim) | xi_flat (nF*dim)]

    zeta: current-step environment parameter (active, may be perturbed)
    xi:   future environment parameter (fixed at zeta_k)

    Returns: (S, A, critic_feat_dim) float array.
    """
    S, A  = env.st_count, env.ac_count
    nF, d = zeta.shape

    # State and action coordinates use zeta (current-step parameter)
    coords = get_state_coords(zeta, env)

    ac_coords = np.zeros((A, d))
    for a in range(A):
        if a == env.idx_delta_action:
            ac_coords[a] = env.dest_loc[0]
        else:
            ac_coords[a] = zeta[env.fac_id_action[a]]

    s_oh = np.eye(S)
    a_oh = np.eye(A)
    zf   = zeta.flatten()
    xf   = xi.flatten()

    feat_dim = d + S + d + A + nF * d + nF * d
    F = np.zeros((S, A, feat_dim))
    F[:, :, :d]                    = coords[:, None, :]
    F[:, :, d:d+S]                 = s_oh[:, None, :]
    F[:, :, d+S:2*d+S]             = ac_coords[None, :, :]
    F[:, :, 2*d+S:2*d+S+A]         = a_oh[None, :, :]
    F[:, :, 2*d+S+A:2*d+S+A+nF*d] = zf                   # zeta_flat
    F[:, :, 2*d+S+A+nF*d:]         = xf                   # xi_flat
    return F


def make_critic_feat_batch_single_sample(
        s: int, a: int, zeta_curr: np.ndarray, xi: np.ndarray,
        env: EnvStructural) -> np.ndarray:
    """
    Construct critic feature vector for a single (s, a) pair.
    Used during critic training where each sample has its own zeta_curr.

    Returns: (critic_feat_dim,) float array.
    """
    d  = env.dim
    S  = env.st_count
    A  = env.ac_count
    nF = env.n_fac

    coords = get_state_coords(zeta_curr, env)

    x_s = coords[s]
    if a == env.idx_delta_action:
        x_a = env.dest_loc[0]
    else:
        x_a = zeta_curr[env.fac_id_action[a]]

    s_oh = np.zeros(S); s_oh[s] = 1.0
    a_oh = np.zeros(A); a_oh[a] = 1.0
    zf   = zeta_curr.flatten()
    xf   = xi.flatten()

    return np.concatenate([x_s, s_oh, x_a, a_oh, zf, xf])


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Two-hidden-layer MLP with Tanh activations, scalar output."""
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Policy and Q evaluation
# ---------------------------------------------------------------------------

def get_pi(actor: MLP, zeta: np.ndarray,
           env: EnvStructural) -> np.ndarray:
    S, A = env.st_count, env.ac_count

    # Guard against NaN zeta
    if not np.all(np.isfinite(zeta)):
        log.warning("NaN zeta in get_pi — returning uniform policy")
        denom = env.valid_ac.sum(1, keepdims=True).astype(float)
        return np.where(
            denom > 0,
            env.valid_ac / np.maximum(denom, 1),
            0.0
        )

    feats = make_actor_feat_batch(zeta, env)

    actor.eval()
    with torch.no_grad():
        f_raw = actor(
            torch.tensor(feats.reshape(S * A, -1),
                         dtype=torch.float32)
        )
        # Check for NaN in actor output
        if not torch.isfinite(f_raw).all():
            log.warning("NaN in actor output — returning uniform policy")
            denom = env.valid_ac.sum(1, keepdims=True).astype(float)
            return np.where(
                denom > 0,
                env.valid_ac / np.maximum(denom, 1),
                0.0
            )
        f = f_raw.numpy().reshape(S, A)

    # Softmax over valid actions
    lg = -f
    lg[~env.valid_ac] = -np.inf

    # Safe max: only over finite values
    finite_mask = np.isfinite(lg)
    row_max = np.where(
        finite_mask.any(axis=1, keepdims=True),
        np.where(finite_mask, lg, -np.inf).max(1, keepdims=True),
        0.0
    )
    lg = lg - row_max
    pi = np.exp(np.where(finite_mask, lg, -700.))
    pi[~env.valid_ac] = 0.0

    row        = pi.sum(1, keepdims=True)
    valid_count = env.valid_ac.sum(1, keepdims=True).astype(float)
    uniform     = np.where(
        valid_count > 0,
        env.valid_ac / np.maximum(valid_count, 1),
        0.0
    )
    pi = np.where(row > 1e-300, pi / row, uniform)

    # Explicitly zero delta state — no valid actions
    pi[env.idx_delta_state, :] = 0.0

    return pi

def get_Q(critic: MLP, zeta: np.ndarray, xi: np.ndarray,
          env: EnvStructural) -> np.ndarray:
    """
    Compute Q[s, a] = Q_phi(s, a, zeta, xi) for all (s, a).

    zeta: current-step environment parameter
    xi:   future environment parameter (fixed at zeta_k)

    Returns: (S, A) float array.
    """
    S, A  = env.st_count, env.ac_count
    feats = make_critic_feat_batch(zeta, xi, env)         # (S, A, fd)

    critic.eval()
    with torch.no_grad():
        Q = critic(
            torch.tensor(feats.reshape(S * A, -1),
                         dtype=torch.float32)
        ).numpy().reshape(S, A)
    return Q

def log_facility_actions(actor: MLP, zeta: np.ndarray,
                         env: EnvStructural) -> None:
    """
    Log the greedy action selected from each UAV state under current policy.
    Greedy action = argmax_a pi(a|s, zeta) over valid actions.
    """
    pi_all = get_pi(actor, zeta, env)   # (S, A)

    log.info("  -- Greedy actions from UAV states --")
    for j in range(env.n_fac):
        s        = env.fac_state_idx[j]
        pi_s     = pi_all[s]                          # (A,)
        valid    = np.where(env.valid_ac[s])[0]       # valid action indices
        a_greedy = valid[np.argmax(pi_s[valid])]      # greedy action index
        a_label  = env.ac_details[a_greedy][0]        # action label e.g. "f3" or "delta"
        pi_valid = pi_s[valid]                        # probabilities over valid actions
        a_labels = [env.ac_details[a][0] for a in valid]

        log.info(
            f"    f{j+1} (s={s}) | greedy: {a_label} "
            f"| probs: { {l: f'{p:.3f}' for l, p in zip(a_labels, pi_valid)} }"
        )

# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def collect_episodes_serial(actor: MLP, zeta: np.ndarray,
                     env: EnvStructural,
                     M: int, eps: float,
                     zeta_noise_std: float) -> list:
    """
    Collect M episodes using epsilon-greedy policy.

    At each step, zeta is perturbed slightly before querying the environment:
        zeta_curr = zeta + noise,  noise ~ N(0, zeta_noise_std^2)

    This perturbation operationalizes the two-copy separation in the EnvPG
    theorem: the critic sees zeta_curr (active zeta argument) while the
    target critic uses the unperturbed zeta_k (frozen xi argument).

    Returns: list of trajectories, where each trajectory is a list of
             (s, a, s_next, cost, zeta_curr) tuples. Storing as trajectories
             enables trajectory-based gradient estimation (sum_t gamma^t * ...)
             that directly mirrors the theoretical expression for nabla J.
    """
    pi_all         = get_pi(actor, zeta, env)             # (S, A)
    greedy_actions = np.argmax(pi_all, axis=1)             # (S,)
    cum_pi         = np.cumsum(pi_all, axis=1)             # for sampling a ~ pi(.|s)
    n_act          = pi_all.shape[1]
    non_term       = np.arange(env.s_low, env.s_high + 1)
    trajectories   = []

    for _ in range(M):
        s    = int(np.random.choice(non_term))
        traj = []
        while True:
            valid = np.where(env.valid_ac[s])[0]
            if len(traj) >= CFG['max_ep_steps']:
                a = int(env.idx_delta_action)                           # time limit: go to delta (valid from UAV states)
            elif np.random.rand() < eps:
                a = int(np.random.choice(valid))                       # random valid action
            elif CFG['sample_actions']:
                a = min(int((cum_pi[s] < np.random.rand() * cum_pi[s, -1]).sum()), n_act - 1)   # a ~ pi(.|s, zeta_k)
            else:
                a = int(greedy_actions[s])                              # argmax (original)

            zeta_curr          = zeta + \
                np.random.randn(*zeta.shape) * zeta_noise_std
            s_next, cost, done = env_step(s, a, zeta_curr, env)
            traj.append((s, a, s_next, cost, zeta_curr))

            if done:
                break
            s = s_next

        trajectories.append(traj)

    return trajectories


def collect_episodes_vec(actor: MLP, zeta: np.ndarray,
                         env: EnvStructural,
                         M: int, eps: float,
                         zeta_noise_std: float) -> list:
    """
    Vectorised version of collect_episodes_serial: all M episodes advance one step
    together with array operations.  Same sampling rules and the same kernel as
    env_step (verified by comparing against the exact kernel), different random stream.
    Returns the same structure: list of trajectories of (s, a, s_next, cost, zeta_curr).
    """
    pi_all  = get_pi(actor, zeta, env)
    cum_pi  = np.cumsum(pi_all, axis=1)
    greedy  = np.argmax(pi_all, axis=1)
    S, A    = pi_all.shape
    nF, d   = zeta.shape
    valid   = env.valid_ac
    delta_s, delta_a = env.idx_delta_state, env.idx_delta_action
    fixed_xy = env.state_coords_fixed
    # action -> intended next state;  action -> UAV index (or -1 for the delta action)
    act_uav = np.where(np.arange(A) == delta_a, -1, env.fac_id_action).astype(int)
    int_state = np.where(act_uav >= 0, env.fac_state_idx[np.clip(act_uav, 0, None)], delta_s).astype(int)
    uav_of_state = -np.ones(S, dtype=int)
    uav_of_state[env.fac_state_idx] = np.arange(nF)
    non_term = np.arange(env.s_low, env.s_high + 1)

    def xy_of_states(st, zc):                       # coordinates of states st (n,) given the perturbed zeta zc (n,nF,d)
        X = fixed_xy[st].copy()
        u = uav_of_state[st]
        m = u >= 0
        if m.any():
            X[m] = zc[np.where(m)[0], u[m]]
        return X

    trajs  = [[] for _ in range(M)]
    ep_ids = np.arange(M)
    s_cur  = np.random.choice(non_term, size=M).astype(int)
    t_step = 0
    while len(ep_ids):
        n  = len(ep_ids)
        # --- action selection (random w.p. eps, else sample from pi / argmax) ---
        rnd_mask = np.random.rand(n) < eps
        if CFG['sample_actions']:
            a_pol = np.minimum((cum_pi[s_cur] < (np.random.rand(n) * cum_pi[s_cur, -1])[:, None]).sum(1), A - 1)
        else:
            a_pol = greedy[s_cur]
        a_rnd = np.argmax(np.where(valid[s_cur], np.random.rand(n, A), -1.0), axis=1)   # uniform over valid actions
        a = np.where(rnd_mask, a_rnd, a_pol).astype(int)
        if t_step >= CFG['max_ep_steps']:                                # time limit: from now on go to delta (valid from UAV states)
            a = np.full(n, delta_a, dtype=int)
        # --- perturbed zeta, one draw per transition ---
        zc = zeta[None] + np.random.randn(n, nF, d) * zeta_noise_std
        # --- next state: intended w.p. kappa, otherwise Gibbs over the other valid actions' intended states ---
        s_next = int_state[a].copy()
        off = np.random.rand(n) >= env.kappa
        if off.any():
            io   = np.where(off)[0]
            alt  = valid[s_cur[io]].copy()
            alt[np.arange(len(io)), a[io]] = False                      # exclude the chosen action
            xs   = xy_of_states(s_cur[io], zc[io])                      # (m,dim)
            zio  = zc[io]
            Xact = np.where((act_uav >= 0)[None, :, None],
                            zio[:, np.clip(act_uav, 0, None), :],
                            fixed_xy[delta_s][None, None, :])           # (m,A,dim): coords of each action's intended state
            d2   = ((Xact - xs[:, None, :]) ** 2).sum(-1)               # (m,A)
            lg   = np.where(alt, -env.eta_tr * d2, -np.inf)
            has  = alt.any(1)
            mx   = np.where(has, lg.max(1), 0.0)
            ex   = np.exp(lg - mx[:, None])
            ps   = ex.sum(1)
            bad  = (ps < 1e-300) | ~np.isfinite(ps)
            if bad.any():                                               # same fallback as env_step: uniform over alternatives
                ex[bad] = alt[bad].astype(float)
                ps[bad] = ex[bad].sum(1)
            cp   = np.cumsum(ex, axis=1)
            pick = np.minimum((cp < (np.random.rand(len(io)) * cp[:, -1])[:, None]).sum(1), A - 1)
            s_off = int_state[pick]
            s_next[io[has]] = s_off[has]                                # rows with no alternative keep the intended state
        # --- cost: squared distance between the (perturbed) coordinates of s and s' ---
        cost = ((xy_of_states(s_next, zc) - xy_of_states(s_cur, zc)) ** 2).sum(1)
        if env.kappa <= 0.75:
            cost = np.where(cost <= 0.1, cost + 1.0, cost)
        cost = cost + env.hop_cost
        done = s_next == delta_s
        # --- record transitions ---
        sl, al, nl, cl = s_cur.tolist(), a.tolist(), s_next.tolist(), cost.tolist()
        for i, e in enumerate(ep_ids.tolist()):
            trajs[e].append((sl[i], al[i], nl[i], cl[i], zc[i]))
        keep   = ~done
        ep_ids = ep_ids[keep]
        s_cur  = s_next[keep]
        t_step+= 1
    return trajs

def collect_episodes(actor, zeta, env, M, eps, zeta_noise_std):
    """Dispatcher: vectorised (default) or the original serial sampler (CFG['vectorized_collect'])."""
    if CFG['vectorized_collect']:
        return collect_episodes_vec(actor, zeta, env, M, eps, zeta_noise_std)
    return collect_episodes_serial(actor, zeta, env, M, eps, zeta_noise_std)


def flatten_trajectories(trajectories: list) -> list:
    """Flatten list of trajectories into a flat list of transitions."""
    return [transition for traj in trajectories for transition in traj]


# ---------------------------------------------------------------------------
# Critic update
# ---------------------------------------------------------------------------

def critic_update(critic: MLP, critic_tgt: MLP,
                  opt: optim.Optimizer,
                  trajectories: list, zeta: np.ndarray,
                  pi: np.ndarray,
                  env: EnvStructural,
                  gamma: float, beta: float,
                  grad_clip: float,
                  n_iters: int,
                  batch_size: int) -> float:
    """
    Entropy-regularized DQN-style critic update.

    TD target:
        y^(i) = c^(i) + beta * log pi(a^(i)|s^(i), zeta_k)
              + gamma * sum_{a'} pi(a'|s'^(i), zeta_k)
                        * Q_phi^-(s'^(i), a', zeta_k, zeta_k)

    The target critic Q_phi^- is evaluated at (zeta_k, zeta_k) — both
    copies at the baseline, implementing xi = zeta_k (frozen future).

    The online critic Q_phi is evaluated at (zeta_curr, zeta_k) — where
    zeta_curr is the perturbed environment parameter stored in each
    transition, implementing the active zeta argument.

    Accepts trajectories (list of lists); flattened internally for TD updates.
    """
    S, A = env.st_count, env.ac_count
    xi   = zeta   # xi fixed at current baseline zeta_k

    # Flatten trajectories for TD updates — order within trajectory
    # does not matter for DQN-style critic training
    data = flatten_trajectories(trajectories)

    # Target soft value: V_tgt(s') = sum_a pi(a|s') * Q_tgt(s', a)
    Q_tgt = get_Q(critic_tgt, zeta, xi, env)              # (S, A)
    V_tgt = (pi * Q_tgt).sum(axis=1)                      # (S,)

    # Log policy for entropy regularization
    lp_all = np.log(np.clip(pi, 1e-12, None))             # (S, A)

    N          = len(data)
    total_loss = 0.0

    for _ in range(n_iters):
        idx   = np.random.choice(N, size=min(batch_size, N), replace=False)
        batch = [data[i] for i in idx]
        B     = len(batch)

        s_b      = np.array([t[0] for t in batch], dtype=int)
        a_b      = np.array([t[1] for t in batch], dtype=int)
        sp_b     = np.array([t[2] for t in batch], dtype=int)
        c_b      = np.array([t[3] for t in batch], dtype=float)
        zcurr_b  = np.stack([t[4] for t in batch])        # (B, nF, dim)

        # Entropy-regularized TD targets
        lp_b = lp_all[s_b, a_b]                           # (B,)
        y    = c_b + beta * lp_b + gamma * V_tgt[sp_b]    # (B,)
        y_t  = torch.tensor(y, dtype=torch.float32)

        # Critic features: each sample uses its own zeta_curr
        feats_b = np.stack([
            make_critic_feat_batch_single_sample(
                s_b[bi], a_b[bi], zcurr_b[bi], xi, env)
            for bi in range(B)
        ])                                                 # (B, feat_dim)
        feats_t = torch.tensor(feats_b, dtype=torch.float32)

        critic.train()
        opt.zero_grad()
        Q_pred = critic(feats_t)                           # (B,)
        loss   = ((Q_pred - y_t) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
        opt.step()
        total_loss += loss.item()

    return total_loss / n_iters


# ---------------------------------------------------------------------------
# Actor update
# ---------------------------------------------------------------------------

def actor_update(actor: MLP, opt: optim.Optimizer,
                 critic: MLP,
                 trajectories: list, zeta: np.ndarray,
                 pi: np.ndarray,
                 env: EnvStructural,
                 gamma: float, beta: float,
                 grad_clip: float,
                 n_steps: int,
                 itr: int,
                 batch_size: int) -> float:
    """
    Entropy-regularized policy gradient update using trajectory-based estimator.

    g_theta = E[ sum_t gamma^t * nabla_theta log pi(a_t|s_t,zeta) *
                 (Q(s_t,a_t,zeta,zeta) + beta) ]

    This directly mirrors the theoretical expression for nabla_theta J,
    using gamma^t discounting along sampled trajectories rather than
    approximating the discounted state visitation distribution w(s).

    The trajectories for w(s) are collected FRESH at every gradient step at the FIXED parameter zeta_k
    (no zeta perturbation) with the CURRENT actor and random-action probability CFG['actor_eps'].
    w(s) is normalised by the total weight, as before.
    """
    S, A   = env.st_count, env.ac_count
    M      = len(trajectories)

    # Q values and advantage
    Q  = get_Q(critic, zeta, zeta, env)                   # (S, A)
    V  = (pi * Q).sum(axis=1, keepdims=True)              # (S, 1)
    Ad = Q - V                                            # (S, A) advantage

    # Precompute actor features for all (s, a)
    feats   = make_actor_feat_batch(zeta, env)             # (S, A, fd)
    feats_t = torch.tensor(
        feats.reshape(S * A, -1), dtype=torch.float32)

    Ad_t = torch.tensor(Ad,           dtype=torch.float32)
    v_t  = torch.tensor(env.valid_ac, dtype=torch.float32)

    # Number of trajectories to sample per gradient step
    #n_traj = max(1, batch_size // 10)   # ~same data volume as before
    n_traj = 140

    total_gn = 0.0

    for _ in range(n_steps):
        # Fresh trajectories at fixed zeta_k with the current actor (eps = actor_eps), redrawn at every step
        sampled = collect_episodes(
            actor, zeta, env,
            M=n_traj,
            eps=CFG['actor_eps'] if itr < 9 else 0.0,
            zeta_noise_std=0.0)

        # Build gamma^t weighted state visitation w(s) from sampled trajectories
        w = torch.zeros(S, dtype=torch.float32)
        total_weight = 0.0
        for traj in sampled:
            for t, transition in enumerate(traj):
                s_t = transition[0]
                wt  = gamma ** t
                w[s_t]       += wt
                total_weight += wt
        if total_weight > 0:
            w /= total_weight

        actor.train()
        opt.zero_grad()

        f    = actor(feats_t).reshape(S, A)
        lg   = (-f).masked_fill(
            torch.tensor(~env.valid_ac), -1e9)
        lp   = lg - torch.logsumexp(lg, dim=1, keepdim=True)
        pi_t = torch.softmax(lg.detach(), dim=1)

        loss = (
            w.unsqueeze(1) * pi_t * lp * (Ad_t + beta) * v_t
        ).sum()

        loss.backward()
        gn = nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
        opt.step()
        total_gn += gn

    return total_gn / n_steps


# ---------------------------------------------------------------------------
# Zeta gradient (central finite differences)
# ---------------------------------------------------------------------------

def zeta_gradient_fd(actor: MLP, critic: MLP,
                     trajectories: list, zeta: np.ndarray,
                     pi: np.ndarray,
                     env: EnvStructural,
                     gamma: float, beta: float,
                     batch_size: int) -> np.ndarray:
    """
    Central finite-difference estimate of the EnvPG gradient using
    trajectory-based estimator:

        g_zeta = E[ sum_t gamma^t * nabla_zeta [
                     log pi(a_t|s_t,zeta) * Q(s_t,a_t,zeta_k,zeta_k)
                   + Q(s_t,a_t,zeta,zeta_k) ] ]

    This directly mirrors the theoretical trajectory-based expression for
    nabla_zeta J, using gamma^t discounting along sampled trajectories.

    xi = zeta_k is held fixed throughout (frozen future parameter).
    Q_fixed = Q(s,a,zeta_k,zeta_k) is precomputed once and never perturbed.
    """
    eps   = CFG['eps_fd']
    xi    = zeta.copy()               # xi fixed at current zeta_k
    nF, d = zeta.shape
    S, A  = env.st_count, env.ac_count
    M     = len(trajectories)

    # Sample trajectories
    n_traj   = max(1, batch_size // 10)   # ~same data volume as before
    traj_idx = np.random.choice(M, size=min(n_traj, M), replace=False)
    sampled  = [trajectories[i] for i in traj_idx]

    # Build gamma^t weighted state visitation w(s) from sampled trajectories
    w            = np.zeros(S)
    total_weight = 0.0
    for traj in sampled:
        for t, transition in enumerate(traj):
            s_t           = transition[0]
            wt            = gamma ** t
            w[s_t]       += wt
            total_weight += wt
    if total_weight > 0:
        w /= total_weight

    # Precompute Q_fixed = Q(s,a,zeta_k,zeta_k) — never perturbed
    # This is Term 1 weight in the EnvPG gradient
    Q_fixed = get_Q(critic, xi, xi, env)                  # (S, A)

    def objective(z: np.ndarray) -> float:
        if not np.all(np.isfinite(z)):
            return np.nan

        pi_ = get_pi(actor, z, env)
        if not np.all(np.isfinite(pi_)):
            return np.nan

        lp_ = np.log(np.clip(pi_, 1e-12, None))
        Q_z_xi = get_Q(critic, z, xi, env)

        if not np.all(np.isfinite(Q_z_xi)):
            return np.nan

        # a_t ~ pi(.|s_t, zeta_k): the sampling weights are the FIXED policy `pi` (= pi at zeta_k, same actor weights that
        # generated the trajectories); only log pi(a|s,z) and Q(s,a,z,xi) are differentiated w.r.t. z.
        val = np.sum(
            w[:, None] * pi *
            (lp_ * Q_fixed + Q_z_xi) *
            env.valid_ac
        )
        return val

    g = np.zeros_like(zeta)
    for i in range(nF):
        for j in range(d):
            zp = zeta.copy(); zp[i, j] += eps
            zm = zeta.copy(); zm[i, j] -= eps
            fp = objective(zp)
            fm = objective(zm)
            if np.isfinite(fp) and np.isfinite(fm):
                g[i, j] = (fp - fm) / (2.0 * eps)
            else:
                g[i, j] = 0.0
                log.warning(f"NaN in FD objective at zeta[{i},{j}] — skipping")
    return g


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(actor: MLP, critic: MLP,
                    zeta: np.ndarray,
                    env: EnvStructural) -> float:
    """
    Estimate J(pi, zeta) = sum_s rho(s) sum_a pi(a|s,zeta) Q(s,a,zeta,zeta).
    rho is uniform over user nodes.
    """
    S    = env.st_count
    pi   = get_pi(actor, zeta, env)
    Q    = get_Q(critic, zeta, zeta, env)
    rho  = np.zeros(S)
    nt   = np.arange(env.s_low, env.s_high + 1)
    rho[nt] = 1.0 / len(nt)
    return float((rho[:, None] * pi * Q).sum())


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(kappa, placement, seed=None, total_iters=None):
    torch.set_num_threads(1)
    CFG['kappa']     = kappa
    CFG['zeta_init'] = get_init(kappa, placement)
    if seed is not None:
        CFG['seed'] = seed
    if total_iters is not None:
        CFG['total_iters'] = total_iters
    tag     = f"k{int(round(kappa*100)):03d}_P{placement}"
    out_dir = setup_run(tag)
    t_start = time.time()
    log.info(f"RUN {tag}  CFG={json.dumps(CFG)}")
    np.random.seed(CFG['seed'])
    torch.manual_seed(CFG['seed'])

    # ── Build environment ─────────────────────────────────────────────────
    st_d, ac_d, dest, zeta_init, fsl, vac = build_env_info(seed=1)
    env = precompute_env(
        st_d, ac_d, dest, fsl, vac,
        eta_tr=CFG['eta_tr'], kappa=CFG['kappa']
    )

    if CFG['zeta_init'] is not None:
        zeta_init = np.array(CFG['zeta_init'], dtype=float).reshape(zeta_init.shape)
    np.random.seed(CFG['seed'])   # build_env_info re-seeds numpy with 1; restore the run seed
    S, A   = env.st_count, env.ac_count
    nF, d  = zeta_init.shape
    gamma  = CFG['gamma']
    beta   = CFG['beta_max']

    # Feature dimensions
    actor_fd  = d + S + d + A + nF * d
    critic_fd = d + S + d + A + nF * d + nF * d

    log.info(f"S={S}  A={A}  nF={nF}  dim={d}  "
             f"actor_fd={actor_fd}  critic_fd={critic_fd}")
    log.info(f"User nodes: [{env.s_low}, {env.s_high}]  "
             f"Delta: {env.idx_delta_state}")

    # ── Networks ──────────────────────────────────────────────────────────
    actor      = MLP(actor_fd,  CFG['hidden_dim'])
    critic     = MLP(critic_fd, CFG['hidden_dim'])
    critic_tgt = copy.deepcopy(critic)
    for p in critic_tgt.parameters():
        p.requires_grad_(False)

    opt_a = optim.Adam(actor.parameters(),  lr=CFG['lr_actor'])
    opt_c = optim.Adam(critic.parameters(), lr=CFG['lr_critic'])

    # ── Zeta — Adam optimizer state ───────────────────────────────────────
    zeta  = zeta_init.copy()
    z_m   = np.zeros_like(zeta)
    z_v   = np.zeros_like(zeta)
    b1, b2, ep_a = 0.9, 0.999, 1e-8
    adam_t = 0   # Adam step counter for zeta
    n_kicks = 0  # number of overlap kicks so far

    # ── Training history ──────────────────────────────────────────────────
    history = dict(J=[], loss_c=[], gz_norm=[], zeta=[], gz=[], gn_a=[], mean_len=[], beta=[], iteration=[], k=[], time=[])

    log.info(f"{'iter':>5} | {'k':>4} | {'phase':>7} | {'J_est':>8} | "
             f"{'loss_c':>8} | {'gn_a':>8} | {'gz':>8} | zeta")
    log.info("-" * 120)

    # ── Main loop ─────────────────────────────────────────────────────────
    for iteration in range(CFG['total_iters']):
        for k in range(CFG['outer_iters']):
            t0     = time.time()
            warmup = (iteration == 0) and (k < CFG['warmup_iters'])

            # ── Data collection ───────────────────────────────────────────
            trajectories = collect_episodes(
                actor, zeta, env,
                M=CFG['M_episodes'],
                eps=CFG['eps_greedy'],
                zeta_noise_std=CFG['zeta_noise_std']
            )

            mean_len = float(np.mean([len(t) for t in trajectories]))     # transitions per episode (slowdown / properness diagnostic)

            # ── Current policy and J estimate ─────────────────────────────
            pi    = get_pi(actor, zeta, env)
            J_est = evaluate_policy(actor, critic, zeta, env)

            # ── Critic update ─────────────────────────────────────────────
            loss_c = critic_update(
                critic, critic_tgt, opt_c,
                trajectories, zeta, pi, env,
                gamma, beta,
                CFG['grad_clip'],
                CFG['critic_iters'],
                CFG['batch_size']
            )

            # ── Polyak update target network ──────────────────────────────
            with torch.no_grad():
                for p, pt in zip(critic.parameters(),
                                 critic_tgt.parameters()):
                    pt.data.mul_(CFG['nu']).add_(
                        (1.0 - CFG['nu']) * p.data)

            # ── Snapshot of pi_{theta_k} (before the actor update) for the zeta step ──
            actor_k = copy.deepcopy(actor) if CFG['zeta_at_theta_k'] else None

            # ── Actor update ──────────────────────────────────────────────
            gn_a = actor_update(
                actor, opt_a, critic,
                trajectories, zeta, pi, env,
                gamma, beta,
                CFG['grad_clip'],
                CFG['actor_steps'],
                iteration,
                CFG['batch_size']
            )
            # After actor update in main loop

            # Log facility actions every 10 iterations
            if k % 10 == 0:
                log_facility_actions(actor, zeta, env)
            # Check for NaN in actor weights
            actor_has_nan = any(
                not torch.isfinite(p).all()
                for p in actor.parameters()
            )
            if actor_has_nan:
                log.warning("NaN detected in actor weights — reinitialising")
                actor = MLP(actor_fd, CFG['hidden_dim'])
                opt_a = optim.Adam(actor.parameters(), lr=CFG['lr_actor'])

            # ── Zeta update (deferred during warmup) ──────────────────────
            if not warmup:
                actor_z = actor_k if CFG['zeta_at_theta_k'] else actor    # pi_{theta_k} (default) or the updated actor
                pi_new  = get_pi(actor_z, zeta, env)
                if CFG['resample_at_zeta_k']:
                    # fresh episodes at zeta_k itself (no zeta noise), generated by the just-updated actor (weights fixed from here on,
                    # the same weights that define pi_new below), with random-action prob. resample_eps; used only for w(s)
                    '''grad_trajs = collect_episodes(
                        actor, zeta, env,
                        M=max(1, CFG['batch_size'] // 10),
                        eps=CFG['resample_eps'],
                        zeta_noise_std=0.0)'''
                    grad_trajs = collect_episodes(
                                    actor_z, zeta, env,
                                    M=max(1, CFG['batch_size'] // 10),
                                    eps=CFG['resample_eps'] if iteration < 9 else 0.0,
                                    zeta_noise_std=0.0)
                else:
                    grad_trajs = trajectories
                gz      = zeta_gradient_fd(
                    actor_z, critic, grad_trajs, zeta,
                    pi_new, env, gamma, beta,
                    CFG['batch_size']
                )
                gz_norm = float(np.linalg.norm(gz))

                # Adam update for zeta
                adam_t += 1
                z_m     = b1 * z_m + (1 - b1) * gz
                z_v     = b2 * z_v + (1 - b2) * gz ** 2
                mh      = z_m / (1 - b1 ** adam_t)
                vh      = z_v / (1 - b2 ** adam_t)
                zeta   -= CFG['lr_zeta'] * mh / (np.sqrt(vh) + ep_a)

                # ── Overlap kick: push apart the first pair of UAVs that are closer than kick_sep ──
                if CFG['kick']:
                    pair = next(((i_, j_) for i_ in range(nF) for j_ in range(i_ + 1, nF)
                                 if np.linalg.norm(zeta[i_] - zeta[j_]) < CFG['kick_sep']), None)
                    if pair is not None:
                        i_, j_ = pair
                        ang  = np.random.uniform(0, 2 * np.pi)
                        u_   = np.array([np.cos(ang), np.sin(ang)])
                        lo_, hi_ = CFG['kick_box']
                        zeta[i_] = np.clip(zeta[i_] + CFG['kick_move'] * u_, lo_, hi_)
                        zeta[j_] = np.clip(zeta[j_] - CFG['kick_move'] * u_, lo_, hi_)
                        for r_ in (i_, j_):                       # clear the Adam moments of the two moved UAVs
                            z_m[r_] = 0.0
                            z_v[r_] = 0.0
                        n_kicks += 1
                        log.info(f"  [kick #{n_kicks}] UAVs f{i_+1},f{j_+1} overlapped -> moved apart by "
                                 f"{2 * CFG['kick_move']:.2f}")
            else:
                gz_norm = 0.0

            # ── Logging ───────────────────────────────────────────────────
            phase = "warmup" if warmup else "joint"
            zs    = np.array2string(
                zeta.flatten(), precision=3, separator=',')
            log.info(
                f"{iteration:5d} | {k:4d} | {phase:>7} | "
                f"{J_est:8.4f} | {loss_c:8.4f} | "
                f"{gn_a:8.3e} | {gz_norm:8.3e} | "
                f"{zs}  [{time.time()-t0:.1f}s]  len={mean_len:.2f}"
            )

            # ── History ───────────────────────────────────────────────────
            history['J'].append(J_est)
            history['loss_c'].append(loss_c)
            history['gz_norm'].append(gz_norm)
            history['zeta'].append(zeta.copy())
            history['gz'].append(gz.flatten().copy() if not warmup else np.zeros(zeta.size))
            history['gn_a'].append(float(gn_a))
            history['mean_len'].append(mean_len)
            history['beta'].append(beta)
            history['iteration'].append(iteration)
            history['k'].append(k)
            history['time'].append(time.time() - t0)

        # ── Temperature annealing ─────────────────────────────────────────
        beta = max(CFG['beta_min'], beta * CFG['beta_anneal'])
        log.info(f"  [anneal] beta -> {beta:.4f}")
        #if (iteration + 1) % 10 == 0 or iteration == CFG['total_iters'] - 1:      # checkpoints (actor, critic, zeta, beta) for the offline gradient-quality validation
        if True:
            torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), critic_tgt=critic_tgt.state_dict(),
                            zeta=zeta, beta=beta, iteration=iteration, kappa=CFG['kappa'],
                            cfg={k_: v_ for k_, v_ in CFG.items() if not k_.startswith('_')}),
                       os.path.join(out_dir, f"ckpt_{tag}_p{iteration+1:03d}.pt"))

    # ── Final output ──────────────────────────────────────────────────────
    log.info("=" * 120)
    log.info(f"Final zeta:\n{zeta}")
    log.info(f"Final J_est: {history['J'][-1]:.4f}")

    # training history (every row of the log) + initial / final placement
    np.savez(os.path.join(out_dir, f"hist_{tag}.npz"),
             zeta=np.array(history['zeta']), J=np.array(history['J']), loss_c=np.array(history['loss_c']),
             gz_norm=np.array(history['gz_norm']), gz=np.array(history['gz']), gn_a=np.array(history['gn_a']),
             mean_len=np.array(history['mean_len']), beta=np.array(history['beta']),
             iteration=np.array(history['iteration']), k=np.array(history['k']), time=np.array(history['time']),
             zeta_init=np.array(CFG['zeta_init']), zeta_final=zeta)

    # final policy and critic over all (s, a): for placement/trajectory figures and for the offline estimator-vs-exact comparison
    pi_final = get_pi(actor, zeta, env)
    Q_final  = get_Q(critic, zeta, zeta, env)
    np.savez(os.path.join(out_dir, f"final_policy_{tag}.npz"), pi=pi_final, Q=Q_final, zeta=zeta, beta=beta,
             valid_ac=env.valid_ac, state_coords=get_state_coords(zeta, env))

    # sample trajectories under the final policy (no random actions, no zeta noise): for the trajectory plot
    rollouts = collect_episodes(actor, zeta, env, M=200, eps=0.0, zeta_noise_std=0.0)
    with open(os.path.join(out_dir, f"rollouts_{tag}.json"), "w") as fh:
        json.dump([[[int(t[0]), int(t[1]), int(t[2]), float(t[3])] for t in traj] for traj in rollouts], fh)

    # exact quantities (need exact_eval.py next to this script; skipped otherwise)
    summary = dict(tag=tag, kappa=kappa, placement=placement, seed=CFG['seed'],
                   zeta_init=list(map(float, CFG['zeta_init'])), zeta_final=list(map(float, zeta.flatten())),
                   J_est_final=float(history['J'][-1]), mean_len_final=float(history['mean_len'][-1]),
                   beta_final=float(beta), runtime_min=(time.time() - t_start) / 60.0,
                   cfg={k_: v_ for k_, v_ in CFG.items() if not k_.startswith('_')})
    try:
        import exact_eval as ee
        summary['J_exact_hard_init']  = float(ee.hard_J(np.array(CFG['zeta_init'], dtype=float).reshape(zeta.shape), env))   # optimal hard cost at the initial placement
        summary['J_exact_hard_final'] = float(ee.hard_J(zeta, env))                                                         # optimal hard cost at the learned placement (policy re-optimised)
        P, C = ee.build_kernel_fast(zeta, env)
        S_, d_ = env.st_count, env.idx_delta_state
        Pp = np.einsum('sa,sap->sp', pi_final, P); Cp = (pi_final * C).sum(1)
        M_ = np.eye(S_) - Pp; b_ = Cp.copy(); M_[d_, :] = 0; M_[d_, d_] = 1; b_[d_] = 0
        V_ = np.linalg.solve(M_, b_)
        nt_ = np.arange(env.s_low, env.s_high + 1)
        summary['J_exact_policy_final'] = float(V_[nt_].mean())            # exact expected cost of the learned (stochastic) policy at the learned placement
        log.info(f"exact: J*(init)={summary['J_exact_hard_init']:.4f}  J*(final zeta)={summary['J_exact_hard_final']:.4f}  "
                 f"J(learned policy, final zeta)={summary['J_exact_policy_final']:.4f}")
    except Exception as e:
        log.info(f"exact evaluation skipped: {e}")
    with open(os.path.join(out_dir, f"summary_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    plot_training_curves(history)
    plt.savefig(os.path.join(out_dir, f"curves_{tag}.png"), dpi=120); plt.close('all')
    try:
        plot_env(env, zeta, title=f"Final UAV Placement ({tag})")
        plt.savefig(os.path.join(out_dir, f"placement_{tag}.png"), dpi=120); plt.close('all')
    except Exception as e:
        log.info(f"plot_env skipped: {e}")
    open(os.path.join(out_dir, "DONE"), "w").write("ok")

    return actor, critic, zeta, env, history


# ---------------------------------------------------------------------------
# Training curve plots
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict) -> None:
    """Plot J, critic loss, and zeta gradient norm over training."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['J'])
    axes[0].set_title("Objective $J(\\pi, \\zeta)$")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("$J$ estimate")
    axes[0].grid(True)

    axes[1].plot(history['loss_c'])
    axes[1].set_title("Critic TD Loss")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("MSE Loss")
    axes[1].grid(True)

    axes[2].plot(history['gz_norm'])
    axes[2].set_title("$\\|\\nabla_\\zeta J\\|$")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Gradient norm")
    axes[2].grid(True)

    plt.tight_layout()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--placement", type=int, required=True, choices=range(1, 7))
    ap.add_argument("--seed", type=int, default=None, help="default: CFG['seed'] (= 0)")
    ap.add_argument("--total_iters", type=int, default=None, help="override (for pilots only)")
    a = ap.parse_args()
    main(a.kappa, a.placement, a.seed, a.total_iters)
