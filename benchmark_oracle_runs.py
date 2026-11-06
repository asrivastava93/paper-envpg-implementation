"""
benchmark_oracle_runs.py -- Model-BASED oracle for the ground-truth zeta*, with the constant hop cost,
runnable for any subset of  kappa x placement  (same initial placements as EnvPG-AC / the score-function baseline).

Algorithm = benchmark_surrogateGPU1.py (first submission), unchanged:
    soft value iteration on the analytic kernel at inverse temperature beta (1 -> 100, x1.1 per level),
    central-FD gradient of J*(zeta), projected gradient descent with Armijo backtracking (25 steps per level),
    uniform noise (+-0.2) added to zeta after every level.
Soft Bellman (cost form):
    Q*(s,a) = C[s,a] + sum_s' P[s,a,s'] V*(s'),   V*(s) = -(1/beta) log sum_a exp(-beta Q*(s,a)),   V*(delta) = 0,  gamma = 1
    C[s,a]  = E_{s'}[ ||x_s - x_s'||^2 (+1 if <= 0.1 and kappa <= 0.75) ]  +  hop_cost        (hop_cost: every hop from a non-terminal state)

What is new with respect to the first-submission script
  1. hop cost: env.hop_cost (env2_paraRLModel.HOP_COST, or --hop_cost) is added to C[s,a] of every non-terminal state.
  2. command line: --kappas, --placements; the initial placement is the one of EnvPG-AC run k<kappa>_P<placement>.
  3. one folder per run  <out>/k085_P1/  (nothing is overwritten; finished runs hold a DONE file and are skipped on restart).
  4. per run it writes: log, per-step csv + npz, summary_*.json (everything needed for the comparison), placement png.
  5. the single hard-coded Tee / output names of the old script are gone (the old script overwrote run1_oracle_kappa100.txt).
  Not changed: the algorithm, its constants, the noise step after every level (so the logged final zeta of the first-submission
  script, which includes that last noise, is reproduced as `zeta_final_logged`; the zeta right after the last gradient step,
  before that noise, is `zeta_star_gd`).

Usage
    python benchmark_oracle_runs.py                                   # all 4 kappas x 6 placements, one after the other
    python benchmark_oracle_runs.py --kappas 0.85 1.0 --placements 1 3
    python benchmark_oracle_runs.py --kappas 0.9 --placements 1 --out my_folder
    python benchmark_oracle_runs.py --kappas 0.9 --placements 1 --beta_max 1.3 --gd_steps 2     # quick pilot
Several processes can be started in parallel on different --kappas/--placements (same --out): every run has its own folder.
"""

from __future__ import annotations
import os
import sys
import csv
import json
import time
import argparse
import logging
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env2_paraRLModel import (
    build_env_info, precompute_env, EnvStructural,
    get_state_coords, plot_env, HOP_COST
)
try:
    import exact_eval as ee          # exact hard-min J* (policy iteration); also reads env.hop_cost
except Exception:
    ee = None

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS']       = '1'

warnings.filterwarnings('ignore', message='invalid value encountered in divide', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='divide by zero encountered in log',   category=RuntimeWarning)

log = logging.getLogger('oracle')
log.setLevel(logging.INFO)
_fmt = logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S')
_sh = logging.StreamHandler(sys.stdout); _sh.setFormatter(_fmt); log.addHandler(_sh)

# ---------------------------------------------------------------------------
# Initial placements (identical to the EnvPG-AC / SF scripts: R1-R5 per kappa, 6 = mean of R1-R5)
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
# Configuration -- matched to the first-submission oracle
# ---------------------------------------------------------------------------

CFG = dict(
    seed         = 0,
    gamma        = 1,        # SSP
    eta_tr       = 100.0,
    kappa        = 1,

    # Entropy annealing: beta here is an INVERSE temperature (1 -> 100)
    beta_max     = 100,
    beta_min     = 1,
    beta_anneal  = 1.1,
)

OCFG = dict(
    svi_eps            = 1e-2,
    svi_max_iter       = 1000,
    fd_h               = 1e-2,   # central-difference step
    gd_lr              = 5e-2,   # initial / max step size (line search shrinks)
    gd_steps_per_level = 25,
    bt_max             = 25,
    bt_armijo          = 1e-4,
    zeta_lo            = 0.0,
    zeta_hi            = 12.0,
    zeta_noise_std     = 0.2,
    bt_min_lr          = 1e-6,
)

# ---------------------------------------------------------------------------
# Analytic transition kernel (+ constant hop cost)
# ---------------------------------------------------------------------------

def build_transition_kernel(zeta: np.ndarray, env: EnvStructural) -> tuple:
    r"""
    P[s,a,s'] and C[s,a] = E[c(s,a,s')] from the env_step kernel:
        p(s'|s,a) = kappa            if s' = intended(a)
                  = (1-kappa)*Gibbs  over intended states of valid(s)\{a}
        c(s,a,s') = ||x_s - x_s'||^2 (+1 if <= 0.1 and kappa <= 0.75)  + hop_cost
    The hop cost is paid on every hop from a non-terminal state (delta has no valid actions, so it is skipped).
    """
    S, A   = env.st_count, env.ac_count
    coords = get_state_coords(zeta, env)

    P = np.zeros((S, A, S))
    C = np.zeros((S, A))

    for s in range(S):
        for a in range(A):
            if not env.valid_ac[s, a]:
                continue

            if a == env.idx_delta_action:
                s_int = env.idx_delta_state
            else:
                s_int = int(env.fac_state_idx[env.fac_id_action[a]])

            valid_actions = np.where(env.valid_ac[s])[0]
            alt_actions   = valid_actions[valid_actions != a]

            if len(alt_actions) == 0:
                alt_probs  = np.array([])
                alt_states = np.array([], dtype=int)
            else:
                alt_states = np.array([
                    env.idx_delta_state
                    if a_ == env.idx_delta_action
                    else int(env.fac_state_idx[env.fac_id_action[a_]])
                    for a_ in alt_actions
                ], dtype=int)
                x_s    = coords[s]
                x_alts = coords[alt_states]
                logits = -env.eta_tr * np.sum((x_alts - x_s) ** 2, axis=1)
                logits -= logits.max()
                p      = np.exp(logits)
                p_sum  = p.sum()
                if p_sum < 1e-300 or not np.isfinite(p_sum):
                    p = np.ones(len(alt_states)) / len(alt_states)
                else:
                    p /= p_sum
                alt_probs = p

            P[s, a, s_int] += env.kappa
            for i, s_alt in enumerate(alt_states):
                P[s, a, s_alt] += (1.0 - env.kappa) * alt_probs[i]

            for sp in range(S):
                if P[s, a, sp] > 0:
                    raw_cost = float(np.sum((coords[sp] - coords[s]) ** 2))
                    if raw_cost <= 0.1 and env.kappa <= 0.75:
                        raw_cost += 1.0
                    C[s, a] += P[s, a, sp] * raw_cost
            C[s, a] += env.hop_cost            # constant cost of the hop (P[s,a,.] sums to 1)

    return P, C

# ---------------------------------------------------------------------------
# Soft value iteration / hard cost under the soft-optimal policy / exact J*
# ---------------------------------------------------------------------------

def soft_value_iteration(P, C, env, beta, eps, max_iter):
    """Soft Bellman optimality (cost form), iterated to max-norm change < eps. Returns (V, Q, pi)."""
    S, A    = env.st_count, env.ac_count
    delta_s = env.idx_delta_state
    valid   = env.valid_ac
    gamma   = CFG['gamma']
    V = np.zeros(S)
    Q = np.zeros((S, A))
    DIVERGENCE_CAP = 1e8

    for it in range(max_iter):
        Q = C + P @ V
        z = -(beta / gamma) * Q
        z = np.where(valid, z, -np.inf)
        z_max = np.max(np.where(valid, z, -np.inf), axis=1, keepdims=True)
        z_max = np.where(np.isfinite(z_max), z_max, 0.0)
        lse = z_max.squeeze(1) + np.log(
            np.sum(np.where(valid, np.exp(z - z_max), 0.0), axis=1))
        V_new = -(gamma / beta) * lse
        V_new[delta_s] = 0.0

        if not np.all(np.isfinite(V_new)):
            raise RuntimeError(f'soft_value_iteration: non-finite V at iter {it} (beta={beta}). '
                               f'Optimal soft policy may be improper.')
        diff = np.max(np.abs(V_new - V))
        V = V_new
        if np.max(np.abs(V)) > DIVERGENCE_CAP:
            raise RuntimeError(f'soft_value_iteration: V diverging at iter {it} (beta={beta}). '
                               f'Likely improper optimal policy.')
        if diff < eps:
            break

    z = -(beta / gamma) * Q
    z = np.where(valid, z, -np.inf)
    z_max = np.max(np.where(valid, z, -np.inf), axis=1, keepdims=True)
    z_max = np.where(np.isfinite(z_max), z_max, 0.0)
    pi = np.where(valid, np.exp(z - z_max), 0.0)
    row = pi.sum(axis=1, keepdims=True)
    denom   = valid.sum(axis=1, keepdims=True).astype(float)
    uniform = np.where(denom > 0, valid / np.maximum(denom, 1), 0.0)
    pi = np.where(row > 1e-300, pi / row, uniform)
    pi[delta_s, :] = 0.0
    return V, Q, pi


def _rho(env):
    rho = np.zeros(env.st_count)
    nt  = np.arange(env.s_low, env.s_high + 1)
    rho[nt] = 1.0 / len(nt)
    return rho


def compute_J_hard(zeta, env, beta, eps, max_iter):
    """Hard routing cost (no entropy terms) under the soft-optimal policy pi* at (zeta, beta)."""
    P, C     = build_transition_kernel(zeta, env)
    _, _, pi = soft_value_iteration(P, C, env, beta, eps, max_iter)

    S       = env.st_count
    delta_s = env.idx_delta_state
    DIVERGENCE_CAP = 1e8
    C_pi = (pi * C).sum(axis=1)
    P_pi = (pi[:, :, None] * P).sum(axis=1)
    V = np.zeros(S)
    for it in range(max_iter):
        V_new = C_pi + P_pi @ V
        V_new[delta_s] = 0.0
        if not np.all(np.isfinite(V_new)):
            raise RuntimeError(f'compute_J_hard: non-finite V at iter {it}.')
        diff = np.max(np.abs(V_new - V))
        V    = V_new
        if np.max(np.abs(V)) > DIVERGENCE_CAP:
            raise RuntimeError(f'compute_J_hard: V diverging at iter {it}. pi* may be improper.')
        if diff < eps:
            break
    return float(_rho(env) @ V)


def compute_J_star(zeta, env, beta, eps, max_iter):
    """Exact entropy-regularised optimum at fixed (zeta, beta); rho0 uniform over the user nodes."""
    P, C = build_transition_kernel(zeta, env)
    V, _, _ = soft_value_iteration(P, C, env, beta, eps, max_iter)
    return float(_rho(env) @ V)


def fd_grad_J_star(zeta, env, beta, h, eps, max_iter):
    """Central finite-difference gradient dJ*/dzeta (beta held fixed)."""
    flat = zeta.flatten().copy()
    dim  = flat.size
    grad = np.zeros(dim)
    for i in range(dim):
        zp = flat.copy(); zp[i] += h
        zm = flat.copy(); zm[i] -= h
        Jp = compute_J_star(zp.reshape(zeta.shape), env, beta, eps, max_iter)
        Jm = compute_J_star(zm.reshape(zeta.shape), env, beta, eps, max_iter)
        grad[i] = (Jp - Jm) / (2.0 * h)
    return grad.reshape(zeta.shape)


def log_facility_routing(zeta, env, pi, log, label=''):
    """Per UAV: probability of every next hop under policy pi."""
    for j in range(env.n_fac):
        s_fac = int(env.fac_state_idx[j])
        valid_actions = np.where(env.valid_ac[s_fac])[0]
        parts = []
        for a in valid_actions:
            if a == env.idx_delta_action:
                target = 'delta'
            else:
                target = f'f_{int(env.fac_id_action[a]) + 1}'
            parts.append(f'{target}:{float(pi[s_fac, a]):.3f}')
        log.info(f'  [routing{" " + label if label else ""}] f_{j+1} -> {{ {", ".join(parts)} }}')

# ---------------------------------------------------------------------------
# Everything needed for the comparison at one (zeta, beta)
# ---------------------------------------------------------------------------

def evaluate_point(zeta, env, beta):
    """
    Metrics at (zeta, beta):
      J_soft            exact soft optimum J*(zeta, beta)
      J_hard            hard cost (distance^2 + hop cost) of the soft-optimal policy pi*(beta)
      mean_hops         expected number of hops from a uniformly drawn user node under pi*(beta)
      J_hard_dist       J_hard - hop_cost * mean_hops  (distance part only)
      mean_entropy, mean_max_prob, greedy routing and its properness
      J_exact_hard      exact hard-min J*(zeta) (policy iteration, exact_eval.hard_J) -- the number EnvPG-AC / SF are judged by
    """
    out = dict(J_soft=np.nan, J_hard=np.nan, mean_hops=np.nan, J_hard_dist=np.nan, mean_entropy=np.nan,
               mean_max_prob=np.nan, greedy={}, proper=None, J_exact_hard=np.nan)
    try:
        P, C = build_transition_kernel(zeta, env)
        V, _, pi = soft_value_iteration(P, C, env, beta, OCFG['svi_eps'], OCFG['svi_max_iter'])
        rho = _rho(env)
        out['J_soft'] = float(rho @ V)
        out['J_hard'] = compute_J_hard(zeta, env, beta, OCFG['svi_eps'], OCFG['svi_max_iter'])
        S, d = env.st_count, env.idx_delta_state
        Pp = (pi[:, :, None] * P).sum(axis=1)
        M = np.eye(S) - Pp; b = np.ones(S); M[d, :] = 0; M[d, d] = 1; b[d] = 0
        hops = np.linalg.solve(M, b)
        out['mean_hops']   = float(rho @ hops)
        out['J_hard_dist'] = out['J_hard'] - env.hop_cost * out['mean_hops']
        nt = np.arange(env.s_low, env.s_high + 1)
        lp = np.where(pi > 0, np.log(pi), 0.0)
        out['mean_entropy']  = -float((pi[nt] * lp[nt]).sum(axis=1).mean())
        out['mean_max_prob'] = float(pi[nt].max(axis=1).mean())
        nxt = {}
        for j in range(env.n_fac):
            s_fac = int(env.fac_state_idx[j])
            va = np.where(env.valid_ac[s_fac])[0]
            ag = int(va[np.argmax(pi[s_fac, va])])
            tgt = 'delta' if ag == env.idx_delta_action else f'f{int(env.fac_id_action[ag]) + 1}'
            out['greedy'][f'f{j+1}'] = dict(target=tgt, prob=float(pi[s_fac, ag]),
                                            probs={('delta' if a == env.idx_delta_action else f'f{int(env.fac_id_action[a]) + 1}'):
                                                   float(pi[s_fac, a]) for a in va})
            nxt[j] = 'delta' if ag == env.idx_delta_action else int(env.fac_id_action[ag])
        proper = True
        for j in range(env.n_fac):
            seen, c = set(), j
            while nxt[c] != 'delta':
                if c in seen:
                    proper = False; break
                seen.add(c); c = nxt[c]
            if not proper: break
        out['proper'] = bool(proper)
    except Exception as e:
        out['error'] = str(e)
    if ee is not None:
        try:
            out['J_exact_hard'] = float(ee.hard_J(zeta, env))
        except Exception as e:
            out['error_exact'] = str(e)
    return out

# ---------------------------------------------------------------------------
# One run: projected gradient descent on the exact J*
# ---------------------------------------------------------------------------

def run_one(kappa, placement, args, root):
    tag     = f"k{int(round(kappa * 100)):03d}_P{placement}"
    out_dir = os.path.join(root, tag)
    if os.path.exists(os.path.join(out_dir, 'DONE')):
        print(f'skip (done): {tag}', flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(out_dir, f'log_oracle_{tag}.txt'), mode='w', encoding='utf-8')
    fh.setFormatter(_fmt); log.addHandler(fh)
    csv_file = None
    try:
        t_start = time.time()
        CFG['kappa'] = kappa
        np.random.seed(CFG['seed'])

        st_d, ac_d, dest, zeta_dummy, fsl, vac = build_env_info(seed=1)      # re-seeds numpy with 1 (as in the first-submission script)
        env = precompute_env(st_d, ac_d, dest, fsl, vac, eta_tr=CFG['eta_tr'], kappa=kappa,
                             hop_cost=(args.hop_cost if args.hop_cost is not None else HOP_COST))
        zeta_init = np.array(get_init(kappa, placement), dtype=float).reshape(zeta_dummy.shape)

        S, A  = env.st_count, env.ac_count
        nF, d = zeta_init.shape
        log.info(f'RUN {tag}  kappa={kappa}  placement={placement}  hop_cost={env.hop_cost}')
        log.info(f'S={S}  A={A}  nF={nF}  dim={d}')
        log.info(f'MODEL-BASED ORACLE: FD-gradient descent on exact J*   (beta = inverse temperature)')
        log.info(f'soft-VI eps={OCFG["svi_eps"]}  fd_h={OCFG["fd_h"]}  gd_lr={OCFG["gd_lr"]}  gd_steps_per_level={OCFG["gd_steps_per_level"]}')
        log.info(f'beta: {CFG["beta_min"]} -> {CFG["beta_max"]}  anneal={CFG["beta_anneal"]}')
        log.info(f'rho0 = uniform over non-terminal states [{env.s_low}, {env.s_high}]')
        log.info(f'zeta_init = {np.array2string(zeta_init.flatten(), precision=3, separator=",")}')

        csv_file   = open(os.path.join(out_dir, f'oracle_Jlog_{tag}.csv'), 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['level', 'step', 'J_star', 'J_hard', 'grad_norm', 'beta', 'zeta'])
        hist = dict(level=[], step=[], J_star=[], J_hard=[], grad_norm=[], beta=[], zeta=[])
        n_skipped, n_ls_fail = 0, 0

        log.info(f"{'lvl':>4} | {'step':>5} | {'J_star':>9} | {'J_hard':>9} | {'grad_norm':>10} | {'beta':>6} | zeta")
        log.info('-' * 120)

        zeta  = zeta_init.copy()
        beta  = CFG['beta_min']
        level = 0
        H     = 0.0
        beta_last, zeta_pre_noise = beta, zeta.copy()
        last_gnorm = np.nan

        while beta <= CFG['beta_max'] + 1e-12:
            t0 = time.time()
            beta_last = beta
            J_lvl = compute_J_star(zeta, env, beta, OCFG['svi_eps'], OCFG['svi_max_iter'])
            log.info(f'  [level {level}] beta={beta:.4f}  J*={J_lvl:.4f} at start of level')

            for step in range(OCFG['gd_steps_per_level']):
                try:
                    grad = fd_grad_J_star(zeta, env, beta, OCFG['fd_h'], OCFG['svi_eps'], OCFG['svi_max_iter'])
                except RuntimeError as e:
                    n_skipped += 1
                    log.info(f'  [level {level}, step {step}] FD hit improper region -- skipping step. ({e})')
                    continue

                gnorm = float(np.linalg.norm(grad)); last_gnorm = gnorm
                J_cur = compute_J_star(zeta, env, beta, OCFG['svi_eps'], OCFG['svi_max_iter'])
                zs = np.array2string(zeta.flatten(), precision=3, separator=',')
                J_hard = compute_J_hard(zeta, env, beta, OCFG['svi_eps'], OCFG['svi_max_iter'])
                log.info(f'{level:4d} | {step:5d} | J*={J_cur:8.4f}  J_hard={J_hard:8.4f} | '
                         f'grad_norm={gnorm:10.6f} | beta={beta:6.4f} | {zs}')
                csv_writer.writerow([level, step, f'{J_cur:.4f}', f'{J_hard:.4f}', f'{gnorm:.6f}', f'{beta:.4f}', zs])
                csv_file.flush()
                for k_, v_ in (('level', level), ('step', step), ('J_star', J_cur), ('J_hard', J_hard),
                               ('grad_norm', gnorm), ('beta', beta)):
                    hist[k_].append(v_)
                hist['zeta'].append(zeta.flatten().copy())

                # Backtracking line search
                step_lr  = OCFG['gd_lr']
                accepted = False
                for _bt in range(OCFG['bt_max']):
                    if step_lr < OCFG['bt_min_lr']:
                        zeta     = np.clip(zeta - step_lr * grad, OCFG['zeta_lo'], OCFG['zeta_hi'])
                        accepted = True
                        break
                    zeta_try = np.clip(zeta - step_lr * grad, OCFG['zeta_lo'], OCFG['zeta_hi'])
                    try:
                        J_try = compute_J_star(zeta_try, env, beta, OCFG['svi_eps'], OCFG['svi_max_iter'])
                    except RuntimeError:
                        step_lr *= 0.5
                        continue
                    if J_try <= J_cur - OCFG['bt_armijo'] * step_lr * gnorm ** 2:
                        zeta     = zeta_try
                        accepted = True
                        break
                    step_lr *= 0.5

                if not accepted:
                    n_ls_fail += 1
                    log.info(f'  [level {level}, step {step}] line search no decrease (||grad||={gnorm:.2e})')

                # Policy entropy diagnostic
                _, _, pi_cur = soft_value_iteration(*build_transition_kernel(zeta, env), env, beta,
                                                    OCFG['svi_eps'], OCFG['svi_max_iter'])
                nt = np.arange(env.s_low, env.s_high + 1)
                lp = np.where(pi_cur > 0, np.log(pi_cur), 0.0)
                H  = -float((pi_cur[nt] * lp[nt]).sum(axis=1).mean())
                max_prob = float(pi_cur[nt].max(axis=1).mean())
                log.info(f'  [diag] mean_entropy={H:.4f}  mean_max_prob={max_prob:.4f}')
                log_facility_routing(zeta, env, pi_cur, log, label=f'lvl{level}_step{step}')

            zeta_pre_noise = zeta.copy()                 # zeta right after the last gradient step of this level
            if H >= 0:                                   # (always true; as in the first-submission script)
                zeta = zeta + np.random.uniform(-1, 1, size=zeta.shape) * OCFG['zeta_noise_std']

            log.info(f'  [level {level}] done in {time.time()-t0:.1f}s')
            beta = beta * CFG['beta_anneal']
            level += 1

        # ---- final evaluation ------------------------------------------------------------------
        zeta_star_gd      = zeta_pre_noise                # oracle zeta* = result of the last gradient step
        zeta_final_logged = zeta                          # what the first-submission script printed (includes the last noise)
        J_final_betamin   = compute_J_star(zeta_final_logged, env, CFG['beta_min'], OCFG['svi_eps'], OCFG['svi_max_iter'])
        log.info('=' * 120)
        log.info(f'Final zeta* (oracle, after last gradient step):\n{zeta_star_gd}')
        log.info(f'zeta after the last noise step (as printed by the first-submission script):\n{zeta_final_logged}')
        log.info(f'Final J* (exact, beta={CFG["beta_min"]}) at the logged final zeta, as in the first-submission script: {J_final_betamin:.4f}')

        m_init  = evaluate_point(zeta_init,         env, beta_last)
        m_star  = evaluate_point(zeta_star_gd,      env, beta_last)
        m_final = evaluate_point(zeta_final_logged, env, beta_last)
        try:
            g_final = fd_grad_J_star(zeta_star_gd, env, beta_last, OCFG['fd_h'], OCFG['svi_eps'], OCFG['svi_max_iter'])
            grad_norm_final = float(np.linalg.norm(g_final))
        except Exception:
            grad_norm_final = float('nan')
        for name, m in (('init', m_init), ('zeta* (gd)', m_star), ('final (logged)', m_final)):
            log.info(f'[{name:>14}] beta={beta_last:.3f}  J_soft={m["J_soft"]:.4f}  J_hard={m["J_hard"]:.4f}  '
                     f'mean_hops={m["mean_hops"]:.3f}  J_hard_dist={m["J_hard_dist"]:.4f}  '
                     f'J_exact_hard={m["J_exact_hard"]:.4f}  proper={m["proper"]}')
        log.info(f'||grad J*|| at zeta* (beta={beta_last:.3f}): {grad_norm_final:.6f}')

        np.savez(os.path.join(out_dir, f'oracle_hist_{tag}.npz'),
                 **{k_: np.array(v_) for k_, v_ in hist.items()},
                 zeta_init=zeta_init.flatten(), zeta_star_gd=zeta_star_gd.flatten(),
                 zeta_final_logged=zeta_final_logged.flatten())

        summary = dict(
            tag=tag, kappa=kappa, placement=placement, hop_cost=float(env.hop_cost),
            zeta_init=list(map(float, zeta_init.flatten())),
            zeta_star_gd=list(map(float, zeta_star_gd.flatten())),
            zeta_final_logged=list(map(float, zeta_final_logged.flatten())),
            beta_last=float(beta_last), n_levels=int(level),
            n_steps_logged=len(hist['step']), n_fd_skipped=int(n_skipped), n_linesearch_fail=int(n_ls_fail),
            grad_norm_final=grad_norm_final, grad_norm_last_step=float(last_gnorm),
            J_exact_hard_init=m_init['J_exact_hard'], J_exact_hard_final=m_star['J_exact_hard'],
            J_final_beta_min_logged_zeta=float(J_final_betamin),
            metrics_init=m_init, metrics_zeta_star_gd=m_star, metrics_final_logged=m_final,
            cfg=dict(CFG), ocfg=dict(OCFG), runtime_min=(time.time() - t_start) / 60.0,
            note=("J_exact_hard_* = exact hard-min J* by policy iteration (same quantity as in the EnvPG-AC / SF summaries); "
                  "J_hard = hard cost under the soft-optimal policy at beta_last; all costs include hop_cost per hop; "
                  "J_hard_dist = J_hard - hop_cost*mean_hops."))
        with open(os.path.join(out_dir, f'summary_oracle_{tag}.json'), 'w') as fh_:
            json.dump(summary, fh_, indent=2)

        try:
            plot_env(env, zeta_star_gd, title=f'Oracle UAV placement ({tag}, hop cost {env.hop_cost})')
            plt.savefig(os.path.join(out_dir, f'placement_oracle_{tag}.png'), dpi=120); plt.close('all')
        except Exception as e:
            log.info(f'plot_env skipped: {e}')

        log.info(f'Saved: log, oracle_Jlog_{tag}.csv, oracle_hist_{tag}.npz, summary_oracle_{tag}.json')
        csv_file.close(); csv_file = None
        open(os.path.join(out_dir, 'DONE'), 'w').write('ok')
    finally:
        if csv_file is not None:
            csv_file.close()
        log.removeHandler(fh); fh.close()


def _default_out(hop):
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    return os.path.join(base, f'results_oracle_h{hop:g}')


if __name__ == '__main__':
    import faulthandler
    faulthandler.enable(file=sys.__stderr__)
    ap = argparse.ArgumentParser()
    ap.add_argument('--kappas', type=float, nargs='+', default=[0.85, 0.9, 0.95, 1.0])
    ap.add_argument('--placements', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    ap.add_argument('--hop_cost', type=float, default=None, help='default: env2_paraRLModel.HOP_COST')
    ap.add_argument('--out', type=str, default=None, help='results folder (default results_oracle_h<hop cost>)')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--beta_min', type=float, default=None, help='first inverse temperature of the schedule (default 1, as in the first-submission script)')
    ap.add_argument('--beta_max', type=float, default=None, help='override (pilots only)')
    ap.add_argument('--gd_steps', type=int, default=None, help='gradient steps per beta level, override (pilots only)')
    a = ap.parse_args()
    if a.seed is not None:           CFG['seed'] = a.seed
    if a.beta_min is not None:       CFG['beta_min'] = a.beta_min
    if a.beta_max is not None:       CFG['beta_max'] = a.beta_max
    if a.gd_steps is not None:       OCFG['gd_steps_per_level'] = a.gd_steps
    hop = a.hop_cost if a.hop_cost is not None else HOP_COST
    root = a.out if a.out is not None else _default_out(hop)
    os.makedirs(root, exist_ok=True)
    jobs = [(k, p) for p in a.placements for k in a.kappas]
    print(f'{len(jobs)} runs (one after the other), hop_cost={hop}, results in {root}', flush=True)
    t0 = time.time()
    for n, (k, p) in enumerate(jobs, 1):
        print(f'[{(time.time()-t0)/60:7.1f} min] ({n}/{len(jobs)}) kappa={k} placement={p}', flush=True)
        try:
            run_one(k, p, a, root)
        except Exception as e:
            print(f'*** run kappa={k} placement={p} FAILED: {e!r}', flush=True)
    print(f'all done in {(time.time()-t0)/3600:.2f} h')
