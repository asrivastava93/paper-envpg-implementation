"""Exact hard-min J*(zeta) via policy iteration (verified against the oracle kernel and converged VI)."""
import numpy as np
from env2_paraRLModel import get_state_coords

def build_kernel_fast(zeta, env):
    S, A = env.st_count, env.ac_count
    coords = get_state_coords(zeta, env)
    D = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)      # (S,S)
    Dc = D + ((D <= 0.1) * 1.0 if env.kappa <= 0.75 else 0.0)
    s_int_of_a = np.array([env.idx_delta_state if a == env.idx_delta_action
                           else int(env.fac_state_idx[env.fac_id_action[a]])
                           for a in range(A)])
    P = np.zeros((S, A, S))
    for s in range(S):
        va = np.where(env.valid_ac[s])[0]
        for a in va:
            P[s, a, s_int_of_a[a]] += env.kappa
            alt = va[va != a]
            if len(alt):
                alt_s = s_int_of_a[alt]
                lg = -env.eta_tr * D[s, alt_s]
                lg -= lg.max()
                p = np.exp(lg); ps = p.sum()
                p = p / ps if ps > 1e-300 else np.ones(len(alt)) / len(alt)
                np.add.at(P[s, a], alt_s, (1.0 - env.kappa) * p)
    C = np.einsum('sap,sp->sa', P, Dc)
    C = C + getattr(env, 'hop_cost', 0.0) * (np.arange(S) != env.idx_delta_state)[:, None]   # constant cost per hop from non-terminal states
    return P, C

def hard_J(zeta, env, max_pi=200):
    """Exact hard-min J* via policy iteration (same quantity as the hard VI used for
    the J* table; VI from V=0 crawls when two UAVs nearly coincide, PI does not)."""
    P, C = build_kernel_fast(zeta, env)
    S, A = env.st_count, env.ac_count
    valid = env.valid_ac; d = env.idx_delta_state
    pol = np.zeros(S, dtype=int)
    pol[:env.n_fac] = env.idx_delta_action                 # UAVs -> destination (proper)
    pol[env.s_low:env.s_high + 1] = 0                      # users -> UAV f1
    idx = np.arange(S)
    for _ in range(max_pi):
        Pp = P[idx, pol]; Cp = C[idx, pol]
        M = np.eye(S) - Pp; b = Cp.copy()
        M[d, :] = 0; M[d, d] = 1; b[d] = 0
        V = np.linalg.solve(M, b)
        Q = np.where(valid, C + P @ V, np.inf)
        cur = Q[idx, pol]
        new = Q.argmin(1)
        better = Q[idx, new] < cur - 1e-12
        better[d] = False
        if not better.any(): break
        pol = np.where(better, new, pol)
    nt = np.arange(env.s_low, env.s_high + 1)
    return float(V[nt].mean())

