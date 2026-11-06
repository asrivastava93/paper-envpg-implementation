"""
env2_paraRLModel.py — MDP environment for EnvPG-AC.

State space:  {f_1,...,f_nF}  (UAV states,   indices 0..nF-1)
            ∪ {n_1,...,n_N}   (user nodes,   indices nF..nF+N-1)
            ∪ {delta}          (destination,  index  nF+N)

Action space: {f_1,...,f_nF}  (jump to UAV,  indices 0..nF-1)
            ∪ {delta}          (jump to dest, index  nF)

Valid actions:
    From n_i  -> any UAV f_j             (not delta)
    From f_j  -> any other UAV f_k, k≠j  (not self)
              ∪ delta
    From delta -> none (absorbing)

Transition kernel:
    p(s'|s, a, zeta) = kappa          if s' = intended(a)
                     = (1-kappa) * Gibbs(||x_s - x_{s'}||^2)
                                       over intended states of A(s)\{a}
                     = 0              otherwise

Cost (as implemented in env_step):
    c(s, a, s', zeta) = ||x_s - x_{s'}||^2  (+1 if that is <= 0.1 and kappa <= 0.75)
                        + HOP_COST          (constant, paid on EVERY hop taken from a non-terminal state)
    The constant HOP_COST does not depend on zeta, so the zeta-gradient expressions are unchanged;
    it makes every cycle strictly positive-cost (so improper policies have infinite cost).

Coordinates:
    x_{n_i}  = fixed_st_loc[i]   (never changes)
    x_{f_j}  = zeta[j]           (optimized)
    x_delta  = dest_loc[0]       (never changes)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt


# Constant cost paid at every hop taken from a non-terminal state (set to 0.0 to recover the original model).
# Needs HOP_COST > beta*ln(#valid actions) (~1.6*beta) at the epsilon switch-off to rule out entropy-driven zero-cost loops.
HOP_COST = 0.2


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class EnvStructural:
    """
    Precomputed zeta-independent quantities. Built once at startup.
    All arrays are numpy, all indices 0-based.

    Critical index conventions:
        State indices:  0..nF-1        -> UAV states f_1..f_nF
                        nF..nF+N-1     -> user node states n_1..n_N
                        nF+N           -> delta state
        Action indices: 0..nF-1        -> UAV actions f_1..f_nF
                        nF             -> delta action
    """
    # Dimensions
    st_count: int                 # S = nF + N + 1
    ac_count: int                 # A = nF + 1
    n_fac:    int                 # nF = number of UAVs
    n_fix:    int                 # N  = number of user nodes
    dim:      int                 # spatial dimension (2)

    # Hyperparameters
    eta_tr:   float               # Gibbs sharpness for off-target transitions
    kappa:    float               # probability of reaching intended next hop

    # Locations
    dest_loc:       np.ndarray    # (1, dim)  destination coordinate
    fixed_st_loc:   np.ndarray    # (N, dim)  user node coordinates

    # Key indices
    idx_delta_state:  int         # = nF + N
    idx_delta_action: int         # = nF

    # State classification
    is_fac_state: np.ndarray      # (S,) bool
    fac_id_state: np.ndarray      # (S,) int, 0-based UAV index
    is_fix_state: np.ndarray      # (S,) bool
    fix_id_state: np.ndarray      # (S,) int, 0-based user node index

    # Action classification
    is_fac_action: np.ndarray     # (A,) bool
    fac_id_action: np.ndarray     # (A,) int, 0-based UAV index

    # Critical lookup: UAV index -> state index
    # fac_state_idx[j] = state index of UAV f_{j+1}
    fac_state_idx: np.ndarray     # (nF,) int

    # Critical lookup: UAV index -> action index
    # fac_action_idx[j] = action index for jumping to UAV f_{j+1}
    # Since actions are ordered f_1..f_nF | delta, fac_action_idx[j] = j
    # Stored explicitly for safety
    fac_action_idx: np.ndarray    # (nF,) int

    # State coordinates with UAV rows zeroed (filled from zeta at runtime)
    state_coords_fixed: np.ndarray  # (S, dim)

    # Valid action mask
    valid_ac: np.ndarray          # (S, A) bool

    # Non-terminal state index range (user nodes only)
    s_low:  int                   # = nF
    s_high: int                   # = nF + N - 1

    # Labels for debugging
    st_details: List[tuple]       # (label, label) per state
    ac_details: List[tuple]       # (label, label) per action

    # Constant cost per hop from a non-terminal state (zeta-independent)
    hop_cost: float = 0.2


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------

def build_env_info(seed: int = 1) -> tuple:
    """
    Build state/action spaces, valid action mask, and initial zeta.

    Returns:
        st_details:   list of (label, label) for each state
        ac_details:   list of (label, label) for each action
        dest_loc:     (1, 2) destination coordinate
        zeta_init:    (nF, 2) initial UAV coordinates
        fixed_st_loc: (N, 2)  user node coordinates
        valid_ac:     (S, A)  bool valid action mask
    """
    np.random.seed(seed)

    # ── User node locations ───────────────────────────────────────────────
    centroids = np.array([
        [1.5, 2.5],
        [3.0, 1.0],
        [5.0, 1.0],
        [8.0, 1.0],
        [10.0, 2.5],
        [9.0,  5.0],
    ], dtype=float)

    fixed_nodes = []
    for c in centroids:
        n = 40 + np.random.randint(0, 10)
        for _ in range(n):
            r     = 0.05 + 0.5 * np.random.rand()
            theta = 2 * np.pi * np.random.rand()
            fixed_nodes.append([c[0] + r * np.cos(theta),
                                 c[1] + r * np.sin(theta)])
    fixed_st_loc = np.array(fixed_nodes, dtype=float)   # (N, 2)
    n_fix = len(fixed_st_loc)

    # ── Destination ───────────────────────────────────────────────────────
    dest_loc = np.array([[4.0, 7.0]], dtype=float)       # (1, 2)

    # ── Initial UAV locations ─────────────────────────────────────────────
    '''zeta_init = np.array([
        [3.7, 3.2],
        [4.5, 5.8],
        [5.5, 4.7],
        [7.2, 3.0],
        [7.3, 5.0],
    ], dtype=float)      
    
    zeta_init = np.array([
        [3.7, 4.2],
        [4.5, 4.8],
        [5.5, 5.7],
        [7.2, 2.0],
        [7.3, 4.0],
    ], dtype=float)                                   # (5, 2)
    n_fac = len(zeta_init)'''
    import secrets
    n_fac = 5
    init_seed = secrets.randbits(32)
    init_rng = np.random.default_rng(init_seed)
    zeta_init = np.column_stack([
        init_rng.uniform(3.0, 7.5, size=n_fac),   # x in [3, 7.5)
        init_rng.uniform(3.0, 5.5, size=n_fac),   # y in [3, 5.5)
    ]).astype(float) 

    # ── State space ───────────────────────────────────────────────────────
    # Order: f_1..f_nF | n_1..n_N | delta
    st_details = []
    for i in range(1, n_fac + 1):
        st_details.append((f"f{i}", f"f{i}"))
    for i in range(1, n_fix + 1):
        st_details.append((f"n{i}", f"n{i}"))
    st_details.append(("delta", "delta"))
    S = len(st_details)
    assert S == n_fac + n_fix + 1, f"Expected S={n_fac+n_fix+1}, got {S}"

    # ── Action space ──────────────────────────────────────────────────────
    # Order: f_1..f_nF | delta
    ac_details = []
    for i in range(1, n_fac + 1):
        ac_details.append((f"f{i}", f"f{i}"))
    ac_details.append(("delta", "delta"))
    A = len(ac_details)
    assert A == n_fac + 1, f"Expected A={n_fac+1}, got {A}"

    # ── Valid action mask ─────────────────────────────────────────────────
    valid_ac         = np.zeros((S, A), dtype=bool)
    delta_ac_idx     = n_fac      # index of delta action
    delta_st_idx     = n_fac + n_fix  # index of delta state

    # UAV state f_j (state index j, 0-based):
    #   valid: all UAV actions except self + delta action
    for j in range(n_fac):
        s = j   # UAV state index = UAV index (by construction)
        for a in range(n_fac):
            if a != j:
                valid_ac[s, a] = True    # other UAV actions
        valid_ac[s, delta_ac_idx] = True  # delta action

    # User node n_i (state index nF + i, 0-based i):
    #   valid: all UAV actions, not delta
    for i in range(n_fix):
        s = n_fac + i
        for a in range(n_fac):
            valid_ac[s, a] = True        # all UAV actions
        # delta action remains False

    # Delta state: no valid actions (absorbing)
    # valid_ac[delta_st_idx, :] remains all False

    return st_details, ac_details, dest_loc, zeta_init, fixed_st_loc, valid_ac


def precompute_env(st_details: list, ac_details: list,
                   dest_loc: np.ndarray, fixed_st_loc: np.ndarray,
                   valid_ac: np.ndarray,
                   eta_tr: float = 100.0,
                   kappa: float = 0.9,
                   hop_cost: float = HOP_COST) -> EnvStructural:
    """
    Precompute all zeta-independent quantities.
    """
    S   = len(st_details)
    A   = len(ac_details)
    dim = dest_loc.shape[1]

    # ── Key indices ───────────────────────────────────────────────────────
    idx_delta_state  = next(i for i, d in enumerate(st_details)
                            if d[0] == "delta")
    idx_delta_action = next(i for i, d in enumerate(ac_details)
                            if d[0] == "delta")

    # ── State classification ──────────────────────────────────────────────
    is_fac_state = np.zeros(S, dtype=bool)
    fac_id_state = np.zeros(S, dtype=int)
    is_fix_state = np.zeros(S, dtype=bool)
    fix_id_state = np.zeros(S, dtype=int)

    for s, (lab1, _) in enumerate(st_details):
        if lab1.startswith("f"):
            is_fac_state[s] = True
            fac_id_state[s] = int(lab1[1:]) - 1   # 0-based UAV index
        elif lab1.startswith("n"):
            is_fix_state[s] = True
            fix_id_state[s] = int(lab1[1:]) - 1   # 0-based user node index

    # ── Action classification ─────────────────────────────────────────────
    is_fac_action = np.zeros(A, dtype=bool)
    fac_id_action = np.zeros(A, dtype=int)

    for a, (lab1, _) in enumerate(ac_details):
        if lab1.startswith("f"):
            is_fac_action[a] = True
            fac_id_action[a] = int(lab1[1:]) - 1   # 0-based UAV index

    # ── Critical lookup arrays ────────────────────────────────────────────
    # fac_state_idx[j]  = state index of UAV f_{j+1}
    # fac_action_idx[j] = action index for jumping to UAV f_{j+1}
    # By construction: fac_state_idx[j] = j, fac_action_idx[j] = j
    # Computed explicitly from labels for safety
    n_fac = int(is_fac_state.sum())

    fac_state_idx  = np.zeros(n_fac, dtype=int)
    fac_action_idx = np.zeros(n_fac, dtype=int)

    for s, (lab1, _) in enumerate(st_details):
        if lab1.startswith("f"):
            j = int(lab1[1:]) - 1
            fac_state_idx[j] = s

    for a, (lab1, _) in enumerate(ac_details):
        if lab1.startswith("f"):
            j = int(lab1[1:]) - 1
            fac_action_idx[j] = a

    # Verify consistency
    for j in range(n_fac):
        assert fac_id_state[fac_state_idx[j]] == j, \
            f"fac_state_idx inconsistency at j={j}"
        assert fac_id_action[fac_action_idx[j]] == j, \
            f"fac_action_idx inconsistency at j={j}"

    # ── Fixed state coordinates ───────────────────────────────────────────
    # UAV rows left as zeros — filled from zeta at runtime via get_state_coords
    n_fix = len(fixed_st_loc)
    state_coords_fixed = np.zeros((S, dim))
    for s, (lab1, _) in enumerate(st_details):
        if lab1 == "delta":
            state_coords_fixed[s] = dest_loc[0]
        elif is_fix_state[s]:
            state_coords_fixed[s] = fixed_st_loc[fix_id_state[s]]
        # UAV states: remain zero, filled from zeta at runtime

    # ── Non-terminal state range ──────────────────────────────────────────
    s_low  = n_fac             # first user node state index
    s_high = n_fac + n_fix - 1 # last user node state index

    # Verify
    assert st_details[s_low][0].startswith("n"), \
        f"Expected user node at s_low={s_low}, got '{st_details[s_low][0]}'"
    assert st_details[s_high][0].startswith("n"), \
        f"Expected user node at s_high={s_high}, got '{st_details[s_high][0]}'"
    assert st_details[idx_delta_state][0] == "delta", \
        f"Expected delta at {idx_delta_state}, got '{st_details[idx_delta_state][0]}'"
    assert idx_delta_state == n_fac + n_fix, \
        f"Delta state should be at index {n_fac+n_fix}, got {idx_delta_state}"
    assert idx_delta_action == n_fac, \
        f"Delta action should be at index {n_fac}, got {idx_delta_action}"

    return EnvStructural(
        st_count=S, ac_count=A,
        n_fac=n_fac, n_fix=n_fix,
        dim=dim, eta_tr=eta_tr, kappa=kappa,
        dest_loc=dest_loc, fixed_st_loc=fixed_st_loc,
        idx_delta_state=idx_delta_state,
        idx_delta_action=idx_delta_action,
        is_fac_state=is_fac_state, fac_id_state=fac_id_state,
        is_fix_state=is_fix_state, fix_id_state=fix_id_state,
        is_fac_action=is_fac_action, fac_id_action=fac_id_action,
        fac_state_idx=fac_state_idx,
        fac_action_idx=fac_action_idx,
        state_coords_fixed=state_coords_fixed,
        valid_ac=valid_ac,
        s_low=s_low, s_high=s_high,
        st_details=st_details, ac_details=ac_details,
        hop_cost=float(hop_cost),
    )


# ---------------------------------------------------------------------------
# Helper: state coordinates
# ---------------------------------------------------------------------------

def get_state_coords(zeta: np.ndarray, env: EnvStructural) -> np.ndarray:
    """
    Return full (S, dim) coordinate array with UAV rows filled from zeta.
    UAV state f_j (state index fac_state_idx[j]) gets coordinate zeta[j].
    """
    coords = env.state_coords_fixed.copy()
    for j in range(env.n_fac):
        coords[env.fac_state_idx[j]] = zeta[j]
    return coords


# ---------------------------------------------------------------------------
# Environment step
# ---------------------------------------------------------------------------

def env_step(s: int, a: int, zeta: np.ndarray,
             env: EnvStructural) -> tuple[int, float, bool]:
    """
    Single MDP step under transition kernel:

        p(s'|s, a, zeta) = kappa          if s' = intended(a)
                         = (1-kappa) * Gibbs(||x_s - x_{s'}||^2)
                                           over intended states of A(s)\{a}
                         = 0              otherwise

    Cost: c(s,a,s',zeta) = ||x_s - x_{s'}||^2  if s' != delta, else 0.
    """
    assert env.valid_ac[s, a], \
        f"Invalid action a={a} ({env.ac_details[a]}) " \
        f"from state s={s} ({env.st_details[s]})"

    # ── State coordinates ─────────────────────────────────────────────────
    coords = get_state_coords(zeta, env)

    # ── Intended next state ───────────────────────────────────────────────
    if a == env.idx_delta_action:
        s_intended = env.idx_delta_state
    else:
        # UAV action a: intended state is fac_state_idx[fac_id_action[a]]
        s_intended = int(env.fac_state_idx[env.fac_id_action[a]])

    # ── Sample next state ─────────────────────────────────────────────────
    if np.random.rand() < env.kappa:
        s_next = s_intended
    else:
        # Off-target: Gibbs over intended states of A(s)\{a}
        valid_actions = np.where(env.valid_ac[s])[0]       # (m,)
        alt_actions   = valid_actions[valid_actions != a]   # exclude chosen

        if len(alt_actions) == 0:
            # No alternatives — fall back to intended
            s_next = s_intended
        else:
            # Map each alternative action to its intended next state
            alt_states = np.array([
                env.idx_delta_state
                if a_ == env.idx_delta_action
                else int(env.fac_state_idx[env.fac_id_action[a_]])
                for a_ in alt_actions
            ], dtype=int)

            # Gibbs distribution over squared distances from current state
            # Gibbs distribution over squared distances from current state
            x_s    = coords[s]
            x_alts = coords[alt_states]
            logits = -env.eta_tr * np.sum(
                (x_alts - x_s) ** 2, axis=1)
            logits -= logits.max()
            p       = np.exp(logits)
            p_sum   = p.sum()

            if p_sum < 1e-300 or not np.isfinite(p_sum):
                # Fallback: uniform over alternatives
                p = np.ones(len(alt_states)) / len(alt_states)
            else:
                p /= p_sum

            s_next = int(alt_states[
                np.random.choice(len(alt_states), p=p)])
        # ── Instantaneous cost ────────────────────────────────────────────────
        '''if s_next == env.idx_delta_state:
            cost = 0.0
        else:
            cost = float(np.sum((coords[s_next] - coords[s]) ** 2))

        done = (s_next == env.idx_delta_state)'''
        # Correct — cost is always ||x_s - x_{s'}||^2, even for the final hop to delta
    cost = float(np.sum((coords[s_next] - coords[s]) ** 2))
    if cost <= 0.1 and env.kappa <= 0.75:
        cost = cost + 1
    cost = cost + env.hop_cost          # constant cost of every hop from a non-terminal state (env_step is never called from delta)

    done = (s_next == env.idx_delta_state)
    return s_next, cost, done
        #return s_next, cost, done


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(env: EnvStructural, zeta: np.ndarray,
                      n_transition_samples: int = 2000) -> None:
    """
    Verify environment structure and transition kernel empirically.
    Raises AssertionError on failure.
    """
    S, A = env.st_count, env.ac_count

    print("Running sanity checks...")

    # ── 1. State ordering ─────────────────────────────────────────────────
    for j in range(env.n_fac):
        assert env.st_details[j][0] == f"f{j+1}", \
            f"Expected f{j+1} at state index {j}, " \
            f"got '{env.st_details[j][0]}'"
    for i in range(env.n_fix):
        assert env.st_details[env.n_fac + i][0] == f"n{i+1}", \
            f"Expected n{i+1} at state index {env.n_fac+i}, " \
            f"got '{env.st_details[env.n_fac+i][0]}'"
    assert env.st_details[S-1][0] == "delta", \
        f"Expected delta at state index {S-1}, " \
        f"got '{env.st_details[S-1][0]}'"
    print("  [PASS] State ordering")

    # ── 2. Action ordering ────────────────────────────────────────────────
    for j in range(env.n_fac):
        assert env.ac_details[j][0] == f"f{j+1}", \
            f"Expected f{j+1} at action index {j}, " \
            f"got '{env.ac_details[j][0]}'"
    assert env.ac_details[A-1][0] == "delta", \
        f"Expected delta at action index {A-1}, " \
        f"got '{env.ac_details[A-1][0]}'"
    print("  [PASS] Action ordering")

    # ── 3. fac_state_idx and fac_action_idx ──────────────────────────────
    for j in range(env.n_fac):
        s = env.fac_state_idx[j]
        a = env.fac_action_idx[j]
        assert env.st_details[s][0] == f"f{j+1}", \
            f"fac_state_idx[{j}]={s} -> '{env.st_details[s][0]}', " \
            f"expected 'f{j+1}'"
        assert env.ac_details[a][0] == f"f{j+1}", \
            f"fac_action_idx[{j}]={a} -> '{env.ac_details[a][0]}', " \
            f"expected 'f{j+1}'"
    print("  [PASS] fac_state_idx and fac_action_idx")

    # ── 4. Valid action mask — UAV states ─────────────────────────────────
    for j in range(env.n_fac):
        s = env.fac_state_idx[j]
        # Delta action must be valid
        assert env.valid_ac[s, env.idx_delta_action], \
            f"Delta action should be valid from UAV state f{j+1} (s={s})"
        # Self-action must be invalid
        assert not env.valid_ac[s, env.fac_action_idx[j]], \
            f"Self-action should be invalid from UAV state f{j+1} (s={s})"
        # All other UAV actions must be valid
        for k in range(env.n_fac):
            if k != j:
                assert env.valid_ac[s, env.fac_action_idx[k]], \
                    f"UAV action f{k+1} should be valid from f{j+1} (s={s})"
    print("  [PASS] Valid actions from UAV states")

    # ── 5. Valid action mask — user nodes ─────────────────────────────────
    for i in range(env.n_fix):
        s = env.n_fac + i
        # All UAV actions must be valid
        for j in range(env.n_fac):
            assert env.valid_ac[s, env.fac_action_idx[j]], \
                f"UAV action f{j+1} should be valid from user node n{i+1} (s={s})"
        # Delta action must be invalid
        assert not env.valid_ac[s, env.idx_delta_action], \
            f"Delta action should be invalid from user node n{i+1} (s={s})"
    print("  [PASS] Valid actions from user nodes")

    # ── 6. Valid action mask — delta state ────────────────────────────────
    assert not env.valid_ac[env.idx_delta_state].any(), \
        "No actions should be valid from delta state"
    print("  [PASS] No valid actions from delta state")

    # ── 7. State coordinates ──────────────────────────────────────────────
    coords = get_state_coords(zeta, env)
    # UAV coordinates should match zeta
    for j in range(env.n_fac):
        s = env.fac_state_idx[j]
        np.testing.assert_array_almost_equal(
            coords[s], zeta[j],
            err_msg=f"UAV f{j+1} coordinate mismatch"
        )
    # Destination coordinate should match dest_loc
    np.testing.assert_array_almost_equal(
        coords[env.idx_delta_state], env.dest_loc[0],
        err_msg="Destination coordinate mismatch"
    )
    print("  [PASS] State coordinates")

    # ── 8. Transition kernel — intended state probability ─────────────────
    print(f"  Checking transition kernel "
          f"({n_transition_samples} samples per (s,a) pair)...")

    # Check a sample of UAV states and user node states
    test_states = (
        list(range(env.n_fac)) +                          # all UAV states
        list(range(env.s_low, env.s_low + 3))             # 3 user nodes
    )

    for s in test_states:
        valid_actions = np.where(env.valid_ac[s])[0]
        for a in valid_actions:
            counts = {}
            for _ in range(n_transition_samples):
                s_next, _, _ = env_step(s, a, zeta, env)
                counts[s_next] = counts.get(s_next, 0) + 1

            # Intended state
            s_intended = (env.idx_delta_state
                          if a == env.idx_delta_action
                          else int(env.fac_state_idx[env.fac_id_action[a]]))

            emp_kappa = counts.get(s_intended, 0) / n_transition_samples
            assert emp_kappa > env.kappa - 0.05, \
                f"State s={s} ({env.st_details[s][0]}), " \
                f"action a={a} ({env.ac_details[a][0]}): " \
                f"empirical kappa={emp_kappa:.3f}, " \
                f"expected ~{env.kappa}"

            # Off-target probability
            emp_off = 1.0 - emp_kappa
            expected_off = 1.0 - env.kappa
            assert abs(emp_off - expected_off) < 0.05, \
                f"Off-target probability mismatch: " \
                f"got {emp_off:.3f}, expected ~{expected_off:.3f}"

    print(f"  [PASS] Transition kernel (kappa={env.kappa})")

    # ── 9. Cost of the final hop to delta = squared distance + hop cost ───
    # Force a transition to delta from a UAV state and check cost
    s = env.fac_state_idx[0]   # UAV f_1
    a = env.idx_delta_action
    costs = []
    for _ in range(100):
        s_next, cost, done = env_step(s, a, zeta, env)
        if s_next == env.idx_delta_state:
            costs.append(cost)
    if costs:
        want = float(np.sum((env.dest_loc[0] - zeta[0]) ** 2)) + env.hop_cost
        assert all(abs(c - want) < 1e-9 for c in costs), \
            f"Cost of the hop to delta should be {want}, got {costs[:3]}"
    print("  [PASS] Cost of the hop to delta (distance^2 + hop cost)")

    print("All sanity checks passed.\n")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_env(env: EnvStructural, zeta: np.ndarray,
             ax: plt.Axes = None,
             title: str = "UAV Network Layout") -> tuple:
    """
    Plot user nodes, UAV locations, and destination.
    Returns (fig, ax) for further annotation.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))
    else:
        fig = ax.get_figure()

    ax.scatter(env.fixed_st_loc[:, 0], env.fixed_st_loc[:, 1],
               s=10, c="steelblue", alpha=0.6, label="User nodes")
    ax.scatter(zeta[:, 0], zeta[:, 1],
               s=100, marker="s", c="orange",
               edgecolors="black", linewidths=0.8, label="UAVs")
    for j, z in enumerate(zeta):
        ax.annotate(f"$f_{{{j+1}}}$", z,
                    textcoords="offset points",
                    xytext=(6, 4), fontsize=10)
    ax.scatter(env.dest_loc[:, 0], env.dest_loc[:, 1],
               s=150, marker="D", c="red",
               edgecolors="black", linewidths=0.8, label="Destination")
    ax.annotate(r"$\delta$", env.dest_loc[0],
                textcoords="offset points",
                xytext=(6, 4), fontsize=10)
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig, ax


def plot_routing(env: EnvStructural, zeta: np.ndarray,
                 paths: list,
                 title: str = "Routing Paths") -> tuple:
    """
    Plot routing paths overlaid on the network layout.

    Args:
        paths: list of lists of state indices representing sample routes
    """
    fig, ax = plot_env(env, zeta, title=title)
    coords  = get_state_coords(zeta, env)
    colors  = plt.cm.tab10(np.linspace(0, 1, max(len(paths), 1)))

    for path, color in zip(paths, colors):
        xs = [coords[s, 0] for s in path]
        ys = [coords[s, 1] for s in path]
        ax.plot(xs, ys, "-o", color=color,
                markersize=4, linewidth=1.5, alpha=0.8)

    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Main — quick verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    st_d, ac_d, dest, zeta_init, fsl, vac = build_env_info(seed=1)
    env = precompute_env(st_d, ac_d, dest, fsl, vac,
                         eta_tr=100.0, kappa=0.9)

    print(f"State space:   S  = {env.st_count}  "
          f"(nF={env.n_fac} UAVs + N={env.n_fix} users + 1 delta)")
    print(f"Action space:  A  = {env.ac_count}  "
          f"({env.n_fac} UAV actions + 1 delta)")
    print(f"User nodes:    [{env.s_low}, {env.s_high}]")
    print(f"Delta state:   {env.idx_delta_state}")
    print(f"Delta action:  {env.idx_delta_action}")
    print(f"fac_state_idx: {env.fac_state_idx}")
    print(f"fac_action_idx:{env.fac_action_idx}")
    print(f"valid_ac shape:{env.valid_ac.shape}")
    print(f"valid_ac row sums (first 10): "
          f"{env.valid_ac[:10].sum(axis=1)}")
    print()

    run_sanity_checks(env, zeta_init, n_transition_samples=2000)
    plot_env(env, zeta_init)
    plt.show()