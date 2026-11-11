# Environment Parameter Gradient (EnvPG) for Co-Design in Reinforcement Learning

Code accompanying *"Environment Parameter Gradient Theorem for Co-Design in Reinforcement Learning"* (arXiv:2607.12590v2).

It contains the EnvPG actor-critic algorithm, the two baselines of the paper (a model-based oracle and a score-function co-design baseline), and the scripts that produce the figure and table of the Simulation section (aerial relay network design).

## Files

| File | Purpose |
|---|---|
| `env2_paraRLModel.py` | The aerial relay network MDP: user nodes, destination, transition kernel (kappa, eta), hop cost, emulator. Imported by every script. |
| `exact_eval.py` | Exact evaluation by policy iteration on the analytic kernel (exact J*, exact gradient). Used for all reported costs and cosines. |
| `train_mf_envpg_resample_runs.py` | **EnvPG** (Algorithm 1): critic, actor and relay-location updates, model-free. |
| `train_mf_sf_runs.py` | **SF baseline**: score-function (REINFORCE) update of a Gaussian design distribution, jointly with the actor. |
| `benchmark_oracle_runs.py` | **Oracle**: model-based soft value iteration with an annealed inverse temperature and projected gradient descent on the relay locations. |
| `generate_paper_figures_v2.py` | Panel (a): network design (relay placement, routing, user-node colors). |
| `plot_trajectory.py` | Panel (b): relay trajectories from the initial to the final positions. |
| `cosine_vs_N.py` | Panel (c): cosine between the estimated and the exact gradient, from saved checkpoints (also computes the stored training gradient with `--stored`). |
| `plot_two_curves.py` | Panel (c): draws the two cosine curves (N = 5,000 estimate and training estimate). |
| `plot_jstar.py` | Panel (d): exact J*(zeta_k) versus update step. |
| `plot_grad_norm_old.py` | Panel (e): norm of the stochastic gradient estimate. |

## Requirements

Python 3.x with `numpy`, `scipy`, `torch` and `matplotlib` (see `requirements.txt`). Everything runs on CPU.

## Experiment

Four values of the link reliability kappa in {0.85, 0.90, 0.95, 1.00} and six initial relay placements P1 to P6 (24 runs per method). The hop cost is 0.2 (set in `env2_paraRLModel.py`).

## Running

Each run is one (kappa, placement) pair and writes to `<this folder>/results*/<tag>/`, where `<tag>` is e.g. `k095_P6`.

```bash
# EnvPG (writes results/<tag>/: hist_<tag>.npz, summary_<tag>.json, checkpoints)
python train_mf_envpg_resample_runs.py --kappa 0.95 --placement 6

# SF baseline (writes results_sf_*/<tag>/)
python train_mf_sf_runs.py --kappa 0.95 --placement 6

# Oracle: all 24 runs one after the other (finished runs are skipped)
python benchmark_oracle_runs.py --beta_min 10 --out results_oracle_h0.2
python benchmark_oracle_runs.py --kappas 0.95 --placements 6 --beta_min 10 --out results_oracle_h0.2   # a single run
```

To run all 24 EnvPG (or SF) runs, loop over `--kappa 0.85 0.90 0.95 1.00` and `--placement 1 ... 6`, for example in a shell loop or with GNU parallel.

In EnvPG the relay-location gradient is estimated with the policy `pi_{theta_k}` of the same iteration as the critic (`zeta_at_theta_k = True` in `CFG`).

## Reproducing the figure

Run the commands from the folder that contains all the scripts above (`cosine_vs_N.py` imports the EnvPG training script and `env2_paraRLModel.py`).

```bash
# (a) network design of one run (kappa = 0.95, placement P6)
python generate_paper_figures_v2.py --envpg results --oracle results_oracle_h0.2 --kappa 0.95 --pick P6

# (b) relay trajectories of the same run
python plot_trajectory.py --root results --tag k095_P6 --panels paths

# (c) cosine with the exact gradient, then the two curves
python cosine_vs_N.py --root results --module train_mf_envpg_resample_runs --stored --workers 4 --out cosine_out
python plot_two_curves.py --dir cosine_out

# (d) exact J* versus update step
python plot_jstar.py --root results --csv jstar_curves.csv --max_it 100

# (e) gradient norm
python plot_grad_norm_old.py --root results --agg pick --pick P4 --window 200 --min_row 10
```

Use the same arguments as for the paper figure for the cosine and plot scripts; `--help` lists all options. The exact values of J* and of the gradient are computed on the analytic kernel for evaluation only. The training of EnvPG never uses it.

## Citation

If you use this code, please cite the paper (reference to be added after publication).
