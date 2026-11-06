# Step 1 changes (leakage audit + model fixes)

## Leakage audit: PASS
- `03_estimate_P.py` fits P, Q, R and the bootstrap only on seasons <= 2022 (913 matches, 16,365 transitions).
- `04_analysis.py` (markov_analytical_results.csv, pi_0) uses only those train matrices and train over-1 states.
- `06_validate.py` LR baseline is trained on <= 2022 and scored on >= 2023 (282 matches, 5,005 over-states).
- Added hard `assert`s in 03, 06, 06b so any future split change that leaks fails loudly.

## Fixes
- 04: removed "exceptional concordance" claim. pi_0*B = empirical win rate in-sample is an identity of the
  count MLE, not validation. Added pi_k = pi_0 P^(k-1) (state probabilities vs time).
- 05: reports fraction of simulated paths with > 19 transitions (horizon violation of homogeneous chain).
- NEW 06b_time_inhomogeneous.py: finite-horizon, per-over chain P_1..P_19 (Dirichlet-smoothed toward
  overs k±2), backward recursion for win prob V_k(s) and expected overs E_k(s), 3-model out-of-sample
  comparison with match-level cluster bootstrap CIs, and absorbed-by-over validation figure.

## Run order
python 01_load_filter.py && python 02_build_states.py && python 03_estimate_P.py && python 04_analysis.py \
 && python 05_simulate.py && python 06_validate.py && python 06b_time_inhomogeneous.py && python 07_tests.py
Real outputs from my run are in report/run_logs/.

# Steps 2-4 (era study, figures, theory)

- 08_era_experiments.py REWRITTEN (original only smoke-tested). Changes vs the original:
  * adds the time-inhomogeneous chain (INH) alongside HOM and LR for every training regime;
  * EB prior weight chosen by Dirichlet-multinomial marginal likelihood on 2023-25 (was a fixed 20; tuning on 2026 would leak);
  * paired Brier differences on identical match resamples (pre vs Impact-only / all / EB; Markov vs LR);
  * leave-one-season-out yardstick for pre-rule seasons;
  * global and phase-stratified G-tests + 1,000 match-level permutation p-values (per-row Holm kept);
  * bootstrap CIs on B_i pre/post and their difference, and on INH win prob at over 10;
  * per-match count tensor -> all refits/bootstraps/permutations in ~50 s.
- 09_figures.py: era heatmaps, B by era, calibration, 2026 traces, LOSO, state diagram,
  state probabilities vs time, INH sample paths + absorption-time histogram, targets by season.
- 10_theory_numbers.py: n-step block identity, Chapman-Kolmogorov, eigenvalues / no unique
  stationary distribution, finite-horizon absorption of the INH product.

Full run order:
python 01_load_filter.py && python 02_build_states.py && python 03_estimate_P.py && python 04_analysis.py \
 && python 05_simulate.py && python 06_validate.py && python 06b_time_inhomogeneous.py && python 07_tests.py \
 && python 08_era_experiments.py && python 09_figures.py && python 10_theory_numbers.py
