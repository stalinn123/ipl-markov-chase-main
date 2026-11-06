# Absorbing Markov Chain Model of IPL T20 Run Chases

A university stochastic-processes project modeling the evolution, win probability, absorption time, and tipping points of Indian Premier League (IPL) second-innings run chases using discrete-time absorbing Markov chains.

---

## 1. Project Overview

In T20 cricket, a run chase in the second innings is a dynamic stochastic process governed by two opposing pressures:
1. **Accumulated run-rate pressure:** The difference between the Required Run Rate (RRR) and Current Run Rate (CRR).
2. **Resource depletion:** The number of wickets lost out of 10 available wickets.

This project formulates the chase as a discrete-time absorbing Markov chain $X_k$ evaluated at the completion of each over $k \in \{1, 2, \dots, 19\}$, transitioning between 15 transient states and 2 absorbing states:
* **Transient States (15):** Cross-product of Wickets Lost (`W_Low`, `W_Mid`, `W_High`) and Pressure Bucket (`P1`, `P2`, `P3`, `P4`, `P5`).
* **Absorbing States (2):** `WIN` (chasing team reaches target) and `LOSS` (10 wickets lost or 20 overs elapse without reaching target).

---

## 2. Directory Structure

```text
ipl-markov-chase/
├── data/
│   ├── raw/
│   │   └── ipl_male_csv/          # 1,243 Cricsheet IPL match CSVs (2008–2026)
│   └── processed/
│       ├── chases.csv              # Filtered match metadata and outcomes
│       ├── over_states.csv         # Ball-by-ball replayed over states
│       ├── transitions.csv         # Empirical over-to-over state transitions
│       ├── transition_matrix_P.csv # 17x17 canonical Markov transition matrix
│       ├── Q_matrix.csv            # 15x15 transient submatrix
│       ├── R_matrix.csv            # 15x2 absorbing submatrix
│       ├── fundamental_matrix_N.csv# Fundamental matrix N = (I - Q)^(-1)
│       ├── markov_analytical_results.csv # Analytical B_i and t_i with 95% CIs
│       ├── tipping_point_states.csv# Knife-edge states (0.40 <= B_i <= 0.60)
│       ├── top_swing_transitions.csv# Probability-weighted swing transitions
│       ├── wicket_penalty_table.csv# Impact of wicket loss holding pressure fixed
│       ├── simulation_vs_theory.csv# Monte Carlo (10,000 runs) vs theory comparison
│       ├── validation_results.csv  # Out-of-sample metrics by over (2023–2026)
│       └── phase_homogeneity_tests.csv # Chi-Square tests by state row
├── figures/
│   ├── transition_matrix.png       # Heatmaps of Q (15x15) and R (15x2)
│   ├── win_probability_heatmap.png # Grid of win probabilities B with 95% CIs
│   ├── expected_overs_heatmap.png  # Grid of expected overs t with 95% CIs
│   ├── tipping_points_analysis.png # Tipping states and top swing transitions
│   ├── mc_vs_theory_win_prob.png   # Monte Carlo vs analytical theory comparison
│   ├── mc_match_length_distribution.png # Match length density from contrasting states
│   ├── validation_calibration.png  # Out-of-sample calibration curve vs baseline
│   ├── validation_brier_by_over.png# Brier score evolution over k = 1..19
│   ├── validation_chase_traces.png # In-play probability curves for 3 test chases
│   └── phase_homogeneity_comparison.png # Direct loss probabilities by phase
├── report/
│   └── results_summary.md          # Comprehensive executive summary with key numbers
├── src/
│   ├── config.py                   # Central tunable settings and bucket cutoffs
│   ├── 01_load_filter.py           # Ingest, filter matches, cross-check targets
│   ├── 02_build_states.py          # Ball-by-ball replay, over-states, transitions
│   ├── 03_estimate_P.py            # Estimate P, Q, R and whole-match bootstrap CIs
│   ├── 04_analysis.py              # Fundamental matrix, B, t, tipping points
│   ├── 05_simulate.py              # Monte Carlo simulation (10,000 chains/state)
│   ├── 06_validate.py              # Out-of-sample validation vs Logistic Regression
│   └── 07_tests.py                 # Time-homogeneity and Markov property tests
├── requirements.txt                # Python package dependencies
└── README.md                       # Complete documentation
```

---

## 3. Installation & Setup

1. **Activate Virtual Environment:**
   ```powershell
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   ```
2. **Install Requirements:**
   ```powershell
   pip install -r requirements.txt
   ```

---

## 4. Execution Pipeline (Run Scripts in Order)

Run scripts from the project root (`ipl-markov-chase/`):

```powershell
# Step 1: Ingest and filter match CSVs
python src/01_load_filter.py

# Step 2: Replay 2nd innings ball-by-ball, construct states & transitions
python src/02_build_states.py

# Step 3: Estimate canonical transition matrix P and bootstrap CIs
python src/03_estimate_P.py

# Step 4: Analytical solution (N, B, t, tipping points) and heatmaps
python src/04_analysis.py

# Step 5: Monte Carlo simulations (150,000 chains) vs theory verification
python src/05_simulate.py

# Step 6: Out-of-sample test validation (Seasons 2023-2026) vs Logistic Regression
python src/06_validate.py

# Step 7: Hypothesis testing of Time-Homogeneity and Markov Property
python src/07_tests.py
```

---

## 5. Summary of Main Findings

1. **Absorption Certainty:** Spectral radius $\rho(Q) = 0.904272 < 1.0$, guaranteeing absorption with probability 1.0. Model-implied overall chase win probability ($\pi_0 \cdot B = 54.87\%$) matches the empirical training win rate (54.87%) to 0.00 percentage points.
2. **Critical Tipping Points:** Identified 5 knife-edge states ($0.40 \le B_i \le 0.60$): `W_Low_P3` (56.7%), `W_Low_P4` (49.5%), `W_Low_P5` (43.4%), `W_Mid_P3` (48.9%), and `W_High_P3` (50.4%).
3. **Wicket Penalty:** While teams with low pressure absorb wicket shocks easily, teams under high pressure suffer a **28.2% to 37.1%** collapse in win probability when dropping from Low to High wickets.
4. **Out-of-Sample Predictive Power:** On held-out seasons 2023–2026 (282 matches, 5,005 observations), the Markov chain achieves **75.22% overall accuracy** and a **Brier score of 0.1711**, reaching **90.38% accuracy** by the 19th over.
5. **Statistical Tests:** Formal hypothesis tests reject strict time-homogeneity ($p < 10^{-16}$) and memorylessness ($p < 10^{-16}$) due to physical cricket dynamics (death-over boundary slogging and bowler spell momentum). Nevertheless, the first-order Markov chain provides an exceptionally accurate, interpretable, and parsimonious stochastic model.
