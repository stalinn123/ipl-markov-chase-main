# Absorbing Markov Chain Model of IPL T20 Run Chases

A university stochastic-processes project modelling second-innings run chases in the Indian Premier League (IPL) as discrete-time absorbing Markov chains: win probability, time to absorption, tipping points, and whether chase dynamics changed in the Impact Player era (from 2023).

**Group:** Stalin Velayudham N, K Sanjay, Shreeya Jetti

---

## 1. Project Overview

A T20 run chase is shaped by two pressures:
1. **Run-rate pressure:** the difference between the Required Run Rate (RRR) and the Current Run Rate (CRR).
2. **Resource depletion:** the number of wickets lost out of 10.

The chase is modelled as a chain $X_k$ observed at the end of each over $k \in \{1, 2, \dots, 19\}$, with 15 transient states and 2 absorbing states:
* **Transient states (15):** wickets lost (`W_Low`, `W_Mid`, `W_High`) × pressure bucket (`P1` to `P5`).
* **Absorbing states (2):** `WIN` (target reached) and `LOSS` (10 wickets lost, or 20 overs used without reaching the target).

Two versions of the chain are compared with a logistic regression baseline:
* **Homogeneous:** one transition matrix for all overs (scripts 03–06).
* **Time-inhomogeneous:** a separate matrix for each over, so chases must end by over 20 (script 06b).

**Data:** ball-by-ball IPL match files from [Cricsheet](https://cricsheet.org/downloads/).

---

## 2. Directory Structure

```text
ipl-markov-chase-main/
├── data/
│   ├── raw/ipl_male_csv/   # 1,243 Cricsheet IPL match CSVs (2008–2026)
│   └── processed/          # Generated tables and matrices, including:
│       ├── chases.csv, over_states.csv, transitions.csv
│       ├── transition_matrix_P.csv, Q_matrix.csv, R_matrix.csv, fundamental_matrix_N.csv
│       ├── markov_analytical_results.csv, simulation_vs_theory.csv
│       ├── validation_results.csv, validation_three_models.csv
│       ├── inhom_win_prob_by_over.csv, inhom_expected_overs_by_over.csv
│       └── era_*.csv       # Impact Player era results
├── figures/                # All generated figures
├── report/
│   ├── results_summary.md  # Results with key numbers
│   └── run_logs/           # Output of each script from the last full run
├── src/
│   ├── config.py                  # Paths, train/test split, state definitions
│   ├── 01_load_filter.py          # Load and filter matches, cross-check targets
│   ├── 02_build_states.py         # Replay chases, build over states and transitions
│   ├── 03_estimate_P.py           # Estimate P, Q, R with whole-match bootstrap
│   ├── 04_analysis.py             # Fundamental matrix, B, t, tipping points
│   ├── 05_simulate.py             # Monte Carlo simulation vs theory
│   ├── 06_validate.py             # Out-of-sample validation vs logistic regression
│   ├── 06b_time_inhomogeneous.py  # Per-over chain and three-model comparison
│   ├── 07_tests.py                # Time-homogeneity and Markov property tests
│   ├── 08_era_experiments.py      # Impact Player era experiments
│   ├── 09_figures.py              # Era, state diagram, sample path and other figures
│   └── 10_theory_numbers.py       # Numerical checks of the theory
├── requirements.txt
└── README.md
```

---

## 3. Setup

Tested on Python 3.13. From the project root in Windows PowerShell (on macOS or Linux, activate with `source .venv/bin/activate`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 4. Running the Pipeline

Run the scripts in this order from the project root. Each one reads the outputs of the earlier ones.

```powershell
python src/01_load_filter.py
python src/02_build_states.py
python src/03_estimate_P.py
python src/04_analysis.py
python src/05_simulate.py
python src/06_validate.py
python src/06b_time_inhomogeneous.py
python src/07_tests.py
python src/08_era_experiments.py
python src/09_figures.py
python src/10_theory_numbers.py
```

The full run takes about six minutes. Random seeds are fixed in `src/config.py`. Very small floating-point differences between machines are normal.

---

## 5. Main Findings

Trained on seasons 2008–2022 and tested on 2023–2026 (282 matches, 5,005 over-level observations). Full details are in `report/results_summary.md`.

| Model | Brier Score | Log-Loss | Accuracy |
| :--- | :---: | :---: | :---: |
| Homogeneous chain | 0.1712 | 0.5161 | 75.60% |
| Time-inhomogeneous chain | 0.1618 | 0.4951 | 76.12% |
| Logistic regression | 0.1431 | 0.4365 | 79.06% |

1. **The over number matters.** Time-homogeneity is rejected ($\chi^2 = 3152.89$, df $= 217$). The homogeneous chain has no limit on chase length, and up to 39% of its simulated paths run past 20 overs. The time-inhomogeneous chain fixes this and predicts better (Brier difference −0.0094, 95% CI [−0.0147, −0.0043]).
2. **Logistic regression predicts best.** It beats the time-inhomogeneous chain (Brier difference 0.0187, 95% CI [0.0027, 0.0344]), most likely because it uses exact values rather than 15 buckets. The chains offer interpretable states and can simulate whole chases.
3. **Control for the over when comparing wickets.** Without it, the states wrongly suggest losing wickets can help. At the end of over 10 in `P3`, win probability is 65.1% with 0–2 wickets down and 44.7% with 3–5 down.
4. **The first-order assumption is rejected** ($G^2 = 1440.34$, df $= 274$). The test cannot separate the over number, bucketing and genuine in-match effects as causes.
5. **Scores rose in the Impact Player era, and chain dynamics shifted only modestly.** Mean targets rose from 164.0 to 190.8. This is a before-and-after comparison, not a causal estimate. Transition matrices differ overall (permutation $p = 0.001$), but no single state is significant after correction, and a chain trained before the rule predicts 2026 about as well as one using recent data.
6. **Known limitations:** some time-inhomogeneous predictions are exactly 0 or 1, and 7 rain-shortened matches are treated as 20-over chases, though removing them does not change the model ranking or significance results (see `report/results_summary.md`, Section 9).
