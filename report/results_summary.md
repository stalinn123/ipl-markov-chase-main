# IPL T20 Run Chases: Absorbing Markov Chain Model Results Summary

This summary provides the core empirical and theoretical results from the university stochastic-processes project modeling second-innings run chases in the Indian Premier League (IPL) as an absorbing Markov chain.

---

## 1. Dataset & Sample Filtering

* **Total Raw Match Files Ingested:** 1,243 CSV files (Cricsheet IPL, 2008–2026).
* **Matches Kept:** 1,195 matches (96.1% retention rate).
* **Matches Dropped (48 total):**
  * Abnormal outcomes / abandoned without winner: 25 matches.
  * Interrupted matches decided by Duckworth-Lewis (D/L): 23 matches.
* **Target Cross-Check:** 100% concordance between `info,target_runs` and `(innings_1_runs + 1)`.
* **Overall Empirical Chase Win Rate:** 54.39% (650 wins / 1,195 chases).
* **Train / Test Partitioning:**
  * **Training Set:** Seasons 2008–2022 (913 matches, 16,365 over-to-over transitions).
  * **Test Set:** Seasons 2023–2026 (282 matches, 5,005 over-level observations).

---

## 2. State Space Specification

* **Transient States (15 total):** Cross-product of Wickets Lost bucket $\times$ Match Pressure bucket:
  * **Wickets Buckets:** `W_Low` (0–2 wkts), `W_Mid` (3–5 wkts), `W_High` (6–9 wkts).
  * **Pressure Buckets ($RRR - CRR$):** `P1` ($< -2.0$), `P2` ($-2.0$ to $0.0$), `P3` ($0.0$ to $2.0$), `P4` ($2.0$ to $4.0$), `P5` ($> 4.0$).
* **Absorbing States (2 total):** `WIN` (target reached) and `LOSS` (all out or 20 overs elapsed without reaching target).
* **Sparsity Validation:** Every single state exhibits $\ge 62$ transitions in the training set (exceeding the 30-transition threshold; no bucket collapsing needed).

---

## 3. Analytical Markov Chain Properties

### Absorption Certainty & Fundamental Matrix
* **Spectral Radius of Transient Submatrix $Q$:** $\rho(Q) = 0.904272 < 1.0$.
* **Invertibility:** $(I - Q)$ is strictly invertible with condition number $1.40 \times 10^1$.
* **Conclusion:** Absorption into either `WIN` or `LOSS` occurs almost surely ($P(\text{Absorption}) = 1.0$) from any starting transient state.
* **Initial Distribution Concordance:** Weighting theoretical win probabilities $B$ by the empirical distribution of states after completed over 1 ($\pi_0$) yields:
  $$\pi_0 \cdot B = 54.87\%$$
  which matches the empirical training set chase win rate of **54.87%** with **0.00% discrepancy**.

### Complete Analytical State Table (with 95% Whole-Match Bootstrap CIs)

| State | Win Prob $B_i$ | 95% Bootstrap CI | Expected Overs $t_i$ | 95% Bootstrap CI | Overs Std $\sigma_t$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **W_Low_P1** | 0.7707 | [0.7429, 0.7972] | 14.77 | [14.25, 15.27] | 12.72 |
| **W_Low_P2** | 0.6497 | [0.6153, 0.6799] | 17.85 | [17.56, 18.14] | 12.26 |
| **W_Low_P3** | 0.5666 | [0.5298, 0.5991] | 18.78 | [18.55, 19.01] | 11.90 |
| **W_Low_P4** | 0.4945 | [0.4570, 0.5309] | 18.80 | [18.59, 19.02] | 11.72 |
| **W_Low_P5** | 0.4343 | [0.3969, 0.4688] | 18.44 | [18.20, 18.67] | 11.62 |
| **W_Mid_P1** | 0.9313 | [0.9099, 0.9499] | 4.10 | [3.64, 4.51] | 4.79 |
| **W_Mid_P2** | 0.7415 | [0.6999, 0.7774] | 7.88 | [7.35, 8.41] | 6.89 |
| **W_Mid_P3** | 0.4894 | [0.4435, 0.5322] | 10.40 | [10.01, 10.82] | 7.34 |
| **W_Mid_P4** | 0.3167 | [0.2790, 0.3564] | 10.61 | [10.28, 10.95] | 7.18 |
| **W_Mid_P5** | 0.2030 | [0.1703, 0.2376] | 9.33 | [8.95, 9.70] | 6.97 |
| **W_High_P1** | 0.8699 | [0.7545, 0.9637] | 1.43 | [1.14, 1.78] | 1.26 |
| **W_High_P2** | 0.7771 | [0.6424, 0.8882] | 2.64 | [1.91, 3.41] | 2.56 |
| **W_High_P3** | 0.5035 | [0.3844, 0.6195] | 4.18 | [3.56, 4.81] | 3.17 |
| **W_High_P4** | 0.2124 | [0.1471, 0.2871] | 4.52 | [4.09, 4.98] | 3.26 |
| **W_High_P5** | 0.0637 | [0.0407, 0.0882] | 3.62 | [3.41, 3.85] | 3.09 |

---

## 4. Tipping-Point States & Critical Swings

### Critical Tipping-Point States ($0.40 \le B_i \le 0.60$)
These 5 states represent match states balanced on a knife-edge:
1. **W_Low_P3** ($B_i = 56.66\%$): 0–2 wkts, balanced pressure.
2. **W_Low_P4** ($B_i = 49.45\%$): 0–2 wkts, moderate mounting pressure.
3. **W_Low_P5** ($B_i = 43.43\%$): 0–2 wkts, severe pressure.
4. **W_Mid_P3** ($B_i = 48.94\%$): 3–5 wkts, balanced pressure.
5. **W_High_P3** ($B_i = 50.35\%$): 6–9 wkts, balanced pressure in closing overs.

### Top Probability-Weighted Transition Swings ($P_{ij} \cdot |\Delta B_{ij}|$)
* **W_High_P2 $\to$ WIN** ($P_{ij} = 0.4324, \Delta B = +0.2229$, Weighted Swing = $0.0964$): Decisive boundary/finish in death overs.
* **W_High_P1 $\to$ WIN** ($P_{ij} = 0.6889, \Delta B = +0.1301$, Weighted Swing = $0.0896$): Immediate target resolution.
* **W_High_P1 $\to$ LOSS** ($P_{ij} = 0.0889, \Delta B = -0.8699$, Weighted Swing = $0.0773$): Catastrophic collapse from winning position.
* **W_High_P3 $\to$ W_High_P4** ($P_{ij} = 0.2533, \Delta B = -0.2911$, Weighted Swing = $0.0738$): Maiden/tight over pushing chasing team toward defeat.
* **W_Mid_P2 $\to$ W_Mid_P3** ($P_{ij} = 0.1993, \Delta B = -0.2521$, Weighted Swing = $0.0502$): Momentum shift in middle overs.

### Wicket Degradation Penalty (Holding Pressure Constant)
* **At Low Pressure (P1, P2):** Batting side maintains strong win expectation even with middle wickets down ($> 74\%$).
* **At High Pressure (P4, P5):** Wicket degradation is devastating:
  * At P4: Dropping from Low to High wickets loses **28.22%** win probability ($49.45\% \to 21.24\%$).
  * At P5: Dropping from Low to High wickets loses **37.06%** win probability ($43.43\% \to 6.37\%$).

---

## 5. Monte Carlo Simulation Verification

* **Simulation Setup:** 10,000 independent synthetic trajectories per starting transient state ($150,000$ chains total).
* **Concordance:** Theoretical win probability falls inside the empirical 95% binomial confidence interval for **14 out of 15 states** (with the 15th state within $0.010$ absolute error, fully consistent with standard multiple-hypothesis binomial variance).
* **Expected Overs:** Simulated average match lengths match analytical $t_i$ to within $0.02$ overs across states.

---

## 6. Out-of-Sample Test Set Validation (Seasons 2023–2026)

Evaluated across 5,005 test observations from 282 held-out IPL matches:

| Model | Brier Score | Log-Loss | Binary Accuracy ($\ge 0.5$) |
| :--- | :---: | :---: | :---: |
| **Absorbing Markov Chain** | **0.1711** | **0.5159** | **75.22%** |
| **Logistic Regression Baseline** | 0.1431 | 0.4365 | 79.02% |

### Evolution Across Match Progression
* **Over 1:** Markov Brier = 0.2383, Accuracy = 61.35% (high residual match uncertainty).
* **Over 6 (End of Powerplay):** Markov Brier = 0.2027, Accuracy = 70.92%.
* **Over 10 (Halfway Mark):** Markov Brier = 0.1714, Accuracy = 77.34%.
* **Over 15 (Start of Death):** Markov Brier = 0.1281, Accuracy = 79.17%.
* **Over 19 (Penultimate Over):** Markov Brier = **0.0813**, Accuracy = **90.38%** (outperforming Logistic Regression's 89.10%).

---

## 7. Statistical Tests of Theoretical Assumptions

### 1. Time-Homogeneity Test (Powerplay vs Middle vs Death)
* **Contingency Chi-Square:** Global $\chi^2 = 3158.16$, $\text{df} = 217$, $p < 10^{-16}$.
* **Plain-English Finding:** Strict time-homogeneity is rejected. Transition probabilities vary across phases due to physical cricket constraints: Powerplay promotes low-risk scoring under field restrictions, while death overs drastically increase direct absorption rates into WIN (boundaries) or LOSS (slogging wickets).

### 2. Markov Property Test (Memorylessness: Order 1 vs Order 2)
* **Likelihood-Ratio Statistic:** $G^2 = 1445.27$, $\text{df} = 190$, $p < 10^{-16}$.
* **Plain-English Finding:** Strict memorylessness is rejected in favor of second-order dependency. Match momentum, bowler spell rhythms, and recent wicket shocks create short-term autocorrelation. Nevertheless, the state features (wickets, CRR, RRR, pressure) encode sufficient cumulative history that the first-order Markov model achieves high out-of-sample accuracy (75.2%) without suffering from the parameter explosion or overfitting of higher-order chains.
