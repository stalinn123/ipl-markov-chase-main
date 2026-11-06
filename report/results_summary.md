# IPL T20 Run Chases: Absorbing Markov Chain Results Summary

Results from the project modelling second-innings IPL run chases as an absorbing Markov chain. All numbers come from the outputs in `data/processed/` and the logs in `report/run_logs/`. Two models are compared with a logistic regression baseline:

* **Homogeneous chain (HOM):** one transition matrix for all overs (scripts 03–06).
* **Time-inhomogeneous chain (INH):** a separate matrix for each over (script 06b).

---

## 1. Dataset & Sample Filtering

* **Source:** Cricsheet IPL ball-by-ball CSV files, 2008–2026 (1,243 match files).
* **Matches kept:** 1,195 (96.1%).
* **Matches dropped (48):** 25 with no winner and 23 decided by Duckworth-Lewis.
* **Target cross-check:** `info,target_runs` equals `innings_1_runs + 1` in every kept match.
* **Wicket cross-check:** final wickets match the official winning margin in all 650 winning chases. Retired hurt is not counted as a wicket.
* **Overall chase win rate:** 54.39% (650 of 1,195).
* **Training set:** seasons 2008–2022 (913 matches, 16,365 transitions; win rate 54.87%).
* **Test set:** seasons 2023–2026 (282 matches, 5,005 over-level observations).

---

## 2. State Space

* **Transient states (15):** wickets bucket × pressure bucket.
  * Wickets: `W_Low` (0–2), `W_Mid` (3–5), `W_High` (6–9).
  * Pressure ($RRR - CRR$): `P1` ($< -2$), `P2` ($-2$ to $0$), `P3` ($0$ to $2$), `P4` ($2$ to $4$), `P5` ($\ge 4$).
* **Absorbing states (2):** `WIN` (target reached) and `LOSS` (all out, or overs used up without reaching the target; see Section 9 on shortened matches).
* **Sparsity:** every state has at least 36 outgoing transitions in the training set (61 across all seasons), above the threshold of 30. Individual state and over combinations are much sparser (see Section 8).

---

## 3. Homogeneous Chain: Analytical Results

* **Spectral radius:** $\rho(Q) = 0.904571 < 1$, so absorption is certain.
* **Condition number of $I - Q$:** $1.40 \times 10^1$.
* **Stationary distribution:** not unique. Eigenvalue 1 of $P$ has multiplicity 2, and every $(0, \dots, 0, a, 1-a)$ is stationary. The limit of $\pi_0 P^n$ depends on the starting distribution (script 10).
* **$\pi_0 \cdot B = 54.87\%$:** this equals the training win rate by construction, because the count-based estimate conserves flow. It checks the code. It is not evidence that the model is valid.

| State | Win Prob $B_i$ | 95% Bootstrap CI | Expected Overs $t_i$ | 95% Bootstrap CI | Overs Std |
| :--- | :---: | :---: | :---: | :---: | :---: |
| W_Low_P1 | 0.7713 | [0.7435, 0.7981] | 14.75 | [14.23, 15.26] | 12.74 |
| W_Low_P2 | 0.6501 | [0.6156, 0.6803] | 17.85 | [17.55, 18.13] | 12.28 |
| W_Low_P3 | 0.5669 | [0.5301, 0.5994] | 18.79 | [18.55, 19.02] | 11.92 |
| W_Low_P4 | 0.4945 | [0.4567, 0.5309] | 18.82 | [18.61, 19.03] | 11.74 |
| W_Low_P5 | 0.4338 | [0.3966, 0.4682] | 18.44 | [18.20, 18.66] | 11.64 |
| W_Mid_P1 | 0.9305 | [0.9087, 0.9492] | 4.09 | [3.64, 4.51] | 4.79 |
| W_Mid_P2 | 0.7410 | [0.6995, 0.7777] | 7.86 | [7.35, 8.39] | 6.88 |
| W_Mid_P3 | 0.4893 | [0.4437, 0.5316] | 10.40 | [9.99, 10.81] | 7.33 |
| W_Mid_P4 | 0.3175 | [0.2800, 0.3574] | 10.59 | [10.26, 10.94] | 7.16 |
| W_Mid_P5 | 0.2020 | [0.1695, 0.2365] | 9.28 | [8.90, 9.66] | 6.94 |
| W_High_P1 | 0.8692 | [0.7539, 0.9634] | 1.44 | [1.14, 1.79] | 1.27 |
| W_High_P2 | 0.7680 | [0.6299, 0.8843] | 2.70 | [1.97, 3.47] | 2.59 |
| W_High_P3 | 0.4939 | [0.3763, 0.6090] | 4.21 | [3.58, 4.83] | 3.19 |
| W_High_P4 | 0.2074 | [0.1404, 0.2810] | 4.52 | [4.08, 4.98] | 3.27 |
| W_High_P5 | 0.0634 | [0.0407, 0.0876] | 3.63 | [3.41, 3.85] | 3.10 |

**Main limitation:** this chain does not know how many overs are left.
* Expected overs from `W_Low` states are about 15 to 19, even though these states occur on average around overs 4 to 8.
* Up to 39% of simulated paths (from `W_Low_P3`) run past the 20-over limit.
* By over 20 the chain has resolved only 64.2% of chases, where the true figure is 100%.

### Tipping points and swings (homogeneous chain)
These do not account for the over number, so treat them as averages across the innings.

* **States with $0.40 \le B_i \le 0.60$:** `W_Low_P3` (56.69%), `W_Low_P4` (49.45%), `W_Low_P5` (43.38%), `W_Mid_P3` (48.93%), `W_High_P3` (49.39%).
* **Largest probability-weighted swings ($P_{ij} \cdot |\Delta B_{ij}|$):**
  * `W_High_P2` → `WIN`: $P_{ij} = 0.4167$, $\Delta B = +0.2320$, weighted 0.0967.
  * `W_High_P1` → `WIN`: $P_{ij} = 0.6889$, $\Delta B = +0.1308$, weighted 0.0901.
  * `W_High_P1` → `LOSS`: $P_{ij} = 0.0889$, $\Delta B = -0.8692$, weighted 0.0773.
  * `W_High_P3` → `W_High_P4`: $P_{ij} = 0.2568$, $\Delta B = -0.2864$, weighted 0.0735.
  * `W_High_P4` → `W_High_P5`: $P_{ij} = 0.4085$, $\Delta B = -0.1441$, weighted 0.0588.

---

## 4. Effect of Wickets

`wicket_penalty_table.csv` compares homogeneous $B_i$ at fixed pressure and gives a misleading result: at `P1` and `P2`, win probability appears to **rise** as wickets fall (at `P1`, from 77.1% at `W_Low` to 93.1% at `W_Mid`). This is a timing effect. `W_Low_P1` occurs around over 8 on average and `W_Mid_P1` around over 15, and later states are closer to the finish.

Holding the over fixed removes the effect. INH win probabilities $V_k(s)$:

| End of over 10 | P1 | P2 | P3 | P4 | P5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| W_Low | 0.938 | 0.800 | 0.651 | 0.471 | 0.302 |
| W_Mid | 0.901 | 0.693 | 0.447 | 0.316 | 0.219 |
| Drop | 0.037 | 0.107 | 0.204 | 0.155 | 0.083 |

| End of over 15 | P1 | P2 | P3 | P4 | P5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| W_Low | 0.985 | 0.919 | 0.775 | 0.585 | 0.393 |
| W_Mid | 0.961 | 0.867 | 0.661 | 0.405 | 0.249 |
| Drop | 0.024 | 0.052 | 0.114 | 0.180 | 0.144 |

At these overs, losing wickets lowers win probability at every pressure level, and the drop is smallest when the team is well ahead (`P1`). Some cells rest on few training observations (for example 10 for `W_Low_P5` at over 15), and these values are smoothed. `W_High` is left out because it has very few observations at over 10 and at low pressure at over 15.

---

## 5. Monte Carlo Simulation

* **Setup:** 10,000 simulated chains per starting state (150,000 total).
* **Result:** the analytical $B_i$ falls inside the simulation's 95% interval for all 15 states. The largest differences are 0.0095 in win probability and 0.20 overs in expected length.
* **What this shows:** the formulas and the code agree. The simulation samples from the same $P$, so it says nothing about how well the model fits real chases. Model fit is tested in Sections 6 and 8.

---

## 6. Time-Inhomogeneous Chain

Each over $k = 1, \dots, 19$ has its own matrix $P_k$, estimated on training seasons only and smoothed toward nearby overs (Dirichlet prior, $\alpha = 10$; fixed, not tuned). Win probability comes from backward recursion:

$$V_{19}(s) = P_{19}(s, \text{WIN}), \qquad V_k(s) = P_k(s, \text{WIN}) + \sum_j Q_k(s, j)\, V_{k+1}(j)$$

Chases must end by over 20, so expected overs can never exceed the overs left.

**P(chase resolved by end of over $k$), training seasons:**

| Over | Actual | Homogeneous | Inhomogeneous |
| :---: | :---: | :---: | :---: |
| 5 | 0.000 | 0.072 | 0.000 |
| 10 | 0.012 | 0.261 | 0.011 |
| 15 | 0.073 | 0.470 | 0.074 |
| 19 | 0.400 | 0.611 | 0.411 |
| 20 | 1.000 | 0.642 | 1.000 |

**Out-of-sample results, seasons 2023–2026 (282 matches, 5,005 observations):**

| Model | Brier Score | Log-Loss | Accuracy |
| :--- | :---: | :---: | :---: |
| Homogeneous chain | 0.1712 | 0.5161 | 75.60% |
| Time-inhomogeneous chain | 0.1618 | 0.4951 | 76.12% |
| Logistic regression | 0.1431 | 0.4365 | 79.06% |

Differences with 95% match-level bootstrap intervals:
* INH − HOM Brier: −0.0094 [−0.0147, −0.0043], so INH is better than HOM.
* INH − LR Brier: +0.0187 [+0.0027, +0.0344], so logistic regression is better than INH. The run log prints the bootstrap mean, +0.0189.

Logistic regression has more information, since it uses exact runs, balls and wickets rather than 15 buckets. The Markov chains offer interpretable states and transitions, and can simulate whole chases, which the regression cannot.

**Brier score by over:**

| Over | HOM | INH | LR |
| :---: | :---: | :---: | :---: |
| 1 | 0.2383 | 0.2399 | 0.2312 |
| 6 | 0.2026 | 0.1890 | 0.1603 |
| 10 | 0.1718 | 0.1593 | 0.1409 |
| 15 | 0.1279 | 0.1158 | 0.0972 |
| 19 | 0.0818 | 0.0893 | 0.0749 |

---

## 7. Tests of Model Assumptions

### Time-homogeneity (Powerplay vs Middle vs Death)
* $\chi^2 = 3152.89$, df $= 217$, $p < 10^{-300}$. Rejected.
* Transition probabilities change through the innings, which is the reason for the time-inhomogeneous chain.

### Markov property (first order vs second order)
* $G^2 = 1440.34$, df $= 274$, $p \approx 2 \times 10^{-157}$. Rejected.
* The previous state carries extra information, but the test cannot say why. Possible causes are the over number (the test pools all overs), bucketing (states merge different scores and wicket counts), and genuine in-match effects such as momentum.
* Some (previous, current) state pairs are rare, so the p-value is approximate. No second-order chain was fitted, so this does not show one would predict better.

---

## 8. Impact Player Era (rule in force from 2023)

**Scoring changed:** mean target rose from 164.0 (2008–22) to 190.8 (2023–26), and targets of 200+ rose from 10.6% to 44.0% of chases.

**Transition matrices changed, modestly:**
* Global test, pre vs post: $G = 189.95$, df $= 121$, permutation $p = 0.001$. With phases stratified: $G = 297.20$, df $= 220$, permutation $p = 0.006$.
* No single state's row is significant after Holm correction, so the change is spread across states rather than concentrated in a few.
* Homogeneous $B_i$ differs between eras in `W_Mid_P2` (−0.121, CI [−0.218, −0.031]) and `W_Mid_P4` (−0.076, CI [−0.149, −0.004]). These intervals are not corrected for checking 15 states.

**Predicting 2026 (71 matches), Brier score by training data:**

| Training data | HOM | INH | LR |
| :--- | :---: | :---: | :---: |
| Pre-rule 2008–22 | 0.1533 | 0.1292 | 0.1345 |
| Impact-only 2023–25 | 0.1630 | 0.1360 | 0.1152 |
| All 2008–25 | 0.1544 | 0.1301 | 0.1275 |
| Empirical Bayes (pre-rule prior + 2023–25) | 0.1543 | 0.1289 | n/a |

* Logistic regression gains from recent data: pre-rule minus Impact-only is +0.0193 [0.0105, 0.0281].
* The INH chain does not: pre-rule minus Impact-only is −0.0068 [−0.0200, 0.0052], and pre-rule minus empirical Bayes is +0.0003 [−0.0014, 0.0018].
* INH (pre-rule) vs LR (Impact-only) on 2026 is +0.0140 [−0.0128, 0.0383], so that difference is not significant. 2026 is only 71 matches.

**Season-by-season INH Brier (pre-rule model):** 2023 0.1732, 2024 0.1688, 2025 0.1757, 2026 0.1292. The pre-rule seasons, each left out in turn, range from 0.1346 to 0.1914 (mean 0.1663). So 2023–25 sit inside the normal range, on the worse side, and 2026 is below it.

---

## 9. Limitations

1. **Coarse states:** 15 buckets discard exact runs, balls and wicket counts, which is the likely reason logistic regression predicts better.
2. **Exact predictions:** 27 of the 156 test observations at over 19 receive a probability of exactly 0 or 1, because every training chase in those states won (`W_Mid_P1` 61 of 61, `W_Mid_P2` 26 of 26, `W_High_P2` 13 of 13). Only 1 of the 27 was wrong. Another 10 test observations at overs 8 to 14, all `W_High_P1`, rest on very few chases, and 3 were wrong. Over 19 is the INH chain's clearest loss to HOM (0.0893 vs 0.0818), and the cause was not isolated.
3. **Second-order dependence:** rejected first-order assumption (Section 7).
4. **Not modelled:** team strength, batters at the crease, bowlers remaining, venue and dew.
5. **Dropped and shortened matches:** Duckworth-Lewis, tied (16) and no-result (9) matches are excluded. Seven rain-shortened matches with no recorded method (336010, 598068, 733971, 829771, 829803, 1426298, 1527686) remain and are treated as 20-over chases. Re-running scripts 01 to 06b without them gives Brier scores of 0.1689 (HOM), 0.1603 (INH) and 0.1410 (LR), with the same ranking and significance conclusions.
6. **Observations within a match are not independent:** intervals therefore resample whole matches.
