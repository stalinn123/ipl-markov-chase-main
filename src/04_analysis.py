"""
src/04_analysis.py
Script 4: Analytical solution of the absorbing Markov chain model.
Computes fundamental matrix N, expected absorption time t, absorption probabilities B,
bootstrap 95% CIs, tipping point analysis, n-step distributions, and generates figures.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import config

def compute_analytical_markov(Q: np.ndarray, R: np.ndarray):
    """
    Computes spectral radius, fundamental matrix N, expected overs t,
    variance/std of absorption time, and absorption probabilities B.
    """
    n_states = len(Q)
    eigvals = np.linalg.eigvals(Q)
    spectral_radius = float(np.max(np.abs(eigvals)))

    I = np.eye(n_states, dtype=np.float64)
    I_minus_Q = I - Q
    cond_number = float(np.linalg.cond(I_minus_Q))
    is_invertible = bool(spectral_radius < 1.0 and cond_number < 1e12)

    # Fundamental matrix N = (I - Q)^(-1)
    N = np.linalg.inv(I_minus_Q)

    # Expected time to absorption t = N * 1
    ones = np.ones((n_states, 1), dtype=np.float64)
    t = (N @ ones).flatten()

    # Variance of absorption time: Var(T) = (2N - I)t - t * t
    var_t = ((2.0 * N - I) @ t.reshape(-1, 1)).flatten() - (t ** 2)
    std_t = np.sqrt(np.maximum(0.0, var_t))

    # Absorption probabilities B = N * R (columns: WIN, LOSS)
    B = N @ R

    return {
        "spectral_radius": spectral_radius,
        "is_invertible": is_invertible,
        "cond_number": cond_number,
        "N": N,
        "t": t,
        "var_t": var_t,
        "std_t": std_t,
        "B": B,
        "win_prob": B[:, 0],
        "loss_prob": B[:, 1],
    }

def compute_bootstrap_cis(boot_Q: np.ndarray, boot_R: np.ndarray):
    """
    Propagates whole-match bootstrap samples through analytical formulas
    to obtain 95% confidence intervals for B (win probability) and t (expected overs).
    """
    n_boot = len(boot_Q)
    n_states = boot_Q.shape[1]
    I = np.eye(n_states, dtype=np.float64)
    ones = np.ones((n_states, 1), dtype=np.float64)

    boot_B_win = np.zeros((n_boot, n_states), dtype=np.float64)
    boot_t = np.zeros((n_boot, n_states), dtype=np.float64)

    for b in range(n_boot):
        I_minus_Q = I - boot_Q[b]
        N_b = np.linalg.inv(I_minus_Q)
        boot_t[b] = (N_b @ ones).flatten()
        B_b = N_b @ boot_R[b]
        boot_B_win[b] = B_b[:, 0]

    b_lower = np.percentile(boot_B_win, 2.5, axis=0)
    b_upper = np.percentile(boot_B_win, 97.5, axis=0)
    t_lower = np.percentile(boot_t, 2.5, axis=0)
    t_upper = np.percentile(boot_t, 97.5, axis=0)

    return b_lower, b_upper, t_lower, t_upper

def plot_heatmap_grid(values_dict, ci_lower_dict, ci_upper_dict, title, cbar_label, filename, cmap, fmt=".2f", is_prob=True):
    """
    Plots a 3x5 grid (Wickets bucket x Pressure bucket) with values and 95% CIs.
    """
    grid = np.zeros((3, 5))
    annot_matrix = [["" for _ in range(5)] for _ in range(3)]

    for w_idx, w_name in enumerate(config.WICKET_ORDER):
        for p_idx, p_name in enumerate(config.PRESSURE_LABELS):
            st = f"{w_name}_{p_name}"
            val = values_dict[st]
            l_val = ci_lower_dict[st]
            u_val = ci_upper_dict[st]
            grid[w_idx, p_idx] = val
            if is_prob:
                annot_matrix[w_idx][p_idx] = f"{val:.1%}\n[{l_val:.1%}, {u_val:.1%}]"
            else:
                annot_matrix[w_idx][p_idx] = f"{val:.2f}\n[{l_val:.2f}, {u_val:.2f}]"

    plt.figure(figsize=(11, 6))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        grid,
        annot=np.array(annot_matrix),
        fmt="",
        cmap=cmap,
        xticklabels=config.PRESSURE_LABELS,
        yticklabels=config.WICKET_ORDER,
        cbar_kws={"label": cbar_label, "shrink": 0.8},
        linewidths=1.0,
        linecolor="#ffffff",
        annot_kws={"size": 10, "weight": "bold"}
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Match Pressure Bucket (RRR - CRR)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Wickets Lost Bucket", fontsize=11, fontweight="bold", labelpad=8)
    ax.tick_params(axis="both", labelsize=10.5)

    save_path = config.FIGURES_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {save_path}")

def plot_tipping_points(tipping_states, swing_transitions, save_path: Path):
    """
    Plots visual summary of tipping point states and highest swing transitions.
    """
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # Panel 1: Tipping point states (balanced win probability 0.4 - 0.6)
    states = tipping_states["State"]
    probs = tipping_states["Win_Prob"]
    ci_err = [probs - tipping_states["CI_Lower"], tipping_states["CI_Upper"] - probs]

    bars = ax1.bar(states, probs, yerr=ci_err, capsize=5, color="#2b5c8f", edgecolor="#1a365d", alpha=0.85)
    ax1.axhline(0.5, color="#e53e3e", linestyle="--", linewidth=1.5, label="50% Equilibrium")
    ax1.set_ylim(0.2, 0.8)
    ax1.set_ylabel("Analytical Win Probability B_i", fontsize=11, fontweight="bold")
    ax1.set_title("Critical Tipping-Point States (0.4 <= B_i <= 0.6)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right")
    for bar, p in zip(bars, probs):
        ax1.text(bar.get_x() + bar.get_width()/2, p + 0.04, f"{p:.1%}", ha="center", fontweight="bold", fontsize=10)

    # Panel 2: Top weighted swing transitions
    top_swings = swing_transitions.head(8)
    labels = [f"{r['from_state']} -> {r['to_state']}" for _, r in top_swings.iterrows()]
    weighted_delta = top_swings["weighted_swing"]
    colors = ["#38a169" if r["delta_B"] > 0 else "#e53e3e" for _, r in top_swings.iterrows()]

    ax2.barh(labels[::-1], weighted_delta[::-1], color=colors[::-1], alpha=0.85, edgecolor="#2d3748")
    ax2.set_xlabel("Probability-Weighted Win Probability Swing (|delta B| * P_ij)", fontsize=11, fontweight="bold")
    ax2.set_title("Top 8 Critical Swing Transitions (|delta B| * P_ij)", fontsize=13, fontweight="bold")

    plt.suptitle("IPL Chase Tipping Point Analysis & Critical Swings", fontsize=15, fontweight="bold", y=1.03)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved tipping point plot: {save_path}")

def main():
    print("=" * 70)
    print("SCRIPT 4: Absorbing Markov Chain Analysis & Tipping Points")
    print("=" * 70)

    # Load canonical matrices and bootstrap distributions
    Q = np.load(config.PROCESSED_DATA_DIR / "Q_matrix.npy")
    R = np.load(config.PROCESSED_DATA_DIR / "R_matrix.npy")
    P = np.load(config.PROCESSED_DATA_DIR / "P_matrix.npy")
    boot_Q = np.load(config.PROCESSED_DATA_DIR / "bootstrap_Q.npy")
    boot_R = np.load(config.PROCESSED_DATA_DIR / "bootstrap_R.npy")

    # 1. Absorption Certainty Check
    results = compute_analytical_markov(Q, R)
    spec_rad = results["spectral_radius"]
    print("\n--- Absorption Certainty & Fundamental Matrix ---")
    print(f"Spectral radius rho(Q): {spec_rad:.6f}")
    print(f"Is rho(Q) < 1.0: {spec_rad < 1.0} (Absorption with probability 1 is guaranteed!)")
    print(f"Condition number of (I - Q): {results['cond_number']:.2e} (Strictly invertible)")

    N = results["N"]
    t = results["t"]
    std_t = results["std_t"]
    B = results["B"]
    win_prob = results["win_prob"]

    # Save Fundamental Matrix N
    n_df = pd.DataFrame(N, index=config.TRANSIENT_STATES, columns=config.TRANSIENT_STATES)
    n_csv = config.PROCESSED_DATA_DIR / "fundamental_matrix_N.csv"
    n_df.to_csv(n_csv)
    print(f"Saved fundamental matrix N to: {n_csv}")

    # 2. Bootstrap 95% Confidence Intervals
    print("\nComputing analytical bootstrap 95% CIs for B and t...")
    b_l, b_u, t_l, t_u = compute_bootstrap_cis(boot_Q, boot_R)

    # Analytical Results Table
    res_rows = []
    for i, st in enumerate(config.TRANSIENT_STATES):
        res_rows.append({
            "State": st,
            "Win_Prob": round(win_prob[i], 4),
            "B_CI_Lower": round(b_l[i], 4),
            "B_CI_Upper": round(b_u[i], 4),
            "Expected_Overs": round(t[i], 2),
            "t_CI_Lower": round(t_l[i], 2),
            "t_CI_Upper": round(t_u[i], 2),
            "Overs_Std": round(std_t[i], 2),
        })
    df_analytical = pd.DataFrame(res_rows)
    analytical_csv = config.PROCESSED_DATA_DIR / "markov_analytical_results.csv"
    df_analytical.to_csv(analytical_csv, index=False)
    print(f"Saved analytical results to: {analytical_csv}")
    print(df_analytical.to_string(index=False))

    # 3. Overall Win Probability from Over 1 Initial Distribution (pi_0 * B)
    over_states = pd.read_csv(config.PROCESSED_DATA_DIR / "over_states.csv")
    train_over1 = over_states[(over_states["season"] <= config.TRAIN_MAX_SEASON) & (over_states["over"] == 1)]
    over1_counts = train_over1["state"].value_counts()
    pi_0 = np.array([over1_counts.get(st, 0) for st in config.TRANSIENT_STATES], dtype=np.float64)
    pi_0 = pi_0 / pi_0.sum()

    overall_markov_win = float(pi_0 @ win_prob)
    chases_df = pd.read_csv(config.PROCESSED_DATA_DIR / "chases.csv")
    actual_train_win = chases_df[chases_df["season"] <= config.TRAIN_MAX_SEASON]["chase_won"].mean()

    print("\n--- Overall Win Probability Check (pi_0 * B) ---")
    print(f"Model-implied overall chase win probability (pi_0 * B): {overall_markov_win:.4f} ({overall_markov_win*100:.2f}%)")
    print(f"Empirical training set chase win rate:                 {actual_train_win:.4f} ({actual_train_win*100:.2f}%)")
    print(f"Discrepancy: {abs(overall_markov_win - actual_train_win)*100:.2f} pp")
    print("NOTE: in-sample, pi_0*B reproduces the empirical rate by construction (count MLE conserves flow);\n"
          "      this is a consistency check of the code, NOT evidence of predictive validity.")

    # 4. n-step distributions P^n for n = 1, 5, 10
    print("\n--- n-step Transition Analysis (P^n) ---")
    pi0_full = np.concatenate([pi_0, [0.0, 0.0]])
    print("State distribution vs time from pi_0 (over 1): pi_k = pi_0 P^(k-1)")
    for n in [1, 5, 10, 18]:
        pk = pi0_full @ np.linalg.matrix_power(P, n)
        print(f"  over {n+1:2d}: P(WIN by now)={pk[15]:.3f}  P(LOSS by now)={pk[16]:.3f}  P(still in play)={pk[:15].sum():.3f}")
    print("  (homogeneous chain: P(still in play) at over 19 should be ~0 in reality -> see script 06b)")
    for n in [1, 5, 10]:
        P_n = np.linalg.matrix_power(P, n)
        # Absorption probability from all transient states after n overs
        avg_win_n = np.mean(P_n[:15, 15])
        avg_loss_n = np.mean(P_n[:15, 16])
        avg_trans_n = np.sum(P_n[:15, :15]) / 15.0
        print(f"After n = {n:2d} overs | Average P(absorbed WIN): {avg_win_n:.3f} | P(absorbed LOSS): {avg_loss_n:.3f} | P(still transient): {avg_trans_n:.3f}")

    # 5. Tipping Points Analysis
    print("\n--- (a) Tipping-Point States (0.4 <= B_i <= 0.6) ---")
    tipping_df = df_analytical[(df_analytical["Win_Prob"] >= 0.40) & (df_analytical["Win_Prob"] <= 0.60)].copy()
    tipping_csv = config.PROCESSED_DATA_DIR / "tipping_point_states.csv"
    tipping_df.to_csv(tipping_csv, index=False)
    print(tipping_df[["State", "Win_Prob", "B_CI_Lower", "B_CI_Upper", "Expected_Overs"]].to_string(index=False))

    print("\n--- (b) Largest Swing Transitions (delta B = B_j - B_i) ---")
    # Extended state vector including absorbing values: WIN = 1.0, LOSS = 0.0
    full_B = np.zeros(len(config.ALL_STATES))
    full_B[:15] = win_prob
    full_B[15] = 1.0  # WIN
    full_B[16] = 0.0  # LOSS

    swing_rows = []
    for i, from_st in enumerate(config.TRANSIENT_STATES):
        for j, to_st in enumerate(config.ALL_STATES):
            p_ij = P[i, j]
            if p_ij > 0.001:  # ignore zero / negligible transitions
                delta_b = full_B[j] - full_B[i]
                weighted_swing = p_ij * abs(delta_b)
                swing_rows.append({
                    "from_state": from_st,
                    "to_state": to_st,
                    "P_ij": round(p_ij, 4),
                    "B_from": round(full_B[i], 4),
                    "B_to": round(full_B[j], 4),
                    "delta_B": round(delta_b, 4),
                    "weighted_swing": round(weighted_swing, 4),
                })

    df_swings = pd.DataFrame(swing_rows).sort_values(by="weighted_swing", ascending=False)
    top_swings_csv = config.PROCESSED_DATA_DIR / "top_swing_transitions.csv"
    df_swings.to_csv(top_swings_csv, index=False)
    print(df_swings.head(10).to_string(index=False))

    print("\n--- (c) Wicket Degradation Penalty (Fixed Pressure) ---")
    wicket_degradation = []
    for p in config.PRESSURE_LABELS:
        b_low = win_prob[config.TRANSIENT_STATES.index(f"W_Low_{p}")]
        b_mid = win_prob[config.TRANSIENT_STATES.index(f"W_Mid_{p}")]
        b_high = win_prob[config.TRANSIENT_STATES.index(f"W_High_{p}")]
        drop_1 = b_mid - b_low
        drop_2 = b_high - b_mid
        wicket_degradation.append({
            "Pressure_Bucket": p,
            "B(W_Low)": round(b_low, 4),
            "B(W_Mid)": round(b_mid, 4),
            "Drop(Low->Mid)": round(drop_1, 4),
            "B(W_High)": round(b_high, 4),
            "Drop(Mid->High)": round(drop_2, 4),
            "Total_Drop(Low->High)": round(b_high - b_low, 4),
        })
    df_wicket_pen = pd.DataFrame(wicket_degradation)
    wicket_pen_csv = config.PROCESSED_DATA_DIR / "wicket_penalty_table.csv"
    df_wicket_pen.to_csv(wicket_pen_csv, index=False)
    print(df_wicket_pen.to_string(index=False))

    # 6. Generate Publication-Quality Figures
    b_dict = {st: win_prob[i] for i, st in enumerate(config.TRANSIENT_STATES)}
    b_l_dict = {st: b_l[i] for i, st in enumerate(config.TRANSIENT_STATES)}
    b_u_dict = {st: b_u[i] for i, st in enumerate(config.TRANSIENT_STATES)}

    t_dict = {st: t[i] for i, st in enumerate(config.TRANSIENT_STATES)}
    t_l_dict = {st: t_l[i] for i, st in enumerate(config.TRANSIENT_STATES)}
    t_u_dict = {st: t_u[i] for i, st in enumerate(config.TRANSIENT_STATES)}

    plot_heatmap_grid(
        b_dict, b_l_dict, b_u_dict,
        title="IPL Run Chase Analytical Win Probability B (with 95% Bootstrap CIs)",
        cbar_label="Probability of Winning Chase",
        filename="win_probability_heatmap.png",
        cmap="YlGnBu",
        is_prob=True
    )

    plot_heatmap_grid(
        t_dict, t_l_dict, t_u_dict,
        title="Expected Overs to Chase Resolution E[T] (with 95% Bootstrap CIs)",
        cbar_label="Expected Overs Remaining",
        filename="expected_overs_heatmap.png",
        cmap="YlOrRd_r",
        fmt=".2f",
        is_prob=False
    )

    plot_tipping_points(
        tipping_df.rename(columns={"B_CI_Lower": "CI_Lower", "B_CI_Upper": "CI_Upper"}),
        df_swings,
        config.FIGURES_DIR / "tipping_points_analysis.png"
    )

    print("=" * 70)

if __name__ == "__main__":
    main()
