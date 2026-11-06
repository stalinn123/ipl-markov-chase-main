"""
src/03_estimate_P.py
Script 3: Estimate transition probability matrix P in canonical form [Q, R; 0, I]
using training seasons (<= 2022). Compute 95% bootstrap confidence intervals
via whole-match resampling, save matrices, and plot heatmaps.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import config

def estimate_transition_matrix(trans_df: pd.DataFrame):
    """
    Computes empirical transition count matrix and normalized probabilities.
    Returns:
        P (17x17), Q (15x15), R (15x2), counts_matrix (15x17)
    """
    n_transient = len(config.TRANSIENT_STATES)
    n_all = len(config.ALL_STATES)

    st_to_idx = {st: i for i, st in enumerate(config.ALL_STATES)}
    counts = np.zeros((n_transient, n_all), dtype=np.float64)

    for _, row in trans_df.iterrows():
        f_st = row["from_state"]
        t_st = row["to_state"]
        if f_st in st_to_idx and t_st in st_to_idx:
            counts[st_to_idx[f_st], st_to_idx[t_st]] += 1.0

    # Row normalize for transient states
    row_sums = counts.sum(axis=1, keepdims=True)
    # Avoid division by zero if any row is empty
    safe_sums = np.where(row_sums == 0, 1.0, row_sums)
    P_transient = counts / safe_sums

    # Construct canonical full P (17 x 17)
    # [ Q   R ]
    # [ 0   I ]
    P = np.zeros((n_all, n_all), dtype=np.float64)
    P[:n_transient, :] = P_transient
    # Absorbing states have self loops of 1
    P[n_transient:, n_transient:] = np.eye(len(config.ABSORBING_STATES))

    Q = P[:n_transient, :n_transient]
    R = P[:n_transient, n_transient:]

    return P, Q, R, counts

def run_match_bootstrap(train_trans: pd.DataFrame, n_boot: int = 1000, random_seed: int = 42):
    """
    Performs whole-match resampling with replacement to compute 95% CIs for P.
    Returns:
        boot_Q (n_boot, 15, 15), boot_R (n_boot, 15, 2), ci_df
    """
    np.random.seed(random_seed)
    match_ids = train_trans["match_id"].unique()
    n_matches = len(match_ids)
    n_transient = len(config.TRANSIENT_STATES)
    n_all = len(config.ALL_STATES)

    mid_to_idx = {mid: i for i, mid in enumerate(match_ids)}
    st_to_idx = {st: i for i, st in enumerate(config.ALL_STATES)}

    # Pre-aggregate transition counts per match into a 3D tensor (n_matches, 15, 17)
    match_tensor = np.zeros((n_matches, n_transient, n_all), dtype=np.float64)
    for _, r in train_trans.iterrows():
        m_i = mid_to_idx[r["match_id"]]
        f_i = st_to_idx[r["from_state"]]
        t_i = st_to_idx[r["to_state"]]
        match_tensor[m_i, f_i, t_i] += 1.0

    boot_Q = np.zeros((n_boot, n_transient, n_transient), dtype=np.float64)
    boot_R = np.zeros((n_boot, n_transient, len(config.ABSORBING_STATES)), dtype=np.float64)
    boot_P_trans = np.zeros((n_boot, n_transient, n_all), dtype=np.float64)

    for b in range(n_boot):
        sampled_indices = np.random.choice(n_matches, size=n_matches, replace=True)
        sample_counts = match_tensor[sampled_indices].sum(axis=0)
        row_sums = sample_counts.sum(axis=1, keepdims=True)
        safe_sums = np.where(row_sums == 0, 1.0, row_sums)
        prob_matrix = sample_counts / safe_sums

        boot_P_trans[b] = prob_matrix
        boot_Q[b] = prob_matrix[:, :n_transient]
        boot_R[b] = prob_matrix[:, n_transient:]

    # Compute 95% Confidence Intervals (2.5% to 97.5%)
    lower_bounds = np.percentile(boot_P_trans, 2.5, axis=0)
    upper_bounds = np.percentile(boot_P_trans, 97.5, axis=0)

    return boot_Q, boot_R, lower_bounds, upper_bounds

def plot_transition_matrices(Q: np.ndarray, R: np.ndarray, save_path: Path):
    """
    Plots high-quality heatmap visualization for Q and R matrices.
    """
    sns.set_theme(style="white")
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 9),
        gridspec_kw={"width_ratios": [15, 2.5]},
        constrained_layout=True
    )

    cmap = "Blues"

    # Heatmap for Q (Transient to Transient)
    sns.heatmap(
        Q,
        ax=ax1,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=0.0,
        vmax=0.6,
        xticklabels=config.TRANSIENT_STATES,
        yticklabels=config.TRANSIENT_STATES,
        cbar_kws={"label": "Transition Probability", "shrink": 0.8},
        annot_kws={"size": 7.5},
        linewidths=0.5,
        linecolor="#f0f0f0"
    )
    ax1.set_title("Transient Transition Matrix Q (15 x 15)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Next State (Over k+1)", fontsize=11, labelpad=8)
    ax1.set_ylabel("Current State (Over k)", fontsize=11, labelpad=8)
    ax1.tick_params(axis="x", rotation=45, labelsize=8.5)
    ax1.tick_params(axis="y", rotation=0, labelsize=8.5)

    # Heatmap for R (Transient to Absorbing)
    sns.heatmap(
        R,
        ax=ax2,
        annot=True,
        fmt=".2f",
        cmap="Oranges",
        vmin=0.0,
        vmax=0.5,
        xticklabels=config.ABSORBING_STATES,
        yticklabels=False,  # shared with ax1
        cbar_kws={"label": "Absorption Probability", "shrink": 0.8},
        annot_kws={"size": 8.5},
        linewidths=0.5,
        linecolor="#f0f0f0"
    )
    ax2.set_title("Absorbing Matrix R (15 x 2)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Absorbing State", fontsize=11, labelpad=8)
    ax2.tick_params(axis="x", rotation=0, labelsize=9.5)

    plt.suptitle("IPL T20 Run Chase Markov Transition Probabilities (Train: Seasons <= 2022)", fontsize=15, fontweight="bold", y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved transition matrix heatmap to: {save_path}")

def main():
    print("=" * 70)
    print("SCRIPT 3: Estimating Transition Matrix P and Bootstrap CIs")
    print("=" * 70)

    transitions_path = config.PROCESSED_DATA_DIR / "transitions.csv"
    df_trans = pd.read_csv(transitions_path)

    # Filter for training set (season <= TRAIN_MAX_SEASON)
    train_trans = df_trans[df_trans["season"] <= config.TRAIN_MAX_SEASON].copy()
    assert train_trans["season"].max() <= config.TRAIN_MAX_SEASON, "LEAKAGE: test seasons in training"
    train_matches = train_trans["match_id"].nunique()
    print(f"Loaded transitions from: {transitions_path}")
    print(f"Training split: Season <= {config.TRAIN_MAX_SEASON}")
    print(f"Train matches: {train_matches} | Train transitions: {len(train_trans)}")

    # Estimate empirical P, Q, R
    P, Q, R, counts = estimate_transition_matrix(train_trans)

    # Save CSVs
    p_df = pd.DataFrame(P, index=config.ALL_STATES, columns=config.ALL_STATES)
    q_df = pd.DataFrame(Q, index=config.TRANSIENT_STATES, columns=config.TRANSIENT_STATES)
    r_df = pd.DataFrame(R, index=config.TRANSIENT_STATES, columns=config.ABSORBING_STATES)

    p_csv = config.PROCESSED_DATA_DIR / "transition_matrix_P.csv"
    q_csv = config.PROCESSED_DATA_DIR / "Q_matrix.csv"
    r_csv = config.PROCESSED_DATA_DIR / "R_matrix.csv"

    p_df.to_csv(p_csv)
    q_df.to_csv(q_csv)
    r_df.to_csv(r_csv)

    # Save NumPy binary files
    np.save(config.PROCESSED_DATA_DIR / "P_matrix.npy", P)
    np.save(config.PROCESSED_DATA_DIR / "Q_matrix.npy", Q)
    np.save(config.PROCESSED_DATA_DIR / "R_matrix.npy", R)

    print("\nSaved canonical matrices:")
    print(f"  - Full P (17x17): {p_csv} and .npy")
    print(f"  - Submatrix Q (15x15): {q_csv} and .npy")
    print(f"  - Submatrix R (15x2): {r_csv} and .npy")

    # Run Match-level Bootstrap for 95% CIs
    print(f"\nRunning whole-match bootstrap (N_BOOT = {config.N_BOOT}, seed = {config.RANDOM_SEED})...")
    boot_Q, boot_R, lower_bounds, upper_bounds = run_match_bootstrap(
        train_trans, n_boot=config.N_BOOT, random_seed=config.RANDOM_SEED
    )

    # Save bootstrap Q and R samples for Script 4
    np.save(config.PROCESSED_DATA_DIR / "bootstrap_Q.npy", boot_Q)
    np.save(config.PROCESSED_DATA_DIR / "bootstrap_R.npy", boot_R)
    print("Saved bootstrap distributions (bootstrap_Q.npy, bootstrap_R.npy) for analytical CI propagation.")

    # Build bootstrap summary table
    ci_rows = []
    n_transient = len(config.TRANSIENT_STATES)
    for i, from_st in enumerate(config.TRANSIENT_STATES):
        for j, to_st in enumerate(config.ALL_STATES):
            p_val = P[i, j]
            ci_l = lower_bounds[i, j]
            ci_u = upper_bounds[i, j]
            ci_rows.append({
                "from_state": from_st,
                "to_state": to_st,
                "p_estimate": round(p_val, 4),
                "ci_lower": round(ci_l, 4),
                "ci_upper": round(ci_u, 4),
            })

    ci_df = pd.DataFrame(ci_rows)
    ci_csv = config.PROCESSED_DATA_DIR / "bootstrap_P_ci.csv"
    ci_df.to_csv(ci_csv, index=False)
    print(f"Saved 95% bootstrap confidence intervals to: {ci_csv}")

    # Plot Heatmap
    plot_file = config.FIGURES_DIR / "transition_matrix.png"
    plot_transition_matrices(Q, R, plot_file)

    # Summary Report
    print("\n--- Summary: Selected Transition Probabilities and 95% CIs ---")
    # Show absorption row probabilities for selected states
    sample_states = ["W_Low_P1", "W_Low_P3", "W_Mid_P3", "W_High_P3", "W_High_P5"]
    sub_ci = ci_df[(ci_df["from_state"].isin(sample_states)) & (ci_df["to_state"].isin(["WIN", "LOSS"]))]
    print(sub_ci.to_string(index=False))

    print("\n--- Row Stochasticity Check ---")
    row_sums = P.sum(axis=1)
    stochastic_pass = np.allclose(row_sums, 1.0)
    print(f"All row sums == 1.0: {stochastic_pass} (min: {row_sums.min():.6f}, max: {row_sums.max():.6f})")
    print("=" * 70)

if __name__ == "__main__":
    main()
