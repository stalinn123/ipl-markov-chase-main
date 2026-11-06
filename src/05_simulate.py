"""
src/05_simulate.py
Script 5: Monte Carlo simulation of the Markov chain.
Simulates N_SIM trajectories from each starting transient state until absorption.
Compares empirical simulation frequencies and absorption times with analytical theory,
computes binomial 95% CIs, and plots match length distributions.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import config

def simulate_markov_chains(P: np.ndarray, start_state_idx: int, n_sim: int = 10000, random_seed: int = 42):
    """
    Simulates n_sim independent chains starting from start_state_idx until absorption.
    Returns:
        win_flags: array of shape (n_sim,) where 1=WIN, 0=LOSS
        steps_taken: array of shape (n_sim,) of overs taken until absorption
    """
    np.random.seed(random_seed)
    cum_P = np.cumsum(P, axis=1)

    curr_states = np.full(n_sim, start_state_idx, dtype=np.int32)
    steps = np.zeros(n_sim, dtype=np.int32)
    active = np.ones(n_sim, dtype=bool)

    while active.any():
        steps[active] += 1
        n_active = np.sum(active)
        rands = np.random.rand(n_active)

        curr_active = curr_states[active]
        next_s = np.array([np.searchsorted(cum_P[s], r) for s, r in zip(curr_active, rands)], dtype=np.int32)

        curr_states[active] = next_s
        # States 15 (WIN) and 16 (LOSS) are absorbing
        active[curr_states >= 15] = False

    win_flags = (curr_states == 15).astype(int)
    return win_flags, steps

def plot_mc_vs_theory(df_comp: pd.DataFrame, save_path: Path):
    """
    Plots analytical B_i vs simulated win probability with binomial 95% CIs.
    """
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    x = np.arange(len(df_comp))
    width = 0.38

    # Win Probability comparison
    ax1.bar(x - width/2, df_comp["Theory_Win_Prob"], width, label="Analytical Theory (B_i)", color="#2b6cb0", alpha=0.85)
    
    # Simulation error bars
    sim_err = [
        df_comp["Sim_Win_Rate"] - df_comp["Sim_CI_Lower"],
        df_comp["Sim_CI_Upper"] - df_comp["Sim_Win_Rate"]
    ]
    ax1.errorbar(
        x + width/2, df_comp["Sim_Win_Rate"], yerr=sim_err,
        fmt="o", color="#c53030", capsize=4, label="Simulation (95% Binomial CI)", markersize=5
    )
    ax1.bar(x + width/2, df_comp["Sim_Win_Rate"], width, color="#e53e3e", alpha=0.45)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df_comp["State"], rotation=45, ha="right", fontsize=9.5)
    ax1.set_ylabel("Win Probability", fontsize=11, fontweight="bold")
    ax1.set_title("Theoretical vs Monte Carlo Win Probability (N = 10,000 per state)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 1.05)

    # Expected absorption overs comparison
    ax2.bar(x - width/2, df_comp["Theory_Overs"], width, label="Analytical Expected Overs (t_i)", color="#2f855a", alpha=0.85)
    ax2.errorbar(
        x + width/2, df_comp["Sim_Mean_Overs"], yerr=1.96 * df_comp["Sim_SE_Overs"],
        fmt="s", color="#d69e2e", capsize=4, label="Simulated Mean Overs (95% CI)", markersize=5
    )
    ax2.bar(x + width/2, df_comp["Sim_Mean_Overs"], width, color="#ecc94b", alpha=0.45)

    ax2.set_xticks(x)
    ax2.set_xticklabels(df_comp["State"], rotation=45, ha="right", fontsize=9.5)
    ax2.set_ylabel("Expected Overs to Resolution", fontsize=11, fontweight="bold")
    ax2.set_title("Theoretical vs Monte Carlo Absorption Time (Overs)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right")

    plt.suptitle("Validation of Analytical Markov Formulas via Monte Carlo Simulation", fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Monte Carlo validation plot to: {save_path}")

def plot_length_distributions(all_steps: dict, save_path: Path):
    """
    Plots empirical distribution of match lengths for selected contrasting starting states.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5.5))

    states_to_plot = ["W_Low_P1", "W_Mid_P3", "W_High_P5"]
    palette = {"W_Low_P1": "#2b6cb0", "W_Mid_P3": "#d69e2e", "W_High_P5": "#c53030"}

    for st in states_to_plot:
        data = all_steps[st]
        sns.kdeplot(
            data,
            label=f"{st} (Mean: {np.mean(data):.2f} ov, Median: {np.median(data):.0f} ov)",
            color=palette[st],
            linewidth=2.5,
            fill=True,
            alpha=0.18,
            bw_adjust=1.2
        )

    plt.title("Simulated Distribution of Overs to Absorption (from Selected States)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Overs Taken to Reach Absorption (WIN or LOSS)", fontsize=11, fontweight="bold")
    plt.ylabel("Probability Density", fontsize=11, fontweight="bold")
    plt.xlim(1, 20)
    plt.legend(fontsize=10.5, loc="upper right")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved match length distribution plot to: {save_path}")

def main():
    print("=" * 70)
    print("SCRIPT 5: Monte Carlo Simulation vs Analytical Theory")
    print("=" * 70)

    P = np.load(config.PROCESSED_DATA_DIR / "P_matrix.npy")
    df_analytical = pd.read_csv(config.PROCESSED_DATA_DIR / "markov_analytical_results.csv")

    n_sim = config.N_SIM
    print(f"Running Monte Carlo simulations: {n_sim} trajectories per transient state (Total: {n_sim * 15:,} chains)...")

    sim_results = []
    all_steps_dict = {}

    for i, st in enumerate(config.TRANSIENT_STATES):
        theory_row = df_analytical[df_analytical["State"] == st].iloc[0]
        theory_win = theory_row["Win_Prob"]
        theory_t = theory_row["Expected_Overs"]

        # Run simulation
        win_flags, steps = simulate_markov_chains(
            P, start_state_idx=i, n_sim=n_sim, random_seed=config.RANDOM_SEED + i
        )
        all_steps_dict[st] = steps

        # Simulation metrics
        sim_wins = np.sum(win_flags)
        sim_win_rate = sim_wins / n_sim
        sim_mean_steps = np.mean(steps)
        sim_std_steps = np.std(steps, ddof=1)
        sim_se_steps = sim_std_steps / np.sqrt(n_sim)
        frac_over_horizon = float(np.mean(steps > 19))  # impossible in a real 20-over chase

        # Binomial 95% Confidence Interval for win probability (Wilson score / Wald)
        se_p = np.sqrt(sim_win_rate * (1.0 - sim_win_rate) / n_sim)
        ci_lower = max(0.0, sim_win_rate - 1.96 * se_p)
        ci_upper = min(1.0, sim_win_rate + 1.96 * se_p)

        # Check if analytical theory falls inside simulation CI
        theory_in_ci = bool(ci_lower <= theory_win <= ci_upper)

        sim_results.append({
            "State": st,
            "Theory_Win_Prob": round(theory_win, 4),
            "Sim_Win_Rate": round(sim_win_rate, 4),
            "Sim_CI_Lower": round(ci_lower, 4),
            "Sim_CI_Upper": round(ci_upper, 4),
            "Theory_In_CI": theory_in_ci,
            "Theory_Overs": round(theory_t, 2),
            "Sim_Mean_Overs": round(sim_mean_steps, 2),
            "Sim_SE_Overs": round(sim_se_steps, 3),
            "Frac_Paths_Over_19_Steps": round(frac_over_horizon, 4),
        })

    df_comp = pd.DataFrame(sim_results)
    sim_csv = config.PROCESSED_DATA_DIR / "simulation_vs_theory.csv"
    df_comp.to_csv(sim_csv, index=False)
    print(f"\nSaved simulation comparison table to: {sim_csv}")

    print("\n--- Simulation vs Theory Comparison Table ---")
    print(df_comp[["State", "Theory_Win_Prob", "Sim_Win_Rate", "Sim_CI_Lower", "Sim_CI_Upper", "Theory_In_CI", "Theory_Overs", "Sim_Mean_Overs"]].to_string(index=False))

    print("\nFraction of simulated paths lasting > 19 transitions (horizon violation of homogeneous chain):")
    print(df_comp[["State", "Frac_Paths_Over_19_Steps"]].to_string(index=False))
    all_in_ci = df_comp["Theory_In_CI"].all()
    count_in_ci = df_comp["Theory_In_CI"].sum()
    print(f"\nTheory inside Simulation 95% CI: {count_in_ci}/{len(df_comp)} states ({'all inside' if all_in_ci else 'expected ~5% misses at 95% level'})")

    # Generate figures
    plot_mc_vs_theory(df_comp, config.FIGURES_DIR / "mc_vs_theory_win_prob.png")
    plot_length_distributions(all_steps_dict, config.FIGURES_DIR / "mc_match_length_distribution.png")

    print("=" * 70)

if __name__ == "__main__":
    main()
