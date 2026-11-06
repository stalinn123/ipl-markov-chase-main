"""
src/06_validate.py
Script 6: Out-of-sample validation on test set (Seasons >= 2023).
Evaluates Brier score, log-loss, and accuracy for Markov chain model vs Logistic Regression baseline,
both overall and by completed over (1..19). Generates calibration curves, Brier score curves,
and in-match win probability trajectories for 3 test chases.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.calibration import calibration_curve

import config

def clip_probs(probs: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Clips probabilities strictly into [eps, 1 - eps] for stable log-loss."""
    return np.clip(probs, eps, 1.0 - eps)

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Computes Brier score, log-loss, and accuracy (at threshold 0.5)."""
    p_clipped = clip_probs(y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, p_clipped)
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    return {
        "Brier": brier,
        "Log_Loss": ll,
        "Accuracy": acc,
    }

def train_baseline_model(train_df: pd.DataFrame):
    """
    Trains a Logistic Regression baseline model on training set over-states.
    Features: wickets, CRR, RRR, runs_needed, balls_left.
    """
    feature_cols = ["wickets", "CRR", "RRR", "runs_needed", "balls_left"]
    X_train = train_df[feature_cols].copy()
    y_train = train_df["chase_won"].values

    # Handle rare infinite or NaN values if any
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    lr_model = LogisticRegression(random_state=config.RANDOM_SEED, max_iter=1000)
    lr_model.fit(X_train, y_train)

    return lr_model, feature_cols

def plot_calibration(y_true: np.ndarray, p_markov: np.ndarray, p_lr: np.ndarray, save_path: Path):
    """Plots reliability / calibration curve for Markov model vs baseline."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6.5))

    prob_true_m, prob_pred_m = calibration_curve(y_true, p_markov, n_bins=10, strategy="uniform")
    prob_true_lr, prob_pred_lr = calibration_curve(y_true, p_lr, n_bins=10, strategy="uniform")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.plot(prob_pred_m, prob_true_m, marker="o", linewidth=2.2, color="#2b6cb0", label="Markov Chain Model")
    plt.plot(prob_pred_lr, prob_true_lr, marker="s", linewidth=2.2, color="#dd6b20", label="Logistic Regression Baseline")

    plt.title("Out-of-Sample Calibration Curve (Test Seasons >= 2023)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Win Probability", fontsize=11, fontweight="bold")
    plt.ylabel("Observed Win Proportion", fontsize=11, fontweight="bold")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(fontsize=10.5, loc="upper left")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved calibration plot: {save_path}")

def plot_brier_by_over(by_over_df: pd.DataFrame, save_path: Path):
    """Plots Brier score by over (1 to 19) for Markov model vs baseline."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 5.5))

    overs = by_over_df["over"]
    plt.plot(overs, by_over_df["Markov_Brier"], marker="o", linewidth=2.2, color="#2b6cb0", label="Markov Model Brier Score")
    plt.plot(overs, by_over_df["LR_Brier"], marker="s", linewidth=2.2, color="#dd6b20", label="Logistic Regression Brier Score")

    plt.title("Predictive Performance Evolution by Completed Over (Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Completed Over (k = 1..19)", fontsize=11, fontweight="bold")
    plt.ylabel("Brier Score (Lower is Better)", fontsize=11, fontweight="bold")
    plt.xticks(np.arange(1, 20))
    plt.legend(fontsize=10.5, loc="upper right")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Brier by over plot: {save_path}")

def plot_chase_trajectories(test_trans_df: pd.DataFrame, markov_win_map: dict, lr_model, feature_cols, save_path: Path):
    """
    Plots win probability curves across the match for 3 selected test-set chases.
    Includes one thrilling close finish, one steady chase, and one defending team turnaround.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True, constrained_layout=True)

    # Pick 3 interesting matches from test set:
    # 1. Close finish: Match 1359543 (Target: 215, runs: 214, 1-run nail-biter) or find from test set
    unique_matches = test_trans_df["match_id"].unique()

    # Find one nail biter (outcome in over 19/20 with tight score), one win, one loss
    sample_ids = []
    for mid in unique_matches:
        sub = test_trans_df[test_trans_df["match_id"] == mid]
        if len(sub) >= 18:
            last_sub = sub.iloc[-1]
            if 0 < last_sub["runs_needed"] <= 12:
                sample_ids.append(mid)
                break

    for mid in unique_matches:
        if mid not in sample_ids:
            sub = test_trans_df[test_trans_df["match_id"] == mid]
            if len(sub) >= 15 and sub.iloc[0]["chase_won"] == 1:
                sample_ids.append(mid)
                break

    for mid in unique_matches:
        if mid not in sample_ids:
            sub = test_trans_df[test_trans_df["match_id"] == mid]
            if len(sub) >= 15 and sub.iloc[0]["chase_won"] == 0:
                sample_ids.append(mid)
                break

    titles = [
        "Nail-Biting Close Finish",
        "Successful Run Chase",
        "Defending Team Victory"
    ]

    for ax, mid, title in zip(axes, sample_ids, titles):
        m_data = test_trans_df[test_trans_df["match_id"] == mid].sort_values(by="over")
        overs = m_data["over"].values
        target = m_data["runs_needed"].iloc[0] + m_data["runs"].iloc[0]
        won = bool(m_data["chase_won"].iloc[0])

        m_p_markov = [markov_win_map[st] for st in m_data["state"]]
        X_sub = m_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        m_p_lr = lr_model.predict_proba(X_sub)[:, 1]

        ax.plot(overs, m_p_markov, marker="o", linewidth=2.4, color="#2b6cb0", label="Markov Model P(Win)")
        ax.plot(overs, m_p_lr, marker="s", linestyle="--", linewidth=2.0, color="#dd6b20", label="Logistic Regression")
        ax.axhline(0.5, color="#e53e3e", linestyle=":", alpha=0.7)

        result_txt = "CHASE WON" if won else "CHASE LOST"
        ax.set_title(f"{title}\nMatch {mid} (Target: {target} | {result_txt})", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Completed Over", fontsize=10.5, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("In-Play Win Probability", fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(np.arange(2, max(overs) + 1, 2))
        ax.legend(fontsize=9, loc="best")

    plt.suptitle("In-Play Win Probability Trajectories: Markov Chain vs Baseline", fontsize=14, fontweight="bold", y=1.03)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved chase trajectories plot: {save_path}")

def main():
    print("=" * 70)
    print("SCRIPT 6: Out-of-Sample Model Validation & Baseline Comparison")
    print("=" * 70)

    # Load data
    over_states = pd.read_csv(config.PROCESSED_DATA_DIR / "over_states.csv")
    chases = pd.read_csv(config.PROCESSED_DATA_DIR / "chases.csv")
    analytical_df = pd.read_csv(config.PROCESSED_DATA_DIR / "markov_analytical_results.csv")

    # Map Markov states to analytical win probability B_i
    markov_win_map = dict(zip(analytical_df["State"], analytical_df["Win_Prob"]))

    # Merge chase outcome
    merged = over_states.merge(chases[["match_id", "chase_won"]], on="match_id")
    # Transient over-end observations (overs 1..19)
    trans_obs = merged[merged["state"].isin(config.TRANSIENT_STATES)].copy()

    # Train / Test split
    train_df = trans_obs[trans_obs["season"] <= config.TRAIN_MAX_SEASON].copy()
    test_df = trans_obs[trans_obs["season"] >= config.TEST_MIN_SEASON].copy()
    assert train_df["season"].max() < test_df["season"].min(), "LEAKAGE: train/test seasons overlap"

    print(f"Training split: Season <= {config.TRAIN_MAX_SEASON} ({len(train_df)} observations across {train_df['match_id'].nunique()} matches)")
    print(f"Testing split:  Season >= {config.TEST_MIN_SEASON} ({len(test_df)} observations across {test_df['match_id'].nunique()} matches)")

    # 1. Train Logistic Regression baseline
    lr_model, feature_cols = train_baseline_model(train_df)
    print(f"\nTrained Logistic Regression baseline with features: {feature_cols}")

    # 2. Predictions on test set
    y_test = test_df["chase_won"].values
    p_markov = np.array([markov_win_map[st] for st in test_df["state"]], dtype=np.float64)

    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    p_lr = lr_model.predict_proba(X_test)[:, 1]

    # Overall Metrics
    overall_markov = compute_metrics(y_test, p_markov)
    overall_lr = compute_metrics(y_test, p_lr)

    print("\n--- Overall Out-of-Sample Test Set Results ---")
    print(f"{'Metric':<15} | {'Markov Model':<15} | {'Logistic Regression':<18}")
    print("-" * 55)
    print(f"{'Brier Score':<15} | {overall_markov['Brier']:<15.4f} | {overall_lr['Brier']:<18.4f}")
    print(f"{'Log-Loss':<15} | {overall_markov['Log_Loss']:<15.4f} | {overall_lr['Log_Loss']:<18.4f}")
    print(f"{'Accuracy (0.5)':<15} | {overall_markov['Accuracy']*100:<14.2f}% | {overall_lr['Accuracy']*100:<17.2f}%")

    # 3. By-Over Breakdown (Overs 1 to 19)
    by_over_records = []
    for ov in range(1, 20):
        sub = test_df[test_df["over"] == ov]
        if len(sub) == 0:
            continue
        y_sub = sub["chase_won"].values
        p_m_sub = np.array([markov_win_map[st] for st in sub["state"]])
        X_sub = sub[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        p_lr_sub = lr_model.predict_proba(X_sub)[:, 1]

        m_res = compute_metrics(y_sub, p_m_sub)
        lr_res = compute_metrics(y_sub, p_lr_sub)

        by_over_records.append({
            "over": ov,
            "observations": len(sub),
            "Markov_Brier": round(m_res["Brier"], 4),
            "LR_Brier": round(lr_res["Brier"], 4),
            "Markov_LogLoss": round(m_res["Log_Loss"], 4),
            "LR_LogLoss": round(lr_res["Log_Loss"], 4),
            "Markov_Acc": round(m_res["Accuracy"], 4),
            "LR_Acc": round(lr_res["Accuracy"], 4),
        })

    by_over_df = pd.DataFrame(by_over_records)

    # Save validation results
    val_csv = config.PROCESSED_DATA_DIR / "validation_results.csv"
    by_over_df.to_csv(val_csv, index=False)
    print(f"\nSaved detailed by-over validation results to: {val_csv}")

    print("\n--- By-Over Performance Breakdown (Sample Overs) ---")
    sample_overs = by_over_df[by_over_df["over"].isin([1, 6, 10, 15, 19])]
    print(sample_overs[["over", "observations", "Markov_Brier", "LR_Brier", "Markov_Acc", "LR_Acc"]].to_string(index=False))

    # 4. Generate Plots
    plot_calibration(y_test, p_markov, p_lr, config.FIGURES_DIR / "validation_calibration.png")
    plot_brier_by_over(by_over_df, config.FIGURES_DIR / "validation_brier_by_over.png")
    plot_chase_trajectories(test_df, markov_win_map, lr_model, feature_cols, config.FIGURES_DIR / "validation_chase_traces.png")

    print("=" * 70)

if __name__ == "__main__":
    main()
