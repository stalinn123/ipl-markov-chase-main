"""
src/06b_time_inhomogeneous.py
Finite-horizon, time-inhomogeneous absorbing Markov chain.

Motivation: the homogeneous chain (scripts 03-06) has no notion of overs remaining,
so E[T] can exceed the overs actually left (e.g. ~18.8 overs from W_Low states) and
script 07 rejects time-homogeneity. Here each over k = 1..19 gets its own transition
matrix P_k (15 x 17), estimated ONLY on training seasons (<= TRAIN_MAX_SEASON).

Estimation: Dirichlet-smoothed MLE. Row s of P_k = (n_k(s,.) + alpha * P_loc,k(s,.)) /
(n_k(s) + alpha), where P_loc,k is the MLE pooled over overs k-2..k+2 (same train data). This is
the posterior mean under a Dirichlet(alpha * P_pool(s,.)) prior and handles sparse rows.

Analytics (backward recursion, horizon = over 19 -> absorption):
  V_19(s) = P_19(s, WIN)                       (transitions out of over 19 always absorb)
  V_k(s)  = P_k(s, WIN) + sum_j Q_k(s, j) V_{k+1}(j)         win probability
  E_19(s) = 1;  E_k(s) = 1 + sum_j Q_k(s, j) E_{k+1}(j)      expected overs to absorption
n-step (Chapman-Kolmogorov, non-homogeneous): P(k -> k+n) = P_k P_{k+1} ... P_{k+n-1}.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
import config

ALPHA = 10.0          # Dirichlet prior strength (pseudo-transitions per row)
N_OVERS = 19
S = config.TRANSIENT_STATES
IDX = {st: i for i, st in enumerate(config.ALL_STATES)}
NT, NA = len(S), len(config.ALL_STATES)


def count_tensor(trans: pd.DataFrame) -> np.ndarray:
    """C[k-1, i, j] = # transitions from state i at end of over k to state j."""
    C = np.zeros((N_OVERS, NT, NA))
    f = trans["from_state"].map(IDX).values
    t = trans["to_state"].map(IDX).values
    k = trans["over_from"].values.astype(int) - 1
    np.add.at(C, (k, f, t), 1.0)
    return C


def fit_inhomogeneous(trans: pd.DataFrame, alpha: float = ALPHA) -> np.ndarray:
    C = count_tensor(trans)
    pooled = C.sum(axis=0)
    P_pool = pooled / np.where(pooled.sum(1, keepdims=True) == 0, 1, pooled.sum(1, keepdims=True))
    # Prior for over k: local pool over neighbouring overs |k'-k| <= 2 (falls back to global pool
    # for empty rows). Local pooling avoids leaking death-over absorption into early overs.
    Pk = np.zeros_like(C)
    for k in range(N_OVERS):
        loc = C[max(0, k - 2): k + 3].sum(axis=0)
        ls = loc.sum(1, keepdims=True)
        prior = np.where(ls > 0, loc / np.where(ls == 0, 1, ls), P_pool)
        Pk[k] = (C[k] + alpha * prior) / (C[k].sum(1, keepdims=True) + alpha)
    # Hard horizon: from over 19 the chain must absorb (true in data; enforce structurally)
    last = Pk[N_OVERS - 1].copy()
    last[:, :NT] = 0.0
    absorb_sum = last[:, NT:].sum(1, keepdims=True)
    fallback = P_pool[:, NT:] / P_pool[:, NT:].sum(1, keepdims=True)
    last[:, NT:] = np.where(absorb_sum > 0, last[:, NT:] / np.where(absorb_sum == 0, 1, absorb_sum), fallback)
    Pk[N_OVERS - 1] = last
    return Pk


def backward(Pk: np.ndarray):
    """Returns V (19 x 15) win probabilities and E (19 x 15) expected overs to absorption."""
    V = np.zeros((N_OVERS, NT))
    E = np.zeros((N_OVERS, NT))
    V[-1] = Pk[-1][:, IDX["WIN"]]
    E[-1] = 1.0
    for k in range(N_OVERS - 2, -1, -1):
        Q, Rw = Pk[k][:, :NT], Pk[k][:, IDX["WIN"]]
        V[k] = Rw + Q @ V[k + 1]
        E[k] = 1.0 + Q @ E[k + 1]
    return V, E


def metrics(y, p):
    return {"Brier": brier_score_loss(y, p), "LogLoss": log_loss(y, np.clip(p, 1e-5, 1 - 1e-5)),
            "Acc": accuracy_score(y, (p >= 0.5).astype(int))}


def main():
    print("=" * 70)
    print("SCRIPT 6b: Time-Inhomogeneous (per-over) Absorbing Markov Chain")
    print("=" * 70)
    trans = pd.read_csv(config.PROCESSED_DATA_DIR / "transitions.csv")
    over_states = pd.read_csv(config.PROCESSED_DATA_DIR / "over_states.csv")
    chases = pd.read_csv(config.PROCESSED_DATA_DIR / "chases.csv")
    hom = pd.read_csv(config.PROCESSED_DATA_DIR / "markov_analytical_results.csv")

    train_trans = trans[trans["season"] <= config.TRAIN_MAX_SEASON]
    assert train_trans["season"].max() <= config.TRAIN_MAX_SEASON, "LEAKAGE: test season in training"
    print(f"Train transitions: {len(train_trans)} from {train_trans['match_id'].nunique()} matches "
          f"(seasons {train_trans['season'].min()}-{train_trans['season'].max()}), alpha = {ALPHA}")

    Pk = fit_inhomogeneous(train_trans)
    assert np.allclose(Pk.sum(2), 1.0)
    V, E = backward(Pk)
    np.save(config.PROCESSED_DATA_DIR / "Pk_inhomogeneous.npy", Pk)

    # Tables: win prob and expected overs by (over, state)
    vdf = pd.DataFrame(V, index=range(1, 20), columns=S); vdf.index.name = "over"
    edf = pd.DataFrame(E, index=range(1, 20), columns=S); edf.index.name = "over"
    vdf.to_csv(config.PROCESSED_DATA_DIR / "inhom_win_prob_by_over.csv")
    edf.to_csv(config.PROCESSED_DATA_DIR / "inhom_expected_overs_by_over.csv")
    print("\nWin probability V_k(s) at selected overs (rows) for selected states:")
    show = ["W_Low_P2", "W_Low_P3", "W_Mid_P3", "W_Mid_P4", "W_High_P3"]
    print(vdf.loc[[1, 5, 10, 15, 19], show].round(3).to_string())
    print("\nExpected overs to absorption E_k(s) (must be <= 20 - k):")
    print(edf.loc[[1, 5, 10, 15, 19], show].round(2).to_string())
    print(f"Max violation of E_k(s) <= 20-k: "
          f"{max(0.0, float((E - (20 - np.arange(1, 20))[:, None]).max())):.4f}")

    # Out-of-sample validation vs homogeneous chain and LR
    obs = over_states.merge(chases[["match_id", "chase_won"]], on="match_id")
    obs = obs[obs["state"].isin(S)].copy()
    tr, te = obs[obs["season"] <= config.TRAIN_MAX_SEASON], obs[obs["season"] >= config.TEST_MIN_SEASON].copy()
    hmap = dict(zip(hom["State"], hom["Win_Prob"]))
    te["p_hom"] = te["state"].map(hmap)
    te["p_inh"] = V[te["over"].astype(int).values - 1, te["state"].map(IDX).values]
    feats = ["wickets", "CRR", "RRR", "runs_needed", "balls_left"]
    lr = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED).fit(
        tr[feats].replace([np.inf, -np.inf], np.nan).fillna(0), tr["chase_won"])
    te["p_lr"] = lr.predict_proba(te[feats].replace([np.inf, -np.inf], np.nan).fillna(0))[:, 1]

    y = te["chase_won"].values
    rows = []
    for name, col in [("Markov homogeneous", "p_hom"), ("Markov time-inhomogeneous", "p_inh"),
                      ("Logistic regression", "p_lr")]:
        m = metrics(y, te[col].values); m["Model"] = name; rows.append(m)
    res = pd.DataFrame(rows)[["Model", "Brier", "LogLoss", "Acc"]]
    res.to_csv(config.PROCESSED_DATA_DIR / "validation_three_models.csv", index=False)
    print(f"\n--- Out-of-sample (seasons >= {config.TEST_MIN_SEASON}, {te['match_id'].nunique()} matches, "
          f"{len(te)} over-states) ---")
    print(res.round(4).to_string(index=False))

    # Match-level cluster bootstrap CI for Brier difference (inhom - LR) and (inhom - hom)
    rng = np.random.default_rng(config.RANDOM_SEED)
    te["se_h"] = (te["p_hom"] - y) ** 2; te["se_i"] = (te["p_inh"] - y) ** 2; te["se_l"] = (te["p_lr"] - y) ** 2
    g = te.groupby("match_id")[["se_h", "se_i", "se_l"]].agg(["sum"]).droplevel(1, axis=1)
    n = te.groupby("match_id").size().values
    G = g.values
    d_il, d_ih = [], []
    for _ in range(config.N_BOOT):
        b = rng.integers(0, len(G), len(G))
        s, nn = G[b].sum(0), n[b].sum()
        d_il.append((s[1] - s[2]) / nn); d_ih.append((s[1] - s[0]) / nn)
    for lab, d in [("Brier(inhom) - Brier(LR)", d_il), ("Brier(inhom) - Brier(hom)", d_ih)]:
        print(f"{lab}: {np.mean(d):+.4f}  95% cluster-bootstrap CI [{np.percentile(d, 2.5):+.4f}, "
              f"{np.percentile(d, 97.5):+.4f}]")

    # Brier by over
    bo = te.groupby("over").apply(lambda d: pd.Series({
        "n": len(d), "hom": brier_score_loss(d.chase_won, d.p_hom),
        "inhom": brier_score_loss(d.chase_won, d.p_inh), "lr": brier_score_loss(d.chase_won, d.p_lr)}),
        include_groups=False)
    bo.to_csv(config.PROCESSED_DATA_DIR / "validation_three_models_by_over.csv")
    plt.figure(figsize=(9, 5.5))
    for c, lab, mk, col in [("hom", "Markov (homogeneous)", "o", "#9a9993"),
                            ("inhom", "Markov (time-inhomogeneous)", "^", "#2a78d6"),
                            ("lr", "Logistic regression", "s", "#eb6834")]:
        plt.plot(bo.index, bo[c], marker=mk, label=lab, color=col, lw=2)
    plt.xlabel("Completed over k"); plt.ylabel("Brier score (lower is better)")
    plt.title(f"Out-of-sample Brier by over (test seasons >= {config.TEST_MIN_SEASON})")
    plt.xticks(range(1, 20)); plt.legend(); plt.grid(alpha=.3)
    plt.savefig(config.FIGURES_DIR / "validation_brier_three_models.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Win-probability surface V_k(s) heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(V.T, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    plt.colorbar(label="P(chase won)")
    plt.yticks(range(NT), S, fontsize=8); plt.xticks(range(19), range(1, 20))
    plt.xlabel("Completed over k"); plt.title("Time-inhomogeneous win probability V_k(s) (train <= 2022)")
    plt.savefig(config.FIGURES_DIR / "inhom_win_prob_surface.png", dpi=200, bbox_inches="tight")
    plt.close()
    # State probabilities vs time: P(absorbed by end of over k+1), model vs empirical (train)
    P_h = np.load(config.PROCESSED_DATA_DIR / "P_matrix.npy")
    o1 = over_states[(over_states["season"] <= config.TRAIN_MAX_SEASON) & (over_states["over"] == 1)]["state"]
    pi = np.array([(o1 == st).mean() for st in config.ALL_STATES])
    ph, pin = pi.copy(), pi.copy()
    absd = over_states[(over_states["season"] <= config.TRAIN_MAX_SEASON) & over_states["state"].isin(["WIN", "LOSS"])]
    n_tr = absd["match_id"].nunique()
    ks, emp, hcurve, icurve = [], [], [], []
    for k in range(1, 20):
        ph = ph @ P_h; pin = pin @ np.vstack([Pk[k - 1], np.hstack([np.zeros((2, NT)), np.eye(2)])])
        ks.append(k + 1); hcurve.append(ph[NT:].sum()); icurve.append(pin[NT:].sum())
        emp.append(((absd["over"] <= k + 1) & (absd["match_id"].isin(o1.index.map(lambda i: over_states.loc[i, "match_id"])))).sum() / len(o1))
    sv = pd.DataFrame({"over": ks, "empirical": emp, "homogeneous": hcurve, "inhomogeneous": icurve})
    sv.to_csv(config.PROCESSED_DATA_DIR / "absorbed_by_over_train.csv", index=False)
    print("\nP(chase resolved by end of over k), train seasons, starting from over-1 distribution:")
    print(sv.iloc[[3, 8, 13, 17, 18]].round(3).to_string(index=False))
    plt.figure(figsize=(8, 5))
    plt.plot(ks, emp, "k-o", label="Empirical"); plt.plot(ks, hcurve, "--", label="Homogeneous chain")
    plt.plot(ks, icurve, "-^", label="Time-inhomogeneous chain")
    plt.xlabel("Over k"); plt.ylabel("P(absorbed by end of over k)"); plt.legend(); plt.grid(alpha=.3)
    plt.title("State probabilities vs time: absorption by over (train)")
    plt.savefig(config.FIGURES_DIR / "absorbed_by_over.png", dpi=200, bbox_inches="tight"); plt.close()
    print("Saved figures: validation_brier_three_models.png, inhom_win_prob_surface.png, absorbed_by_over.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
