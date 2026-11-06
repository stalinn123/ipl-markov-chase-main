"""
src/09_figures.py
Report figures for the era study and the time-inhomogeneous chain (run after 06b and 08).
  fig_era_transition_heatmaps.png   P pre, P post, difference (homogeneous, 15 x 17)
  fig_era_B_by_era.png              B_i pre vs post with 95% bootstrap CIs + delta
  fig_era_calibration.png           reliability diagrams on 2023-25 and 2026
  fig_era_2026_traces.png           in-play win probability for three 2026 chases
  fig_era_loso_brier.png            season-by-season Brier vs pre-era LOSO range
  fig_state_transition_diagram.png  17-state transition graph (edges with p >= 0.10)
  fig_state_prob_vs_time.png        state probabilities vs over: INH chain vs empirical
  fig_sample_paths.png              simulated sample paths + absorption-time distribution
"""
import importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import config

E = importlib.import_module("08_era_experiments")
S, ALL, NT, IDX = E.S, E.ALL, E.NT, E.IDX
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GREY, INK, INK2 = "#9a9993", "#0b0b0b", "#52514e"
FIG = config.FIGURES_DIR
PROC = config.PROCESSED_DATA_DIR
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7",
                     "grid.linewidth": 0.8, "lines.linewidth": 2.0, "legend.frameon": False,
                     "figure.dpi": 100, "savefig.dpi": 220, "savefig.bbox": "tight"})


def short(s):
    return s.replace("W_Low_", "L").replace("W_Mid_", "M").replace("W_High_", "H")


def save(name):
    plt.savefig(FIG / name); plt.close(); print("saved", name)


def main():
    tr, obs, T, mids, season = E.load()
    pre, post = (season <= 2022), (season >= 2023)
    Ck_pre, Ck_post = E.counts(T, pre), E.counts(T, post)
    P_pre, P_post = E.hom_P(Ck_pre), E.hom_P(Ck_post)

    # 1 ---------------------------------------------------------------- heatmaps
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.2), constrained_layout=True)
    labs = [short(s) for s in ALL]
    for a, M, t in [(ax[0], P_pre, "Pre-rule 2008-22 (913 chases)"),
                    (ax[1], P_post, "Impact Player era 2023-26 (282 chases)")]:
        im = a.imshow(M, cmap="Blues", vmin=0, vmax=0.6)
        a.set_title(t, color=INK, fontweight="bold")
        for i in range(NT):
            for j in range(len(ALL)):
                if M[i, j] >= 0.05:
                    a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                           color="white" if M[i, j] > 0.35 else INK)
    fig.colorbar(im, ax=ax[:2], shrink=0.75, label="Transition probability")
    D = P_post - P_pre
    im2 = ax[2].imshow(D, cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    ax[2].set_title("Difference (post - pre)", color=INK, fontweight="bold")
    for i in range(NT):
        for j in range(len(ALL)):
            if abs(D[i, j]) >= 0.03:
                ax[2].text(j, i, f"{D[i, j]:+.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(D[i, j]) > 0.07 else INK)
    fig.colorbar(im2, ax=ax[2], shrink=0.75, label="Probability difference")
    for a in ax:
        a.set_xticks(range(len(ALL))); a.set_xticklabels(labs, rotation=90, fontsize=8)
        a.set_yticks(range(NT)); a.set_yticklabels(labs[:NT], fontsize=8); a.grid(False)
        a.set_xlabel("State at end of next over"); a.set_ylabel("State at end of over k")
    fig.text(0.01, -0.03, "L/M/H = 0-2 / 3-5 / 6-9 wickets; P1..P5 = pressure (RRR - CRR) bucket. "
             "Cells < 0.05 (left) or |diff| < 0.03 (right) unlabelled.", fontsize=8.5, color=INK2)
    save("fig_era_transition_heatmaps.png")

    # 2 ---------------------------------------------------------------- B by era
    b = pd.read_csv(PROC / "era_B_comparison.csv")
    x = np.arange(NT)
    fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]},
                           constrained_layout=True)
    for off, col, c, lab in [(-0.15, "pre", BLUE, "Pre-rule 2008-22"), (0.15, "post", ORANGE, "Impact era 2023-26")]:
        ax[0].errorbar(x + off, b[col], yerr=[b[col] - b[f"{col}_lo"], b[f"{col}_hi"] - b[col]],
                       fmt="o", ms=6, color=c, capsize=3, lw=1.5, label=lab)
    ax[0].set_ylabel("Win probability $B_i$"); ax[0].set_ylim(0, 1); ax[0].legend(loc="upper right")
    ax[0].set_title("Absorption probability into WIN by state and era (95% match-bootstrap CIs)",
                    color=INK, fontweight="bold", loc="left")
    sig = b["delta_CI_excludes_0"].values
    ax[1].axhline(0, color=GREY, lw=1)
    ax[1].errorbar(x[~sig], b["delta"][~sig], yerr=[(b["delta"] - b["delta_lo"])[~sig], (b["delta_hi"] - b["delta"])[~sig]],
                   fmt="o", color=INK2, capsize=3, lw=1.3, label="CI includes 0")
    ax[1].errorbar(x[sig], b["delta"][sig], yerr=[(b["delta"] - b["delta_lo"])[sig], (b["delta_hi"] - b["delta"])[sig]],
                   fmt="D", color=ORANGE, capsize=3, lw=1.5, ms=7, label="CI excludes 0")
    ax[1].set_ylabel("Post - pre"); ax[1].legend(loc="lower left", ncol=2)
    ax[1].set_xticks(x); ax[1].set_xticklabels(S, rotation=45, ha="right", fontsize=8.5)
    save("fig_era_B_by_era.png")

    # 3 ---------------------------------------------------------------- calibration
    from sklearn.linear_model import LogisticRegression
    fits = {k: E.fit_all(T, season, obs, *v) for k, v in {"pre": E.PRE, "impact": E.IMPACT}.items()}
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
    for a, (lo, hi, ttl) in zip(ax, [(2023, 2025, "Test 2023-25 (211 chases)"), (2026, 2026, "Test 2026 (71 chases)")]):
        te = obs[obs["season"].between(lo, hi)]
        y = te["chase_won"].values
        a.plot([0, 1], [0, 1], "--", color=GREY, lw=1.2, label="Perfect calibration")
        series = [("INH", "pre", BLUE, "o", "Markov INH (pre-rule)"), ("LR", "pre", ORANGE, "s", "LR (pre-rule)")]
        if lo == 2026:
            series.append(("LR", "impact", AQUA, "^", "LR (Impact-only 2023-25)"))
        for mname, reg, c, mk, lab in series:
            ph, pi, pl = E.predict(te, fits[reg][1], fits[reg][2], fits[reg][3])
            p = {"INH": pi, "LR": pl}[mname]
            q = pd.qcut(p, 10, duplicates="drop")
            g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q", observed=True).mean()
            a.plot(g["p"], g["y"], marker=mk, color=c, ms=7, label=lab)
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_aspect("equal")
        a.set_xlabel("Predicted win probability (decile mean)"); a.set_ylabel("Observed win rate")
        a.set_title(ttl, color=INK, fontweight="bold", loc="left"); a.legend(loc="upper left", fontsize=8.5)
    save("fig_era_calibration.png")

    # 4 ---------------------------------------------------------------- 2026 traces
    p26 = pd.read_csv(PROC / "era_predictions_2026.csv")
    allst = pd.read_csv(PROC / "over_states.csv")
    fin = allst[allst["state"].isin(["WIN", "LOSS"]) & (allst["season"] == 2026)].set_index("match_id")
    ch = pd.read_csv(PROC / "chases.csv").set_index("match_id")
    mx = p26.groupby("match_id")["INH_pre"].agg(["max", "min", "size"])
    won = fin["state"] == "WIN"
    close = fin[won & (fin["balls_left"] <= 1) & (mx.reindex(fin.index)["size"] >= 18)].index
    if len(close) == 0:
        close = fin[(fin["over"] == 20)].index
    comfy = fin[won].sort_values("balls_left", ascending=False).index
    comfy = [m for m in comfy if mx.loc[m, "size"] >= 12]
    turn = mx.reindex(fin[~won].index).sort_values("max", ascending=False).index
    picks = [(close[0], "Last-over finish"), (comfy[0], "Comfortable chase"), (turn[0], "Collapse from a winning position")]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharey=True, constrained_layout=True)
    for a, (mid, ttl) in zip(ax, picks):
        d = p26[p26["match_id"] == mid].sort_values("over")
        a.plot(d["over"], d["INH_pre"], "-o", color=BLUE, ms=5, label="Markov INH, pre-rule")
        a.plot(d["over"], d["INH_eb"], "--", color=AQUA, label="Markov INH, EB")
        a.plot(d["over"], d["LR_impact"], "-s", color=ORANGE, ms=4, lw=1.5, label="LR, Impact-only")
        a.axhline(0.5, color=GREY, lw=1, ls=":")
        res = "chase won" if ch.loc[mid, "chase_won"] == 1 else "chase lost"
        a.set_title(f"{ttl}\nmatch {mid}, target {ch.loc[mid, 'target']}, {res}", color=INK, fontsize=10, loc="left")
        a.set_xlabel("Completed over"); a.set_ylim(-0.03, 1.03); a.set_xticks(range(1, 20, 2))
        for _, r in d.iloc[::4].iterrows():
            a.annotate(f"{int(r['runs'])}/{int(r['wickets'])}", (r["over"], r["INH_pre"]), textcoords="offset points",
                       xytext=(0, -15), ha="center", fontsize=7.5, color=INK2)
    ax[0].set_ylabel("P(chase won)"); ax[0].legend(loc="lower left", fontsize=8.5)
    save("fig_era_2026_traces.png")

    # 5 ---------------------------------------------------------------- LOSO season Brier
    lo_df = pd.read_csv(PROC / "era_loso_seasons.csv")
    prs, pst = lo_df[lo_df.season <= 2022], lo_df[lo_df.season >= 2023]
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    ax.axhspan(prs["Brier_INH"].min(), prs["Brier_INH"].max(), color=BLUE, alpha=0.08,
               label="Range of pre-era seasons (Markov INH, LOSO)")
    ax.plot(lo_df.season, lo_df["Brier_INH"], "-o", color=BLUE, label="Markov INH")
    ax.plot(lo_df.season, lo_df["Brier_LR"], "-s", color=ORANGE, lw=1.5, ms=5, label="Logistic regression")
    ax.axvline(2022.5, color=INK2, lw=1, ls="--")
    ax.text(2022.6, ax.get_ylim()[1], "Impact Player rule", va="top", fontsize=9, color=INK2)
    ax.set_xticks(lo_df.season); ax.set_xticklabels(lo_df.season, rotation=45)
    ax.set_ylabel("Brier score (lower = better)")
    ax.set_title("Season-by-season predictive error: leave-one-season-out (<=2022) vs pre-rule model (>=2023)",
                 color=INK, fontweight="bold", loc="left", fontsize=10.5)
    ax.legend(loc="lower left", fontsize=8.5)
    save("fig_era_loso_brier.png")

    # 5b --------------------------------------------------------------- targets by season
    ch_all = pd.read_csv(PROC / "chases.csv")
    g = ch_all.groupby("season").agg(mean_target=("target", "mean"), p200=("target", lambda t: (t >= 200).mean()))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
    cols = [ORANGE if yv >= 2023 else BLUE for yv in g.index]
    ax[0].plot(g.index, g["mean_target"], "-", color=GREY, lw=1.2, zorder=1)
    ax[0].scatter(g.index, g["mean_target"], c=cols, s=55, zorder=2)
    ax[0].set_ylabel("Mean target (runs)")
    ax[0].set_title("Mean chase target by season", color=INK, fontweight="bold", loc="left")
    ax[1].bar(g.index, 100 * g["p200"], color=cols, width=0.7)
    ax[1].set_ylabel("% of chases with target >= 200")
    ax[1].set_title("Share of 200+ targets (orange = Impact Player era)", color=INK, fontweight="bold", loc="left")
    for a in ax:
        a.set_xticks(g.index); a.set_xticklabels(g.index, rotation=45, fontsize=8.5)
    save("fig_targets_by_season.png")

    # 6 ---------------------------------------------------------------- state diagram
    Ptr = P_pre
    pos = {}
    for wi, w in enumerate(config.WICKET_ORDER):
        for pi_, p in enumerate(config.PRESSURE_LABELS):
            pos[f"{w}_{p}"] = (pi_ * 2.0, -wi * 2.2)
    pos["WIN"], pos["LOSS"] = (11.0, -0.6), (11.0, -3.8)
    fig, ax = plt.subplots(figsize=(14, 7.5)); ax.set_axis_off()
    for i, s in enumerate(S):
        for j, t in enumerate(ALL):
            p = Ptr[i, j]
            if p < 0.10:
                continue
            if s == t:
                continue
            col = AQUA if t == "WIN" else (ORANGE if t == "LOSS" else BLUE)
            arr = FancyArrowPatch(pos[s], pos[t], arrowstyle="-|>", mutation_scale=11, lw=0.6 + 5 * p,
                                  color=col, alpha=0.55, connectionstyle="arc3,rad=0.15", shrinkA=17, shrinkB=17)
            ax.add_patch(arr)
    for s, (x0, y0) in pos.items():
        fc = "#dff3ea" if s == "WIN" else ("#fde6dc" if s == "LOSS" else "#eaf2fc")
        ax.add_patch(plt.Circle((x0, y0), 0.55, fc=fc, ec=INK2, lw=1.0, zorder=3))
        ax.text(x0, y0 + (0.12 if s in S else 0), short(s) if s in S else s, ha="center", va="center",
                fontsize=9, zorder=4, fontweight="bold", color=INK)
        if s in S:
            ax.text(x0, y0 - 0.22, f"stay {Ptr[S.index(s), S.index(s)]:.2f}", ha="center", va="center",
                    fontsize=7, zorder=4, color=INK2)
    for wi, w in enumerate(["0-2 wkts", "3-5 wkts", "6-9 wkts"]):
        ax.text(-1.3, -wi * 2.2, w, ha="right", va="center", fontsize=9, color=INK2)
    for pi_, p in enumerate(["P1 RRR-CRR<-2", "P2 [-2,0)", "P3 [0,2)", "P4 [2,4)", "P5 >=4"]):
        ax.text(pi_ * 2.0, 1.2, p, ha="center", fontsize=8.5, color=INK2)
    ax.set_xlim(-3, 12.3); ax.set_ylim(-5.6, 1.8); ax.set_aspect("equal")
    ax.set_title("State-transition diagram, pre-rule chain (edges with p >= 0.10; width proportional to p; "
                 "stay = self-loop p)", color=INK, fontsize=10.5, loc="left")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=3, label="to transient state"),
                       Line2D([], [], color=AQUA, lw=3, label="to WIN"),
                       Line2D([], [], color=ORANGE, lw=3, label="to LOSS")], loc="lower right", fontsize=9)
    save("fig_state_transition_diagram.png")

    # 7 ---------------------------------------------------------------- state probabilities vs time
    Pk = E.inh_P(Ck_pre)
    ost = allst[allst["season"] <= 2022]
    o1 = ost[ost["over"] == 1]
    pi = np.array([(o1["state"] == s).mean() for s in ALL])
    groups = {"0-2 wkts": [IDX[s] for s in S if s.startswith("W_Low")],
              "3-5 wkts": [IDX[s] for s in S if s.startswith("W_Mid")],
              "6-9 wkts": [IDX[s] for s in S if s.startswith("W_High")],
              "WIN": [IDX["WIN"]], "LOSS": [IDX["LOSS"]]}
    model = {g: [] for g in groups}; emp = {g: [] for g in groups}
    mids_tr = set(o1["match_id"])
    absr = ost[ost["state"].isin(["WIN", "LOSS"]) & ost["match_id"].isin(mids_tr)].set_index("match_id")
    n = len(mids_tr)
    ks = list(range(1, 21))
    v = pi.copy()
    absorb_block = np.hstack([np.zeros((2, NT)), np.eye(2)])
    for k in ks:
        for g, ix in groups.items():
            model[g].append(v[ix].sum())
        # empirical: state at end of over k (absorbed if absorption over <= k)
        at = ost[(ost["over"] == k) & ost["state"].isin(S) & ost["match_id"].isin(mids_tr)]
        for g in ["0-2 wkts", "3-5 wkts", "6-9 wkts"]:
            emp[g].append(at["state"].isin([ALL[i] for i in groups[g]]).sum() / n)
        emp["WIN"].append(((absr["state"] == "WIN") & (absr["over"] <= k)).sum() / n)
        emp["LOSS"].append(((absr["state"] == "LOSS") & (absr["over"] <= k)).sum() / n)
        if k <= 19:
            v = v @ np.vstack([Pk[k - 1], absorb_block])
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    for (g, c) in zip(groups, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]):
        ax.plot(ks, model[g], color=c, label=f"{g} (model)")
        ax.plot(ks, emp[g], "o", color=c, ms=5, mfc="white", mew=1.5)
    ax.plot([], [], "o", color=INK2, mfc="white", label="empirical (open markers)")
    ax.set_xlabel("End of over k"); ax.set_ylabel("Probability"); ax.set_xticks(ks)
    ax.set_title("State probabilities vs time: $\\pi_k = \\pi_1 P_1 P_2 \\cdots P_{k-1}$ (time-inhomogeneous, train 2008-22)",
                 color=INK, fontweight="bold", loc="left", fontsize=10.5)
    ax.legend(loc="center left", fontsize=8.5, ncol=1)
    save("fig_state_prob_vs_time.png")
    pd.DataFrame({"over": ks, **{f"model_{g}": model[g] for g in groups},
                  **{f"emp_{g}": emp[g] for g in groups}}).to_csv(PROC / "state_prob_vs_time.csv", index=False)

    # 8 ---------------------------------------------------------------- sample paths (INH simulation)
    rng = np.random.default_rng(config.RANDOM_SEED)
    nsim = 10000
    full = [np.vstack([Pk[k], absorb_block]) for k in range(19)]
    cum = [np.cumsum(M, 1) for M in full]
    st = rng.choice(len(ALL), size=nsim, p=pi)
    paths = np.zeros((nsim, 20), dtype=int); paths[:, 0] = st
    for k in range(19):
        u = rng.random(nsim)
        cur = paths[:, k]
        paths[:, k + 1] = np.minimum((u[:, None] > cum[k][cur]).sum(1), len(ALL) - 1)
    absorbed = paths >= NT
    absorb_over = np.where(absorbed.any(1), absorbed.argmax(1) + 1, 21)  # over in which absorption happened
    sim_win = (paths[:, -1] == IDX["WIN"]).mean()
    emp_over = absr["over"].values
    emp_win = (absr["state"] == "WIN").mean()
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4), constrained_layout=True, gridspec_kw={"width_ratios": [1.3, 1]})
    yv = {s: i for i, s in enumerate(ALL)}
    for i in range(25):
        p = paths[i]
        end = np.argmax(p >= NT) if (p >= NT).any() else 19
        xs = np.arange(1, end + 2); ys = p[:end + 1] + rng.uniform(-0.15, 0.15, end + 1)
        c = AQUA if p[end] == IDX["WIN"] else ORANGE
        ax[0].plot(xs, ys, "-", color=c, lw=1.0, alpha=0.75)
        ax[0].plot(xs[-1], ys[-1], "o", color=c, ms=5)
    ax[0].set_yticks(range(len(ALL))); ax[0].set_yticklabels([short(s) for s in ALL], fontsize=8)
    ax[0].set_xlabel("End of over k"); ax[0].set_xticks(range(1, 21))
    ax[0].set_title("25 simulated sample paths (green = WIN, orange = LOSS)", color=INK, fontweight="bold", loc="left")
    bins = np.arange(0.5, 21.5)
    ax[1].hist(absorb_over, bins=bins, density=True, color=BLUE, alpha=0.75, label=f"Simulated (n={nsim:,})",
               rwidth=0.85)
    ax[1].hist(emp_over, bins=bins, density=True, histtype="step", color=INK, lw=1.8,
               label=f"Empirical train (n={len(emp_over)})")
    ax[1].set_xlabel("Over in which the chase ended"); ax[1].set_ylabel("Probability")
    ax[1].set_title(f"Absorption time: sim P(win)={sim_win:.3f} vs empirical {emp_win:.3f}",
                    color=INK, fontweight="bold", loc="left")
    ax[1].legend(loc="upper left")
    save("fig_sample_paths.png")
    print(f"INH simulation: P(win) sim = {sim_win:.4f}, empirical train = {emp_win:.4f}; "
          f"mean absorption over sim = {absorb_over.mean():.3f}, empirical = {emp_over.mean():.3f}")
    from scipy.stats import ks_2samp
    print(f"Absorption-over distributions: KS stat = {ks_2samp(absorb_over, emp_over).statistic:.4f}")


if __name__ == "__main__":
    main()
