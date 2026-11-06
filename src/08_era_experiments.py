"""
src/08_era_experiments.py
Impact-Player-era experiments (rule in force from IPL 2023).

Research question: did the Impact Player rule change chase dynamics, and does a
pre-2023 Markov model still hold?

Models (each re-estimated on the chosen training seasons only -> no leakage):
  HOM  : time-homogeneous absorbing chain, B = (I-Q)^-1 R                (scripts 03/04)
  INH  : finite-horizon per-over chain P_1..P_19, backward recursion     (script 06b)
  LR   : logistic regression on wickets, CRR, RRR, runs_needed, balls_left
Training regimes: pre-rule (2008-22), Impact-only (2023-25), all (2008-25), and
empirical Bayes (EB): Dirichlet prior centred on the pre-rule matrix, updated with
2023-25 counts. EB prior weight w is chosen by maximising the Dirichlet-multinomial
marginal likelihood of the 2023-25 counts (type-II ML) -> never looks at 2026.

Outputs (data/processed/):
  era_validation_results.csv     Brier/log-loss/acc + match-level cluster-bootstrap CIs
  era_paired_differences.csv     paired Brier differences on the SAME resampled matches
  era_loso_seasons.csv           leave-one-season-out Brier for pre seasons + post seasons
  era_homogeneity_tests.csv      per-row G-tests (Holm), global G, phase-stratified G,
                                 match-level permutation p-values
  era_B_comparison.csv           B_i pre vs post with bootstrap CIs on B and on the delta
  era_V_over10_comparison.csv    INH win prob at end of over 10, pre vs post, with CIs
  era_eb_weight.csv              marginal likelihood vs prior weight
  era_predictions_2026.csv       per-observation predictions (for figures)
"""
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
import config

FEATURES = ["wickets", "CRR", "RRR", "runs_needed", "balls_left"]
S = config.TRANSIENT_STATES
ALL = config.ALL_STATES
NT, NA, NO = len(S), len(ALL), 19
IDX = {s: i for i, s in enumerate(ALL)}
WIN = IDX["WIN"]
ALPHA_INH = 10.0        # same as 06b
ALPHA_HOM = 1.0         # total pseudo-count per row (tiny; keeps empty rows valid)
PRE, IMPACT, ALLTR = (2008, 2022), (2023, 2025), (2008, 2025)
N_BOOT = config.N_BOOT
N_PERM = 1000
RNG = np.random.default_rng(config.RANDOM_SEED)
PHASES = {"PP": range(1, 6), "MID": range(6, 15), "DEATH": range(15, 20)}  # by over_from (as in 07)


# ------------------------------------------------------------------ data
def load():
    tr = pd.read_csv(config.PROCESSED_DATA_DIR / "transitions.csv")
    os_ = pd.read_csv(config.PROCESSED_DATA_DIR / "over_states.csv")
    ch = pd.read_csv(config.PROCESSED_DATA_DIR / "chases.csv")
    obs = os_.merge(ch[["match_id", "chase_won"]], on="match_id")
    obs = obs[obs["state"].isin(S)].copy()
    for c in FEATURES:
        obs[c] = obs[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # per-match count tensor T[m, over-1, from, to] -> fast refits and bootstraps
    mids = np.sort(tr["match_id"].unique())
    mpos = {m: i for i, m in enumerate(mids)}
    T = np.zeros((len(mids), NO, NT, NA), dtype=np.float32)
    np.add.at(T, (tr["match_id"].map(mpos).values, tr["over_from"].values - 1,
                  tr["from_state"].map(IDX).values, tr["to_state"].map(IDX).values), 1.0)
    season = tr.groupby("match_id")["season"].first().reindex(mids).values
    return tr, obs, T, mids, season


def counts(T, mask_or_weights):
    """Per-over counts C[k, i, j] from selected / weighted matches."""
    w = np.asarray(mask_or_weights, dtype=np.float64)
    return np.tensordot(w, T, axes=(0, 0))


# ------------------------------------------------------------------ models
def hom_P(Ck, prior=None, w=0.0):
    C = Ck.sum(0)
    C = C + (w * prior if prior is not None else ALPHA_HOM / NA)
    return C / C.sum(1, keepdims=True)


def hom_B(P):
    return np.linalg.solve(np.eye(NT) - P[:, :NT], P[:, WIN])


def inh_P(Ck, prior_k=None, w=0.0):
    """Per-over matrices. Without prior: Dirichlet smoothing toward local pool (overs k±2)
    exactly as 06b. With prior_k: EB posterior mean (Ck + w*prior_k)/(n + w)."""
    pooled = Ck.sum(0)
    gp = pooled / np.where(pooled.sum(1, keepdims=True) == 0, 1, pooled.sum(1, keepdims=True))
    gp[gp.sum(1) == 0] = 1.0 / NA
    P = np.zeros_like(Ck)
    for k in range(NO):
        if prior_k is None:
            loc = Ck[max(0, k - 2): k + 3].sum(0)
            ls = loc.sum(1, keepdims=True)
            pr = np.where(ls > 0, loc / np.where(ls == 0, 1, ls), gp)
            P[k] = (Ck[k] + ALPHA_INH * pr) / (Ck[k].sum(1, keepdims=True) + ALPHA_INH)
        else:
            P[k] = (Ck[k] + w * prior_k[k]) / (Ck[k].sum(1, keepdims=True) + w)
    last = P[-1].copy(); last[:, :NT] = 0.0                     # hard horizon at over 19
    s = last[:, NT:].sum(1, keepdims=True)
    fb = gp[:, NT:] / np.maximum(gp[:, NT:].sum(1, keepdims=True), 1e-12)
    last[:, NT:] = np.where(s > 0, last[:, NT:] / np.where(s == 0, 1, s), fb)
    P[-1] = last
    return P


def inh_V(P):
    V = np.zeros((NO, NT)); V[-1] = P[-1][:, WIN]
    for k in range(NO - 2, -1, -1):
        V[k] = P[k][:, WIN] + P[k][:, :NT] @ V[k + 1]
    return V


def floor_prior(p, eps=1e-6):
    p = p + eps
    return p / p.sum(-1, keepdims=True)


def log_ml(C, prior, w):
    """Dirichlet-multinomial log marginal likelihood, summed over rows (any leading dims)."""
    a = w * prior
    n = C.sum(-1)
    return float(np.sum(gammaln(w) - gammaln(w + n)) +
                 np.sum(gammaln(a + C) - gammaln(a)))


def tune_w(C, prior, grid):
    ll = np.array([log_ml(C, prior, w) for w in grid])
    return grid[int(np.argmax(ll))], ll


# ------------------------------------------------------------------ evaluation
def metrics(y, p):
    return {"Brier": brier_score_loss(y, p), "LogLoss": log_loss(y, np.clip(p, 1e-5, 1 - 1e-5)),
            "Acc": accuracy_score(y, (p >= 0.5).astype(int)), "MeanPred": float(np.mean(p)),
            "ObsRate": float(np.mean(y))}


def boot_index(test, n_boot):
    groups = list(test.groupby("match_id").indices.values())
    idxs = []
    for _ in range(n_boot):
        pick = RNG.integers(0, len(groups), len(groups))
        idxs.append(np.concatenate([groups[g] for g in pick]))
    return idxs


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def fit_all(T, season, obs, lo, hi):
    m = (season >= lo) & (season <= hi)
    Ck = counts(T, m)
    tr = obs[obs["season"].between(lo, hi)]
    lr = LogisticRegression(max_iter=2000, random_state=config.RANDOM_SEED).fit(tr[FEATURES], tr["chase_won"])
    return Ck, hom_B(hom_P(Ck)), inh_V(inh_P(Ck)), lr


def predict(test, B, V, lr):
    si = test["state"].map(IDX).values
    return B[si], V[test["over"].values.astype(int) - 1, si], lr.predict_proba(test[FEATURES])[:, 1]


# ------------------------------------------------------------------ era tests
def G_stat(a, b):
    """G = 2 * sum O ln(O/E) for a 2 x K table (rows a, b); returns (G, dof)."""
    tab = np.vstack([a, b]); tab = tab[:, tab.sum(0) > 0]
    if tab.shape[1] < 2 or (tab.sum(1) == 0).any():
        return 0.0, 0
    E = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    nz = tab > 0
    return float(2 * np.sum(tab[nz] * np.log(tab[nz] / E[nz]))), tab.shape[1] - 1


def era_G(Ck_a, Ck_b, stratified):
    """Global G over rows (and phases if stratified). Returns total G, dof, per-row G."""
    per_row, Gt, dft = [], 0.0, 0
    for i in range(NT):
        if stratified:
            g = d = 0
            for ks in PHASES.values():
                kk = [k - 1 for k in ks]
                gg, dd = G_stat(Ck_a[kk, i].sum(0), Ck_b[kk, i].sum(0)); g += gg; d += dd
        else:
            g, d = G_stat(Ck_a[:, i].sum(0), Ck_b[:, i].sum(0))
        per_row.append((g, d)); Gt += g; dft += d
    return Gt, dft, per_row


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------ main
def main():
    print("=" * 78)
    print("SCRIPT 8: Impact Player era experiments")
    print("=" * 78)
    tr, obs, T, mids, season = load()
    pre_mask = (season >= PRE[0]) & (season <= PRE[1])
    imp_mask = (season >= IMPACT[0]) & (season <= IMPACT[1])
    post_mask = season >= 2023
    print(f"Matches: pre 2008-22 = {pre_mask.sum()}, 2023-25 = {imp_mask.sum()}, "
          f"2026 = {(season == 2026).sum()}")

    # ---------------- fits
    fits = {name: fit_all(T, season, obs, lo, hi) for name, (lo, hi) in
            {"pre": PRE, "impact": IMPACT, "all": ALLTR}.items()}
    Ck_pre, Ck_imp = fits["pre"][0], fits["impact"][0]

    # ---------------- EB: tune prior weight on 2023-25 counts only (type-II ML)
    grid = np.unique(np.round(np.logspace(0, 4.5, 46), 2))
    P_pre_h = floor_prior(hom_P(Ck_pre))
    w_h, ll_h = tune_w(Ck_imp.sum(0), P_pre_h, grid)
    P_pre_k = floor_prior(inh_P(Ck_pre))
    Ck_imp_body = Ck_imp.copy()
    w_k, ll_k = tune_w(Ck_imp_body, P_pre_k, grid)
    pd.DataFrame({"w": grid, "logML_hom": ll_h, "logML_inh": ll_k}).to_csv(
        config.PROCESSED_DATA_DIR / "era_eb_weight.csv", index=False)
    print(f"\nEB prior weight by marginal likelihood (2023-25 data only): HOM w* = {w_h:g}, INH w* = {w_k:g}")
    print("  (w* -> large means 2023-25 data are consistent with the pre-rule matrix; small means they differ)")
    B_eb = hom_B(hom_P(Ck_imp, prior=P_pre_h, w=w_h))
    V_eb = inh_V(inh_P(Ck_imp, prior_k=P_pre_k, w=w_k))
    fits["eb"] = (None, B_eb, V_eb, fits["impact"][3])   # LR has no EB analogue -> Impact-only LR

    # ---------------- validation table
    test_sets = {"2023-25": obs[obs["season"].between(2023, 2025)].reset_index(drop=True),
                 "2026": obs[obs["season"] == 2026].reset_index(drop=True)}
    regimes = [("pre", "2023-25"), ("pre", "2026"), ("impact", "2026"), ("all", "2026"), ("eb", "2026")]
    label = {"pre": "pre-rule 2008-22", "impact": "Impact-only 2023-25", "all": "all 2008-25",
             "eb": "EB (pre prior + 2023-25)"}
    rows, preds26, boots = [], {}, {}
    for tname in test_sets:
        boots[tname] = boot_index(test_sets[tname], N_BOOT)
    for reg, tname in regimes:
        te = test_sets[tname]; y = te["chase_won"].values
        _, B, V, lr = fits[reg]
        ph, pi, pl = predict(te, B, V, lr)
        for mname, p in [("HOM", ph), ("INH", pi), ("LR", pl)]:
            m = metrics(y, p)
            bs = [brier_score_loss(y[ix], p[ix]) for ix in boots[tname]]
            lo, hi = ci(bs)
            rows.append({"train": label[reg], "test": tname, "model": mname,
                         "n_matches": te["match_id"].nunique(), "n_obs": len(te),
                         "Brier": m["Brier"], "Brier_lo": lo, "Brier_hi": hi,
                         "LogLoss": m["LogLoss"], "Acc": m["Acc"],
                         "MeanPred": m["MeanPred"], "ObsWinRate": m["ObsRate"]})
            if tname == "2026":
                preds26[f"{mname}_{reg}"] = p
    res = pd.DataFrame(rows)
    res.to_csv(config.PROCESSED_DATA_DIR / "era_validation_results.csv", index=False)
    print("\n--- Out-of-sample validation (match-level cluster bootstrap 95% CI) ---")
    print(res[["train", "test", "model", "n_matches", "Brier", "Brier_lo", "Brier_hi", "LogLoss", "Acc",
               "MeanPred", "ObsWinRate"]].round(4).to_string(index=False))

    te26 = test_sets["2026"].copy()
    for k, v in preds26.items():
        te26[k] = v
    te26.to_csv(config.PROCESSED_DATA_DIR / "era_predictions_2026.csv", index=False)

    # ---------------- paired differences on 2026 (same bootstrap resamples)
    y26 = te26["chase_won"].values
    pairs = [("INH_pre", "INH_impact"), ("INH_pre", "INH_all"), ("INH_pre", "INH_eb"),
             ("INH_pre", "HOM_pre"), ("INH_pre", "LR_pre"), ("INH_pre", "LR_impact"), ("LR_pre", "LR_impact"), ("LR_pre", "LR_all")]
    prow = []
    for a, b in pairs:
        d0 = brier_score_loss(y26, preds26[a]) - brier_score_loss(y26, preds26[b])
        ds = [brier_score_loss(y26[ix], preds26[a][ix]) - brier_score_loss(y26[ix], preds26[b][ix])
              for ix in boots["2026"]]
        lo, hi = ci(ds)
        prow.append({"A": a, "B": b, "Brier_A_minus_B": d0, "lo": lo, "hi": hi,
                     "significant": not (lo <= 0 <= hi)})
    pr = pd.DataFrame(prow)
    pr.to_csv(config.PROCESSED_DATA_DIR / "era_paired_differences.csv", index=False)
    print("\n--- Paired Brier differences on 2026 (negative = A better) ---")
    print(pr.round(4).to_string(index=False))

    # ---------------- leave-one-season-out yardstick
    lrow = []
    for s in range(2008, 2027):
        if s <= 2022:
            msk = pre_mask & (season != s)
            trn = obs[obs["season"].between(*PRE) & (obs["season"] != s)]
        else:
            msk = pre_mask
            trn = obs[obs["season"].between(*PRE)]
        Ck = counts(T, msk)
        lr = LogisticRegression(max_iter=2000, random_state=config.RANDOM_SEED).fit(trn[FEATURES], trn["chase_won"])
        te = obs[obs["season"] == s]
        ph, pi, pl = predict(te, hom_B(hom_P(Ck)), inh_V(inh_P(Ck)), lr)
        y = te["chase_won"].values
        lrow.append({"season": s, "era": "pre (LOSO)" if s <= 2022 else "post (pre-rule model)",
                     "n_matches": te["match_id"].nunique(), "win_rate": te.groupby("match_id")["chase_won"].first().mean(),
                     "Brier_HOM": brier_score_loss(y, ph), "Brier_INH": brier_score_loss(y, pi),
                     "Brier_LR": brier_score_loss(y, pl)})
    lo_df = pd.DataFrame(lrow)
    lo_df.to_csv(config.PROCESSED_DATA_DIR / "era_loso_seasons.csv", index=False)
    print("\n--- Season-by-season Brier (pre seasons: leave-one-season-out; post: pre-rule model) ---")
    print(lo_df.round(4).to_string(index=False))
    pre_b = lo_df.loc[lo_df.season <= 2022, "Brier_INH"]
    print(f"Pre-era LOSO INH Brier: mean {pre_b.mean():.4f}, range [{pre_b.min():.4f}, {pre_b.max():.4f}]")
    for s in range(2023, 2027):
        b = lo_df.loc[lo_df.season == s, "Brier_INH"].iloc[0]
        print(f"  {s}: INH Brier {b:.4f} -> rank among pre seasons: worse than "
              f"{(pre_b < b).sum()}/{len(pre_b)}")

    # ---------------- era homogeneity tests (pre 2008-22 vs post 2023-26)
    Ck_post = counts(T, post_mask)
    G_u, df_u, rows_u = era_G(Ck_pre, Ck_post, stratified=False)
    G_s, df_s, rows_s = era_G(Ck_pre, Ck_post, stratified=True)
    # permutation null: shuffle era labels across matches (keeps within-match dependence)
    sub = pre_mask | post_mask
    Tsub, lab = T[sub], post_mask[sub].astype(float)
    perm_u, perm_s, perm_rows = [], [], []
    for _ in range(N_PERM):
        L = RNG.permutation(lab)
        A, Bc = counts(Tsub, 1 - L), counts(Tsub, L)
        gu, _, ru = era_G(A, Bc, False); gs, _, _ = era_G(A, Bc, True)
        perm_u.append(gu); perm_s.append(gs); perm_rows.append([r[0] for r in ru])
    perm_rows = np.array(perm_rows)
    p_perm_u = (1 + np.sum(np.array(perm_u) >= G_u)) / (N_PERM + 1)
    p_perm_s = (1 + np.sum(np.array(perm_s) >= G_s)) / (N_PERM + 1)
    hrows = []
    for i, s in enumerate(S):
        g, d = rows_u[i]
        hrows.append({"state": s, "n_pre": int(Ck_pre[:, i].sum()), "n_post": int(Ck_post[:, i].sum()),
                      "G": g, "dof": d, "p_chi2": chi2.sf(g, d) if d > 0 else np.nan,
                      "p_perm": (1 + np.sum(perm_rows[:, i] >= g)) / (N_PERM + 1)})
    hd = pd.DataFrame(hrows)
    hd["p_chi2_holm"] = holm(hd["p_chi2"].fillna(1).values)
    hd["p_perm_holm"] = holm(hd["p_perm"].values)
    glob = pd.DataFrame([
        {"state": "GLOBAL (unstratified)", "G": G_u, "dof": df_u, "p_chi2": chi2.sf(G_u, df_u), "p_perm": p_perm_u},
        {"state": "GLOBAL (phase-stratified)", "G": G_s, "dof": df_s, "p_chi2": chi2.sf(G_s, df_s), "p_perm": p_perm_s}])
    pd.concat([hd, glob]).to_csv(config.PROCESSED_DATA_DIR / "era_homogeneity_tests.csv", index=False)
    print("\n--- Era homogeneity of transition rows: pre 2008-22 vs post 2023-26 ---")
    print(hd.round(4).to_string(index=False))
    print(glob.round(6).to_string(index=False))
    print(f"(permutation tests: {N_PERM} match-level shuffles of era labels)")

    # ---------------- B_i and V_10 by era with bootstrap CIs (match resampling within era)
    def era_boot(mask):
        ids = np.where(mask)[0]; Bs, Vs = [], []
        for _ in range(N_BOOT):
            w = np.bincount(RNG.choice(ids, len(ids)), minlength=len(mask))
            Ck = counts(T, w); Bs.append(hom_B(hom_P(Ck))); Vs.append(inh_V(inh_P(Ck))[9])
        return np.array(Bs), np.array(Vs)
    Bpre_b, Vpre_b = era_boot(pre_mask)
    Bpost_b, Vpost_b = era_boot(post_mask)
    B_pre, B_post = hom_B(hom_P(Ck_pre)), hom_B(hom_P(Ck_post))
    V_pre, V_post = inh_V(inh_P(Ck_pre))[9], inh_V(inh_P(Ck_post))[9]
    n10_pre = Ck_pre[9].sum(1); n10_post = Ck_post[9].sum(1)
    for nm, est_a, est_b, ba, bb, fn, extra in [
            ("B (homogeneous absorption prob.)", B_pre, B_post, Bpre_b, Bpost_b, "era_B_comparison.csv", None),
            ("V_10 (INH win prob at end of over 10)", V_pre, V_post, Vpre_b, Vpost_b, "era_V_over10_comparison.csv",
             (n10_pre, n10_post))]:
        d = bb - ba
        rr = []
        for i, s in enumerate(S):
            r = {"state": s, "pre": est_a[i], "pre_lo": ci(ba[:, i])[0], "pre_hi": ci(ba[:, i])[1],
                 "post": est_b[i], "post_lo": ci(bb[:, i])[0], "post_hi": ci(bb[:, i])[1],
                 "delta": est_b[i] - est_a[i], "delta_lo": ci(d[:, i])[0], "delta_hi": ci(d[:, i])[1]}
            r["delta_CI_excludes_0"] = not (r["delta_lo"] <= 0 <= r["delta_hi"])
            if extra is not None:
                r["n_obs_over10_pre"], r["n_obs_over10_post"] = int(extra[0][i]), int(extra[1][i])
            rr.append(r)
        df = pd.DataFrame(rr)
        df.to_csv(config.PROCESSED_DATA_DIR / fn, index=False)
        print(f"\n--- {nm}: pre 2008-22 vs post 2023-26 (95% bootstrap CIs) ---")
        print(df.round(3).to_string(index=False))
    print("=" * 78)


if __name__ == "__main__":
    main()
