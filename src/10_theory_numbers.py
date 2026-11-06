"""
src/10_theory_numbers.py
Numerical checks of the theory used in the report (train seasons <= 2022 only):
  * canonical form P = [[Q, R], [0, I]] and the n-step block identity
        P^n = [[Q^n, (I + Q + ... + Q^{n-1}) R], [0, I]]
  * Chapman-Kolmogorov P^(m+n) = P^m P^n
  * absorption probabilities B = N R, expected steps t = N 1, Var(T) = (2N - I) t - t^2
  * why there is NO unique stationary distribution (absorbing => reducible):
        eigenvalue 1 has multiplicity 2; every pi = (0,...,0, a, 1-a) is stationary;
        lim pi_0 P^n = (0, pi_0 B) depends on pi_0; convergence rate = rho(Q)^n
  * time-inhomogeneous chain: n-step = product P_k ... P_{k+n-1}; absorption by over 20 is certain
"""
import importlib
import numpy as np
import pandas as pd
import config

E = importlib.import_module("08_era_experiments")
S, ALL, NT = E.S, E.ALL, E.NT


def main():
    print("=" * 74); print("SCRIPT 10: Theory checks (train 2008-22)"); print("=" * 74)
    P = np.load(config.PROCESSED_DATA_DIR / "P_matrix.npy")
    Q, R = P[:NT, :NT], P[:NT, NT:]
    I = np.eye(NT)
    # n-step block identity and Chapman-Kolmogorov
    for n in [2, 5, 10]:
        Pn = np.linalg.matrix_power(P, n)
        geo = sum(np.linalg.matrix_power(Q, m) for m in range(n)) @ R
        print(f"n={n:2d}: max|P^n[T,T]-Q^n| = {np.abs(Pn[:NT, :NT]-np.linalg.matrix_power(Q, n)).max():.2e}, "
              f"max|P^n[T,A]-(sum Q^m)R| = {np.abs(Pn[:NT, NT:]-geo).max():.2e}")
    ck = np.abs(np.linalg.matrix_power(P, 7) - np.linalg.matrix_power(P, 3) @ np.linalg.matrix_power(P, 4)).max()
    print(f"Chapman-Kolmogorov P^7 = P^3 P^4: max error {ck:.2e}")
    # worked example: from W_Low_P3
    i = S.index("W_Low_P3")
    print("\nWorked example, start in W_Low_P3 (homogeneous chain):")
    for n in [1, 2, 5, 10, 20, 50]:
        r = np.linalg.matrix_power(P, n)[i]
        top = np.argsort(r[:NT])[::-1][:2]
        print(f"  n={n:2d}: P(WIN)={r[NT]:.3f} P(LOSS)={r[NT+1]:.3f} P(transient)={r[:NT].sum():.3f}  "
              f"most likely transient: {S[top[0]]} {r[top[0]]:.3f}, {S[top[1]]} {r[top[1]]:.3f}")
    N = np.linalg.inv(I - Q)
    B = N @ R; t = N @ np.ones(NT); var = (2 * N - I) @ t - t ** 2
    print(f"  B(W_Low_P3) = {B[i,0]:.4f} (WIN), {B[i,1]:.4f} (LOSS);  t = {t[i]:.2f} overs, sd = {np.sqrt(var[i]):.2f}")
    print(f"  N[W_Low_P3, W_Low_P3] = {N[i,i]:.2f} = expected visits (incl. start) to W_Low_P3")
    # stationary distributions
    ev, evec = np.linalg.eig(P.T)
    ones = np.isclose(ev, 1.0)
    print(f"\nEigenvalues of P equal to 1: {ones.sum()} (multiplicity) -> stationary space is 2-dimensional")
    print(f"Spectral radius of Q: {np.max(np.abs(np.linalg.eigvals(Q))):.4f} -> P(still in play after n overs) "
          f"<= C * {np.max(np.abs(np.linalg.eigvals(Q))):.3f}^n")
    for a in [0.0, 0.3, 1.0]:
        pi = np.zeros(len(ALL)); pi[NT] = a; pi[NT + 1] = 1 - a
        print(f"  pi = (0,..,0,{a},{1-a}): max|pi P - pi| = {np.abs(pi @ P - pi).max():.1e}  (stationary)")
    over1 = pd.read_csv(config.PROCESSED_DATA_DIR / "over_states.csv")
    over1 = over1[(over1.season <= 2022) & (over1.over == 1)]
    pi0 = np.array([(over1.state == s).mean() for s in ALL])
    lim = pi0 @ np.linalg.matrix_power(P, 500)
    print(f"  lim pi_0 P^n (pi_0 = over-1 distribution): WIN {lim[NT]:.4f}, LOSS {lim[NT+1]:.4f}, "
          f"transient {lim[:NT].sum():.1e}; equals pi_0 B = {pi0[:NT] @ B[:,0]:.4f}")
    e1 = np.zeros(len(ALL)); e1[S.index("W_High_P5")] = 1
    l2 = e1 @ np.linalg.matrix_power(P, 500)
    print(f"  lim from W_High_P5: WIN {l2[NT]:.4f}, LOSS {l2[NT+1]:.4f} -> limit depends on start: no unique stationary law")
    # inhomogeneous chain: product of matrices
    tr, obs, T, mids, season = E.load()
    Pk = E.inh_P(E.counts(T, season <= 2022))
    blk = np.hstack([np.zeros((2, NT)), np.eye(2)])
    full = [np.vstack([Pk[k], blk]) for k in range(19)]
    prod = np.eye(len(ALL))
    for k in range(19):
        prod = prod @ full[k]
    print(f"\nTime-inhomogeneous chain: P_1 P_2 ... P_19 restricted to transient columns: max = "
          f"{prod[:NT, :NT].max():.1e}  -> absorption by end of over 20 is certain (finite horizon)")
    p10 = np.eye(len(ALL))
    for k in range(9, 19):
        p10 = p10 @ full[k]
    print(f"From W_Low_P3 at end of over 10: P(WIN) = {p10[i, NT]:.4f} via product P_10...P_19 "
          f"(= backward-recursion V_10 = {E.inh_V(Pk)[9, i]:.4f})")
    print("=" * 74)


if __name__ == "__main__":
    main()
