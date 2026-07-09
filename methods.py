# -*- coding: utf-8 -*-
"""SG-ACO methods: corpus, GBT surrogate, real TreeSHAP signed weights, ACO variants."""
import time
import numpy as np
import shap
from sklearn.ensemble import GradientBoostingClassifier
from engine import (NF, TWK_FACTOR, construct, decode_order, greedy_rule,
                    replay_features, tabu_search)


MAX_TRAIN = 3000


def due_dates(pt):
    return (TWK_FACTOR * pt.sum(axis=1)).astype(np.int64)


def build_corpus(nj, nm, mach, pt, due, seed, n_greedy=300, n_keep=10, n_poor=10):
    """Randomized greedy constructions with random feature-weight vectors + adjacent-swap
    local search; keep best n_keep as elite corpus, plus n_poor random permutations as poor class."""
    rng = np.random.default_rng(seed)
    O = nj * nm
    tau = np.ones((O + 1, O)) # unused weight path: heur-only via beta large
    order = np.zeros(O, np.int64)
    sols = []
    mu = np.zeros(NF); sd = np.ones(NF)
    for i in range(n_greedy):
        w = rng.normal(0, 1, NF)
        st = np.array([rng.integers(1, 2**62)], np.uint64)
        v = construct(nj, nm, mach, pt, due, tau, 0.0, 1.0, 1, w, mu, sd, 0.35,
                      0.0, 0, 0.0, 0.0, st, order)
        sols.append((v, order.copy()))
    sols.sort(key=lambda t: t[0])
    # light first-improvement adjacent-swap local search on the best few
    elites = []
    for v, o in sols[:n_keep * 3]:
        v2, o2 = _local_search(nj, nm, mach, pt, o, v, 40 * O)
        elites.append((v2, o2))
    elites.sort(key=lambda t: t[0])
    elites = elites[:n_keep]
    poor = []
    for i in range(n_poor):
        o = np.repeat(np.arange(nj), nm)
        rng.shuffle(o)
        poor.append((decode_order(nj, nm, mach, pt, o), o))
    return elites, poor


def train_surrogate(nj, nm, mach, pt, due, elites, poor, rng):
    Xs, ys = [], []
    for v, o in elites:
        Xs.append(replay_features(nj, nm, mach, pt, due, o)); ys.append(np.ones(nj * nm))
    for v, o in poor:
        Xs.append(replay_features(nj, nm, mach, pt, due, o)); ys.append(np.zeros(nj * nm))
    X = np.vstack(Xs); y = np.concatenate(ys)
    if len(y) > MAX_TRAIN:
        idx = rng.choice(len(y), MAX_TRAIN, replace=False)
        X, y = X[idx], y[idx]
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.05, subsample=0.9, random_state=0)
    t0 = time.perf_counter()
    clf.fit(X, y)
    t_fit = time.perf_counter() - t0
    acc = clf.score(X, y)
    return clf, X, y, acc, t_fit


def shap_signed_weights(clf, X):
    """Real TreeSHAP via shap.TreeExplainer. Returns (w, mu, sd, wall_clock_seconds)."""
    t0 = time.perf_counter()
    ex = shap.TreeExplainer(clf)
    sv = ex.shap_values(X)
    t = time.perf_counter() - t0
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    phi = np.abs(sv).mean(axis=0)                      # Eq. (1)
    s = np.zeros(NF)
    for f in range(NF):
        if X[:, f].std() > 0 and sv[:, f].std() > 0:
            s[f] = np.sign(np.corrcoef(X[:, f], sv[:, f])[0, 1])   # Eq. (2)
    w = phi * s                                        # Eq. (3)
    mu = X.mean(axis=0)
    sd = X.std(axis=0); sd[sd == 0] = 1.0
    return w, mu, sd, t


def run_aco(nj, nm, mach, pt, due, variant, heur, w, mu, sd, seed,
            n_ants=12, iters=100, alpha=1.0, beta=2.0, rho=0.1, lam=1.0,
            refresh_G=0, elites0=None, poor0=None):
    """variant: 'as' (Ant System, top-3 deposit), 'mmas', 'acs'.
    heur: 'inv_pt' or 'shap'. refresh_G>0 enables the SG-ACO closed loop.
    Fixed budget = n_ants*iters schedule evaluations.
    Returns dict with best makespan, timings."""
    O = nj * nm
    rng = np.random.default_rng(seed)
    hm = 0 if heur == 'inv_pt' else 1
    if w is None:
        w = np.zeros(NF); mu = np.zeros(NF); sd = np.ones(NF)
    w = w.copy(); mu = mu.copy(); sd = sd.copy()
    tau0 = 1.0
    q0, acs_local, xi = 0.0, 0, 0.1
    if variant == 'acs':
        q0, acs_local = 0.9, 1
        tmp = np.zeros(O, np.int64)
        c_ref = greedy_rule(nj, nm, mach, pt, due, 1, tmp)  # MWKR reference
        tau0 = 1.0 / (O * float(c_ref))
    tau = np.full((O + 1, O), tau0)
    best_v = np.int64(1 << 60); order = np.zeros(O, np.int64)
    curve = np.zeros(iters, np.int64)
    iter_pool = []  # (value, order) of recent elites for refresh
    elites = list(elites0) if elites0 else []
    poor = list(poor0) if poor0 else []
    shap_time = 0.0; fit_time = 0.0; n_ref = 0
    t0 = time.perf_counter()
    for t in range(1, iters + 1):
        results = []
        for k in range(n_ants):
            st = np.array([rng.integers(1, 2**62)], np.uint64)
            v = construct(nj, nm, mach, pt, due, tau, alpha, beta, hm, w, mu, sd, lam,
                          q0, acs_local, tau0, xi, st, order)
            results.append((v, order.copy()))
        results.sort(key=lambda x: x[0])
        if results[0][0] < best_v:
            best_v, best_o = results[0][0], results[0][1].copy()
        curve[t - 1] = best_v
        # pheromone update
        if variant == 'as':
            tau *= (1.0 - rho)
            for v, o in results[:3]:
                _deposit(tau, o, nj, nm, 1.0 / v)
        elif variant == 'mmas':
            tau *= (1.0 - rho)
            _deposit(tau, best_o, nj, nm, 1.0 / best_v)
            tmax = 1.0 / (rho * best_v) * 5.0
            tmin = tmax / (2.0 * O)
            np.clip(tau, tmin, tmax, out=tau)
        else:  # ACS: decay-and-deposit on global-best edges only
            _acs_global(tau, best_o, nj, nm, rho, 1.0 / best_v)
        iter_pool.extend(results[:3])
        # closed-loop refresh
        if refresh_G > 0 and t % refresh_G == 0:
            thr = 1.02 * best_v
            new = [(v, o) for v, o in iter_pool if v <= thr]
            iter_pool = []
            if new:
                elites = sorted(elites + new, key=lambda x: x[0])[:30]
                clf, X, y, acc, tf = train_surrogate(nj, nm, mach, pt, due, elites, poor, rng)
                w2, mu2, sd2, ts = shap_signed_weights(clf, X)
                w, mu, sd = w2, mu2, sd2
                fit_time += tf; shap_time += ts; n_ref += 1
    return dict(best=int(best_v), wall=time.perf_counter() - t0, curve=curve,
                shap_time=shap_time, fit_time=fit_time, refreshes=n_ref)


def _deposit(tau, order, nj, nm, amt):
    prev = nj * nm
    nxt = np.zeros(nj, np.int64)
    for j in order:
        op = j * nm + nxt[j]
        tau[prev, op] += amt
        nxt[j] += 1
        prev = op


def _local_search(nj, nm, mach, pt, order, val, budget):
    cur = order.copy(); cur_v = val; evals = 0; improved = True
    O = len(cur)
    while improved and evals < budget:
        improved = False
        for i in range(O - 1):
            if cur[i] == cur[i + 1]:
                continue
            cur[i], cur[i + 1] = cur[i + 1], cur[i]
            nv = decode_order(nj, nm, mach, pt, cur)
            evals += 1
            if nv < cur_v:
                cur_v = nv; improved = True
            else:
                cur[i], cur[i + 1] = cur[i + 1], cur[i]
            if evals >= budget:
                break
    return cur_v, cur


def _acs_global(tau, order, nj, nm, rho, amt):
    prev = nj * nm
    nxt = np.zeros(nj, np.int64)
    for j in order:
        op = j * nm + nxt[j]
        tau[prev, op] = (1.0 - rho) * tau[prev, op] + rho * amt
        nxt[j] += 1
        prev = op
