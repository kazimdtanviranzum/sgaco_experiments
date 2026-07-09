# -*- coding: utf-8 -*-
"""JSSP engine: construction, features, ACO variants (AS/MMAS/ACS), SHAP-guided ACO,
tabu search, dispatching rules. Numba-compiled inner loops.
Due dates for JobSlack are synthetic: d_j = TWK_FACTOR * total processing time of job j (TWK rule)."""
import numpy as np
from numba import njit

TWK_FACTOR = 1.5
NF = 8  # ProcTime, JobRemWork, MachRemWork, EarliestStart, MachIdle, QueueLen, OpsLeftJob, JobSlack


def load_instance(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        rows.append([int(x) for x in line.split()])
    n, m = rows[0][0], rows[0][1]
    mach = np.zeros((n, m), np.int64)
    pt = np.zeros((n, m), np.int64)
    for j in range(n):
        r = rows[1 + j]
        for o in range(m):
            mach[j, o] = r[2 * o]
            pt[j, o] = r[2 * o + 1]
    return n, m, mach, pt


@njit(cache=True)
def features_at_state(nj, nm, mach, pt, nxt, job_ready, mach_ready, job_rem, mach_rem, due, out):
    """Compute NF features for each candidate job (one schedulable op per unfinished job).
    Returns number of candidates and fills out[c, NF]; also returns cand job ids."""
    cands = np.empty(nj, np.int64)
    k = 0
    for j in range(nj):
        if nxt[j] < nm:
            cands[k] = j
            k += 1
    # queue length per machine among candidates
    qlen = np.zeros(nm, np.int64)
    for c in range(k):
        j = cands[c]
        qlen[mach[j, nxt[j]]] += 1
    for c in range(k):
        j = cands[c]
        o = nxt[j]
        mm = mach[j, o]
        p = pt[j, o]
        es = job_ready[j] if job_ready[j] > mach_ready[mm] else mach_ready[mm]
        idle = es - mach_ready[mm]
        out[c, 0] = p
        out[c, 1] = job_rem[j]
        out[c, 2] = mach_rem[mm]
        out[c, 3] = es
        out[c, 4] = idle
        out[c, 5] = qlen[mm]
        out[c, 6] = nm - o
        out[c, 7] = due[j] - (es + job_rem[j])
    return k, cands


@njit(cache=True)
def construct(nj, nm, mach, pt, due, tau, alpha, beta, heur_mode, w, mu, sd, lam,
              q0, acs_local, tau0, xi, rng_state, order_out):
    """Build one schedule. heur_mode: 0=1/pt, 1=SHAP exp(lam*w.z), 2=greedy rule vector w on raw feats.
    q0>0 enables ACS pseudo-random-proportional; acs_local=1 applies ACS local pheromone update.
    Returns makespan. order_out receives the flat op sequence (job ids)."""
    O = nj * nm
    nxt = np.zeros(nj, np.int64)
    job_ready = np.zeros(nj, np.int64)
    mach_ready = np.zeros(nm, np.int64)
    job_rem = np.zeros(nj, np.int64)
    mach_rem = np.zeros(nm, np.int64)
    for j in range(nj):
        for o in range(nm):
            job_rem[j] += pt[j, o]
            mach_rem[mach[j, o]] += pt[j, o]
    feats = np.empty((nj, NF), np.float64)
    prev = O  # start node index
    cmax = 0
    for step in range(O):
        k, cands = features_at_state(nj, nm, mach, pt, nxt, job_ready, mach_ready, job_rem, mach_rem, due, feats)
        # desirability
        eta = np.empty(k, np.float64)
        for c in range(k):
            if heur_mode == 0:
                eta[c] = 1.0 / feats[c, 0]
            else:
                s = 0.0
                for f in range(NF):
                    z = (feats[c, f] - mu[f]) / sd[f]
                    s += w[f] * z
                v = lam * s
                if v > 30.0:
                    v = 30.0
                elif v < -30.0:
                    v = -30.0
                eta[c] = np.exp(v)
        # scores
        sc = np.empty(k, np.float64)
        tot = 0.0
        best_c = 0
        best_v = -1.0
        for c in range(k):
            j = cands[c]
            op = j * nm + nxt[j]
            v = (tau[prev, op] ** alpha) * (eta[c] ** beta)
            sc[c] = v
            tot += v
            if v > best_v:
                best_v = v
                best_c = c
        # choose
        rng_state[0] = (rng_state[0] * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        u = (rng_state[0] >> 11) / 9007199254740992.0
        if q0 > 0.0 and u < q0:
            chosen = best_c
        else:
            rng_state[0] = (rng_state[0] * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            r = ((rng_state[0] >> 11) / 9007199254740992.0) * tot
            acc = 0.0
            chosen = k - 1
            for c in range(k):
                acc += sc[c]
                if r <= acc:
                    chosen = c
                    break
        j = cands[chosen]
        o = nxt[j]
        mm = mach[j, o]
        p = pt[j, o]
        es = job_ready[j] if job_ready[j] > mach_ready[mm] else mach_ready[mm]
        fin = es + p
        job_ready[j] = fin
        mach_ready[mm] = fin
        job_rem[j] -= p
        mach_rem[mm] -= p
        nxt[j] += 1
        if fin > cmax:
            cmax = fin
        op = j * nm + o
        if acs_local == 1:
            tau[prev, op] = (1.0 - xi) * tau[prev, op] + xi * tau0
        order_out[step] = j
        prev = op
    return cmax


@njit(cache=True)
def decode_order(nj, nm, mach, pt, order):
    nxt = np.zeros(nj, np.int64)
    job_ready = np.zeros(nj, np.int64)
    mach_ready = np.zeros(nm, np.int64)
    cmax = 0
    for s in range(order.shape[0]):
        j = order[s]
        o = nxt[j]
        mm = mach[j, o]
        es = job_ready[j] if job_ready[j] > mach_ready[mm] else mach_ready[mm]
        fin = es + pt[j, o]
        job_ready[j] = fin
        mach_ready[mm] = fin
        nxt[j] += 1
        if fin > cmax:
            cmax = fin
    return cmax


@njit(cache=True)
def greedy_rule(nj, nm, mach, pt, due, rule, order_out):
    """Deterministic dispatching: rule 0=SPT, 1=MWKR, 2=MOR (most operations remaining)."""
    nxt = np.zeros(nj, np.int64)
    job_ready = np.zeros(nj, np.int64)
    mach_ready = np.zeros(nm, np.int64)
    job_rem = np.zeros(nj, np.int64)
    for j in range(nj):
        for o in range(nm):
            job_rem[j] += pt[j, o]
    cmax = 0
    for s in range(nj * nm):
        best_j = -1
        best_key = 1e18
        for j in range(nj):
            if nxt[j] < nm:
                o = nxt[j]
                if rule == 0:
                    key = pt[j, o]
                elif rule == 1:
                    key = -job_rem[j]
                else:
                    key = -(nm - o)
                if key < best_key:
                    best_key = key
                    best_j = j
        j = best_j
        o = nxt[j]
        mm = mach[j, o]
        es = job_ready[j] if job_ready[j] > mach_ready[mm] else mach_ready[mm]
        fin = es + pt[j, o]
        job_ready[j] = fin
        mach_ready[mm] = fin
        job_rem[j] -= pt[j, o]
        nxt[j] += 1
        order_out[s] = j
        if fin > cmax:
            cmax = fin
    return cmax


@njit(cache=True)
def tabu_search(nj, nm, mach, pt, order0, budget, tenure, seed):
    """Simple tabu search on operation-list encoding with adjacent-swap neighborhood
    restricted to swaps that change machine sequences; tabu on swapped position pairs.
    Fixed budget in schedule evaluations."""
    O = nj * nm
    cur = order0.copy()
    best = order0.copy()
    cur_v = decode_order(nj, nm, mach, pt, cur)
    best_v = cur_v
    tabu = np.zeros((O, O), np.int64)
    it = 0
    evals = 0
    rng = np.int64(seed * 2654435761 + 12345)
    while evals < budget:
        it += 1
        best_i = -1
        best_nv = np.int64(1 << 60)
        # sample candidate swap positions
        trials = 3 * O if 3 * O < budget - evals else budget - evals
        for t in range(trials):
            rng = (rng * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            i = int((rng >> 33) % (O - 1))
            if cur[i] == cur[i + 1]:
                continue
            cur[i], cur[i + 1] = cur[i + 1], cur[i]
            nv = decode_order(nj, nm, mach, pt, cur)
            evals += 1
            cur[i], cur[i + 1] = cur[i + 1], cur[i]
            if (tabu[cur[i], cur[i + 1]] > it and nv >= best_v):
                continue
            if nv < best_nv:
                best_nv = nv
                best_i = i
            if evals >= budget:
                break
        if best_i < 0:
            continue
        i = best_i
        a = cur[i]
        b = cur[i + 1]
        cur[i], cur[i + 1] = b, a
        cur_v = best_nv
        tabu[b, a] = it + tenure
        if cur_v < best_v:
            best_v = cur_v
            best = cur.copy()
    return best_v


def replay_features(nj, nm, mach, pt, due, order):
    """Replay a schedule, returning the feature vector of the chosen op at each step."""
    X = np.zeros((nj * nm, NF))
    nxt = np.zeros(nj, np.int64)
    job_ready = np.zeros(nj, np.int64)
    mach_ready = np.zeros(nm, np.int64)
    job_rem = pt.sum(axis=1).astype(np.int64)
    mach_rem = np.zeros(nm, np.int64)
    for j in range(nj):
        for o in range(nm):
            mach_rem[mach[j, o]] += pt[j, o]
    feats = np.empty((nj, NF))
    for s, j in enumerate(order):
        k, cands = features_at_state(nj, nm, mach, pt, nxt, job_ready, mach_ready, job_rem, mach_rem, due, feats)
        ci = int(np.where(cands[:k] == j)[0][0])
        X[s] = feats[ci]
        o = nxt[j]
        mm = mach[j, o]
        es = max(job_ready[j], mach_ready[mm])
        fin = es + pt[j, o]
        job_ready[j] = fin
        mach_ready[mm] = fin
        job_rem[j] -= pt[j, o]
        mach_rem[mm] -= pt[j, o]
        nxt[j] += 1
    return X
