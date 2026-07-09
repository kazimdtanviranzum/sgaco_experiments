# -*- coding: utf-8 -*-
"""Resumable experimental grid. Run repeatedly; each invocation works for ~MAX_SEC then exits.
State: results.csv (append), inst_cache/*.npz (corpus+weights), curves_*.npz chunks, timings.csv."""
import json, time, csv, os, sys, glob
import numpy as np
from engine import load_instance, greedy_rule, tabu_search
from methods import due_dates, build_corpus, train_surrogate, shap_signed_weights, run_aco

INSTANCES = ['ft06','la01','la02','la03','la04','la05','ft10','abz5','orb01','ta01']
SEEDS = list(range(1, 26))
N_ANTS, ITERS, G = 12, 250, 25
BUDGET = N_ANTS * ITERS
MAX_SEC = 260
t_start = time.time()

os.makedirs('inst_cache', exist_ok=True)
meta = {d['name']: d for d in json.load(open('instances.json'))}

done = set()
if os.path.exists('results.csv'):
    with open('results.csv') as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            if row: done.add((row[0], row[1], int(row[2])))
else:
    with open('results.csv','w',newline='') as f:
        csv.writer(f).writerow(['instance','method','seed','best','wall','shap_time','fit_time','refreshes'])
if not os.path.exists('timings.csv'):
    with open('timings.csv','w',newline='') as f:
        csv.writer(f).writerow(['instance','refreshes','fit_s','treeshap_s','run_wall_s'])

res_f = open('results.csv','a',newline=''); res_w = csv.writer(res_f)
tim_f = open('timings.csv','a',newline=''); tim_w = csv.writer(tim_f)
chunk_id = len(glob.glob('curves_*.npz'))
curves = {}

def flush_exit(code):
    res_f.flush(); tim_f.flush()
    if curves:
        np.savez_compressed(f'curves_{chunk_id}.npz', **curves)
    ii = []
    for nm in INSTANCES:
        p = f'inst_cache/{nm}.npz'
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            ii.append(dict(name=nm, jobs=int(z['nj']), machines=int(z['nm_']),
                bks=meta[nm]['optimum'], corpus_best=int(z['elite_vals'][0]),
                surrogate_acc=float(z['acc']), n_train=int(z['n_train']),
                fit_s=float(z['tfit']), treeshap_s=float(z['tshap']),
                weights=[float(x) for x in z['w']]))
    json.dump(ii, open('inst_info.json','w'), indent=1)
    print('EXIT', code, 'elapsed', round(time.time()-t_start), flush=True)
    sys.exit(code)

def get_instance_pack(name):
    p = f'inst_cache/{name}.npz'
    nj, nm_, mach, pt = load_instance(f'instances/{name}')
    due = due_dates(pt)
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        elites = [(int(v), o.astype(np.int64)) for v, o in zip(z['elite_vals'], z['elite_ords'])]
        poor = [(int(v), o.astype(np.int64)) for v, o in zip(z['poor_vals'], z['poor_ords'])]
        return nj, nm_, mach, pt, due, elites, poor, z['w'], z['mu'], z['sd']
    rng = np.random.default_rng(1000)
    elites, poor = build_corpus(nj, nm_, mach, pt, due, seed=7)
    clf, X, y, acc, tfit = train_surrogate(nj, nm_, mach, pt, due, elites, poor, rng)
    w, mu, sd, tshap = shap_signed_weights(clf, X)
    np.savez_compressed(p, nj=nj, nm_=nm_,
        elite_vals=np.array([v for v,_ in elites]), elite_ords=np.stack([o for _,o in elites]),
        poor_vals=np.array([v for v,_ in poor]), poor_ords=np.stack([o for _,o in poor]),
        w=w, mu=mu, sd=sd, acc=acc, n_train=len(y), tfit=tfit, tshap=tshap)
    print(f'[{name}] corpus best={elites[0][0]} bks={meta[name]["optimum"]} acc={acc:.3f} '
          f'fit={tfit:.2f}s TreeSHAP={tshap:.3f}s', flush=True)
    return nj, nm_, mach, pt, due, elites, poor, w, mu, sd

for name in INSTANCES:
    todo = [(m, s) for s in SEEDS for m in
            (['AS-std','MMAS','ACS','Tabu','SHAP-off','SG-ACO'] +
             (['Mag-only'] if name in ('ft06','la02') else []))
            if (name, m, s) not in done]
    rules_todo = [r for r in ['SPT','MWKR','MOR'] if (name, r, 0) not in done]
    if not todo and not rules_todo:
        continue
    nj, nm_, mach, pt, due, elites, poor, w, mu, sd = get_instance_pack(name)
    O = nj * nm_
    order = np.zeros(O, np.int64)
    for r, rn in enumerate(['SPT','MWKR','MOR']):
        if rn in rules_todo:
            v = greedy_rule(nj, nm_, mach, pt, due, r, order)
            res_w.writerow([name, rn, 0, int(v), 0, 0, 0, 0])
    mwkr_order = np.zeros(O, np.int64); greedy_rule(nj, nm_, mach, pt, due, 1, mwkr_order)
    for m, seed in todo:
        if m == 'AS-std':
            r = run_aco(nj,nm_,mach,pt,due,'as','inv_pt',None,None,None,seed,N_ANTS,ITERS)
            curves[f'{name}|AS-std|{seed}'] = r['curve']
        elif m == 'MMAS':
            r = run_aco(nj,nm_,mach,pt,due,'mmas','inv_pt',None,None,None,seed,N_ANTS,ITERS)
        elif m == 'ACS':
            r = run_aco(nj,nm_,mach,pt,due,'acs','inv_pt',None,None,None,seed,N_ANTS,ITERS)
        elif m == 'Tabu':
            t0 = time.perf_counter()
            bv = tabu_search(nj, nm_, mach, pt, mwkr_order, BUDGET, 10, seed)
            r = dict(best=int(bv), wall=time.perf_counter()-t0, shap_time=0, fit_time=0, refreshes=0)
        elif m == 'SHAP-off':
            r = run_aco(nj,nm_,mach,pt,due,'as','shap',w,mu,sd,seed,N_ANTS,ITERS)
            curves[f'{name}|SHAP-off|{seed}'] = r['curve']
        elif m == 'SG-ACO':
            r = run_aco(nj,nm_,mach,pt,due,'as','shap',w,mu,sd,seed,N_ANTS,ITERS,
                        refresh_G=G, elites0=elites, poor0=poor)
            curves[f'{name}|SG-ACO|{seed}'] = r['curve']
            tim_w.writerow([name, r['refreshes'], f"{r['fit_time']:.3f}",
                            f"{r['shap_time']:.3f}", f"{r['wall']:.2f}"])
        else:  # Mag-only
            r = run_aco(nj,nm_,mach,pt,due,'as','shap',np.abs(w),mu,sd,seed,N_ANTS,ITERS)
        res_w.writerow([name, m, seed, r['best'], f"{r['wall']:.2f}",
                        f"{r.get('shap_time',0):.3f}", f"{r.get('fit_time',0):.3f}",
                        r.get('refreshes',0)])
        if time.time() - t_start > MAX_SEC:
            flush_exit(42)
    print(f'[{name}] complete, elapsed {time.time()-t_start:.0f}s', flush=True)
flush_exit(0)
