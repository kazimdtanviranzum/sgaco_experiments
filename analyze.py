# -*- coding: utf-8 -*-
import json, csv, itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
import scikit_posthocs as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')
meta = {d['name']: d for d in json.load(open('instances.json'))}
info = {d['name']: d for d in json.load(open('inst_info.json'))}
INSTANCES = ['ft06','la01','la02','la03','la04','la05','ft10','abz5','orb01','ta01']
STOCH = ['AS-std','MMAS','ACS','Tabu','SHAP-off','SG-ACO']
LBL = {'AS-std':'Standard ACO (AS)','MMAS':'MAX-MIN Ant System','ACS':'Ant Colony System',
       'Tabu':'Tabu search','SHAP-off':'SHAP-ACO (offline)','SG-ACO':'SG-ACO (closed loop)',
       'SPT':'SPT rule','MWKR':'MWKR rule','MOR':'MOR rule','Mag-only':'Magnitude-only SHAP'}

# ---------- Table A: per-instance summary ----------
rows = []
for inst in INSTANCES:
    bks = meta[inst]['optimum']
    r = {'instance': inst, 'size': f"{meta[inst]['jobs']}x{meta[inst]['machines']}", 'BKS': bks}
    for m in ['SPT','MWKR','MOR']:
        v = df[(df.instance==inst)&(df.method==m)].best.iloc[0]
        r[m] = v
    for m in STOCH:
        s = df[(df.instance==inst)&(df.method==m)].best
        r[m+'_mean'] = s.mean(); r[m+'_std'] = s.std(ddof=1); r[m+'_min'] = s.min()
        r[m+'_gap'] = 100*(s.mean()-bks)/bks
    rows.append(r)
summary = pd.DataFrame(rows)
summary.to_csv('summary.csv', index=False)

# ---------- Wilcoxon paired tests (per instance and pooled) ----------
def rank_biserial(x, y):
    d = np.asarray(x)-np.asarray(y); d = d[d!=0]
    if len(d)==0: return 0.0
    from scipy.stats import rankdata
    r = rankdata(np.abs(d)); rp = r[d>0].sum(); rm = r[d<0].sum()
    return (rp-rm)/(rp+rm)

def holm(ps):
    idx = np.argsort(ps); out = np.empty_like(ps, dtype=float); mlt = len(ps)
    running = 0
    for rank,i in enumerate(idx):
        adj = (len(ps)-rank)*ps[i]; running = max(running, adj)
        out[i] = min(1.0, running)
    return out

wil = []
pairs = [('SG-ACO', b) for b in ['AS-std','MMAS','ACS','Tabu','SHAP-off']] + [('SHAP-off','AS-std')]
for inst in INSTANCES:
    ps = []
    for a,b in pairs:
        xa = df[(df.instance==inst)&(df.method==a)].sort_values('seed').best.values
        xb = df[(df.instance==inst)&(df.method==b)].sort_values('seed').best.values
        if np.all(xa==xb):
            p = 1.0; W = np.nan
        else:
            W,p = wilcoxon(xa, xb, zero_method='wilcox')
        ps.append(p)
        wil.append(dict(instance=inst, A=a, B=b, mean_A=xa.mean(), mean_B=xb.mean(),
                        W=W, p=p, r_rb=rank_biserial(xb, xa)))
    padj = holm(np.array(ps))
    for k,(a,b) in enumerate(pairs):
        wil[-len(pairs)+k]['p_holm'] = padj[k]
wildf = pd.DataFrame(wil); wildf.to_csv('wilcoxon.csv', index=False)

# pooled across all (instance,seed) blocks
pooled = []
for a,b in pairs:
    xa = df[df.method==a].sort_values(['instance','seed']).best.values.astype(float)
    xb = df[df.method==b].sort_values(['instance','seed']).best.values.astype(float)
    # normalize by BKS per block so instances are comparable
    bksv = df[df.method==a].sort_values(['instance','seed']).instance.map(lambda i: meta[i]['optimum']).values
    W,p = wilcoxon(xa/bksv, xb/bksv)
    pooled.append(dict(A=a,B=b,W=W,p=p,r_rb=rank_biserial(xb/bksv, xa/bksv)))
pooled = pd.DataFrame(pooled); pooled['p_holm']=holm(pooled.p.values); pooled.to_csv('wilcoxon_pooled.csv', index=False)

# ---------- Friedman + Nemenyi over 250 blocks ----------
mat = np.stack([df[df.method==m].sort_values(['instance','seed']).best.values.astype(float) for m in STOCH], axis=1)
bksv = df[df.method=='SG-ACO'].sort_values(['instance','seed']).instance.map(lambda i: meta[i]['optimum']).values
matn = mat / bksv[:,None]
chi, pfr = friedmanchisquare(*[matn[:,i] for i in range(len(STOCH))])
nem = sp.posthoc_nemenyi_friedman(matn)
nem.columns = STOCH; nem.index = STOCH
nem.to_csv('nemenyi.csv')
avg_rank = pd.DataFrame(matn).rank(axis=1).mean().values
json.dump(dict(chi2=float(chi), p=float(pfr), n_blocks=int(matn.shape[0]),
               avg_ranks={m: float(r) for m,r in zip(STOCH, avg_rank)}),
          open('friedman.json','w'), indent=1)

# ---------- Magnitude-only ablation ----------
abl = []
for inst in ['ft06','la02']:
    for m in ['AS-std','Mag-only','SHAP-off','SG-ACO']:
        s = df[(df.instance==inst)&(df.method==m)].best
        abl.append(dict(instance=inst, method=m, mean=s.mean(), std=s.std(ddof=1)))
    xa = df[(df.instance==inst)&(df.method=='SHAP-off')].sort_values('seed').best.values
    xb = df[(df.instance==inst)&(df.method=='Mag-only')].sort_values('seed').best.values
    W,p = wilcoxon(xa,xb)
    abl.append(dict(instance=inst, method='Wilcoxon SHAP-off vs Mag-only', mean=W, std=p))
pd.DataFrame(abl).to_csv('ablation.csv', index=False)

# ---------- TreeSHAP timing table ----------
tim = pd.read_csv('timings.csv')
tt = tim.groupby('instance').agg(refreshes=('refreshes','mean'),
     fit_total_s=('fit_s','mean'), treeshap_total_s=('treeshap_s','mean'),
     run_wall_s=('run_wall_s','mean')).reindex(INSTANCES)
tt['treeshap_per_refresh_ms'] = 1000*tt.treeshap_total_s/tt.refreshes
tt['fit_per_refresh_s'] = tt.fit_total_s/tt.refreshes
tt.to_csv('treeshap_timing.csv')

# ---------- Figures ----------
plt.rcParams.update({'font.size':9})
# Fig A: % gap to BKS
fig, ax = plt.subplots(figsize=(9,3.6))
x = np.arange(len(INSTANCES)); wd = 0.13
for i,m in enumerate(STOCH):
    g = [summary[summary.instance==inst][m+'_gap'].iloc[0] for inst in INSTANCES]
    ax.bar(x+(i-2.5)*wd, g, wd, label=LBL[m])
ax.set_xticks(x); ax.set_xticklabels([i.upper() for i in INSTANCES])
ax.set_ylabel('Mean makespan gap to BKS (%)'); ax.legend(ncol=3, fontsize=8)
ax.set_title('Percentage gap to best-known solution (25 seeds, 3000 evaluations)')
plt.tight_layout(); plt.savefig('fig_gap.png', dpi=200); plt.close()

# Fig B: convergence ft10 & ta01
import glob
cur = {}
for f in glob.glob('curves_*.npz'):
    z = np.load(f)
    for k in z.files: cur[k] = z[k]
fig, axes = plt.subplots(1,2, figsize=(9,3.2))
for axi, inst in zip(axes, ['ft10','ta01']):
    for m,c in [('AS-std','tab:gray'),('SHAP-off','tab:blue'),('SG-ACO','tab:red')]:
        cs = np.stack([cur[f'{inst}|{m}|{s}'] for s in range(1,26)])
        axi.plot(np.arange(1,251), cs.mean(0), color=c, label=LBL[m])
    for gline in range(25,251,25):
        axi.axvline(gline, color='tab:red', ls=':', lw=0.5, alpha=0.4)
    axi.axhline(meta[inst]['optimum'], color='k', ls='--', lw=0.8)
    axi.set_title(inst.upper()); axi.set_xlabel('Iteration'); axi.set_ylabel('Best makespan')
axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig('fig_conv.png', dpi=200); plt.close()

# Fig C: signed weights per instance
FEAT = ['ProcTime','JobRemWork','MachRemWork','EarliestStart','MachIdle','QueueLen','OpsLeftJob','JobSlack']
W = np.array([info[i]['weights'] for i in INSTANCES])
fig, ax = plt.subplots(figsize=(7.5,3.6))
im = ax.imshow(W, cmap='RdBu_r', vmin=-np.abs(W).max(), vmax=np.abs(W).max(), aspect='auto')
ax.set_xticks(range(8)); ax.set_xticklabels(FEAT, rotation=35, ha='right')
ax.set_yticks(range(len(INSTANCES))); ax.set_yticklabels([i.upper() for i in INSTANCES])
for r in range(W.shape[0]):
    for c in range(W.shape[1]):
        ax.text(c, r, f'{W[r,c]:+.2f}', ha='center', va='center', fontsize=6.5)
fig.colorbar(im, label='signed SHAP weight (TreeSHAP)')
ax.set_title('Signed SHAP weights per instance (real TreeSHAP)')
plt.tight_layout(); plt.savefig('fig_weights.png', dpi=200); plt.close()
print('analysis done')
