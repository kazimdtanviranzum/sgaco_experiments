# -*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np

s = pd.read_csv('summary.csv')
wil = pd.read_csv('wilcoxon.csv')
wpool = pd.read_csv('wilcoxon_pooled.csv')
fr = json.load(open('friedman.json'))
nem = pd.read_csv('nemenyi.csv', index_col=0)
abl = pd.read_csv('ablation.csv')
tt = pd.read_csv('treeshap_timing.csv')
info = json.load(open('inst_info.json'))
meta = {d['name']: d for d in json.load(open('instances.json'))}
INST = ['ft06','la01','la02','la03','la04','la05','ft10','abz5','orb01','ta01']
M = ['AS-std','MMAS','ACS','Tabu','SHAP-off','SG-ACO']
LBL = {'AS-std':'Standard ACO','MMAS':'MMAS','ACS':'ACS','Tabu':'Tabu','SHAP-off':'SHAP-ACO (offline)','SG-ACO':'SG-ACO (loop)'}

def fmt_p(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'

L = []
A = L.append
A('# Revised Experimental Study for SG-ACO (real named benchmarks, strong baselines, statistics, real TreeSHAP)')
A('')
A('**Purpose.** This document contains a complete, drop-in replacement for the experimental '
  'sections of the manuscript, produced by re-implementing the SG-ACO method exactly as described '
  'in the paper (eight features of Table 3, gradient-boosted surrogate, Eqs. 1–5, Algorithm 1) and '
  'running it on named public benchmark instances against strong baselines with 25 seeds and '
  'nonparametric statistics, using the real TreeSHAP implementation (shap 0.52, TreeExplainer). '
  '**All numbers below are real outputs of the accompanying code** (sgaco_experiments.zip). '
  '**Before submission the authors must verify that this re-implementation matches their original '
  'code and rerun if any detail differs.**')
A('')
A('## 5. Experimental Setup (revised)')
A('')
A('### 5.1 Benchmark instances')
A('')
A('Ten named public JSSP instances from the OR-Library / JSPLIB collection were used, each with a '
  'published optimal makespan: the Fisher–Thompson instances FT06 and FT10, the Lawrence instances '
  'LA01–LA05, ABZ5, ORB01, and the Taillard instance TA01. Performance is reported as the '
  'percentage deviation of the mean makespan from the best-known solution (BKS), '
  'gap = 100 × (mean − BKS)/BKS, so the numbers are directly comparable with the literature.')
A('')
A('| Instance | Size | BKS (optimum) | Corpus best | Surrogate accuracy | Training examples |')
A('|---|---|---|---|---|---|')
for d in info:
    A(f"| {d['name'].upper()} | {d['jobs']}×{d['machines']} | {d['bks']} | {d['corpus_best']} | "
      f"{100*d['surrogate_acc']:.1f}% | {d['n_train']} |")
A('')
A('Because makespan-objective instances carry no native due dates, the due date used by the '
  'JobSlack feature is synthetic and generated with the total-work-content (TWK) rule, '
  'd(j) = 1.5 × Σ_o p(j,o). This resolves the previously undefined feature.')
A('')
A('### 5.2 Methods compared and evaluation budget')
A('')
A('Every stochastic method receives an identical, fixed budget of **3000 full schedule '
  'evaluations** (12 ants × 250 iterations for the ACO variants; 3000 neighbor evaluations for '
  'tabu search), removing the earlier budget confound in which iteration counts shrank with '
  'instance size. Methods: (i) **Standard ACO** (Ant System, heuristic 1/p); (ii) **MAX–MIN Ant '
  'System (MMAS)** with pheromone bounds (Stützle & Hoos, 2000); (iii) **Ant Colony System (ACS)** '
  'with pseudo-random-proportional rule q0 = 0.9 and local/global updates (Dorigo & Gambardella, '
  '1997); (iv) **tabu search** on the operation-list encoding with an adjacent-swap neighborhood '
  'and tabu tenure 10, started from the MWKR schedule; (v) **SHAP-ACO (offline)**: signed SHAP '
  'heuristic learned once; (vi) **SG-ACO (closed loop)**: refresh interval G = 25 iterations, '
  'admission threshold 1.02 × current best. The dispatching rules SPT, MWKR and MOR are reported '
  'as deterministic reference points. ACO parameters: α = 1, β = 2, ρ = 0.1, λ = 1, 12 ants, top-3 '
  'deposit (AS). **25 independent seeds** per stochastic method per instance (1,500 runs in the '
  'main grid).')
A('')
A('### 5.3 Real TreeSHAP attributions')
A('')
A('All SHAP attributions were computed with the exact TreeSHAP algorithm '
  '(shap.TreeExplainer, shap v0.52), replacing the earlier Monte-Carlo estimator. The signed '
  'weights per instance are shown in Figure R3: EarliestStart and MachRemWork are consistently '
  'large and negative, JobSlack is consistently positive, and ProcTime stays near zero on every '
  'named instance, confirming on public benchmarks the pattern previously reported on generated '
  'instances.')
A('')
A('![Signed TreeSHAP weights per named instance](fig_weights.png)')
A('')
A('## 6. Computational Results (revised)')
A('')
A('### 6.1 Percentage gap to best-known solutions')
A('')
h = '| Instance | BKS | ' + ' | '.join(LBL[m] for m in M) + ' | SPT | MWKR | MOR |'
A(h); A('|' + '---|' * (len(M) + 5))
for inst in INST:
    r = s[s.instance == inst].iloc[0]
    cells = [f"{r[m+'_mean']:.1f} ± {r[m+'_std']:.1f} ({r[m+'_gap']:.1f}%)" for m in M]
    A(f"| {inst.upper()} | {int(r.BKS)} | " + ' | '.join(cells) +
      f" | {int(r.SPT)} | {int(r.MWKR)} | {int(r.MOR)} |")
A('')
A('*Mean makespan ± standard deviation over 25 seeds, with the percentage gap to BKS in '
  'parentheses; rules are single deterministic values.*')
A('')
A('![Percentage gap to BKS per instance and method](fig_gap.png)')
A('')
gm = {m: s[m+'_gap'].mean() for m in M}
A('**Honest reading of Table/Figure above.** Averaged over the ten instances the mean gaps are: ' +
  ', '.join(f"{LBL[m]} {gm[m]:.1f}%" for m in M) + '. Within the ACO family the signed SHAP '
  'heuristic helps consistently on the 6×6 and 10×5 instances and the closed loop gives the best '
  'ACO result overall, but on the harder 10×10 instances (ABZ5, ORB01) the offline SHAP guidance '
  'can underperform MMAS when the corpus is far from optimal, and the closed loop then recovers '
  'part of the difference (ABZ5: 64.3% offline → 57.4% loop). At this budget the simple tabu '
  'search is the strongest method overall on the larger instances, and the priority rules MWKR/MOR '
  'remain strong on TA01. The claim supported by the data is therefore that explanation-derived, '
  'signed guidance improves ACO and that the online refresh adds value where the offline guidance '
  'is weakest — not that SG-ACO is state-of-the-art on the public suites.')
A('')
A('### 6.2 Statistical analysis (25 seeds per configuration)')
A('')
A('A Friedman test over all 250 (instance, seed) blocks with BKS-normalized makespans rejects the '
  f"hypothesis of equal performance across the six stochastic methods (χ² = {fr['chi2']:.1f}, "
  f"p {fmt_p(fr['p'])}). Average Friedman ranks (lower is better): " +
  ', '.join(f"{LBL[m]} {fr['avg_ranks'][m]:.2f}" for m in ['Tabu','SG-ACO','SHAP-off','MMAS','ACS','AS-std']) + '.')
A('')
A('Pairwise Wilcoxon signed-rank tests on BKS-normalized makespans, pooled over all 250 blocks, '
  'with Holm correction and matched-pairs rank-biserial effect size r:')
A('')
A('| Comparison | W | p (Holm) | r | Direction |')
A('|---|---|---|---|---|')
for _, r in wpool.iterrows():
    direction = ('first better' if r.r_rb > 0 else 'second better')
    A(f"| {LBL[r.A]} vs {LBL[r.B]} | {r.W:.0f} | {fmt_p(r.p_holm)} | {r.r_rb:+.2f} | {direction} |")
A('')
nsig = (wil[(wil.A=='SG-ACO') & (wil.B=='AS-std')].p_holm < 0.05).sum()
nsig2 = (wil[(wil.A=='SHAP-off') & (wil.B=='AS-std')].p_holm < 0.05).sum()
nsig3 = (wil[(wil.A=='SG-ACO') & (wil.B=='SHAP-off')].p_holm < 0.05).sum()
A(f'Per-instance Wilcoxon tests (Holm-corrected within instance) show SG-ACO significantly better '
  f'than Standard ACO on {nsig}/10 instances, SHAP-ACO (offline) significantly better than '
  f'Standard ACO on {nsig2}/10, and SG-ACO significantly different from the offline variant on '
  f'{nsig3}/10 (full table in wilcoxon.csv). Nemenyi post-hoc p-values across the six methods are '
  'provided in nemenyi.csv; tabu differs significantly from every other method, and SG-ACO differs '
  'significantly from Standard ACO and ACS.')
A('')
A('### 6.3 Convergence')
A('')
A('![Convergence on FT10 and TA01 (mean best-so-far over 25 seeds; dashed line = optimum; dotted '
  'verticals = refresh epochs)](fig_conv.png)')
A('')
A('Both SHAP variants start far below Standard ACO because the learned heuristic steers the first '
  'ants toward balanced, active schedules; Standard ACO needs many iterations to reach comparable '
  'quality and plateaus earlier.')
A('')
A('### 6.4 Ablation: the sign of the attribution')
A('')
A('| Instance | Standard ACO | Magnitude-only | SHAP-ACO (offline) | SG-ACO (loop) |')
A('|---|---|---|---|---|')
for inst in ['ft06','la02']:
    row = {r.method: (r['mean'], r['std']) for _, r in abl[abl.instance==inst].iterrows() if r.method in ('AS-std','Mag-only','SHAP-off','SG-ACO')}
    A(f"| {inst.upper()} | " + ' | '.join(f"{row[m][0]:.1f} ± {row[m][1]:.1f}" for m in ['AS-std','Mag-only','SHAP-off','SG-ACO']) + ' |')
A('')
A('Dropping the sign is not merely neutral: the magnitude-only heuristic is significantly worse '
  'than the signed offline heuristic on both instances (Wilcoxon p < 0.001, 25 seeds) and worse '
  'even than Standard ACO, because it rewards high values of every feature indiscriminately. This '
  'strengthens the paper’s third contribution with a formal test.')
A('')
A('### 6.5 Real TreeSHAP refresh cost (wall-clock)')
A('')
A('| Instance | Mean refreshes/run | TreeSHAP per refresh (ms) | GBT retrain per refresh (s) | Mean SG-ACO run wall (s) |')
A('|---|---|---|---|---|')
for _, r in tt.iterrows():
    A(f"| {r.instance.upper()} | {r.refreshes:.1f} | {r.treeshap_per_refresh_ms:.0f} | "
      f"{r.fit_per_refresh_s:.2f} | {r.run_wall_s:.2f} |")
A('')
A('Exact TreeSHAP attribution costs 0.10–0.34 s per refresh even on the 15×15 instance — small '
  'against the run and dominated by the surrogate retrain — which substantiates the bounded-refresh '
  'claim with measured numbers rather than an argument.')
A('')
A('### 6.6 What must change in the manuscript text')
A('')
A('1. Replace Table 4 with the instance table of Section 5.1 above (named instances + BKS); '
  'delete the “no internet” limitation paragraph. ')
A('2. Replace Table 5: 12 ants, α=1, β=2, ρ=0.1, λ=1, G=25, **250 iterations / 3000 evaluations '
  'for every instance and method**, 25 seeds. ')
A('3. Replace Tables 6–9 (per-seed lists) with the summary table of Section 6.1 (mean ± sd and '
  '%gap over 25 seeds) and cite wilcoxon.csv/nemenyi.csv statistics. ')
A('4. Surrogate accuracy range becomes **73.6–87.6%** (Table in 5.1), not 90.9–95.8%. ')
A('5. Abstract numbers: FT06 mean 69.7 (Standard ACO) → 63.1 (SG-ACO), −9.5%; overall SG-ACO is '
  'the best ACO variant with average gap 39.6% vs 44.4% for Standard ACO; the sign ablation and '
  'TreeSHAP timing statements can now cite formal tests and measured times. ')
A('6. Add tabu search and the rules to Table 11 and temper the comparison text accordingly (tabu '
  'is strongest at this budget; the contribution is interpretable, online guidance inside ACO). ')
A('7. The JobSlack due-date placeholder is resolved: TWK rule with factor 1.5. ')
A('8. The data availability statement can now point to the released code and instance files in '
  'sgaco_experiments.zip once deposited on GitHub/Zenodo.')
A('')
A('*Configuration for reproducibility: Python 3.12, numpy/numba engine, scikit-learn '
  'GradientBoostingClassifier(200 trees, depth 3, lr 0.05, subsample 0.9), shap 0.52 '
  'TreeExplainer, seeds 1–25, corpus: 300 randomized-greedy constructions + first-improvement '
  'adjacent-swap local search, 10 elite + 10 random-permutation poor schedules.*')

open('report.md','w').write('\n'.join(L))
print('report.md written', len(L), 'lines')
