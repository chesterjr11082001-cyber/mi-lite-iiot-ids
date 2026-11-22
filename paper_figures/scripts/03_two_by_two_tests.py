from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "paper_figures" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)
"""Pruebas corregidas del diseño 2x2 (atributos x modelo) y de las variantes de la TCN (E4 + E5 + S3)."""
import json, glob, sys
import numpy as np, pandas as pd
from scipy import stats

N_TR, N_TE = 1_336_769, 572_902
NORMAL = 7

def t_corr(d):
    d = np.asarray(d, float); J = len(d); m = d.mean(); s2 = d.var(ddof=1)
    se = np.sqrt((1 / J + N_TE / N_TR) * s2); t = m / se
    h = stats.t.ppf(0.975, J - 1) * se
    return m, t, 2 * stats.t.sf(abs(t), J - 1), (m - h, m + h)

def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); adj[k] = min(1, run)
    return adj

# --- TCN de Gao (C) y árboles v3 por REP, desde las fichas de S3
tcn = {}
for f in glob.glob(str(RES/'v3_protocolC_S3'/'ficha_REP-*.json')):
    d = json.load(open(f)); r = d['tabla_14_2']['repeticion']
    m = d['modelos']['tcn_u']; cm = np.asarray(d['matrices_confusion']['tcn_u'])
    assert cm[NORMAL].sum() == 409200, cm[NORMAL].sum()
    ata = [i for i in range(15) if i != NORMAL]
    miss = cm[ata, NORMAL].sum() / cm[ata].sum() * 100
    tcn[r] = dict(f1_macro=m['f1_macro'], tfp=m['tfp'], ataque_normal_pct=miss, exactitud=m['exactitud'])
tcn = pd.DataFrame(tcn).T.sort_index()

e4 = pd.read_csv(RES/'s11e_robustness'/'E4_ablaciones.csv')
e5 = pd.read_csv(RES/'s11e_robustness'/'E5_tcn.csv')
def serie(df, **kw):
    q = df.copy()
    for k, v in kw.items(): q = q[q[k] == v]
    return q.set_index('rep').sort_index()

S = {
    ('TCN', 'gao18'): tcn,
    ('TCN-cw', 'gao18'): serie(e5, variante='tcn_cw'),
    ('TCN', 'mi25'): serie(e5, variante='tcn_mi25'),
}
for conj in ['gao18', 'mi25', 'sin_id', 'sin_id_cont', 'todos']:
    for mod in ['rf', 'xgb']:
        S[(mod.upper(), conj)] = serie(e4, conjunto=conj, modelo=mod)

print('Resumen por celda (media ± DE; min–max):')
for k, v in S.items():
    print(f'{k[0]:7s} {k[1]:12s} F1 {v.f1_macro.mean():6.2f} ± {v.f1_macro.std(ddof=1):5.2f}  '
          f'TFP {v.tfp.mean():7.3f} ± {v.tfp.std(ddof=1):6.3f} [{v.tfp.min():.3f}–{v.tfp.max():.3f}]  '
          f'Miss {v.ataque_normal_pct.mean():6.2f} ± {v.ataque_normal_pct.std(ddof=1):5.2f}  '
          f'Acc {v.exactitud.mean():6.2f}')

comps = [
    ('Feature effect, TCN', ('TCN', 'gao18'), ('TCN', 'mi25')),
    ('Feature effect, RF', ('RF', 'gao18'), ('RF', 'mi25')),
    ('Feature effect, XGB', ('XGB', 'gao18'), ('XGB', 'mi25')),
    ('Model effect, Gao-18: TCN->RF', ('TCN', 'gao18'), ('RF', 'gao18')),
    ('Model effect, Gao-18: TCN->XGB', ('TCN', 'gao18'), ('XGB', 'gao18')),
    ('Model effect, MI-25: TCN->RF', ('TCN', 'mi25'), ('RF', 'mi25')),
    ('Model effect, MI-25: TCN->XGB', ('TCN', 'mi25'), ('XGB', 'mi25')),
    ('Class weights: TCN->TCN-cw', ('TCN', 'gao18'), ('TCN-cw', 'gao18')),
]
filas = []
for met in ['f1_macro', 'tfp', 'ataque_normal_pct']:
    for nom, a, b in comps:
        d = S[b][met].values - S[a][met].values
        m, t, p, ci = t_corr(d)
        w = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        filas.append(dict(metrica=met, comparacion=nom, a=f'{a[0]}/{a[1]}', b=f'{b[0]}/{b[1]}', media_a=S[a][met].mean(),
                          media_b=S[b][met].mean(), delta=m, ic_lo=ci[0], ic_hi=ci[1], t=t, p_corr=p, p_wilcoxon=w))
R = pd.DataFrame(filas)
R['p_corr_holm'] = np.nan
for met, g in R.groupby('metrica'):
    R.loc[g.index, 'p_corr_holm'] = holm(g.p_corr.values)
# interacción: (XGB - TCN | gao18) - (XGB - TCN | mi25)
for mod in ['RF', 'XGB']:
    d = (S[(mod, 'gao18')].f1_macro.values - S[('TCN', 'gao18')].f1_macro.values) - \
        (S[(mod, 'mi25')].f1_macro.values - S[('TCN', 'mi25')].f1_macro.values)
    m, t, p, ci = t_corr(d)
    print(f'Interacción {mod}: {m:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] p_corr={p:.2e}')
# Levene FPR TCN-mi25 vs XGB
print('Levene TFP TCN-mi25 vs XGB-mi25:', stats.levene(S[('TCN','mi25')].tfp, S[('XGB','mi25')].tfp).pvalue)
print('TCN-mi25 reps con TFP>1%:', int((S[('TCN','mi25')].tfp > 1).sum()), '| TCN-cw reps con TFP>1%:', int((S[('TCN-cw','gao18')].tfp > 1).sum()))
pd.set_option('display.width', 250)
print(R[['metrica','comparacion','media_a','media_b','delta','ic_lo','ic_hi','p_corr','p_corr_holm','p_wilcoxon']].round(4).to_string())
R.to_csv(RES/'s11e_robustness'/'analisis_2x2.csv', index=False)
json.dump({f'{k[0]}|{k[1]}': {c: [float(S[k][c].mean()), float(S[k][c].std(ddof=1)), float(S[k][c].min()), float(S[k][c].max())]
           for c in ['f1_macro', 'tfp', 'ataque_normal_pct', 'exactitud']} for k in S}, open(RES/'s11e_robustness'/'celdas_2x2.json', 'w'), indent=1)
