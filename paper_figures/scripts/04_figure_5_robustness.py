from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "paper_figures" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)
"""Fig. 5 del paper: robustez bajo el protocolo C (ablación de entradas, TCN más justa y cota de conflictos de etiqueta)."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
C = json.load(open(RES/'s11e_robustness'/'celdas_2x2.json'))
R = json.load(open(RES/'s11e_robustness'/'S11e_resultados.json'))
C_T, C_R, C_X = '#2a78d6', '#eb6834', '#1baf7a'; INK, INK2, GRID = '#1d1d1b', '#55534e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.fontsize': 6.8,
                     'pdf.fonttype': 42})
def clean(ax, grid='x'):
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis=grid, color=GRID, lw=0.6); ax.set_axisbelow(True)
def punto(ax, x, y, col, mk, filled=True, s=26, xerr=None):
    if xerr is not None:
        ax.errorbar(x, y, xerr=xerr, fmt='none', ecolor=col, elinewidth=0.8, capsize=1.8, zorder=3)
    kw = dict(color=col, edgecolors='white', linewidths=0.5) if filled else dict(facecolors='white', edgecolors=col, linewidths=1.0)
    ax.scatter(x, y, s=s, marker=mk, zorder=4, **kw)
def rot(ax, txt, x, y, dx, dy, ha='left', col=INK):
    ax.annotate(txt, (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=6.3, color=col, ha=ha, va='center')
g = lambda k, m='f1_macro', i=0: C[k][m][i]
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw={'width_ratios': [1, 1]})
# ---------------- (a) F1 macro por conjunto de entradas (gráfico de puntos)
ax = axs[0]
filas = [('gao18', "Gao's 18 inputs"), ('mi25', 'MI top-25'), ('sin_id', 'MI top-25,\nno identifiers'),
         ('sin_id_cont', 'MI top-25, no identi-\nfiers or counters'), ('todos', 'All 88 inputs')]
ys = np.arange(len(filas))[::-1].astype(float)
for (s, _), y in zip(filas, ys):
    punto(ax, g(f'RF|{s}'), y + 0.13, C_R, 's', xerr=g(f'RF|{s}', i=1))
    punto(ax, g(f'XGB|{s}'), y - 0.13, C_X, '^', xerr=g(f'XGB|{s}', i=1))
    if s in R['E3']:
        ax.scatter(R['E3'][s]['f1_macro_oraculo'], y, s=30, color=INK, marker='x', linewidths=1.1, zorder=4)
y0, y1 = ys[0], ys[1]
punto(ax, g('TCN|gao18'), y0 + 0.13, C_T, 'o', xerr=g('TCN|gao18', i=1))
punto(ax, g('TCN-cw|gao18'), y0 - 0.13, C_T, 'o', filled=False, xerr=g('TCN-cw|gao18', i=1))
punto(ax, g('TCN|mi25'), y1, C_T, 'D', s=24, xerr=g('TCN|mi25', i=1))
ax.set_yticks(ys); ax.set_yticklabels([f for _, f in filas], fontsize=6.7); ax.set_ylim(ys[-1] - 0.55, ys[0] + 0.55)
ax.set_xlim(0, 102); ax.set_xlabel('Macro-F1 (%), mean $\\pm$ SD over 10 repetitions'); clean(ax, 'x')
ax.set_title('(a) Macro-F1 by input set', fontsize=7.5, loc='left')
# ---------------- (b) TFP frente a ataques no detectados
ax = axs[1]
v = lambda k: (C[k]['tfp'][0], C[k]['ataque_normal_pct'][0])
punto(ax, *v('TCN|gao18'), C_T, 'o'); rot(ax, "TCN, Gao's 18", *v('TCN|gao18'), 0, 8, 'center')
punto(ax, *v('TCN-cw|gao18'), C_T, 'o', filled=False); rot(ax, 'TCN + class\nweights', *v('TCN-cw|gao18'), -2, 12, 'right')
punto(ax, *v('TCN|mi25'), C_T, 'D', s=24); rot(ax, 'TCN, MI top-25', *v('TCN|mi25'), -7, 0, 'right')
for m, col, mk in [('RF', C_R, 's'), ('XGB', C_X, '^')]:
    punto(ax, *v(f'{m}|gao18'), col, mk); punto(ax, *v(f'{m}|mi25'), col, mk)
rot(ax, "trees, Gao's 18", 0.083, 15.4, 0, 8, 'center')
rot(ax, 'MI-LiteRF, MI-LiteXGB', 0.073, 10.9, 0, -9, 'center')
bal = {'RF': (13.52, 1.14), 'XGB': (16.08, 0.09)}            # configuraciones -bal (T = 2 000) de la tabla principal
punto(ax, *bal['RF'], C_R, 's', filled=False); punto(ax, *bal['XGB'], C_X, '^', filled=False)
rot(ax, 'trees,\nrebalanced', 17.0, 0.6, 7, 3, 'left')
for key, txt, dx, dy, ha in [('gao18', "bound, Gao's 18", 5, -7, 'left'), ('mi25', 'bound,\nMI top-25', 6, 0, 'left')]:
    fx, my = R['E3'][key]['tfp_oraculo'], R['E3'][key]['ataque_normal_oraculo_pct']
    ax.scatter(fx, my, s=30, color=INK, marker='x', linewidths=1.1, zorder=4); rot(ax, txt, fx, my, dx, dy, ha, INK2)
ax.set_xscale('log'); ax.set_xlim(0.0012, 300); ax.set_ylim(-3, 37)
ax.set_xlabel('False-positive rate (%, log scale)'); ax.set_ylabel('Attacks classified as Normal (%)'); clean(ax, 'both')
ax.set_title('(b) False alarms vs. missed attacks (means)', fontsize=7.5, loc='left')
h = [Line2D([], [], marker='o', ls='', color=C_T, mec='white', label='TCN (Gao et al.)'),
     Line2D([], [], marker='o', ls='', mfc='white', mec=C_T, mew=1.0, label='TCN + class weights'),
     Line2D([], [], marker='D', ls='', color=C_T, mec='white', ms=5, label='TCN on MI top-25'),
     Line2D([], [], marker='s', ls='', color=C_R, mec='white', label='Random forest (MI-LiteRF config.)'),
     Line2D([], [], marker='^', ls='', color=C_X, mec='white', label='XGBoost (MI-LiteXGB config.)'),
     Line2D([], [], marker='s', ls='', mfc='white', mec=INK2, mew=1.0, label='Open tree marker: rebalanced, $T$ = 2,000'),
     Line2D([], [], marker='x', ls='', color=INK, mew=1.1, label='Label-conflict bound (in-sample oracle)')]
fig.legend(handles=h, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06), columnspacing=1.1, handletextpad=0.3)
fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=2.0)
fig.savefig(OUT / 'fig_robustness.pdf', bbox_inches='tight'); fig.savefig(OUT / 'fig_robustness.png', dpi=220, bbox_inches='tight')
print('ok')
