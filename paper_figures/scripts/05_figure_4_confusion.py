from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "paper_figures" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)
"""Fig. 4 del paper: matrices de confusión normalizadas por fila bajo el protocolo C (10 repeticiones sumadas)."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle
R = RES / 'v3_protocolC_S3'
F = {r: json.load(open(R / f'ficha_REP-{r:02d}.json')) for r in range(1, 11)}
CL = F[1]['preprocesamiento']['clases']
S = {m: sum(np.array(F[r]['matrices_confusion'][m], float) for r in range(1, 11)) for m in ['tcn_u', 'xgb_u3']}
ORDEN = ['Normal', 'DDoS_HTTP', 'SQL_injection', 'XSS', 'Uploading', 'Password',
         'Backdoor', 'Ransomware', 'Vulnerability_scanner', 'Port_Scanning', 'Fingerprinting', 'MITM',
         'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP']
NAME = {'Backdoor': 'Backdoor', 'DDoS_HTTP': 'DDoS HTTP', 'DDoS_ICMP': 'DDoS ICMP', 'DDoS_TCP': 'DDoS TCP SYN', 'DDoS_UDP': 'DDoS UDP',
        'Fingerprinting': 'OS fingerprinting', 'MITM': 'MITM', 'Normal': 'Normal', 'Password': 'Password', 'Port_Scanning': 'Port scanning',
        'Ransomware': 'Ransomware', 'SQL_injection': 'SQL injection', 'Uploading': 'Uploading', 'Vulnerability_scanner': 'Vuln. scanning', 'XSS': 'XSS'}
idx = [CL.index(c) for c in ORDEN]
INK, INK2 = '#1d1d1b', '#55534e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'pdf.fonttype': 42})
# Rampa secuencial de un solo tono (violeta), distinta de los colores de modelo (azul, naranja, verde)
cmap = LinearSegmentedColormap.from_list('seq', ['#ffffff', '#efedf5', '#bcbddc', '#807dba', '#54278f', '#3f007d'])
norm = PowerNorm(gamma=0.45, vmin=0, vmax=100)
fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.55), gridspec_kw={'wspace': 0.08})
for k, (ax, m, tit) in enumerate(zip(axs, ['xgb_u3', 'tcn_u'], ['(a) MI-LiteXGB', '(b) TCN (Gao et al.)'])):
    M = S[m][np.ix_(idx, idx)]
    P = M / M.sum(1, keepdims=True) * 100
    im = ax.imshow(P, cmap=cmap, norm=norm, aspect='equal')
    for i in range(15):
        for j in range(15):
            v = P[i, j]
            if v >= 0.5:
                txt = '100' if v >= 99.95 else (f'{v:.1f}' if (v < 9.95 or v >= 99.0) else f'{v:.0f}')
                ax.text(j, i, txt, ha='center', va='center', fontsize=4.6,
                        color='white' if norm(v) > 0.55 else INK)
    ax.set_xticks(range(15)); ax.set_yticks(range(15))
    ax.set_xticklabels([NAME[c] for c in ORDEN], rotation=55, ha='right', rotation_mode='anchor', fontsize=6)
    ax.set_yticklabels([NAME[c] for c in ORDEN] if k == 0 else [], fontsize=6)
    ax.tick_params(length=1.5, width=0.5)
    # Bloque Normal + ataques web
    ax.add_patch(Rectangle((-0.5, -0.5), 6, 6, fill=False, ec=INK, lw=1.0, zorder=5))
    ax.set_xlabel('Predicted class', fontsize=6.8)
    if k == 0:
        ax.set_ylabel('True class', fontsize=6.8)
    ax.set_title(tit, fontsize=7.5, loc='left')
    for s in ax.spines.values():
        s.set_linewidth(0.5)
cax = fig.add_axes([0.915, 0.30, 0.012, 0.55])
cb = fig.colorbar(im, cax=cax, ticks=[0, 1, 5, 10, 25, 50, 100])
cb.ax.tick_params(labelsize=5.8, length=1.5, width=0.5); cb.outline.set_linewidth(0.4)
cb.set_label('Share of the true class (%)', fontsize=6.2)
fig.subplots_adjust(left=0.10, right=0.905, bottom=0.22, top=0.95)
fig.savefig(OUT / 'fig_confusion.pdf', bbox_inches='tight')
print('ok', int(S['xgb_u3'].sum()))
