from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "paper_figures" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)
"""Fig. 6 del paper: (a) tiempo hasta la primera alarma vs tasa de llegada; (b) memoria y (c) tiempo por
paquete vs tamaño de lote. (a) usa los tiempos de la tabla de costo (Xeon 2,20 GHz, media de 10 REP) y E[N] de la tabla principal;
(b, c) usan el barrido de lotes de results/s14_batch_sweep/ (edge_benchmark/batch_sweep.py en un Xeon de 2,30 GHz, ORT 1.29,
un hilo; mediana de 3 corridas)."""
import json, glob, collections, statistics as st, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R = RES / 'v3_protocolC_S3'
LOTES_DIR = RES / 's14_batch_sweep'
F3 = {r: json.load(open(R / f'ficha_REP-{r:02d}.json')) for r in range(1, 11)}
g = lambda m, k: np.array([F3[r]['modelos'][m][k] for r in range(1, 11)], float)
C_T, C_R, C_X = '#2a78d6', '#eb6834', '#1baf7a'; INK, INK2, GRID = '#1d1d1b', '#55534e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.2, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'legend.fontsize': 6.4, 'pdf.fonttype': 42})
def clean(ax, grid='both'):
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis=grid, color=GRID, lw=0.5, which='major'); ax.set_axisbelow(True)
MOD = [('rf_u3', 'MI-LiteRF', C_R, 's'), ('xgb_u3', 'MI-LiteXGB', C_X, '^'), ('tcn_u', 'TCN', C_T, 'o')]
# ---- parámetros de (a): tiempos (s) por paquete y E[N] promedio de las 14 clases (finitas para la TCN)
t_resp = {m: g(m, 'respuesta_ms').mean() / 1e3 for m, *_ in MOD}
t_1024 = {m: g(m, 'latencia_ms').mean() / 1e3 for m, *_ in MOD}
EN = {}
CL = F3[1]['preprocesamiento']['clases']; NOR = CL.index('Normal')
for m, *_ in MOD:
    vals = []
    for r in range(1, 11):
        M = np.array(F3[r]['matrices_confusion'][m], float)
        for c in range(15):
            if c == NOR:
                continue
            mc = M[c, NOR] / M[c].sum()
            if mc < 1:
                vals.append(mc / (1 - mc))
    EN[m] = float(np.mean(vals))
EN = {'rf_u3': 0.157, 'xgb_u3': 0.164, 'tcn_u': 3.56}   # valores de la tabla principal (para que el texto y la figura coincidan)
print('t_resp us', {k: round(v * 1e6, 1) for k, v in t_resp.items()}, 't_1024 us', {k: round(v * 1e6, 2) for k, v in t_1024.items()})
print('E[N]', {k: round(v, 3) for k, v in EN.items()})
B = 1024
def T1(lam, m):
    rho = lam * t_resp[m]
    return np.where(rho < 1, EN[m] / lam + t_resp[m] + rho * t_resp[m] / (2 * (1 - rho)), np.nan)
def TB(lam, m):
    return np.where(lam * t_1024[m] < 1, EN[m] / lam + (B - 1) / (2 * lam) + B * t_1024[m], np.nan)
# ---- datos de (b, c)
D = collections.defaultdict(list)
for f in glob.glob(str(LOTES_DIR / '*_r[0-9].json')):
    d = json.load(open(f)); D[(d['modelo'], d['lote'])].append(d)
LOTES = sorted({b for (_, b) in D})
mem = {m: [st.median([max(x['pico_mb'], x['rss_final_mb']) - x['rss_numpy_mb'] for x in D[(m, b)]]) for b in LOTES] for m, *_ in MOD}
tpp = {m: [st.median([x['us_por_paquete_mediana'] for x in D[(m, b)]]) for b in LOTES] for m, *_ in MOD}
json.dump({'lotes': LOTES, 'memoria_mb': mem, 'us_por_paquete': tpp, 'EN': EN, 't_resp_us': {k: v * 1e6 for k, v in t_resp.items()},
           't_1024_us': {k: v * 1e6 for k, v in t_1024.items()}}, open(LOTES_DIR / 'escala_resumen.json', 'w'), indent=1)
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.45), gridspec_kw={'width_ratios': [1.25, 1, 1], 'wspace': 0.42})
# (a)
ax = axs[0]
lam = np.logspace(1, 6, 600)
for m, lab, col, mk in MOD:
    y1 = T1(lam, m); ax.plot(lam, y1, color=col, lw=1.3)
    k = np.where(np.isfinite(y1))[0][-1]; ax.plot(lam[k], y1[k], marker='|', color=col, ms=6, mew=1.2)
    yb = TB(lam, m); ax.plot(lam, yb, color=col, lw=1.1, ls=(0, (3.5, 1.6)))
    k = np.where(np.isfinite(yb))[0][-1]; ax.plot(lam[k], yb[k], marker='|', color=col, ms=6, mew=1.2)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(10, 1e6); ax.set_ylim(2e-5, 100)
ax.set_xlabel('Packet arrival rate $\\lambda$ (packets/s)'); ax.set_ylabel('Expected time to first alarm (s)')
ax.axhline(1, color=INK2, lw=0.6, ls=':'); ax.text(13, 1.35, '1 s', fontsize=6, color=INK2)
clean(ax); ax.set_title('(a) Time to the first alarm', fontsize=7.4, loc='left')
h = [Line2D([], [], color=col, lw=1.3, label=lab) for _, lab, col, _ in MOD] + \
    [Line2D([], [], color=INK2, lw=1.1, label='packet by packet'), Line2D([], [], color=INK2, lw=1.1, ls=(0, (3.5, 1.6)), label='batches of 1,024')]
ax.legend(handles=h, frameon=False, loc='upper right', fontsize=5.9, handlelength=2.0, labelspacing=0.25)
# (b)
ax = axs[1]
for m, lab, col, mk in MOD:
    ax.plot(LOTES, mem[m], color=col, marker=mk, ms=3.2, lw=1.0, mec='white', mew=0.4)
ax.set_xscale('log', base=2); ax.set_xticks([1, 32, 1024, 32768]); ax.set_xticklabels(['1', '32', '1,024', '32,768']); ax.minorticks_off()
ax.set_ylim(0, 105); ax.set_xlabel('Batch size (packets)'); ax.set_ylabel('Peak memory above interpreter (MB)')
clean(ax); ax.set_title('(b) Memory', fontsize=7.4, loc='left')
for m, lab, col, mk in MOD:
    yy = mem[m][0]
    ax.annotate(lab, (1, yy), xytext=(3, {'rf_u3': 5, 'xgb_u3': 5, 'tcn_u': -9}[m]), textcoords='offset points', fontsize=6, color=col)
# (c)
ax = axs[2]
for m, lab, col, mk in MOD:
    ax.plot(LOTES, tpp[m], color=col, marker=mk, ms=3.2, lw=1.0, mec='white', mew=0.4)
ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xticks([1, 32, 1024, 32768]); ax.set_xticklabels(['1', '32', '1,024', '32,768']); ax.minorticks_off()
ax.set_ylim(0.4, 40); ax.set_yticks([0.5, 1, 2, 5, 10, 20]); ax.set_yticklabels(['0.5', '1', '2', '5', '10', '20'])
ax.set_xlabel('Batch size (packets)'); ax.set_ylabel('Time per packet (µs)')
clean(ax); ax.set_title('(c) Time per packet', fontsize=7.4, loc='left')
fig.savefig(OUT / 'fig_scaling.pdf', bbox_inches='tight')
# Valores para el texto
for lam_ in [10, 100, 1000, 10000, 30000, 100000]:
    print(lam_, {m: (float(T1(np.array([lam_]), m)[0]), float(TB(np.array([lam_]), m)[0])) for m, *_ in MOD})
print('capacidad 1/t_resp', {m: round(1 / t_resp[m]) for m, *_ in MOD}, '1/t_1024', {m: round(1 / t_1024[m]) for m, *_ in MOD})
