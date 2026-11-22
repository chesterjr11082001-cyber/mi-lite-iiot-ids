from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "paper_figures" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)
import json, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
F3={json.load(open(f))['tabla_14_2']['repeticion']: json.load(open(f)) for f in glob.glob(str(RES/'v3_protocolC_S3'/'ficha_REP-*.json'))}
F2={json.load(open(f))['tabla_14_2']['repeticion']: json.load(open(f)) for f in glob.glob(str(RES/'v3_protocolO_S2'/'ficha_REP-*.json'))}
R=range(1,11); PC=pd.DataFrame(json.load(open(ROOT/'paper_figures'/'nums_pc.json'))['pc']).set_index('clase')
C_T,C_R,C_X='#2a78d6','#eb6834','#1baf7a'; INK,INK2,GRID='#1d1d1b','#55534e','#e4e3df'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.edgecolor':INK2,'axes.labelcolor':INK,'xtick.color':INK2,'ytick.color':INK2,
                     'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'legend.fontsize':7,'pdf.fonttype':42})
NAME={'Backdoor':'Backdoor','DDoS_HTTP':'DDoS HTTP','DDoS_ICMP':'DDoS ICMP','DDoS_TCP':'DDoS TCP SYN','DDoS_UDP':'DDoS UDP','Fingerprinting':'OS fingerprinting',
      'MITM':'MITM','Normal':'Normal','Password':'Password','Port_Scanning':'Port scanning','Ransomware':'Ransomware','SQL_injection':'SQL injection',
      'Uploading':'Uploading','Vulnerability_scanner':'Vuln. scanning','XSS':'XSS'}
def clean(ax,grid='x'):
    ax.spines[['top','right']].set_visible(False); ax.grid(axis=grid,color=GRID,lw=0.6); ax.set_axisbelow(True)
# ---------- Fig 2: per-class F1, O vs C (dumbbell) ----------
order=PC.sort_values('tcn_O').index.tolist()
fig,ax=plt.subplots(figsize=(3.45,3.6))
yy=np.arange(len(order))
for i,c in enumerate(order):
    for (a,b,col,dy) in [('tcn_O','tcn_C',C_T,0.17),('rf3_O','rf3_C',C_R,-0.17)]:
        ax.plot([PC.loc[c,a],PC.loc[c,b]],[i+dy,i+dy],color=col,lw=1.0,alpha=0.55,zorder=1)
        ax.scatter(PC.loc[c,a],i+dy,s=16,facecolors='white',edgecolors=col,linewidths=1.0,zorder=3)
        ax.scatter(PC.loc[c,b],i+dy,s=16,color=col,zorder=3)
    ax.scatter(PC.loc[c,'xgb3_C'],i-0.17,marker='^',s=14,color=C_X,zorder=4)
ax.set_yticks(yy); ax.set_yticklabels([NAME[c] for c in order]); ax.set_xlim(-3,103); ax.set_xlabel('Per-class F1 (%)'); clean(ax)
h=[Line2D([],[],marker='o',ls='',mfc='white',mec=C_T,label='TCN, original (O)'),Line2D([],[],marker='o',ls='',color=C_T,label='TCN, corrected (C)'),
   Line2D([],[],marker='o',ls='',mfc='white',mec=C_R,label='MI-LiteRF, O'),Line2D([],[],marker='o',ls='',color=C_R,label='MI-LiteRF, C'),
   Line2D([],[],marker='^',ls='',color=C_X,label='MI-LiteXGB, C')]
ax.legend(handles=h,loc='upper center',bbox_to_anchor=(0.42,-0.13),ncol=2,frameon=False,handletextpad=0.3,columnspacing=0.8)
fig.tight_layout(); fig.savefig(OUT/'fig_perclass_f1.pdf',bbox_inches='tight'); fig.savefig(OUT/'fig_perclass_f1.png',dpi=200,bbox_inches='tight'); plt.close(fig)
# ---------- Fig 3: (a) FPR per repetition (log), (b) attacks predicted Normal per class ----------
tfp={m:[F3[r]['modelos'][m]['tfp'] for r in R] for m in ['tcn_u','rf_u3','xgb_u3']}
CL=[x['Clase'] for x in F3[1]['tabla_14_4'] if not str(x['Clase']).startswith('Total')]; NOR=CL.index('Normal'); ATK=[c for c in CL if c!='Normal']
miss={m:{c:np.mean([np.array(F3[r]['matrices_confusion'][m])[CL.index(c),NOR]/np.array(F3[r]['matrices_confusion'][m])[CL.index(c)].sum()*100 for r in R]) for c in ATK} for m in ['tcn_u','rf_u3','xgb_u3']}
fig,axs=plt.subplots(1,2,figsize=(7.1,2.75),gridspec_kw={'width_ratios':[1,2.1]})
ax=axs[0]
for j,(m,col,lab,mk) in enumerate([('tcn_u',C_T,'TCN','o'),('rf_u3',C_R,'MI-LiteRF','s'),('xgb_u3',C_X,'MI-LiteXGB','^')]):
    x=np.full(10,j)+np.linspace(-0.18,0.18,10)
    ax.scatter(x,tfp[m],s=14,color=col,marker=mk,zorder=3)
ax.axhline(1.0,color=INK2,lw=0.7,ls='--'); ax.text(2.45,1.15,'1%',fontsize=6.5,color=INK2,ha='right')
ax.set_yscale('log'); ax.set_xticks([0,1,2]); ax.set_xticklabels(['TCN','MI-LiteRF','MI-LiteXGB']); ax.set_ylabel('False-positive rate (%)')
ax.set_xlim(-0.5,2.5); clean(ax,'y'); ax.set_title('(a) FPR in each of the 10 repetitions',fontsize=7.5,loc='left')
ax=axs[1]
ordA=sorted(ATK,key=lambda c:-miss['tcn_u'][c]); x=np.arange(len(ordA)); w=0.27
for k,(m,col,lab) in enumerate([('tcn_u',C_T,'TCN'),('rf_u3',C_R,'MI-LiteRF'),('xgb_u3',C_X,'MI-LiteXGB')]):
    ax.bar(x+(k-1)*w,[miss[m][c] for c in ordA],w*0.95,color=col,label=lab)
ax.set_xticks(x); ax.set_xticklabels([NAME[c] for c in ordA],rotation=40,ha='right'); ax.set_ylabel('Attack packets classified as Normal (%)')
ax.set_ylim(0,100); clean(ax,'y'); ax.legend(frameon=False,ncol=3,loc='upper right'); ax.set_title('(b) Missed attack packets by class',fontsize=7.5,loc='left')
fig.tight_layout(); fig.savefig(OUT/'fig_fpr_miss.pdf',bbox_inches='tight'); fig.savefig(OUT/'fig_fpr_miss.png',dpi=200,bbox_inches='tight'); plt.close(fig)
# ---------- Fig 4: (a) macro-F1 vs per-packet latency; (b) batch-1 vs batch-1024 latency ----------
g=lambda m,k: np.array([F3[r]['modelos'][m][k] for r in R],float)
fig,axs=plt.subplots(1,2,figsize=(7.1,2.6))
ax=axs[0]
for m,col,lab,mk in [('tcn_u',C_T,'TCN','o'),('rf_u3',C_R,'MI-LiteRF','s'),('xgb_u3',C_X,'MI-LiteXGB','^')]:
    l1=g(m,'lote1_ms')*1000; f1=g(m,'f1_macro'); kb=g(m,'tamano_onnx_gzip_kb').mean()
    ax.errorbar(l1.mean(),f1.mean(),xerr=l1.std(ddof=1),yerr=f1.std(ddof=1),fmt=mk,color=col,ms=6,mec='white',mew=0.8,capsize=2,elinewidth=0.9)
    off={'tcn_u':(-62,10),'rf_u3':(-4,10),'xgb_u3':(6,-24)}[m]
    ax.annotate(f'{lab}\n{kb:.0f} KB ONNX (gzip)',(l1.mean(),f1.mean()),xytext=off,textcoords='offset points',fontsize=6.5,color=INK)
ax.set_xlabel('Per-packet latency, batch size 1 (µs)'); ax.set_ylabel('Macro-F1 (%)'); ax.set_xlim(0,80); ax.set_ylim(25,100); clean(ax,'both')
ax.set_title('(a) Detection quality vs. per-packet latency (C)',fontsize=7.5,loc='left')
ax=axs[1]
mods=[('tcn_u',C_T,'TCN'),('rf_u3',C_R,'MI-LiteRF'),('xgb_u3',C_X,'MI-LiteXGB')]
b1=[g(m,'lote1_ms').mean()*1000 for m,_,_ in mods]; b1024=[g(m,'latencia_ms').mean()*1000 for m,_,_ in mods]; tr=[g(m,'respuesta_ms').mean()*1000 for m,_,_ in mods]
x=np.arange(3); w=0.26
for k,(vals,lab,hatch) in enumerate([(b1,'Batch 1 (inference)',None),(tr,'Batch 1 (scaling + inference, $t_{resp}$)','////'),(b1024,'Batch 1,024 (per packet)','....')]):
    bars=ax.bar(x+(k-1)*w,vals,w*0.95,color=[c for _,c,_ in mods],edgecolor='white',hatch=hatch,lw=0.4,label=lab)
    for xi,v in zip(x+(k-1)*w,vals): ax.text(xi,v*1.08,f'{v:.1f}',ha='center',fontsize=6)
ax.set_yscale('log'); ax.set_ylim(1,900); ax.set_xticks(x); ax.set_xticklabels([l for _,_,l in mods]); ax.set_ylabel('Time per packet (µs, log scale)'); clean(ax,'y')
h=[plt.Rectangle((0,0),1,1,fc='#bdbbb4',ec='white',hatch=hh) for hh in [None,'////','....']]
ax.legend(h,['Batch 1, inference','Batch 1, scaling + inference ($t_{resp}$)','Batch 1,024, per packet'],frameon=False,fontsize=6.3,loc='upper right')
ax.set_title('(b) Streaming vs. batched inference (ONNX Runtime, 1 thread)',fontsize=7.5,loc='left')
fig.tight_layout(); fig.savefig(OUT/'fig_latency.pdf',bbox_inches='tight'); fig.savefig(OUT/'fig_latency.png',dpi=200,bbox_inches='tight'); plt.close(fig)
# correlation of reproduced TCN per-class F1 vs Gao published
r=np.corrcoef(PC['gao_pub'],PC['tcn_O'])[0,1]; mad=np.abs(PC['gao_pub']-PC['tcn_O']).mean()
print('corr gao vs repro', round(r,4), 'MAD', round(mad,2))
print({m:[round(v,1) for v in tfp[m]] for m in tfp})
