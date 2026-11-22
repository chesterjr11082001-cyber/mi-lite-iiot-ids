from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "paper_figures" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)
import json, glob, numpy as np, pandas as pd
F2={json.load(open(f))['tabla_14_2']['repeticion']: json.load(open(f)) for f in glob.glob(str(RES/'v3_protocolO_S2'/'ficha_REP-*.json'))}
F3={json.load(open(f))['tabla_14_2']['repeticion']: json.load(open(f)) for f in glob.glob(str(RES/'v3_protocolC_S3'/'ficha_REP-*.json'))}
R=range(1,11)
t3=F3[1]['tabla_14_4']; t2=F2[1]['tabla_14_4']
print(json.dumps(t3[0],ensure_ascii=False)); print(json.dumps(t2[0],ensure_ascii=False)); print(json.dumps(t2[-1],ensure_ascii=False))
CL=[x['Clase'] for x in t3 if not str(x['Clase']).startswith('Total')]
dist=pd.DataFrame([{'clase':x['Clase'],'train':x['Flujos de entrenamiento'],'test':x['Flujos de prueba'],'gao_support':x['Soporte de prueba (Gao)'],'gao_f1':x['F1 de referencia (%)']} for x in t3 if x['Clase'] in CL])
dist['total']=dist.train+dist.test; dist['pct']=dist.total/dist.total.sum()*100
print(dist.to_string())
# per-class F1 means
pcO_tcn={c: np.mean([next(x for x in F2[r]['tabla_14_4'] if x['Clase']==c)['F1 O1 TCN (%)'] for r in R]) for c in CL}
P2=pd.DataFrame(json.load(open(RES/'v3_protocolO_S2'/'S2_summary.json'))['por_rep'])
pcO_rf3={c: P2[f'pc3_{c}'].mean() for c in CL}; pcO_rf1={c: P2[f'pc1_{c}'].mean() for c in CL}
pcC={m:{c: np.mean([F3[r]['modelos'][m]['f1_por_clase'][c] for r in R]) for c in CL} for m in ['tcn_u','rf_u3','xgb_u3','rf_u1','xgb_u1']}
T=pd.DataFrame({'gao_pub':dist.set_index('clase').gao_f1,'tcn_O':pcO_tcn,'rf1_O':pcO_rf1,'rf3_O':pcO_rf3,'tcn_C':pcC['tcn_u'],'rf3_C':pcC['rf_u3'],'xgb3_C':pcC['xgb_u3'],'rf1_C':pcC['rf_u1'],'xgb1_C':pcC['xgb_u1']})
print(T.round(2).to_string()); print(T.mean().round(2).to_string())
# S2 tabla_14_3 indicators
for x in F2[1]['tabla_14_3']: print(json.dumps(x,ensure_ascii=False)[:300])
json.dump({'dist':dist.to_dict('records'),'pc':T.reset_index().rename(columns={'index':'clase'}).to_dict('records')},open(ROOT/'paper_figures'/'nums_pc.json','w'),indent=1)
