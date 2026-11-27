"""Mide, en un proceso nuevo, la memoria pico y el tiempo por paquete de un modelo ONNX para un tamaño de lote dado.
Cada llamada clasifica paquetes distintos (bloques consecutivos de las 10 000 muestras del paquete de borde),
como bench_edge.py, para no favorecer a los lotes pequeños con la caché.
Uso: python edge_benchmark/batch_sweep.py <modelo> <lote> <salida.json>   (modelo: rf_u3 | xgb_u3 | tcn_u)
El barrido completo (3 modelos x 9 lotes x 3 corridas) está en edge_benchmark/run_batch_sweep.sh."""
import sys, os, json, time, resource
modelo, lote, salida = sys.argv[1], int(sys.argv[2]), sys.argv[3]
BASE = os.path.dirname(os.path.abspath(__file__))


def rss_mb():
    with open('/proc/self/status') as f:
        for linea in f:
            if linea.startswith('VmRSS:'):
                return int(linea.split()[1]) / 1024


def pico_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


import numpy as np
r_np = rss_mb()
import onnxruntime as ort
r_ort = rss_mb()
so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
s = ort.InferenceSession(os.path.join(BASE, modelo + '.onnx'), so, providers=['CPUExecutionProvider'])
r_ses = rss_mb()
ent = s.get_inputs()[0]
es_3d = len(ent.shape) == 3
d = np.load(os.path.join(BASE, 'muestras.npz'))
X0 = np.ascontiguousarray(d['x_' + modelo], dtype=np.float32)
# Bloques de paquetes distintos: hasta ~20 000 paquetes (o un solo lote si el lote es mayor)
total = max(lote, (20000 // lote) * lote)
reps = int(np.ceil(total / len(X0)))
XA = np.ascontiguousarray(np.tile(X0, (reps, 1))[:total]) if reps > 1 else np.ascontiguousarray(X0[:total])
if es_3d:
    XA = XA.reshape(len(XA), 1, -1)
bloques = [XA[i:i + lote] for i in range(0, len(XA) - lote + 1, lote)]
r_datos = rss_mb()
n_cal = min(len(bloques), 3 if lote >= 256 else 50)
for k in range(n_cal):
    s.run(None, {ent.name: bloques[k]})
ts = []
t_ini = time.perf_counter()
k = 0
while (len(ts) < 5 or time.perf_counter() - t_ini < 0.6) and len(ts) < 20000:
    b = bloques[k % len(bloques)]
    k += 1
    c = time.perf_counter_ns()
    s.run(None, {ent.name: b})
    ts.append(time.perf_counter_ns() - c)
ts = np.array(ts, float)
res = dict(modelo=modelo, lote=lote, n_llamadas=len(ts), paquetes_distintos=int(min(total, len(X0))),
           us_por_paquete_mediana=float(np.median(ts) / 1e3 / lote), us_por_paquete_media=float(ts.mean() / 1e3 / lote),
           us_por_lote_mediana=float(np.median(ts) / 1e3),
           rss_numpy_mb=r_np, rss_tras_ort_mb=r_ort, rss_tras_sesion_mb=r_ses, rss_tras_datos_mb=r_datos,
           rss_final_mb=rss_mb(), pico_mb=pico_mb(), entrada_lote_mb=bloques[0].nbytes / 1024 ** 2,
           ort=ort.__version__)
json.dump(res, open(salida, 'w'), indent=1)
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}))
