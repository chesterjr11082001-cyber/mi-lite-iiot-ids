
import argparse, json, os, platform, time
import numpy as np
import onnxruntime as ort
try:
    import psutil
    rss = lambda: psutil.Process().memory_info().rss / 1024 ** 2
except ImportError:
    rss = lambda: float('nan')
p = argparse.ArgumentParser()
p.add_argument('--n', type=int, default=10000)
p.add_argument('--calentamiento', type=int, default=500)
p.add_argument('--hilos', type=int, default=1)
p.add_argument('--salida', default='resultado_borde.json')
a = p.parse_args()
base = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(base, 'muestras.npz'))
res = {}
for nombre, tipo in [('xgb_u3', 'arbol'), ('rf_u3', 'arbol'), ('tcn_u', 'tcn')]:
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.hilos
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    r0 = rss()
    s = ort.InferenceSession(os.path.join(base, nombre + '.onnx'), so, providers=['CPUExecutionProvider'])
    r1 = rss()
    ent = s.get_inputs()[0]
    es_3d = len(ent.shape) == 3
    X = np.ascontiguousarray(d['x_' + nombre][:a.n], dtype=np.float32)
    def predecir(x):
        out = s.run(None, {ent.name: x.reshape(len(x), 1, -1) if es_3d else x})[0]
        return np.argmax(out, axis=1) if tipo == 'tcn' else np.asarray(out).astype(np.int64).ravel()
    for i in range(min(a.calentamiento, len(X))):
        predecir(X[i:i + 1])
    t = np.empty(len(X)); pr = np.empty(len(X), dtype=np.int64)
    for i in range(len(X)):
        c = time.perf_counter_ns()
        pr[i] = predecir(X[i:i + 1])[0]
        t[i] = time.perf_counter_ns() - c
    t /= 1e3
    res[nombre] = {'mediana_us': float(np.median(t)), 'p95_us': float(np.percentile(t, 95)), 'p99_us': float(np.percentile(t, 99)),
                   'media_us': float(t.mean()), 'paquetes_por_s': float(1e6 / t.mean()), 'memoria_modelo_mb': r1 - r0,
                   'rss_mb': rss(), 'acuerdo_con_esperado': float((pr == d['esperado_' + nombre][:len(X)]).mean()),
                   'tamano_kb': os.path.getsize(os.path.join(base, nombre + '.onnx')) / 1024}
    print(nombre, {k: round(v, 3) for k, v in res[nombre].items()})
info = {'maquina': platform.machine(), 'procesador': platform.processor(), 'sistema': platform.platform(),
        'python': platform.python_version(), 'onnxruntime': ort.__version__, 'hilos': a.hilos, 'n': a.n, 'resultados': res}
json.dump(info, open(os.path.join(base, a.salida), 'w'), indent=2)
print('Guardado en', a.salida)
