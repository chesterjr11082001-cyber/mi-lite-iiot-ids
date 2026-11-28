"""Reexporta la LSTM compacta (E8, REP-01) a un ONNX que corre en CPU (x86 o ARM), sin el operador CudnnRNNV3.

El lstm.onnx de E8 se exportó en GPU: tf2onnx dejó el nodo CudnnRNNV3, que ONNX Runtime no implementa en CPU,
por eso la LSTM no se cronometró en E9. Los pesos entrenados sí están dentro de ese archivo (el vector
empaquetado de cuDNN, 4 480 valores). Este script los desempaqueta y arma el mismo modelo con el operador
estándar LSTM de ONNX (opset 17). No necesita TensorFlow ni GPU, y no reentrena: son los mismos pesos.

Formato de cuDNN tal como lo arma Keras 3 (backend TF, _cudnn_lstm), compuertas en orden i, f, c, o:
  W_i, W_f, W_c, W_o  -> 4 bloques [unidades x entradas]   (kernel transpuesto)
  R_i, R_f, R_c, R_o  -> 4 bloques [unidades x unidades]  (recurrent_kernel transpuesto)
  bW (4 x unidades, ceros puestos por Keras) y bR (4 x unidades, el sesgo de Keras)
ONNX LSTM usa el orden i, o, f, c; se reordena.

Uso: python edge_benchmark/export_lstm_cpu.py --gpu lstm.onnx --salida edge_benchmark/lstm_cpu.onnx
Verifica (1) que el ONNX nuevo no tenga operadores fuera del estándar, (2) que su salida coincida con una
implementación en NumPy que lee el vector de cuDNN directamente (otra ruta de código) y, si hay muestras.npz,
(3) la exactitud sobre los 10 000 paquetes de borde como control de orden de columnas."""
import argparse, json, os
import numpy as np
import onnx
from onnx import helper, numpy_helper as nh, TensorProto

p = argparse.ArgumentParser()
p.add_argument('--gpu', required=True, help='lstm.onnx exportado en GPU (con CudnnRNNV3)')
p.add_argument('--salida', required=True)
p.add_argument('--muestras', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'muestras.npz'))
a = p.parse_args()

m = onnx.load(a.gpu)
g = m.graph
ini = {t.name: nh.to_array(t) for t in g.initializer}
nodo = [n for n in g.node if n.op_type == 'CudnnRNNV3']
assert len(nodo) == 1, 'No se encontró un único nodo CudnnRNNV3'
nodo = nodo[0]
attr = {x.name: helper.get_attribute_value(x) for x in nodo.attribute}
assert attr['rnn_mode'] == b'lstm' and attr['direction'] == b'unidirectional' and attr['time_major'] == 0, attr
params = ini[nodo.input[3]].astype(np.float32)

# Densa final: MatMul (unidades x clases) + Add (clases)
mm = [n for n in g.node if n.op_type == 'MatMul'][0]
ad = [n for n in g.node if n.op_type == 'Add'][0]
Wd = ini[mm.input[1]].astype(np.float32)
bd = ini[[i for i in ad.input if i in ini][0]].astype(np.float32)
H, C = Wd.shape
n_atr = g.input[0].type.tensor_type.shape.dim[1].dim_value   # 25 entradas = 25 pasos de 1 canal
E = 1
esperado = 4 * H * E + 4 * H * H + 8 * H
assert params.size == esperado, f'vector cuDNN de {params.size} valores, se esperaban {esperado}'

# --- Desempaquetar (orden cuDNN i, f, c, o) --------------------------------------------------------
o = 0
W = []
for _ in range(4):
    W.append(params[o:o + H * E].reshape(H, E)); o += H * E
R = []
for _ in range(4):
    R.append(params[o:o + H * H].reshape(H, H)); o += H * H
bW = params[o:o + 4 * H].reshape(4, H); o += 4 * H
bR = params[o:o + 4 * H].reshape(4, H); o += 4 * H
assert np.all(bW == 0), 'Se esperaban sesgos de entrada en cero (Keras los pone así)'
assert bR[1].mean() > 0.5, 'La compuerta de olvido debería tener sesgo ≈ 1 (unit_forget_bias): revisa el orden'

# --- ONNX estándar (orden i, o, f, c) ----------------------------------------------------------------
orden = [0, 3, 1, 2]
W_onnx = np.concatenate([W[k] for k in orden])[None]                     # [1, 4H, E]
R_onnx = np.concatenate([R[k] for k in orden])[None]                     # [1, 4H, H]
B_onnx = np.concatenate([bW[orden].ravel(), bR[orden].ravel()])[None]    # [1, 8H]

nodos = [
    # ONNX Runtime en CPU solo admite layout = 0 (tiempo primero): [N, 25] -> [25, N] -> [25, N, 1]
    helper.make_node('Transpose', ['x'], ['xt'], perm=[1, 0]),
    helper.make_node('Unsqueeze', ['xt', 'eje2'], ['x3']),
    helper.make_node('LSTM', ['x3', 'W', 'R', 'B'], ['', 'h_n'], hidden_size=H),            # h_n: [1, N, H]
    helper.make_node('Squeeze', ['h_n', 'eje0'], ['h']),                                    # [N, H]
    helper.make_node('MatMul', ['h', 'Wd'], ['z0']),
    helper.make_node('Add', ['z0', 'bd'], ['z']),
    helper.make_node('Softmax', ['z'], ['salida'], axis=-1),
]
inits = [nh.from_array(np.array([2], np.int64), 'eje2'), nh.from_array(np.array([0], np.int64), 'eje0'),
         nh.from_array(W_onnx, 'W'), nh.from_array(R_onnx, 'R'), nh.from_array(B_onnx, 'B'),
         nh.from_array(Wd, 'Wd'), nh.from_array(bd, 'bd')]
grafo = helper.make_graph(nodos, 'LSTM_compacta_cpu',
                          [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['N', n_atr])],
                          [helper.make_tensor_value_info('salida', TensorProto.FLOAT, ['N', C])], inits)
nuevo = helper.make_model(grafo, opset_imports=[helper.make_opsetid('', 17)], producer_name='export_lstm_cpu.py')
nuevo.ir_version = 8
onnx.checker.check_model(nuevo, full_check=True)
onnx.save(nuevo, a.salida)

n_par = sum(int(np.prod(t.shape)) for t in (*W, *R)) + bR.size + Wd.size + bd.size
ops = sorted({n.op_type for n in nuevo.graph.node})
print(f'Guardado {a.salida}: {os.path.getsize(a.salida) / 1024:.1f} KB · {n_par:,} parámetros (Keras: 4,847) · operadores {ops}')


# --- Verificación 1: NumPy leyendo el formato cuDNN (i, f, c, o) ---------------------------------------
def sig(v):
    return 1 / (1 + np.exp(-v))


def lstm_numpy(X):
    h = np.zeros((len(X), H), np.float64); c = np.zeros_like(h)
    for t in range(n_atr):
        xt = X[:, t:t + 1].astype(np.float64)
        gi, gf, gc, go = (xt @ W[k].T + h @ R[k].T + bW[k] + bR[k] for k in range(4))
        c = sig(gf) * c + sig(gi) * np.tanh(gc)
        h = sig(go) * np.tanh(c)
    z = h @ Wd + bd
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


import onnxruntime as ort
s = ort.InferenceSession(a.salida, providers=['CPUExecutionProvider'])
rng = np.random.default_rng(0)
if os.path.exists(a.muestras):
    d = np.load(a.muestras)
    X = np.ascontiguousarray(d['x_xgb_u3'], dtype=np.float32)   # las mismas 25 entradas de IM (REP-01, protocolo C)
else:
    X = rng.random((2000, n_atr), dtype=np.float32)
P_onnx = s.run(None, {'x': X})[0]
P_np = lstm_numpy(X)
dif = float(np.abs(P_onnx - P_np).max())
acuerdo = float((P_onnx.argmax(1) == P_np.argmax(1)).mean())
info = {'origen': os.path.basename(a.gpu), 'salida': os.path.basename(a.salida), 'parametros': int(n_par),
        'onnx_kb': os.path.getsize(a.salida) / 1024, 'operadores': ops,
        'dif_max_prob_vs_numpy': dif, 'acuerdo_argmax_vs_numpy': acuerdo, 'n_verificacion': int(len(X))}
print(f'Verificación vs NumPy (formato cuDNN): dif. máx. de probabilidad {dif:.2e} · acuerdo argmax {acuerdo:.4f}')
assert dif < 1e-4 and acuerdo > 0.999, 'El ONNX nuevo no reproduce la LSTM: no lo uses'
if os.path.exists(a.muestras) and 'y' in d.files:
    pred = P_onnx.argmax(1)
    info['exactitud_muestras'] = float((pred == d['y']).mean())
    info['clases_predichas'] = int(len(np.unique(pred)))
    print(f"Control de orden de columnas: exactitud {info['exactitud_muestras'] * 100:.2f} % en {len(X):,} paquetes "
          f"({info['clases_predichas']} clases predichas)")
json.dump(info, open(a.salida.replace('.onnx', '_export.json'), 'w'), indent=2)
