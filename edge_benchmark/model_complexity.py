"""Costo de cómputo por paquete, independiente del hardware, de los cinco modelos ONNX.

Árboles (MI-LiteXGB, MI-LiteRF): recorre cada árbol del TreeEnsembleClassifier con los 10,000 paquetes de muestras.npz
y cuenta las comparaciones realmente hechas por paquete (media y máximo), además de árboles, nodos, hojas y profundidad.
Control: con las hojas alcanzadas reconstruye la clase predicha y la compara con la esperada (debe dar 1.0).

Redes (TCN, LSTM, Transformer): ejecuta el grafo con lote 1 exponiendo todos los tensores intermedios y cuenta
multiplicaciones-acumulaciones (MAC) de MatMul/Gemm/Conv/LSTM y el resto de operaciones elementales (sumas,
activaciones, normalizaciones, softmax), además de los parámetros.

Solo necesita numpy, onnx y onnxruntime. Uso: python3 edge_benchmark/model_complexity.py
Salida: results/complexity/model_complexity.json"""
import json, os
import numpy as np
import onnx
from onnx import helper, numpy_helper as nh, shape_inference
import onnxruntime as ort

BASE = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(BASE)
MODELOS = [('MI-LiteXGB', 'xgb_u3.onnx', 'x_xgb_u3', 'esperado_xgb_u3'),
           ('MI-LiteRF', 'rf_u3.onnx', 'x_rf_u3', 'esperado_rf_u3'),
           ('TCN', 'tcn_u.onnx', 'x_tcn_u', 'esperado_tcn_u'),
           ('LSTM', 'lstm_cpu.onnx', 'x_xgb_u3', None),
           ('Transformer', 'transformer.onnx', 'x_xgb_u3', None)]


# ------------------------------------------------------------------------------------------------------ árboles
def analizar_arboles(m, X, esperado):
    nodo = [n for n in m.graph.node if n.op_type == 'TreeEnsembleClassifier'][0]
    a = {x.name: helper.get_attribute_value(x) for x in nodo.attribute}
    tid, nid = np.array(a['nodes_treeids']), np.array(a['nodes_nodeids'])
    feat, val = np.array(a['nodes_featureids']), np.array(a['nodes_values'], np.float32)
    tn, fn = np.array(a['nodes_truenodeids']), np.array(a['nodes_falsenodeids'])
    modos = [x.decode() if isinstance(x, bytes) else x for x in a['nodes_modes']]
    clases = np.array(a['classlabels_int64s'])
    C = len(clases)
    puntaje = np.tile(np.array(a.get('base_values', [0.0] * C), np.float64), (len(X), 1))
    hoja_peso = {}
    for t, n, c, w in zip(a['class_treeids'], a['class_nodeids'], a['class_ids'], a['class_weights']):
        hoja_peso.setdefault((t, n), []).append((c, w))
    comparaciones = np.zeros(len(X), np.int64)
    profundidades, hojas = [], 0
    for t in np.unique(tid):
        sel = np.where(tid == t)[0]
        idx = {int(nid[i]): i for i in sel}            # id de nodo -> fila
        es_hoja = {int(nid[i]): modos[i] == 'LEAF' for i in sel}
        hojas += sum(es_hoja.values())
        actual = np.full(len(X), int(nid[sel].min()), np.int64)   # raíz = id 0
        prof = 0
        while True:
            activos = np.array([not es_hoja[int(n)] for n in actual])
            if not activos.any():
                break
            filas = np.array([idx[int(n)] for n in actual])
            f, v = feat[filas], val[filas]
            x = X[np.arange(len(X)), f]
            modo = np.array([modos[r] for r in filas])
            verdad = np.where(modo == 'BRANCH_LT', x < v, x <= v)
            sig = np.where(verdad, tn[filas], fn[filas])
            comparaciones += activos
            actual = np.where(activos, sig, actual)
            prof += 1
        profundidades.append(prof)
        for n in np.unique(actual):
            for c, w in hoja_peso.get((int(t), int(n)), []):
                puntaje[actual == n, c] += w
    pred = clases[puntaje.argmax(1)]
    return {'tipo': 'arboles', 'arboles': int(len(np.unique(tid))), 'nodos': int(len(nid)), 'hojas': int(hojas),
            'nodos_internos': int(len(nid) - hojas), 'profundidad_max': int(max(profundidades)),
            'comparaciones_media': float(comparaciones.mean()), 'comparaciones_p99': float(np.percentile(comparaciones, 99)),
            'comparaciones_max': int(comparaciones.max()),
            'valores_hoja': int(len(a['class_weights'])), 'post_transform': a.get('post_transform', b'NONE').decode()
            if isinstance(a.get('post_transform'), bytes) else a.get('post_transform', 'NONE'),
            'acuerdo_reconstruccion': float((pred == esperado).mean()) if esperado is not None else None}


# -------------------------------------------------------------------------------------------------------- redes
ELEMENTALES = {'Add', 'Sub', 'Mul', 'Div', 'Relu', 'Neg', 'Sqrt', 'Reciprocal', 'Sigmoid', 'Tanh', 'Erf', 'Pow', 'PRelu'}


def analizar_red(m, x1):
    m = shape_inference.infer_shapes(m)
    tipos = {v.name: v.type.tensor_type.elem_type for v in list(m.graph.value_info) + list(m.graph.output)}
    ini = {t.name: nh.to_array(t) for t in m.graph.initializer}
    salidas_orig = [o.name for o in m.graph.output]
    nombres = []
    for n in m.graph.node:
        for o in n.output:
            if o and o not in salidas_orig and o not in ini and tipos.get(o):
                m.graph.output.append(helper.make_tensor_value_info(o, tipos[o], None)); nombres.append(o)
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(m.SerializeToString(), so, providers=['CPUExecutionProvider'])
    ent = s.get_inputs()[0]
    x = x1.reshape(1, 1, -1) if len(ent.shape) == 3 else x1.reshape(1, -1)
    outs = s.run(None, {ent.name: x.astype(np.float32)})
    forma = {o.name: np.shape(v) for o, v in zip(s.get_outputs(), outs)}
    forma[ent.name] = x.shape
    forma.update({k: v.shape for k, v in ini.items()})
    mac, otras, por_op = 0, 0, {}
    for n in m.graph.node:
        op, sal = n.op_type, (forma.get(n.output[0]) if n.output and n.output[0] else None)
        k = 0
        if op in ('MatMul', 'Gemm') and sal is not None:
            k = int(np.prod(sal)) * int(forma[n.input[0]][-1 if op == 'MatMul' else
                                         (0 if any(a.name == 'transA' and a.i for a in n.attribute) else -1)])
            mac += k
        elif op == 'Conv' and sal is not None:
            k = int(np.prod(sal)) * int(np.prod(forma[n.input[1]][1:])); mac += k
        elif op == 'LSTM':
            sx, sw, sr = forma[n.input[0]], forma[n.input[1]], forma[n.input[2]]
            seq, H4 = sx[0], sw[1]
            k = int(seq * H4 * (sw[2] + sr[2])); mac += k
            otras += int(seq * (H4 * 2 + (H4 // 4) * 5))       # sesgos, 3 sigmoides + 2 tanh, productos de celda
        elif sal is not None:
            if op in ELEMENTALES:
                k = int(np.prod(sal))
            elif op == 'Softmax':
                k = 3 * int(np.prod(sal))
            elif op in ('ReduceMean', 'GlobalAveragePool'):
                k = int(np.prod(forma[n.input[0]]))
            otras += k
        if k:
            por_op[op] = por_op.get(op, 0) + k
    param = int(sum(v.size for v in ini.values() if v.dtype in (np.float32, np.float64) and v.size > 1))
    return {'tipo': 'red', 'parametros': param, 'mac_por_paquete': int(mac), 'otras_ops_por_paquete': int(otras),
            'ops_por_tipo': por_op}


def principal():
    d = np.load(os.path.join(BASE, 'muestras.npz'))
    res = {}
    for nombre, archivo, cx, ce in MODELOS:
        ruta = os.path.join(BASE, archivo)
        if not os.path.exists(ruta):
            print(f'falta {archivo}, se omite'); continue
        m = onnx.load(ruta)
        X = np.ascontiguousarray(d[cx], dtype=np.float32)
        r = (analizar_arboles(m, X, d[ce] if ce else None) if any(n.op_type == 'TreeEnsembleClassifier' for n in m.graph.node)
             else analizar_red(m, X[0]))
        r.update(archivo=archivo, onnx_kb=os.path.getsize(ruta) / 1024, entradas=int(X.shape[1]))
        res[nombre] = r
        if r['tipo'] == 'arboles':
            print(f"{nombre:11s} {r['arboles']:4d} árboles · {r['nodos_internos']:,} nodos internos · prof. máx {r['profundidad_max']} · "
                  f"comparaciones por paquete: media {r['comparaciones_media']:.0f}, máx {r['comparaciones_max']} · "
                  f"acuerdo de la reconstrucción {r['acuerdo_reconstruccion']}")
        else:
            print(f"{nombre:11s} {r['parametros']:,} parámetros · {r['mac_por_paquete']:,} MAC/paquete · "
                  f"{r['otras_ops_por_paquete']:,} otras ops/paquete")
    os.makedirs(os.path.join(RAIZ, 'results', 'complexity'), exist_ok=True)
    ruta = os.path.join(RAIZ, 'results', 'complexity', 'model_complexity.json')
    json.dump({'script': 'edge_benchmark/model_complexity.py', 'paquetes': 'muestras.npz (10,000; árboles)',
               'modelos': res}, open(ruta, 'w'), indent=2, ensure_ascii=False)
    print('Guardado', ruta)


if __name__ == '__main__':
    principal()
