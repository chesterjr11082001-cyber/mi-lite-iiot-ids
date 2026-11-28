"""Benchmark de borde portátil (protocolo E9 + batch_sweep.py) para ARM64: Raspberry Pi, Jetson, MacBook (Apple Silicon) o x86.

Mide, para cada modelo ONNX y en un PROCESO NUEVO por modelo y ronda (ONNX Runtime, CPUExecutionProvider, 1 hilo):
  - Latencia con lote de 1: mediana, p95 y p99 por paquete sobre n paquetes distintos, tras `calentamiento` llamadas descartadas (E9).
  - Tiempo por paquete con lote de 1 024: bloques consecutivos de paquetes distintos, una pasada de calentamiento y luego
    llamadas hasta reunir >= 20 bloques y >= 1 s; se reporta la mediana por bloque / 1 024 (como batch_sweep.py).
  - Memoria: RSS pico del proceso completo (Python + NumPy + ONNX Runtime + modelo + datos + inferencia) y el aumento de RSS
    al cargar la sesión (huella del modelo en el runtime).
  - Tamaño del archivo ONNX y su SHA-256.
  - Control de corrección: acuerdo con las predicciones esperadas (árboles y TCN) o exactitud sobre las etiquetas.
Además guarda la ficha del equipo (modelo de placa/CPU, SO, Python, NumPy, ONNX Runtime, temperatura si existe).

Solo necesita python3, numpy y onnxruntime (psutil es opcional). El mismo archivo corre sin cambios en macOS y Linux.

Uso (desde la raíz del repositorio):
  python3 edge_benchmark/arm_benchmark.py --etiqueta macbook-m5
  python3 edge_benchmark/arm_benchmark.py --etiqueta rpi5 --rondas 3
  python3 edge_benchmark/arm_benchmark.py --etiqueta rpi5 --energia rf_u3 --segundos 60 --reposo 30   (con medidor USB)
Salida: results/arm/<etiqueta>_<fecha>.json"""
import argparse, datetime, hashlib, json, os, platform, subprocess, sys, time

BASE = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(BASE)
# nombre en el paper, archivo ONNX, clave de entradas en muestras.npz, clave de predicciones esperadas (o None)
MODELOS = {
    'xgb_u3':      ('MI-LiteXGB',  'xgb_u3.onnx',      'x_xgb_u3', 'esperado_xgb_u3'),
    'rf_u3':       ('MI-LiteRF',   'rf_u3.onnx',       'x_rf_u3',  'esperado_rf_u3'),
    'tcn_u':       ('TCN',         'tcn_u.onnx',       'x_tcn_u',  'esperado_tcn_u'),
    'transformer': ('Transformer', 'transformer.onnx', 'x_xgb_u3', None),   # mismas 25 entradas de IM que MI-LiteXGB
    'lstm':        ('LSTM',        'lstm_cpu.onnx',    'x_xgb_u3', None),   # reexportada desde CPU (export_lstm_cpu.py)
}
HILOS_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                               'NUMEXPR_NUM_THREADS')}


# ----------------------------------------------------------------------------------------------------- utilidades
def pico_mb():
    """RSS pico del proceso. ru_maxrss está en bytes en macOS y en KB en Linux."""
    import resource
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 1024 ** 2 if sys.platform == 'darwin' else v / 1024


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 2
    except ImportError:
        pass
    try:
        with open('/proc/self/status') as f:
            for l in f:
                if l.startswith('VmRSS:'):
                    return int(l.split()[1]) / 1024
    except OSError:
        pass
    return float('nan')   # macOS sin psutil: solo queda el pico


def memoria_linux_mb():
    """Linux: RSS privado (anónimo: lo que debe quedarse en RAM) y RSS de archivos (bibliotecas compartidas y el ONNX
    mapeado, que el kernel puede descartar y releer del disco) del proceso actual. En macOS no existe: None."""
    try:
        with open('/proc/self/status') as f:
            v = {l.split(':')[0]: int(l.split()[1]) / 1024 for l in f if l.startswith(('RssAnon', 'RssFile', 'RssShmem', 'VmHWM'))}
        return {'rss_anon_mb': v.get('RssAnon'), 'rss_archivos_mb': v.get('RssFile'), 'vm_hwm_mb': v.get('VmHWM')}
    except OSError:
        return {'rss_anon_mb': None, 'rss_archivos_mb': None, 'vm_hwm_mb': None}


def leer(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def leer_archivo(ruta):
    try:
        with open(ruta, 'rb') as f:
            return f.read().decode(errors='ignore').strip('\x00').strip()
    except OSError:
        return None


def temperatura_c():
    t = leer_archivo('/sys/class/thermal/thermal_zone0/temp')
    try:
        return int(t) / 1000 if t else None
    except ValueError:
        return None


def ficha_equipo():
    import numpy, onnxruntime
    f = {'maquina': platform.machine(), 'sistema': platform.platform(), 'nodo': platform.node(),
         'python': platform.python_version(), 'numpy': numpy.__version__, 'onnxruntime': onnxruntime.__version__,
         'proveedores_ort': onnxruntime.get_available_providers(), 'nucleos_logicos': os.cpu_count()}
    if sys.platform == 'darwin':
        f['cpu'] = leer(['sysctl', '-n', 'machdep.cpu.brand_string'])
        f['modelo_equipo'] = leer(['sysctl', '-n', 'hw.model'])
        f['ram_gb'] = round(int(leer(['sysctl', '-n', 'hw.memsize']) or 0) / 1024 ** 3, 1)
        f['nucleos_rendimiento'] = leer(['sysctl', '-n', 'hw.perflevel0.physicalcpu'])
        f['nucleos_eficiencia'] = leer(['sysctl', '-n', 'hw.perflevel1.physicalcpu'])
        f['macos'] = leer(['sw_vers', '-productVersion'])
        f['alimentacion'] = leer(['pmset', '-g', 'batt'])
        f['modo_bajo_consumo'] = leer(['pmset', '-g'])
        if f['modo_bajo_consumo']:
            f['modo_bajo_consumo'] = next((l.split()[-1] for l in f['modo_bajo_consumo'].splitlines() if 'lowpowermode' in l), None)
    else:
        f['modelo_equipo'] = leer_archivo('/proc/device-tree/model')   # Raspberry Pi / Jetson
        cpu = leer_archivo('/proc/cpuinfo') or ''
        f['cpu'] = next((l.split(':', 1)[1].strip() for l in cpu.splitlines()
                         if l.lower().startswith(('model name', 'hardware', 'cpu part'))), platform.processor())
        mem = leer_archivo('/proc/meminfo') or ''
        kb = next((int(l.split()[1]) for l in mem.splitlines() if l.startswith('MemTotal')), 0)
        f['ram_gb'] = round(kb / 1024 ** 2, 1)
        f['gobernador_cpu'] = leer_archivo('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor')
        f['frec_max_mhz'] = leer_archivo('/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq')
        f['throttled'] = leer(['vcgencmd', 'get_throttled'])          # solo Raspberry Pi
        f['jetson_nvpmodel'] = leer(['nvpmodel', '-q'])              # solo Jetson
        f['kernel'] = platform.release()
    f['temperatura_c'] = temperatura_c()
    return f


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def fijar_nucleo(n):
    if n is not None and hasattr(os, 'sched_setaffinity'):   # Linux; macOS no permite fijar núcleos
        try:
            os.sched_setaffinity(0, {n})
            return n
        except OSError:
            return None
    return None


# ----------------------------------------------------------------------------------------- medición (proceso hijo)
def trabajador(a):
    nucleo = fijar_nucleo(a.nucleo)
    r0 = rss_mb()
    import numpy as np
    import onnxruntime as ort
    r_ort = rss_mb()
    nombre, archivo, clave_x, clave_esp = MODELOS[a.trabajador]
    ruta = os.path.join(BASE, archivo)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    t0 = time.perf_counter()
    s = ort.InferenceSession(ruta, so, providers=['CPUExecutionProvider'])
    t_carga = time.perf_counter() - t0
    r_ses = rss_mb()
    ent = s.get_inputs()[0]
    es_3d = len(ent.shape) == 3
    salida0 = s.get_outputs()[0].name
    d = np.load(os.path.join(BASE, 'muestras.npz'))
    X = np.ascontiguousarray(d[clave_x][:a.n], dtype=np.float32)
    if es_3d:
        X = X.reshape(len(X), 1, -1)

    def correr(L):
        return s.run([salida0], {ent.name: L})[0]

    # Predicciones (control de corrección), fuera del cronómetro
    P = np.concatenate([np.asarray(correr(X[i:i + a.lote])) for i in range(0, len(X), a.lote)])   # mismo lote que la medición
    pred = P.argmax(1) if P.ndim == 2 else P.astype(np.int64).ravel()
    control = {'exactitud_muestras': float((pred == d['y'][:len(X)]).mean())}
    if clave_esp:
        control['acuerdo_con_esperado'] = float((pred == d[clave_esp][:len(X)]).mean())
    r_pre = rss_mb(); p_pre = pico_mb()

    # --- Lote de 1 (E9): calentamiento + mediana por paquete -----------------------------------------------------
    for i in range(min(a.calentamiento, len(X))):
        correr(X[i:i + 1])
    t1 = np.empty(len(X))
    for i in range(len(X)):
        c = time.perf_counter_ns()
        correr(X[i:i + 1])
        t1[i] = time.perf_counter_ns() - c
    t1 /= 1e3

    # --- Lote de 1 024 (batch_sweep.py): bloques de paquetes distintos ------------------------------------------
    B = a.lote
    total = max(B, (20480 // B) * B)
    XA = np.ascontiguousarray(np.tile(X, (int(np.ceil(total / len(X))),) + (1,) * (X.ndim - 1))[:total])
    bloques = [XA[i:i + B] for i in range(0, total - B + 1, B)]
    for b in bloques:
        correr(b)
    tb, k, ini = [], 0, time.perf_counter()
    while len(tb) < 20 or time.perf_counter() - ini < 1.0:
        b = bloques[k % len(bloques)]; k += 1
        c = time.perf_counter_ns()
        correr(b)
        tb.append(time.perf_counter_ns() - c)
    tb = np.array(tb, float) / 1e3 / B

    res = {'modelo': a.trabajador, 'nombre': nombre, 'archivo': archivo, 'onnx_kb': os.path.getsize(ruta) / 1024,
           'onnx_sha256': sha256(ruta), 'entradas': int(X.shape[-1]), 'n_lote1': int(len(t1)),
           'lote1_mediana_us': float(np.median(t1)), 'lote1_p95_us': float(np.percentile(t1, 95)),
           'lote1_p99_us': float(np.percentile(t1, 99)), 'lote1_media_us': float(t1.mean()),
           'paquetes_por_s_lote1': float(1e6 / np.median(t1)),
           'lote': B, 'n_llamadas_lote': int(len(tb)), 'lote_us_por_paquete_mediana': float(np.median(tb)),
           'carga_sesion_s': t_carga, 'pico_proceso_mb': pico_mb(), 'aumento_pico_en_inferencia_mb': pico_mb() - p_pre,
           'rss_tras_imports_mb': r_ort, 'rss_inicial_mb': r0, 'huella_sesion_mb': r_ses - r_ort, 'rss_antes_medir_mb': r_pre,
           'rss_final_mb': rss_mb(), **memoria_linux_mb(), 'nucleo_fijado': nucleo, 'temperatura_final_c': temperatura_c(),
           **control}
    json.dump(res, open(a.salida_trabajador, 'w'), indent=1)


# ------------------------------------------------------------------------------------------- modo energía (hijo)
def energia(a):
    """Inferencia sostenida con lote de 1 para leer el medidor USB: reposo y carga en ventanas con hora marcada."""
    fijar_nucleo(a.nucleo)
    import numpy as np
    import onnxruntime as ort
    nombre, archivo, clave_x, _ = MODELOS[a.energia]
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    s = ort.InferenceSession(os.path.join(BASE, archivo), so, providers=['CPUExecutionProvider'])
    ent = s.get_inputs()[0]
    X = np.ascontiguousarray(np.load(os.path.join(BASE, 'muestras.npz'))[clave_x], dtype=np.float32)
    if len(ent.shape) == 3:
        X = X.reshape(len(X), 1, -1)
    for i in range(500):
        s.run(None, {ent.name: X[i:i + 1]})
    hora = lambda: datetime.datetime.now().strftime('%H:%M:%S')
    print(f'[{hora()}] REPOSO {a.reposo} s: anota la potencia media del medidor (W) en esta ventana', flush=True)
    r_ini = hora(); time.sleep(a.reposo); r_fin = hora()
    print(f'[{hora()}] CARGA {a.segundos} s: {nombre}, lote de 1, 1 hilo. Anota la potencia media (W)', flush=True)
    c_ini, n, t0 = hora(), 0, time.perf_counter()
    while time.perf_counter() - t0 < a.segundos:
        s.run(None, {ent.name: X[n % len(X):n % len(X) + 1]}); n += 1
    dur = time.perf_counter() - t0
    print(f'[{hora()}] FIN: {n:,} inferencias en {dur:.1f} s ({dur / n * 1e6:.1f} µs c/u)', flush=True)
    return {'modelo': a.energia, 'nombre': nombre, 'reposo': {'inicio': r_ini, 'fin': r_fin, 'segundos': a.reposo},
            'carga': {'inicio': c_ini, 'segundos': dur, 'inferencias': n, 'us_por_inferencia': dur / n * 1e6},
            'potencia_reposo_w': None, 'potencia_carga_w': None,
            'formula': 'energia_por_paquete_uJ = (potencia_carga_w - potencia_reposo_w) * us_por_inferencia'}


# ------------------------------------------------------------------------------------------------- orquestador
def principal(a):
    os.makedirs(a.salida, exist_ok=True)
    marca = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    if a.energia:
        res = {'etiqueta': a.etiqueta, 'fecha': marca, 'equipo': ficha_equipo(), 'energia': energia(a)}
        ruta = os.path.join(a.salida, f'{a.etiqueta}_energia_{a.energia}_{marca}.json')
        json.dump(res, open(ruta, 'w'), indent=2, ensure_ascii=False)
        print('Guardado', ruta, '(completa potencia_reposo_w y potencia_carga_w con las lecturas del medidor)')
        return
    modelos = [m for m in a.modelos if os.path.exists(os.path.join(BASE, MODELOS[m][1]))]
    for m in set(a.modelos) - set(modelos):
        print(f'⚠️ Falta edge_benchmark/{MODELOS[m][1]}: se omite {MODELOS[m][0]}')
    equipo = ficha_equipo()
    print(f"Equipo: {equipo.get('modelo_equipo') or ''} {equipo.get('cpu') or ''} · {equipo['sistema']} · "
          f"ONNX Runtime {equipo['onnxruntime']} · {a.rondas} rondas × {len(modelos)} modelos, 1 hilo")
    env = dict(os.environ, **HILOS_ENV)
    tmp = os.path.join(a.salida, f'.tmp_{marca}')
    os.makedirs(tmp, exist_ok=True)
    corridas = []
    for r in range(1, a.rondas + 1):
        orden = modelos[r - 1:] + modelos[:r - 1]   # rota el orden entre rondas
        for m in orden:
            sal = os.path.join(tmp, f'{m}_r{r}.json')
            cmd = [sys.executable, os.path.abspath(__file__), '--trabajador', m, '--salida-trabajador', sal,
                   '--n', str(a.n), '--calentamiento', str(a.calentamiento), '--lote', str(a.lote)]
            if a.nucleo is not None:
                cmd += ['--nucleo', str(a.nucleo)]
            p = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if p.returncode != 0:
                print(f'❌ {m} ronda {r} falló:\n{p.stderr[-1500:]}')
                continue
            d = json.load(open(sal)); d['ronda'] = r; corridas.append(d)
            ctrl = d.get('acuerdo_con_esperado', d['exactitud_muestras'])
            print(f"  r{r} {d['nombre']:11s} lote 1: {d['lote1_mediana_us']:8.1f} µs (p99 {d['lote1_p99_us']:8.1f}) · "
                  f"lote {a.lote}: {d['lote_us_por_paquete_mediana']:7.2f} µs/paq · pico {d['pico_proceso_mb']:6.1f} MB · "
                  f"ONNX {d['onnx_kb']:7.1f} KB · {'acuerdo' if 'acuerdo_con_esperado' in d else 'exactitud'} {ctrl:.4f}", flush=True)
            time.sleep(a.pausa)
    import numpy as np
    resumen = {}
    for m in modelos:
        c = [d for d in corridas if d['modelo'] == m]
        if not c:
            continue
        med = lambda k: float(np.median([d[k] for d in c]))
        rango = lambda k: [float(min(d[k] for d in c)), float(max(d[k] for d in c))]
        resumen[MODELOS[m][0]] = {
            'n_rondas': len(c), 'lote1_mediana_us': med('lote1_mediana_us'), 'lote1_mediana_us_rango': rango('lote1_mediana_us'),
            'lote1_p99_us': med('lote1_p99_us'), f'lote{a.lote}_us_por_paquete': med('lote_us_por_paquete_mediana'),
            f'lote{a.lote}_us_por_paquete_rango': rango('lote_us_por_paquete_mediana'),
            'pico_proceso_mb': med('pico_proceso_mb'), 'huella_sesion_mb': med('huella_sesion_mb'),
            'onnx_kb': c[0]['onnx_kb'], 'paquetes_por_s_lote1': med('paquetes_por_s_lote1'),
            'acuerdo_con_esperado': c[0].get('acuerdo_con_esperado'), 'exactitud_muestras': c[0]['exactitud_muestras']}
    salida = {'etiqueta': a.etiqueta, 'fecha': marca, 'script': 'edge_benchmark/arm_benchmark.py',
              'protocolo': {'runtime': 'ONNX Runtime, CPUExecutionProvider', 'hilos': 1, 'proceso_nuevo_por_modelo_y_ronda': True,
                            'rondas': a.rondas, 'n_lote1': a.n, 'calentamiento_lote1': a.calentamiento, 'lote': a.lote,
                            'paquetes': 'edge_benchmark/muestras.npz (prueba REP-01, protocolo C)',
                            'estadistico': 'mediana por paquete en cada ronda; resumen = mediana de las rondas [min, max]',
                            'nucleo_fijado': a.nucleo if hasattr(os, 'sched_setaffinity') else 'no (macOS)'},
              'equipo': equipo, 'temperatura_final_c': temperatura_c(), 'resumen': resumen, 'corridas': corridas}
    ruta = os.path.join(a.salida, f'{a.etiqueta}_{marca}.json')
    json.dump(salida, open(ruta, 'w'), indent=2, ensure_ascii=False)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print(f'\nResumen (mediana de {a.rondas} rondas):')
    print(f"{'Modelo':12s} {'lote 1 (µs)':>12s} {'p99 (µs)':>10s} {'lote ' + str(a.lote) + ' (µs/paq)':>20s} {'pico (MB)':>10s} {'ONNX (KB)':>10s}")
    for n, v in resumen.items():
        print(f"{n:12s} {v['lote1_mediana_us']:12.1f} {v['lote1_p99_us']:10.1f} {v[f'lote{a.lote}_us_por_paquete']:20.2f} "
              f"{v['pico_proceso_mb']:10.1f} {v['onnx_kb']:10.1f}")
    print('Guardado', ruta)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--etiqueta', default=platform.node().split('.')[0] or 'equipo', help='nombre corto del equipo, p. ej. rpi5')
    p.add_argument('--modelos', nargs='+', default=list(MODELOS), choices=list(MODELOS))
    p.add_argument('--rondas', type=int, default=3)
    p.add_argument('--n', type=int, default=10000, help='paquetes para la latencia con lote de 1')
    p.add_argument('--calentamiento', type=int, default=500)
    p.add_argument('--lote', type=int, default=1024)
    p.add_argument('--nucleo', type=int, default=None, help='Linux: fija el proceso a este núcleo (como taskset -c)')
    p.add_argument('--pausa', type=float, default=2.0, help='segundos entre procesos (enfriamiento)')
    p.add_argument('--salida', default=os.path.join(RAIZ, 'results', 'arm'))
    p.add_argument('--energia', choices=list(MODELOS), help='modo medidor USB: inferencia sostenida de este modelo')
    p.add_argument('--segundos', type=float, default=60)
    p.add_argument('--reposo', type=float, default=30)
    p.add_argument('--trabajador', help=argparse.SUPPRESS)
    p.add_argument('--salida-trabajador', dest='salida_trabajador', help=argparse.SUPPRESS)
    a = p.parse_args()
    trabajador(a) if a.trabajador else principal(a)
