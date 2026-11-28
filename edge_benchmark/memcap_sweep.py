"""¿Caben los modelos en la RAM de una placa pequeña? Barrido de límites de memoria con Docker.

Corre cada modelo (el mismo trabajador de arm_benchmark.py: lote 1 y lote 1,024, 1 hilo) dentro de un contenedor
Linux con un límite DURO de memoria (sin swap) y 1 CPU, para límites decrecientes (1,024 ... 64 MB). Si el proceso
excede el límite, el kernel lo mata (OOM) y se registra. Resultado por modelo: el menor límite con el que corre
completo y su RSS pico.

Qué es válido reportar: si el modelo cabe o no bajo cada límite, su RSS pico y su memoria privada (no dependen de
la velocidad del procesador). Qué NO es válido: los tiempos medidos aquí (son los del equipo anfitrión).
Ojo: el RSS incluye páginas de bibliotecas compartidas y del ONNX mapeado, que el kernel puede descartar y releer
bajo presión; por eso un modelo puede «caber» con un límite menor que su RSS pico. La memoria privada (RssAnon) es
la que sí debe quedar en RAM.

Solo usa la biblioteca estándar de Python y Docker (Docker Desktop o Colima en macOS; docker en Linux).
En un Mac con Apple Silicon el contenedor es Linux ARM64 nativo (aarch64), como una Raspberry Pi.

Uso (desde la raíz del repositorio):
  python3 edge_benchmark/memcap_sweep.py                       # construye la imagen la primera vez (~1 min)
  python3 edge_benchmark/memcap_sweep.py --limites 512 256 128 --modelos xgb_u3 rf_u3
Salida: results/memcap/memcap_<etiqueta>_<fecha>.json"""
import argparse, datetime, json, os, platform, subprocess, sys, time

BASE = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(BASE)
MODELOS = ['xgb_u3', 'rf_u3', 'tcn_u', 'transformer', 'lstm']
NOMBRES = {'xgb_u3': 'MI-LiteXGB', 'rf_u3': 'MI-LiteRF', 'tcn_u': 'TCN', 'transformer': 'Transformer', 'lstm': 'LSTM'}
DOCKERFILE = """FROM python:3.12-slim
RUN pip install --no-cache-dir numpy psutil onnxruntime=={ort}
"""


def docker(*args, **kw):
    return subprocess.run(['docker', *args], capture_output=True, text=True, **kw)


def asegurar_imagen(a):
    if docker('image', 'inspect', a.imagen).returncode == 0:
        return
    print(f'Construyendo la imagen {a.imagen} (python:3.12-slim + numpy + onnxruntime {a.ort}), una sola vez...', flush=True)
    p = subprocess.run(['docker', 'build', '--platform', a.plataforma, '-t', a.imagen, '-'],
                       input=DOCKERFILE.format(ort=a.ort), text=True, capture_output=True)
    if p.returncode != 0:
        sys.exit(f'No se pudo construir la imagen:\n{p.stderr[-2000:]}')


def una_corrida(a, modelo, limite_mb):
    nombre = f'memcap_{modelo}_{limite_mb}_{os.getpid()}'
    sal = f'results/memcap/.tmp/{modelo}_{limite_mb}.json'
    if os.path.exists(os.path.join(RAIZ, sal)):
        os.remove(os.path.join(RAIZ, sal))
    cmd = ['run', '--name', nombre, '--platform', a.plataforma, f'--memory={limite_mb}m', f'--memory-swap={limite_mb}m',
           f'--cpus={a.cpus}', '-e', 'OMP_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1', '-v', f'{RAIZ}:/work', '-w', '/work',
           a.imagen, 'python', 'edge_benchmark/arm_benchmark.py', '--trabajador', modelo, '--salida-trabajador', sal,
           '--n', str(a.n), '--calentamiento', str(min(500, a.n)), '--lote', str(a.lote)]
    t0 = time.time()
    p = docker(*cmd)
    estado = json.loads(docker('inspect', '--format', '{{json .State}}', nombre).stdout or '{}')
    docker('rm', '-f', nombre)
    res = {'modelo': modelo, 'nombre': NOMBRES[modelo], 'limite_mb': limite_mb, 'codigo_salida': p.returncode,
           'oom_killed': bool(estado.get('OOMKilled')), 'segundos': round(time.time() - t0, 1)}
    ruta = os.path.join(RAIZ, sal)
    if p.returncode == 0 and os.path.exists(ruta):
        d = json.load(open(ruta))
        res.update(estado='cabe', pico_proceso_mb=d['pico_proceso_mb'], huella_sesion_mb=d['huella_sesion_mb'],
                   rss_anon_mb=d.get('rss_anon_mb'), rss_archivos_mb=d.get('rss_archivos_mb'),
                   control=d.get('acuerdo_con_esperado', d['exactitud_muestras']),
                   lote1_mediana_us_NO_REPORTAR=d['lote1_mediana_us'])
        os.remove(ruta)
    else:
        res['estado'] = 'OOM' if (res['oom_killed'] or p.returncode == 137) else 'error'
        res['error'] = (p.stderr or p.stdout)[-600:]
    return res


def principal(a):
    if docker('version').returncode != 0:
        sys.exit('Docker no responde: abre Docker Desktop (o `colima start`) y vuelve a intentar.')
    asegurar_imagen(a)
    os.makedirs(os.path.join(RAIZ, 'results', 'memcap', '.tmp'), exist_ok=True)
    info = json.loads(docker('info', '--format', '{{json .}}').stdout or '{}')
    anfitrion = {'maquina': platform.machine(), 'sistema': platform.platform(),
                 'docker_server': info.get('ServerVersion'), 'docker_os': info.get('OperatingSystem'),
                 'docker_arquitectura': info.get('Architecture'), 'docker_ncpu': info.get('NCPU'),
                 'docker_mem_total_mb': round((info.get('MemTotal') or 0) / 1024 ** 2), 'cgroup': info.get('CgroupVersion')}
    arq = docker('run', '--rm', '--platform', a.plataforma, a.imagen, 'python', '-c',
                 'import platform, onnxruntime as o; print(platform.machine(), o.__version__)').stdout.split()
    anfitrion['contenedor_arquitectura'], anfitrion['contenedor_onnxruntime'] = (arq + [None, None])[:2]
    print(f"Contenedor {anfitrion['contenedor_arquitectura']} · ONNX Runtime {anfitrion['contenedor_onnxruntime']} · "
          f"{a.cpus} CPU · límites {a.limites} MB · lote 1 y {a.lote}")
    corridas = []
    for m in a.modelos:
        for lim in sorted(a.limites, reverse=True):
            r = una_corrida(a, m, lim)
            corridas.append(r)
            extra = f"pico {r['pico_proceso_mb']:.1f} MB (privada {r['rss_anon_mb'] or 0:.1f})" if r['estado'] == 'cabe' else ('memoria agotada' if r['estado'] == 'OOM' else r['error'][-120:].replace('\n', ' '))
            print(f"  {NOMBRES[m]:11s} límite {lim:5d} MB: {r['estado']:5s} · {extra}", flush=True)
            if r['estado'] != 'cabe' and not a.seguir:
                break   # límites menores tampoco alcanzarán
    resumen = {}
    for m in a.modelos:
        ok = [r for r in corridas if r['modelo'] == m and r['estado'] == 'cabe']
        no = [r for r in corridas if r['modelo'] == m and r['estado'] != 'cabe']
        resumen[NOMBRES[m]] = {'limite_minimo_que_cabe_mb': min((r['limite_mb'] for r in ok), default=None),
                               'mayor_limite_que_no_cabe_mb': max((r['limite_mb'] for r in no), default=None),
                               'pico_proceso_mb': max((r['pico_proceso_mb'] for r in ok), default=None),
                               'rss_anon_mb_sin_limite': next((r['rss_anon_mb'] for r in sorted(ok, key=lambda r: -r['limite_mb'])), None),
                               'rss_archivos_mb_sin_limite': next((r['rss_archivos_mb'] for r in sorted(ok, key=lambda r: -r['limite_mb'])), None),
                               'control': ok[0]['control'] if ok else None}
    marca = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    salida = {'etiqueta': a.etiqueta, 'fecha': marca, 'script': 'edge_benchmark/memcap_sweep.py',
              'protocolo': {'limite': 'docker --memory = --memory-swap (sin swap)', 'cpus': a.cpus, 'n_lote1': a.n,
                            'lote': a.lote, 'proceso': 'trabajador de arm_benchmark.py (un modelo por contenedor)',
                            'validez': 'cabe/no cabe y memoria pico; los tiempos son del anfitrión y no se reportan'},
              'anfitrion': anfitrion, 'resumen': resumen, 'corridas': corridas}
    os.makedirs(os.path.join(RAIZ, 'results', 'memcap'), exist_ok=True)
    ruta = os.path.join(RAIZ, 'results', 'memcap', f'memcap_{a.etiqueta}_{marca}.json')
    json.dump(salida, open(ruta, 'w'), indent=2, ensure_ascii=False)
    print('\nModelo        cabe desde   no cabe con   pico RSS (MB)   privada (MB)   bibliotecas/archivos (MB)')
    for n, v in resumen.items():
        f = lambda x: f'{x} MB' if x else '—'
        print(f"{n:12s} {f(v['limite_minimo_que_cabe_mb']):>11s} {f(v['mayor_limite_que_no_cabe_mb']):>13s} "
              f"{v['pico_proceso_mb'] or float('nan'):15.1f} {v['rss_anon_mb_sin_limite'] or float('nan'):14.1f} "
              f"{v['rss_archivos_mb_sin_limite'] or float('nan'):27.1f}")
    print('Guardado', ruta)


if __name__ == '__main__':
    nativa = 'linux/arm64' if platform.machine().lower() in ('arm64', 'aarch64') else 'linux/amd64'
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--etiqueta', default=platform.node().split('.')[0] or 'equipo')
    p.add_argument('--limites', nargs='+', type=int, default=[1024, 512, 256, 192, 160, 128, 96, 64])
    p.add_argument('--modelos', nargs='+', default=MODELOS, choices=MODELOS)
    p.add_argument('--n', type=int, default=2000, help='paquetes con lote 1 (basta poco: aquí importa la memoria)')
    p.add_argument('--lote', type=int, default=1024)
    p.add_argument('--cpus', type=float, default=1.0)
    p.add_argument('--plataforma', default=nativa)
    p.add_argument('--imagen', default='mi-lite-edge:py312')
    p.add_argument('--ort', default='1.32.0')
    p.add_argument('--seguir', action='store_true', help='probar todos los límites aunque uno falle')
    principal(p.parse_args())
