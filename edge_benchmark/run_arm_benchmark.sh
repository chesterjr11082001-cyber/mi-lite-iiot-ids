#!/bin/bash
# Crea (una sola vez) un entorno virtual con numpy + onnxruntime y corre el benchmark de borde.
# Sirve igual en macOS (Apple Silicon) y en Linux ARM64 (Raspberry Pi OS 64 bits, Jetson).
# Uso (desde la raíz del repositorio):  bash edge_benchmark/run_arm_benchmark.sh <etiqueta> [más opciones de arm_benchmark.py]
#   ej.: bash edge_benchmark/run_arm_benchmark.sh macbook-m5
#        bash edge_benchmark/run_arm_benchmark.sh rpi5 --nucleo 2
set -e
cd "$(dirname "$0")/.."
ETIQUETA=${1:-$(hostname -s)}; shift || true
ORT_VERSION=${ORT_VERSION:-1.32.0}          # la de las sesiones del paper (requirements.txt)
if [ ! -d .venv-edge ]; then
  python3 -m venv .venv-edge
  .venv-edge/bin/pip install -q --upgrade pip
  .venv-edge/bin/pip install -q numpy psutil "onnxruntime==${ORT_VERSION}" \
    || { echo "onnxruntime ${ORT_VERSION} no está para esta plataforma; se instala la última"; .venv-edge/bin/pip install -q numpy psutil onnxruntime; }
fi
.venv-edge/bin/python -c "import platform, numpy, onnxruntime as o; print('Python', platform.python_version(), '·', platform.machine(), '· numpy', numpy.__version__, '· onnxruntime', o.__version__)"
if [ "$(uname)" = "Darwin" ]; then
  # Evita que el Mac entre en reposo durante la corrida (macOS no permite fijar núcleos)
  exec caffeinate -i .venv-edge/bin/python edge_benchmark/arm_benchmark.py --etiqueta "$ETIQUETA" "$@"
else
  exec .venv-edge/bin/python edge_benchmark/arm_benchmark.py --etiqueta "$ETIQUETA" "$@"
fi
