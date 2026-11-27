#!/bin/bash
# Barrido de memoria y tiempo por paquete según el tamaño del lote (Fig. 6b-c del paper).
# Cada combinación corre en un proceso nuevo, con un hilo y fijada a un núcleo; tres corridas por combinación.
# Uso (desde la raíz del repositorio): bash edge_benchmark/run_batch_sweep.sh [carpeta_de_salida]
cd "$(dirname "$0")/.."
SALIDA=${1:-results/s14_batch_sweep}
mkdir -p "$SALIDA"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for ronda in 1 2 3; do
 for m in rf_u3 xgb_u3 tcn_u; do
  for b in 1 4 16 64 256 1024 4096 16384 65536; do
    taskset -c 1 python3 edge_benchmark/batch_sweep.py $m $b "$SALIDA/${m}_${b}_r${ronda}.json" > /dev/null
  done
 done
done
