# Benchmark de borde en ARM (S15)

Archivos nuevos en `edge_benchmark/`:

| Archivo | Qué es |
|---|---|
| `arm_benchmark.py` | Benchmark portátil (protocolo E9 + `batch_sweep.py`): lote 1 y lote 1,024, 1 hilo, memoria pico, tamaño ONNX. Un proceso nuevo por modelo y ronda. Mismo archivo en macOS, Raspberry Pi y Jetson. |
| `run_arm_benchmark.sh` | Crea `.venv-edge` (numpy, psutil, onnxruntime 1.32.0) la primera vez y corre el benchmark. |
| `export_lstm_cpu.py` | Reexporta la LSTM de E8 (REP-01) sin `CudnnRNNV3`, con los mismos pesos, usando el operador LSTM estándar de ONNX. |
| `lstm_gpu.onnx` | LSTM original de E8 (REP-01), exportada en GPU con `CudnnRNNV3`; no corre en CPU, es la entrada de `export_lstm_cpu.py`: `python3 edge_benchmark/export_lstm_cpu.py --gpu edge_benchmark/lstm_gpu.onnx --salida edge_benchmark/lstm_cpu.onnx`. |
| `lstm_cpu.onnx` (+ `_export.json`) | Resultado de lo anterior: 4,847 parámetros, 19.9 KB. Coincide con una implementación en NumPy del formato cuDNN (dif. máx. 1.7e-5, acuerdo argmax 1.0). |
| `transformer.onnx` | Transformer de E8 (REP-01), copiado de Drive (`mejoras/S14_recurrentes/transformer/REP-01/`). |
| `xgb_u3.onnx`, `rf_u3.onnx`, `tcn_u.onnx`, `muestras.npz` | Los mismos del repo (protocolo C, REP-01). |

Las 25 entradas de LSTM y Transformer son las de MI-LiteXGB (`x_xgb_u3`); la TCN usa sus 18 entradas (`x_tcn_u`).
Control de corrección: árboles y TCN, acuerdo con las predicciones esperadas (debe ser 1.0); LSTM y Transformer,
exactitud sobre las etiquetas de los 10,000 paquetes (88.5 % y 89.8 % en x86).

## MacBook Pro M5 (dato complementario, no es un equipo de borde)

```bash
# 1. Python 3.10 o superior (en macOS viene con las Command Line Tools)
python3 --version            # si no existe: xcode-select --install   (o brew install python@3.12)

# 2. Repositorio
cd ~/Documents
git clone https://github.com/chesterjr11082001-cyber/mi-lite-iiot-ids.git   # si aún no lo tienes
cd mi-lite-iiot-ids

# 3. Antes de medir: cargador conectado, "Modo de bajo consumo" desactivado,
#    cierra navegador y apps pesadas, y no uses la Mac mientras corre (~5-10 min)

# 4. Medir (3 rondas x 5 modelos)
bash edge_benchmark/run_arm_benchmark.sh macbook-m5
```

Resultado: `results/arm/macbook-m5_<fecha>.json`, con la ficha del equipo (chip, núcleos P/E, macOS, versiones,
alimentación), cada corrida y el resumen (mediana de las 3 rondas con su rango). macOS no permite fijar el
proceso a un núcleo; con 1 hilo y prioridad normal el sistema lo pone en los núcleos de rendimiento.

## Raspberry Pi 4/5 o Jetson (equipo de borde, para la tabla principal)

```bash
# Raspberry Pi OS de 64 bits (aarch64). Disipador o ventilador puesto.
sudo apt update && sudo apt install -y git python3-venv
git clone https://github.com/chesterjr11082001-cyber/mi-lite-iiot-ids.git && cd mi-lite-iiot-ids
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor   # opcional, se anota en el JSON
bash edge_benchmark/run_arm_benchmark.sh rpi5 --nucleo 2
vcgencmd get_throttled        # debe dar throttled=0x0; si no, hubo bajada de frecuencia: repetir con mejor enfriamiento
```

### Energía con el medidor USB (entre la fuente y la placa)

```bash
.venv-edge/bin/python edge_benchmark/arm_benchmark.py --etiqueta rpi5 --energia rf_u3 --segundos 60 --reposo 30
```

El script marca la hora de la ventana de reposo (30 s) y de la de carga (60 s, lote 1, 1 hilo). Anota la potencia
media del medidor en cada ventana y escríbela en `potencia_reposo_w` y `potencia_carga_w` del JSON:
energía por paquete (µJ) = (P_carga − P_reposo) × µs por inferencia. Repetir con `xgb_u3`, `tcn_u`, `transformer` y `lstm`.

## Notas

- La latencia de la M5 y de QEMU no se reportan como latencia de borde. QEMU solo sirve para comprobar que corre.
- Si `onnxruntime==1.32.0` no está para la plataforma, el script instala la última y su versión queda en el JSON.
- Opciones: `--rondas`, `--modelos xgb_u3 rf_u3 ...`, `--n`, `--calentamiento`, `--lote`, `--pausa`, `--salida`.
