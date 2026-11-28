# MI-Lite: leakage-free, entropy-guided early intrusion detection for the IIoT edge

Code, per-repetition result records and an edge benchmark for the article

> **Lightweight entropy-guided early intrusion detection for IIoT edge network monitoring: a leakage-free reassessment on Edge-IIoTset**
> Leonardo Leoncio Villanueva Fonseca, Ricardo Franco Baldoceda Rios and Yury Tello Canchapoma
> Facultad de Ingeniería Eléctrica y Electrónica, Universidad Nacional de Ingeniería (Lima, Peru)
> Manuscript in preparation for *Future Generation Computer Systems*.

## What the study shows

Edge-IIoTset, a reference dataset for IIoT intrusion detection, writes an absent protocol field as `"0"` in normal captures and as `"0.0"` in attack captures. After one-hot encoding, three such columns carry the full entropy of the normal/attack label (0.863 bits). We compare an entropy-guided detector with a faithful reproduction of a recent temporal convolutional network (TCN; Gao et al., 2026) under the original recipe (protocol **O**) and under a corrected recipe that unifies the placeholder (protocol **C**), over ten repeated stratified splits of 1,909,671 packets.

| Protocol C (mean ± SD, 10 repetitions) | Macro-F1 (%) | FPR (%) | Latency, batch 1 (µs) |
|---|---|---|---|
| TCN (Gao et al., reproduced) | 35.62 ± 2.58 | 14.25 ± 16.70 (0.009–45.8) | 64.2 |
| **MI-LiteXGB** (MI top-25 + XGBoost) | **89.34 ± 0.26** | 0.096 ± 0.004 | 38.2 |
| MI-LiteRF (MI top-25 + random forest) | 88.81 ± 0.68 | 0.055 ± 0.004 | 16.4 |

* With the shortcut (protocol O) the TCN reaches 68.5% and MI-LiteRF 93.2%: removing it costs the TCN 32.9 points and the trees 4.4.
* Given the TCN's own 18 inputs, the trees still reach 77.6–79.5%; the TCN on the 25 MI inputs reaches 64.6%; class weighting does not rescue it.
* Without four capture identifiers the trees keep 81.6–82.3%; without TCP session counters as well they fall to 46–47%.
* Latency: ONNX Runtime, one CPU thread, Intel Xeon 2.20 GHz. In 1,024-packet batches the TCN is faster per packet.
* A compact Transformer (5,551 parameters) and an LSTM (4,847) trained on the same 25 MI inputs reach 65.6% and 48.4%, 24 and 41 points below MI-LiteXGB.
* External validation on CICIoT2023 (5 of its 169 CSV files, 1,032,490 records, 8 classes) with the same procedure and configurations: the trees again lead the TCN by 20–29 points (MI-LiteRF 81.4%), but the FPR rises to 2.5–12%, and choosing rebalancing by macro-F1 alone doubles the FPR of MI-LiteXGB.

## Repository layout

```
notebooks/        Google Colab notebooks, in execution order (Spanish comments)
results/          per-repetition records ("fichas") and summaries of every analysis
edge_benchmark/   ONNX models, 10,000 test packets and scripts to measure latency, memory and energy on edge devices
paper_figures/    scripts that regenerate Figures 2-7 of the article from results/
```

| Notebook | Content |
|---|---|
| `01_v1_development_REP01` | Pre-registered pipeline (protocol O, rebalancing T = 2,000): preprocessing, TCN baseline, MI ranking, CV, training, measurement and the record of repetition 1 |
| `02_v1_runs_REP02-10` | Same pipeline for repetitions 2–10 |
| `03_v1_consolidation_and_M1-M4` | Consolidation and tests of v1; M1 ONNX deployment benchmark, M2 input ablation, M3 compact forests, M4 placeholder sensitivity |
| `04_complementary_M4-M7` | M5/M7 per-packet latency and earliness bound, M6 compact variants, M4 without the placeholder |
| `05_v3_protocols_O_and_C` | S2: rebalancing chosen by CV (protocol O); S3–S5: protocol C with the TCN retrained on identical splits |
| `06_robustness_S11e` | E1 corrected resampled t-test, E2 information audit, E3 label conflicts, E4 input ablations, E5 fairer TCN, E6 edge package, E7 summary |
| `07_external_validation_CICIoT2023_S14` | External validation on CICIoT2023: entropy audit, MI ranking, rebalancing chosen by CV, MI-LiteRF and MI-LiteXGB with the Edge-IIoTset configurations, TCN on the same MI inputs, input ablation, ONNX latency and paired tests (`results/s14_external_ciciot2023/`) |
| `07b_rebalancing_false_positives_S14` | Follow-up on CICIoT2023, with rules fixed before it was run: MI-LiteXGB with each fixed rebalancing cap, the cap chosen by CV under a false-positive constraint, and a prior correction of the class probabilities |
| `08_lstm_transformer_S14` | Compact LSTM and Transformer baselines on the 25 MI inputs of protocol C (same splits), same-session ONNX latency and corrected paired tests (`results/s14_recurrent_baselines/`) |

`notebooks/executed_html/` keeps HTML exports of notebooks 07, 07b and 08 as they were run in Google Colab (26–27 November 2026), with all their outputs.

## Data

Edge-IIoTset is **not redistributed** here (Ferrag et al., *Edge-IIoTset*, IEEE Access 10, 40281–40306, 2022). The notebooks download `DNN-EdgeIIoT-dataset.csv` from Kaggle with `kagglehub` when it is not found (Kaggle may ask for your credentials) and read and write inside the Google Drive folder `MyDrive/Articulo_Grupal_EdgeIIoT`, set in the configuration cell.

CICIoT2023 (Neto et al., *Sensors* 23, 5941, 2023) is not redistributed either. Notebook 07 downloads the original CSV files from the Kaggle mirror `madhavmalhotra/unb-cic-iot-dataset` with `kagglehub` (about 2.8 GB) and uses 5 of the 169 files, chosen at regular intervals of the sorted file list.

## Reproducing the experiments

1. Open the notebooks in Google Colab in the order above. Notebooks 01, 02, 05, 07 (part A3) and 08 (part E8), and part E5 of notebook 06, need a GPU for the neural networks; everything else runs on a high-RAM CPU session. Notebook 07b reuses what notebook 07 saved.
2. Repetitions use seeds `1000 + r`, `r = 1..10`, with a stratified 70/30 split (1,336,769 training and 572,902 test packets).
3. Each notebook writes one JSON record per repetition and a summary; `REANUDAR = True` reuses what is already saved, so long runs can be split across sessions (the S11e notebook shows how).

To regenerate the figures from the records in this repository (no dataset needed):

```bash
pip install numpy pandas scipy matplotlib
python paper_figures/scripts/01_per_class_summary.py
python paper_figures/scripts/02_figures_2_to_4.py
python paper_figures/scripts/03_two_by_two_tests.py
python paper_figures/scripts/04_figure_5_robustness.py
python paper_figures/scripts/05_figure_4_confusion.py
python paper_figures/scripts/06_figure_6_scaling.py
```

In the current manuscript, `02_figures_2_to_4.py` draws Figures 2, 3 and 5, `05_figure_4_confusion.py` Figure 4, `06_figure_6_scaling.py` Figure 6 and `04_figure_5_robustness.py` Figure 7.

## Edge benchmark

`edge_benchmark/` contains the protocol-C models of repetition 1 (`xgb_u3.onnx` = MI-LiteXGB, `rf_u3.onnx` = MI-LiteRF, `tcn_u.onnx` = TCN), 10,000 scaled test packets with their expected predictions (`muestras.npz`) and `bench_edge.py`. On a Raspberry Pi or any gateway:

```bash
pip install numpy onnxruntime psutil
python edge_benchmark/bench_edge.py            # 1 thread, 10,000 packets, 500 warm-up calls
```

It writes `resultado_borde.json` with the median, p95 and p99 latency per packet, throughput, memory and the agreement with the expected predictions. `reference_result_colab_epyc.json` is the reference run on an AMD EPYC 7B12 (medians of 13.4, 27.2 and 48.0 µs for MI-LiteRF, MI-LiteXGB and the TCN; 100% agreement). `reference_result_xeon230.json` is a second run on an Intel Xeon at 2.30 GHz with ONNX Runtime 1.29 (medians of 6.8, 19.3 and 26.2 µs).

To measure peak memory and time per packet as a function of the batch size (Figure 6b–c), each combination in a new process:

```bash
bash edge_benchmark/run_batch_sweep.sh     # 3 models x 9 batch sizes (1 to 65,536) x 3 runs -> results/s14_batch_sweep/
```

### ARM benchmark (Raspberry Pi, Jetson, Apple Silicon)

`arm_benchmark.py` extends the benchmark to the five models of the article (MI-LiteXGB, MI-LiteRF, TCN, and the compact Transformer and LSTM of notebook 08). Each model runs in a new process, with one thread, in three rounds: latency with batch size 1 (median, p95 and p99 over 10,000 packets after 500 warm-up calls), time per packet with batches of 1,024, peak memory of the whole process and ONNX size, plus a description of the device. The same file runs on macOS and Linux (x86 or ARM64):

```bash
bash edge_benchmark/run_arm_benchmark.sh rpi5 --nucleo 2     # creates .venv-edge once -> results/arm/rpi5_<date>.json
python3 edge_benchmark/arm_benchmark.py --etiqueta rpi5 --energia rf_u3 --segundos 60 --reposo 30   # with a USB power meter
```

The LSTM of notebook 08 was exported from a GPU session and its graph (`lstm_gpu.onnx`) keeps the cuDNN operator `CudnnRNNV3`, which ONNX Runtime does not run on CPU. `export_lstm_cpu.py` rebuilds it as `lstm_cpu.onnx` with the standard ONNX LSTM operator and the same trained weights, and checks it against an independent NumPy implementation of the cuDNN weight layout (maximum difference 1.7e-5, identical predicted classes). Step-by-step instructions, in Spanish, are in `edge_benchmark/README_ARM.md`.

Two further measurements do not depend on the processor speed. `memcap_sweep.py` runs each model in a Docker container with a hard memory limit and no swap, from 1,024 down to 64 MB, and reports the smallest limit under which it completes and its private (anonymous) memory. `model_complexity.py` counts the comparisons per packet that the tree ensembles actually make on the 10,000 packets and the multiply-accumulate operations of the networks (`results/complexity/model_complexity.json`).

Reference runs on an Apple M5 laptop (ARM64; a complementary reference, not an edge device): `results/arm/macbook-m5_20261128_1123.json` (batch-1 medians of 2.7, 6.7, 18.4, 34.0 and 49.9 µs for MI-LiteRF, MI-LiteXGB, the TCN, the LSTM and the Transformer; 100% agreement for the trees and the TCN) and `results/memcap/memcap_macbook-m5_20261128_1119.json` (Linux aarch64 container: MI-LiteXGB, the TCN and the LSTM complete under 64 MB, MI-LiteRF and the Transformer under 96 MB).

## Reading the records

Records and notebooks were written for a Spanish-language course, so keys are in Spanish:

| Key | Meaning |
|---|---|
| `ficha_REP-xx.json` | record of repetition xx |
| `repeticion`, `semilla` | repetition, seed |
| `f1_macro`, `f1_por_clase`, `exactitud`, `f1_ponderado` | macro-F1, per-class F1, accuracy, weighted F1 (%) |
| `tfp`, `falsas_alarmas` | false-positive rate (%), number of false alarms |
| `ataque_normal_pct` | attack packets classified as Normal (%) |
| `lote1_ms`, `respuesta_ms`, `latencia_ms` | latency per packet with batch 1, response time (scaling + inference), per-packet time in 1,024-packet batches |
| `tamano_kb`, `tamano_onnx_kb`, `memoria_total_onnx_mb` | model size (native, ONNX), process memory with ONNX Runtime |
| `tcn_u`, `rf_u3`, `xgb_u3` | TCN, MI-LiteRF and MI-LiteXGB under protocol C (`_u` = unified placeholder) |
| `rf_u1`, `xgb_u1` | pre-specified configurations with rebalancing (T = 2,000) |
| `resumen`, `por_rep`, `pruebas` | in the S14 records: mean and SD per model, per-repetition values, paired tests (`media_d`, `ic95_corr`, `p_holm`) |
| `ataques_como_normal_pct`, `E_N_registros` | attack records classified as benign (%), expected attack records before the first alarm (CICIoT2023) |
| `auditoria`, `im_binaria`, `im_clase`, `pureza` | entropy audit: MI with the normal/attack label and with the class label (bits), one-column purity |

## Citation and license

Please cite the article (see `CITATION.cff`). Code and result records are released under the MIT License. Parts of the analysis code were developed with the assistance of Claude (Anthropic); the authors reviewed all code and results.
