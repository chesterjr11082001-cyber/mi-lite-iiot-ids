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

## Repository layout

```
notebooks/        Google Colab notebooks, in execution order (Spanish comments)
results/          per-repetition records ("fichas") and summaries of every analysis
edge_benchmark/   ONNX models, 10,000 test packets and a script to measure latency on an edge device
paper_figures/    scripts that regenerate Figures 2-5 of the article from results/
```

| Notebook | Content |
|---|---|
| `01_v1_development_REP01` | Pre-registered pipeline (protocol O, rebalancing T = 2,000): preprocessing, TCN baseline, MI ranking, CV, training, measurement and the record of repetition 1 |
| `02_v1_runs_REP02-10` | Same pipeline for repetitions 2–10 |
| `03_v1_consolidation_and_M1-M4` | Consolidation and tests of v1; M1 ONNX deployment benchmark, M2 input ablation, M3 compact forests, M4 placeholder sensitivity |
| `04_complementary_M4-M7` | M5/M7 per-packet latency and earliness bound, M6 compact variants, M4 without the placeholder |
| `05_v3_protocols_O_and_C` | S2: rebalancing chosen by CV (protocol O); S3–S5: protocol C with the TCN retrained on identical splits |
| `06_robustness_S11e` | E1 corrected resampled t-test, E2 information audit, E3 label conflicts, E4 input ablations, E5 fairer TCN, E6 edge package, E7 summary |

## Data

Edge-IIoTset is **not redistributed** here (Ferrag et al., *Edge-IIoTset*, IEEE Access 10, 40281–40306, 2022). The notebooks download `DNN-EdgeIIoT-dataset.csv` from Kaggle with `kagglehub` when it is not found (Kaggle may ask for your credentials) and read and write inside the Google Drive folder `MyDrive/Articulo_Grupal_EdgeIIoT`, set in the configuration cell.

## Reproducing the experiments

1. Open the notebooks in Google Colab in the order above. Notebooks 01, 02 and 05, and part E5 of notebook 06, need a GPU for the TCN; everything else runs on a high-RAM CPU session.
2. Repetitions use seeds `1000 + r`, `r = 1..10`, with a stratified 70/30 split (1,336,769 training and 572,902 test packets).
3. Each notebook writes one JSON record per repetition and a summary; `REANUDAR = True` reuses what is already saved, so long runs can be split across sessions (the S11e notebook shows how).

To regenerate the figures from the records in this repository (no dataset needed):

```bash
pip install numpy pandas scipy matplotlib
python paper_figures/scripts/01_per_class_summary.py
python paper_figures/scripts/02_figures_2_to_4.py
python paper_figures/scripts/03_two_by_two_tests.py
python paper_figures/scripts/04_figure_5_robustness.py
```

## Edge benchmark

`edge_benchmark/` contains the protocol-C models of repetition 1 (`xgb_u3.onnx` = MI-LiteXGB, `rf_u3.onnx` = MI-LiteRF, `tcn_u.onnx` = TCN), 10,000 scaled test packets with their expected predictions (`muestras.npz`) and `bench_edge.py`. On a Raspberry Pi or any gateway:

```bash
pip install numpy onnxruntime psutil
python edge_benchmark/bench_edge.py            # 1 thread, 10,000 packets, 500 warm-up calls
```

It writes `resultado_borde.json` with the median, p95 and p99 latency per packet, throughput, memory and the agreement with the expected predictions. `reference_result_colab_epyc.json` is the reference run on an AMD EPYC 7B12 (medians of 13.4, 27.2 and 48.0 µs for MI-LiteRF, MI-LiteXGB and the TCN; 100% agreement).

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

## Citation and license

Please cite the article (see `CITATION.cff`). Code and result records are released under the MIT License. Parts of the analysis code were developed with the assistance of Claude (Anthropic); the authors reviewed all code and results.
