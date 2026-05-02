# NYC-EMS-Traffic V4 (NET-V4)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A spatiotemporal benchmark dataset aligning New York City DOT Automated Traffic Recorder (ATR) observations with FDNY EMS dispatch records for intervention-aware traffic forecasting.

---

## Overview

| Property | Value |
|:--|:--|
| Samples | 33,076 (20,000 EV-active + 13,076 control) |
| Graph Nodes | 363 NYC traffic sensor segments |
| Time Span | January 2018 -- December 2023 |
| Time Resolution | 15-minute intervals, 24-step windows (12 input + 12 output) |
| Spatial Coverage | 5 NYC boroughs (Bronx, Brooklyn, Manhattan, Queens, Staten Island) |
| Train/Val/Test | 22,028 / 5,523 / 5,525 (67/17/17) stratified by EV/Control label |

---

## Why This Dataset?

Existing spatiotemporal traffic forecasting benchmarks (METR-LA, PeMS, LargeST) evaluate models under steady-state conditions. NET-V4 adds operational event labels---emergency vehicle dispatch records aligned with traffic sensor readings---enabling the study of forecasting behavior under localized operational disruptions. The dataset is designed to support:

- Spatiotemporal forecasting under observed distribution shift
- Robustness evaluation with operational intervention signals
- Causal confounding analysis in labeled benchmarks
- Geographic fairness auditing across boroughs

---

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from net_dataset import load_net

train, val, test, adj = load_net(data_dir="path/to/data/", split="all")
# train['x'].shape -> (N_train, K, T_in, C)
# adj.shape        -> (363, 363)
```

Or use the PyTorch DataLoader directly:

```python
from net_dataset.loader import NETDataset
from torch.utils.data import DataLoader

dataset = NETDataset(data_dir="path/to/data/", split="train")
loader  = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## Dataset Files

| File | Size | Description |
|:--|:--|:--|
| `NYC_EMS_Traffic_V4.csv` | ~14 MB | Full tabular data (33,076 rows $\times$ 22 columns) |
| `NYC_EMS_Traffic_V4.npz` | ~17 MB | NumPy tensor archive for direct loading |
| `NYC_EMS_Traffic_V4_graph.npz` | ~593 KB | Adjacency matrix and node metadata |
| `DATASET_CARD.md` | ~6 KB | Detailed dataset documentation |

---

## Data Sources

| Source | Provider | Description |
|:--|:--|:--|
| Automated Traffic Volume Counts (ATR) | NYC Department of Transportation | Traffic volume at 15-minute intervals from permanent sensor stations |
| EMS Incident Dispatch Data | NYC Fire Department (FDNY) | Emergency medical service dispatch records with timestamps and locations |

Both sources are publicly available under [NYC OpenData Terms of Use](https://www.nyc.gov/home/terms-of-use.page).

---

## Reproducing the Dataset

```bash
git clone https://anonymous.4open.science/r/net-v4-dataset-903C
cd net-v4-dataset
pip install -r requirements.txt
python data_preprocessing_v4/fetch_and_build_nyc_dataset_v4.py
```

The build script fetches raw data from the NYC OpenData API, performs spatial alignment (R-tree nearest-neighbor matching within 0.02-degree radius), and writes the compiled `.npz` and `.csv` files. The build process takes approximately 30 minutes and requires an internet connection.

---

## License

- **Dataset**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE)
- **Code**: [MIT License](LICENSE-CODE)
- **Underlying data**: Subject to [NYC OpenData Terms of Use](https://www.nyc.gov/home/terms-of-use.page)

---

## Citation

```bibtex
@dataset{net_v4_2026,
  title     = {{NET-V4}: NYC-EMS-Traffic Benchmark for Intervention-Aware
               Spatiotemporal Forecasting},
  author    = {Anonymous},
  year      = {2026},
  note      = {Dataset available at anonymous.4open.science/r/net-v4-dataset-903C},
  version   = {4.0},
  license   = {CC BY 4.0}
}
```
