# NET-V4 Dataset Card

**Version:** 4.0  
**Release Date:** 2026-05  
**License:** CC BY 4.0 (data) / MIT (code)  
**URL:** https://anonymous.4open.science/r/net-v4-dataset-903C

---

## Dataset Overview

NYC-EMS-Traffic V4 (NET-V4) is a spatiotemporal benchmark dataset constructed by cross-aligning two independent public data sources: NYC Department of Transportation Automated Traffic Recorder (ATR) observations and NYC Fire Department (FDNY) EMS dispatch records.

| Property | Value |
|:--|:--|
| Total samples | 33,076 |
| EV-active samples | 20,000 (60.47%) |
| Control samples | 13,076 (39.53%) |
| Graph nodes | 363 NYC traffic sensor segments |
| Graph edges | 14,388 (mean degree 39.6) |
| Time span | January 2018 -- December 2023 |
| Time resolution | 15-minute intervals |
| Window size | 24 timesteps (12 input + 12 output = 6 hours) |
| Subgraph size | 8 nodes (1 center + 7 nearest neighbors) |
| Geographic coverage | 5 NYC boroughs |
| Train / Val / Test | 22,028 / 5,523 / 5,525 (67/17/17) |

---

## Motivation

Existing spatiotemporal traffic forecasting benchmarks (METR-LA, PeMS, LargeST) evaluate models under steady-state, passive conditions. NET-V4 introduces real operational event labels---emergency vehicle dispatch records aligned with traffic sensor readings---enabling the study of forecasting behavior under localized operational disruptions.

The dataset is specifically designed to support:
- Spatiotemporal forecasting under observed distribution shift
- Robustness evaluation with operational intervention signals
- Causal confounding analysis in labeled benchmarks
- Geographic fairness auditing across boroughs

---

## Data Sources

Both data sources are publicly available under NYC OpenData Terms of Use.

| Source | Provider | Description |
|:--|:--|:--|
| Automated Traffic Volume Counts (ATR) | NYC Department of Transportation | Traffic volume at 15-minute intervals from permanent sensor stations across NYC |
| EMS Incident Dispatch Data | NYC Fire Department (FDNY) | Emergency medical service dispatch records with timestamps, locations, severity levels |

Raw data URLs:
- ATR: `https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt`
- EMS: `https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj`

---

## Construction Methodology

### Spatial Alignment

EMS dispatch locations are mapped to the nearest traffic sensor within a 0.02-degree (~2.2 km) radius using R-tree spatial indexing. 96.37% of EMS events map to an available sensor.

### Temporal Windowing

Each sample is a 24-timestep window centered on an EMS dispatch time. The window is split into 12 input steps (3 hours before dispatch) and 12 output steps (3 hours after dispatch). Timestamps are quantized to 15-minute bins.

### Control Sample Construction

Control samples are drawn from time windows on the same sensor segments with no active EMS dispatch within $\pm$180 minutes (3 hours) and outside a spatial exclusion radius. This prevents geographic spillover from active interventions.

### Train/Validation/Test Split

The dataset is split 67/17/17, stratified by EV/Control label to maintain consistent class balance across partitions. Incident identifiers are isolated to individual partitions to prevent chronological leakage.

---

## File Descriptions

### `NYC_EMS_Traffic_V4.npz`

Primary NumPy archive. Keys:
- `volume_sequences` (33076, 24): Traffic volume at center node
- `spatial_volumes` (33076, K, 24): Traffic volume at K neighbor nodes
- `ev_masks` (33076, 24): Binary EV activity indicators per timestep
- `is_ev_event` (33076,): Binary sample-level EV label
- `segment_ids` (33076,): NYC DOT segment identifiers
- `baseline_vols` (33076,): Per-segment baseline volume
- `incident_years`, `incident_hours`, `incident_dows`: Temporal identifiers
- `severity_scores`: EMS incident severity
- `spatial_distances`: Distance to center node for each neighbor

### `NYC_EMS_Traffic_V4.csv`

Tabular version (33,076 rows $\times$ 22 columns) for non-Python analysis.

### `NYC_EMS_Traffic_V4_graph.npz`

Graph structure. Keys:
- `adjacency` (363, 363): Binary adjacency matrix
- `node_lats`, `node_lons`: GPS coordinates (EPSG:4326)
- `node_streets`, `node_boroughs`: Segment metadata
- `seg_ids`: NYC DOT segment identifiers
- `profiles`: Smoothed day-of-week temporal profiles (363, 7, 96)

---

## Usage Notes

### Appropriate Uses
- Spatiotemporal traffic forecasting benchmark
- OOD generalization evaluation under operational disruptions
- Causal confounding detection in labeled benchmarks
- Geographic fairness analysis across boroughs

### Inappropriate Uses
- Micro-kinematic simulation (15-minute aggregation is too coarse)
- Causal effect estimation without covariate adjustment (EV labels are operational correlates, not randomized treatments)
- Fair resource allocation without neighborhood-level adjustment

---

## Citation

```bibtex
@dataset{net_v4_2026,
  title     = {{NET-V4}: NYC-EMS-Traffic Benchmark for Intervention-Aware
               Spatiotemporal Forecasting},
  author    = {Anonymous},
  year      = {2026},
  note      = {Available at anonymous.4open.science/r/net-v4-dataset-903C},
  version   = {4.0},
  license   = {CC BY 4.0}
}
```

---

## Maintenance

The authorship commits to lifecycle stewardship including continuous baseline integration, redundant cloud dataset survivability, and GitHub issue tracking for a minimum of 5 years. Static benchmark leaderboard versioning will be maintained; alignment bug fixes will be released as minor versions (e.g., NET-V4.1).
