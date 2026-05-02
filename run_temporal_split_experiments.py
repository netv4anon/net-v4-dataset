"""
run_temporal_split_experiments.py
=================================
Chronological train/val/test split experiment for NET-V4.
  - Train: 2018-2021
  - Val:   2022
  - Test:  2023

Runs the same baselines as run_full_neurips_experiments.py but with
temporal split to assess concept drift and COVID-era generalization.

Also produces per-year MAE breakdown for trend analysis.

Usage:
    python experiments/run_temporal_split_experiments.py
"""

import os
import sys
import json
import numpy as np
import torch
from collections import defaultdict

# ── Path setup ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NPZ_PATH = os.path.join(ROOT, "data_preprocessing_v4", "NYC_EMS_Traffic_V4.npz")
OUT_DIR = os.path.join(ROOT, "NIPS_DnB_Submission_Package", "results")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456]


def load_data():
    data = np.load(NPZ_PATH, allow_pickle=True)
    vols = data["volume_sequences"]       # (33076, 24)
    spatial = data["spatial_volumes"]      # (33076, 8, 24)
    ev_masks = data["ev_masks"]            # (33076, 24)
    is_ev = data["is_ev_event"]            # (33076,)
    years = data["incident_years"]         # (33076,)
    severity = data["severity_numeric"]    # (33076,)
    return vols, spatial, ev_masks, is_ev, years, severity


def temporal_split(years):
    """Chronological split: train 2018-2021, val 2022, test 2023."""
    train_mask = years <= 2021
    val_mask = years == 2022
    test_mask = years == 2023
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    return train_idx, val_idx, test_idx


def compute_metrics(pred, target, ev_mask_out):
    """Compute Task-1 MAE, Task-2 MAE, Task-2 RMSE."""
    results = {}

    # Task-1: all output steps
    diff = np.abs(pred - target)
    results["T1_MAE"] = float(np.mean(diff))
    results["T1_RMSE"] = float(np.sqrt(np.mean(diff ** 2)))

    # Task-2: shock window (steps where ev_mask is active in output)
    shock_mask = ev_mask_out > 0.5
    if shock_mask.sum() > 0:
        s_diff = np.abs(pred[shock_mask] - target[shock_mask])
        results["T2_MAE"] = float(np.mean(s_diff))
        results["T2_RMSE"] = float(np.sqrt(np.mean(s_diff ** 2)))
    else:
        results["T2_MAE"] = float("nan")
        results["T2_RMSE"] = float("nan")

    return results


# ── Baseline implementations (same as run_full_neurips_experiments.py) ──

def baseline_histmean(train_vols, test_vols_in):
    """Historical average: predict training set mean for all steps."""
    mean_val = np.mean(train_vols[:, :12])
    N = test_vols_in.shape[0]
    return np.full((N, 12), mean_val)


def baseline_lastvalue(test_vols_in):
    """Copy the last input value forward."""
    last = test_vols_in[:, -1:]
    return np.tile(last, (1, 12))


def baseline_eventawarelast(test_vols_in, test_ev_in, train_vols, train_ev_masks, train_is_ev):
    """LastValue adjusted only for ongoing events visible at forecast origin."""
    # Compute shock profile from training EV samples
    ev_train_idx = np.where(train_is_ev == 1)[0]
    if len(ev_train_idx) == 0:
        return baseline_lastvalue(test_vols_in)

    ev_vols = train_vols[ev_train_idx]  # (N_ev, 24)
    ev_out = ev_vols[:, 12:]            # (N_ev, 12)
    ev_in_last = ev_vols[:, 11:12]      # (N_ev, 1)
    denom = np.clip(ev_in_last, 1.0, None)
    rel_change = (ev_out - ev_in_last) / denom  # relative change per step
    shock_profile = np.mean(rel_change, axis=0)  # (12,)

    last = test_vols_in[:, -1:]  # (N, 1)
    pred = np.tile(last, (1, 12))
    ongoing_event = test_ev_in[:, -1] > 0.5
    if ongoing_event.any():
        pred[ongoing_event] = last[ongoing_event] * (1.0 + shock_profile[np.newaxis, :])
    return pred


def baseline_lineardrift(test_vols_in):
    """Linear extrapolation from last two input values."""
    v_last = test_vols_in[:, -1]
    v_prev = test_vols_in[:, -2]
    drift = v_last - v_prev
    steps = np.arange(1, 13)[np.newaxis, :]
    pred = v_last[:, np.newaxis] + drift[:, np.newaxis] * steps
    return pred


def baseline_mlp(train_X, train_Y, test_X, seed=42):
    """2-layer MLP regressor."""
    from sklearn.neural_network import MLPRegressor as MLP
    model = MLP(hidden_layer_sizes=(128, 64), max_iter=200,
                random_state=seed, early_stopping=True,
                validation_fraction=0.15)
    model.fit(train_X, train_Y)
    return model.predict(test_X)


def baseline_ridge(train_X, train_Y, test_X):
    """Ridge regression baseline."""
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(train_X, train_Y)
    return model.predict(test_X)


def run_all_baselines(vols, ev_masks, is_ev, train_idx, test_idx, seed=42):
    """Run all baselines and return dict of {model_name: predictions}."""
    train_vols = vols[train_idx]
    test_vols = vols[test_idx]
    train_ev_masks = ev_masks[train_idx]
    train_is_ev = is_ev[train_idx]

    test_in = test_vols[:, :12]
    test_out = test_vols[:, 12:]
    test_ev_in = ev_masks[test_idx][:, :12]
    test_ev_out = ev_masks[test_idx][:, 12:]

    results = {}

    # HistMean
    pred = baseline_histmean(train_vols, test_in)
    results["HistMean"] = compute_metrics(pred, test_out, test_ev_out)

    # LastValue
    pred = baseline_lastvalue(test_in)
    results["LastValue"] = compute_metrics(pred, test_out, test_ev_out)

    # EventAwareLast
    pred = baseline_eventawarelast(test_in, test_ev_in, train_vols, train_ev_masks, train_is_ev)
    results["EventAwareLast"] = compute_metrics(pred, test_out, test_ev_out)

    # LinearDrift
    pred = baseline_lineardrift(test_in)
    results["LinearDrift"] = compute_metrics(pred, test_out, test_ev_out)

    # MLPRegressor
    train_X = train_vols[:, :12]
    train_Y = train_vols[:, 12:]
    pred = baseline_mlp(train_X, train_Y, test_in, seed=seed)
    results["MLPRegressor"] = compute_metrics(pred, test_out, test_ev_out)

    # RidgeLearned
    pred = baseline_ridge(train_X, train_Y, test_in)
    results["RidgeLearned"] = compute_metrics(pred, test_out, test_ev_out)

    return results


def per_year_analysis(vols, ev_masks, is_ev, years, train_idx):
    """Compute per-year MAE for EventAwareLast (the most interpretable baseline)."""
    train_vols = vols[train_idx]
    train_ev_masks = ev_masks[train_idx]
    train_is_ev = is_ev[train_idx]

    year_results = {}
    for yr in sorted(np.unique(years)):
        yr_idx = np.where(years == yr)[0]
        yr_vols = vols[yr_idx]
        yr_ev_out = ev_masks[yr_idx][:, 12:]

        yr_in = yr_vols[:, :12]
        yr_out = yr_vols[:, 12:]

        # EventAwareLast
        yr_ev_in = ev_masks[yr_idx][:, :12]
        pred_ea = baseline_eventawarelast(yr_in, yr_ev_in, train_vols, train_ev_masks, train_is_ev)
        ea_metrics = compute_metrics(pred_ea, yr_out, yr_ev_out)

        # LastValue
        pred_lv = baseline_lastvalue(yr_in)
        lv_metrics = compute_metrics(pred_lv, yr_out, yr_ev_out)

        year_results[int(yr)] = {
            "n_samples": len(yr_idx),
            "mean_vol": float(np.mean(yr_vols)),
            "EventAwareLast_T1_MAE": ea_metrics["T1_MAE"],
            "LastValue_T1_MAE": lv_metrics["T1_MAE"],
        }

    return year_results


def main():
    print("=" * 60)
    print("NET-V4 Temporal Split Experiments")
    print("=" * 60)

    vols, spatial, ev_masks, is_ev, years, severity = load_data()
    print(f"Loaded {len(vols)} samples, years {int(years.min())}-{int(years.max())}")

    # --- Chronological Split ---
    train_idx, val_idx, test_idx = temporal_split(years)
    print(f"\nChronological split:")
    print(f"  Train (2018-2021): {len(train_idx)} samples")
    print(f"  Val   (2022):      {len(val_idx)} samples")
    print(f"  Test  (2023):      {len(test_idx)} samples")

    # EV distribution per split
    for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        ev_frac = is_ev[idx].mean() * 100
        print(f"  {name} EV fraction: {ev_frac:.1f}%")

    # Run baselines with temporal split
    print("\n--- Running baselines (Chronological Split) ---")
    all_chrono = {}
    for seed in SEEDS:
        res = run_all_baselines(vols, ev_masks, is_ev, train_idx, test_idx, seed=seed)
        for model, metrics in res.items():
            if model not in all_chrono:
                all_chrono[model] = defaultdict(list)
            for k, v in metrics.items():
                all_chrono[model][k].append(v)

    # Aggregate
    chrono_summary = {}
    for model, metric_lists in all_chrono.items():
        summary = {}
        for k, vals in metric_lists.items():
            arr = np.array(vals)
            summary[k] = {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr))}
        chrono_summary[model] = summary

    print("\n=== Chronological Split Results ===")
    print(f"{'Model':<20} {'T1 MAE':>12} {'T2 MAE':>12}")
    print("-" * 46)
    for model in sorted(chrono_summary.keys(),
                        key=lambda m: chrono_summary[m]["T1_MAE"]["mean"]):
        t1 = chrono_summary[model]["T1_MAE"]
        t2 = chrono_summary[model]["T2_MAE"]
        print(f"{model:<20} {t1['mean']:8.2f}+/-{t1['std']:.2f}  "
              f"{t2['mean']:8.2f}+/-{t2['std']:.2f}")

    # --- Per-Year Analysis ---
    print("\n--- Per-Year Analysis (trained on 2018-2021) ---")
    year_res = per_year_analysis(vols, ev_masks, is_ev, years, train_idx)
    print(f"\n{'Year':<6} {'N':>6} {'Mean Vol':>10} {'EA T1 MAE':>10} {'LV T1 MAE':>10}")
    print("-" * 46)
    for yr in sorted(year_res.keys()):
        r = year_res[yr]
        print(f"{yr:<6} {r['n_samples']:>6} {r['mean_vol']:>10.1f} "
              f"{r['EventAwareLast_T1_MAE']:>10.2f} {r['LastValue_T1_MAE']:>10.2f}")

    # --- Save results ---
    output = {
        "experiment": "temporal_split",
        "split": {"train": "2018-2021", "val": "2022", "test": "2023"},
        "chrono_results": chrono_summary,
        "per_year": year_res,
    }
    out_path = os.path.join(OUT_DIR, "temporal_split_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
