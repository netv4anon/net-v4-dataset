"""
run_severity_and_extreme_analysis.py
=====================================
Additional analyses for NET-V4 NeurIPS D&B submission:

1. Severity-stratified Task-2 performance
2. Extreme event analysis (top-10%/25% by vol_drop)
3. Incident class breakdown (Medical vs NonMedical)
4. Year-level volume distribution statistics

Populates Table 7 and related figures for the paper.

Usage:
    python experiments/run_severity_and_extreme_analysis.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NPZ_PATH = os.path.join(ROOT, "data_preprocessing_v4", "NYC_EMS_Traffic_V4.npz")
CSV_PATH = os.path.join(ROOT, "data_preprocessing_v4", "NYC_EMS_Traffic_V4.csv")
OUT_DIR = os.path.join(ROOT, "NIPS_DnB_Submission_Package", "results")
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    data = np.load(NPZ_PATH, allow_pickle=True)
    vols = data["volume_sequences"]      # (33076, 24)
    ev_masks = data["ev_masks"]           # (33076, 24)
    is_ev = data["is_ev_event"]           # (33076,)
    years = data["incident_years"]        # (33076,)
    severity = data["severity_numeric"]    # (33076,) integer 0-5
    return vols, ev_masks, is_ev, years, severity


def load_csv_metadata():
    """Load CSV for incident_class_group, alarm_level, vol_drop."""
    df = pd.read_csv(CSV_PATH, low_memory=False)
    return df


def stratified_random_split(is_ev, seed=42):
    """Replicate the canonical stratified split from run_full_neurips_experiments."""
    rng = np.random.RandomState(seed)
    ev_idx = np.where(is_ev == 1)[0].copy()
    ctrl_idx = np.where(is_ev == 0)[0].copy()
    rng.shuffle(ev_idx)
    rng.shuffle(ctrl_idx)

    def ratio_split(idx, ratios=(0.666, 0.167, 0.167)):
        n = len(idx)
        n_train = int(n * ratios[0])
        n_val = int(n * (ratios[0] + ratios[1]))
        return idx[:n_train], idx[n_train:n_val], idx[n_val:]

    ev_tr, ev_va, ev_te = ratio_split(ev_idx)
    ct_tr, ct_va, ct_te = ratio_split(ctrl_idx)

    train_idx = np.concatenate([ev_tr, ct_tr])
    val_idx = np.concatenate([ev_va, ct_va])
    test_idx = np.concatenate([ev_te, ct_te])
    return train_idx, val_idx, test_idx


def baseline_lastvalue(vols_in):
    return np.tile(vols_in[:, -1:], (1, 12))


def baseline_eventawarelast(vols_in, ev_in, train_vols, train_ev_masks, train_is_ev):
    ev_train_idx = np.where(train_is_ev == 1)[0]
    if len(ev_train_idx) == 0:
        return baseline_lastvalue(vols_in)
    ev_vols = train_vols[ev_train_idx]
    ev_out = ev_vols[:, 12:]
    ev_in_last = ev_vols[:, 11:12]
    denom = np.clip(ev_in_last, 1.0, None)
    rel_change = (ev_out - ev_in_last) / denom
    shock_profile = np.mean(rel_change, axis=0)
    last = vols_in[:, -1:]
    pred = np.tile(last, (1, 12))
    ongoing_event = ev_in[:, -1] > 0.5
    if ongoing_event.any():
        pred[ongoing_event] = last[ongoing_event] * (1.0 + shock_profile[np.newaxis, :])
    return pred


def baseline_mlp(train_X, train_Y, test_X, seed=42):
    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200,
                         random_state=seed, early_stopping=True,
                         validation_fraction=0.15)
    model.fit(train_X, train_Y)
    return model.predict(test_X)


def compute_task2_mae(pred, target, ev_mask_out):
    shock_mask = ev_mask_out > 0.5
    if shock_mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[shock_mask] - target[shock_mask])))


def compute_task1_mae(pred, target):
    return float(np.mean(np.abs(pred - target)))


def main():
    print("=" * 60)
    print("NET-V4 Severity & Extreme Event Analysis")
    print("=" * 60)

    vols, ev_masks, is_ev, years, severity = load_data()
    print(f"Loaded {len(vols)} samples")

    # Load CSV for extra metadata
    df = load_csv_metadata()
    print(f"CSV has {len(df)} rows, columns: {list(df.columns[:10])}...")

    # Split
    train_idx, val_idx, test_idx = stratified_random_split(is_ev, seed=42)
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_vols = vols[train_idx]
    train_ev_masks = ev_masks[train_idx]
    train_is_ev = is_ev[train_idx]

    test_vols = vols[test_idx]
    test_in = test_vols[:, :12]
    test_out = test_vols[:, 12:]
    test_ev_in = ev_masks[test_idx][:, :12]
    test_ev_out = ev_masks[test_idx][:, 12:]
    test_is_ev = is_ev[test_idx]

    # Generate predictions for 3 models
    print("\nGenerating predictions...")
    preds = {}
    preds["LastValue"] = baseline_lastvalue(test_in)
    preds["EventAwareLast"] = baseline_eventawarelast(
        test_in, test_ev_in, train_vols, train_ev_masks, train_is_ev)
    preds["MLPRegressor"] = baseline_mlp(
        train_vols[:, :12], train_vols[:, 12:], test_in, seed=42)

    # ── 1. Severity-Stratified Analysis ──
    print("\n--- Severity-Stratified Task-2 MAE ---")
    test_severity = severity[test_idx]

    # Group by severity level
    sev_levels = [(0, 0.5, "Control (sev=0)"),
                  (0.5, 1.5, "First Alarm (sev=1)"),
                  (1.5, 6.0, "Higher Alarm (sev>=2)")]

    sev_results = {}
    for lo, hi, label in sev_levels:
        mask = (test_severity >= lo) & (test_severity < hi) & (test_is_ev == 1)
        n = mask.sum()
        if n == 0:
            continue
        sev_results[label] = {"n": int(n)}
        for mname, pred in preds.items():
            mae = compute_task2_mae(pred[mask], test_out[mask], test_ev_out[mask])
            sev_results[label][f"{mname}_T2_MAE"] = mae
        print(f"  {label} (N={n}):")
        for mname in preds:
            print(f"    {mname}: T2 MAE = {sev_results[label].get(f'{mname}_T2_MAE', 'N/A'):.2f}")

    # ── 2. Incident Class Stratification ──
    print("\n--- Incident Class Stratification ---")
    if "incident_class_group" in df.columns:
        test_class = df["incident_class_group"].values[test_idx]
        classes = ["Medical Emergencies", "NonMedical Emergencies"]
        class_results = {}
        for cls in classes:
            mask = (test_class == cls) & (test_is_ev == 1)
            n = mask.sum()
            if n == 0:
                continue
            class_results[cls] = {"n": int(n)}
            for mname, pred in preds.items():
                mae = compute_task2_mae(pred[mask], test_out[mask], test_ev_out[mask])
                class_results[cls][f"{mname}_T2_MAE"] = mae
            print(f"  {cls} (N={n}):")
            for mname in preds:
                v = class_results[cls].get(f"{mname}_T2_MAE", float("nan"))
                print(f"    {mname}: T2 MAE = {v:.2f}")
    else:
        class_results = {}
        print("  incident_class_group not found in CSV")

    # ── 3. Extreme Event Analysis ──
    print("\n--- Extreme Event Analysis (by vol_drop) ---")
    if "vol_drop" in df.columns:
        test_vol_drop = df["vol_drop"].values[test_idx].astype(float)
        ev_test_mask = test_is_ev == 1
        ev_drops = test_vol_drop[ev_test_mask]

        # Top-10% and Top-25% most severe drops
        for pct_label, pct in [("Top-10%", 90), ("Top-25%", 75)]:
            threshold = np.nanpercentile(ev_drops, pct)
            extreme_mask = ev_test_mask & (test_vol_drop >= threshold)
            n = extreme_mask.sum()
            print(f"\n  {pct_label} extreme events (vol_drop >= {threshold:.2f}, N={n}):")
            for mname, pred in preds.items():
                t1 = compute_task1_mae(pred[extreme_mask], test_out[extreme_mask])
                t2 = compute_task2_mae(pred[extreme_mask], test_out[extreme_mask],
                                       test_ev_out[extreme_mask])
                print(f"    {mname}: T1 MAE = {t1:.2f}, T2 MAE = {t2:.2f}")
    else:
        print("  vol_drop not found in CSV")

    # ── 4. Year-Level Volume Distribution ──
    print("\n--- Year-Level Volume Statistics ---")
    year_stats = {}
    for yr in sorted(np.unique(years)):
        yr_mask = years == yr
        yr_vols_flat = vols[yr_mask].flatten()
        year_stats[int(yr)] = {
            "n_samples": int(yr_mask.sum()),
            "mean_vol": float(np.mean(yr_vols_flat)),
            "std_vol": float(np.std(yr_vols_flat)),
            "median_vol": float(np.median(yr_vols_flat)),
            "ev_fraction": float(is_ev[yr_mask].mean()),
        }
        r = year_stats[int(yr)]
        print(f"  {yr}: N={r['n_samples']}, mean={r['mean_vol']:.1f}, "
              f"std={r['std_vol']:.1f}, EV%={r['ev_fraction']*100:.1f}%")

    # ── 5. Alarm Level Distribution ──
    print("\n--- Alarm Level Distribution ---")
    if "alarm_level" in df.columns:
        alarm_counts = df["alarm_level"].value_counts()
        for level, count in alarm_counts.items():
            print(f"  {level}: {count} ({count/len(df)*100:.1f}%)")

    # ── Save all results ──
    output = {
        "severity_stratified": sev_results,
        "incident_class": class_results,
        "year_stats": year_stats,
    }
    out_path = os.path.join(OUT_DIR, "severity_extreme_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
