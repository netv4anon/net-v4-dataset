"""
Propensity-Score Matched Experiment:
Build a balanced EV/Control dataset where both groups have identical
distributions of (segment, DOW, hour-bin, volume-bin).
Then re-run baselines to see if EV is genuinely harder to predict.

Two matching strategies:
  A) Exact match on (segment, DOW, 3h-hour-bin, volume-quartile) 
  B) Nearest-neighbor propensity matching with caliper

Then run LastValue, MLP, EventAwareLast on the balanced set.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats

print("=" * 70)
print("PROPENSITY-MATCHED EXPERIMENT")
print("=" * 70)

# --- Load ---
d = np.load('data_preprocessing_v4/NYC_EMS_Traffic_V4.npz', allow_pickle=True)
df = pd.read_csv('data_preprocessing_v4/NYC_EMS_Traffic_V4.csv', low_memory=False)

vol = d['volume_sequences']       # (33076, 24)
ev_masks = d['ev_masks']          # (33076, 24)
is_ev = d['is_ev_event']          # (33076,)
seg_ids = d['segment_ids']        # (33076,)

dt = pd.to_datetime(df['incident_datetime'])
dow = dt.dt.dayofweek.values
hour = dt.dt.hour.values

# Volume features
input_mean_vol = vol[:, :12].mean(axis=1)
input_std_vol = vol[:, :12].std(axis=1)

# --- Strategy A: Exact covariate matching ---
print("\n--- Strategy A: Exact Covariate Matching ---")

# Discretize covariates
hour_bin = (hour // 4).astype(int)  # 6 bins: 0-3, 4-7, 8-11, 12-15, 16-19, 20-23
vol_quartiles = pd.qcut(input_mean_vol, q=4, labels=False, duplicates='drop')

# Build strata
def make_key(i):
    return (str(seg_ids[i]), int(dow[i]), int(hour_bin[i]), int(vol_quartiles[i]))

ev_all = np.where(is_ev == 1)[0]
ctrl_all = np.where(is_ev == 0)[0]

# Group by strata
ev_strata = defaultdict(list)
ctrl_strata = defaultdict(list)

for i in ev_all:
    ev_strata[make_key(i)].append(i)
for i in ctrl_all:
    ctrl_strata[make_key(i)].append(i)

# Match: for each stratum, take min(n_ev, n_ctrl) from each
matched_ev_A = []
matched_ctrl_A = []
strata_used = 0

for key in ev_strata:
    if key in ctrl_strata:
        n_ev = len(ev_strata[key])
        n_ctrl = len(ctrl_strata[key])
        n_match = min(n_ev, n_ctrl)
        if n_match > 0:
            # Random subsample
            np.random.seed(42)
            ev_sub = np.random.choice(ev_strata[key], n_match, replace=False)
            ctrl_sub = np.random.choice(ctrl_strata[key], n_match, replace=False)
            matched_ev_A.extend(ev_sub)
            matched_ctrl_A.extend(ctrl_sub)
            strata_used += 1

matched_ev_A = np.array(matched_ev_A)
matched_ctrl_A = np.array(matched_ctrl_A)

print(f"Strata with both EV & Ctrl: {strata_used}")
print(f"Matched EV:   {len(matched_ev_A)}")
print(f"Matched Ctrl: {len(matched_ctrl_A)}")
print(f"Coverage:     {len(matched_ev_A)}/{len(ev_all)} = {100*len(matched_ev_A)/len(ev_all):.1f}% of EV")

# Balance check
print(f"\nBalance check (Strategy A):")
print(f"  EV mean vol:    {input_mean_vol[matched_ev_A].mean():.2f}")
print(f"  Ctrl mean vol:  {input_mean_vol[matched_ctrl_A].mean():.2f}")
print(f"  EV mean hour:   {hour[matched_ev_A].mean():.1f}")
print(f"  Ctrl mean hour: {hour[matched_ctrl_A].mean():.1f}")
print(f"  EV mean DOW:    {dow[matched_ev_A].mean():.2f}")
print(f"  Ctrl mean DOW:  {dow[matched_ctrl_A].mean():.2f}")
print(f"  EV vol std:     {input_std_vol[matched_ev_A].mean():.2f}")
print(f"  Ctrl vol std:   {input_std_vol[matched_ctrl_A].mean():.2f}")

# --- Run baselines on balanced set ---
print(f"\n{'='*70}")
print("BASELINE RESULTS ON PROPENSITY-MATCHED SET")
print(f"{'='*70}")

# Combined balanced set
all_matched = np.concatenate([matched_ev_A, matched_ctrl_A])
is_ev_matched = np.concatenate([np.ones(len(matched_ev_A)), np.zeros(len(matched_ctrl_A))])

# Shuffle and split 70/15/15
np.random.seed(42)
perm = np.random.permutation(len(all_matched))
all_matched = all_matched[perm]
is_ev_matched = is_ev_matched[perm]

n = len(all_matched)
n_train = int(0.7 * n)
n_val = int(0.85 * n)

train_idx = all_matched[:n_train]
val_idx = all_matched[n_train:n_val]
test_idx = all_matched[n_val:]
test_ev = is_ev_matched[n_val:]

print(f"\nBalanced split: train={n_train}, val={n_val-n_train}, test={n-n_val}")
print(f"Test EV rate: {test_ev.mean()*100:.1f}%")

x_test = vol[test_idx, :12]
y_test = vol[test_idx, 12:]
masks_test = ev_masks[test_idx, 12:]

# --- Baseline 1: LastValue ---
lv_pred = np.repeat(x_test[:, -1:], 12, axis=1)
lv_err = np.abs(y_test - lv_pred)

ev_test_mask = test_ev == 1
ctrl_test_mask = test_ev == 0

lv_ev_mae = lv_err[ev_test_mask].mean()
lv_ctrl_mae = lv_err[ctrl_test_mask].mean()

# Task 2 (shock-window MAE for EV only)
shock_errors_lv = []
for i in range(len(test_idx)):
    if not ev_test_mask[i]:
        continue
    out_mask = masks_test[i]
    if out_mask.sum() == 0:
        continue
    shock_steps = out_mask > 0
    shock_errors_lv.append(np.abs(y_test[i, shock_steps] - lv_pred[i, shock_steps]).mean())

lv_t2 = np.mean(shock_errors_lv) if shock_errors_lv else float('nan')

print(f"\n--- LastValue ---")
print(f"  EV full-output MAE:   {lv_ev_mae:.3f}")
print(f"  Ctrl full-output MAE: {lv_ctrl_mae:.3f}")
print(f"  Gap (EV - Ctrl):      {lv_ev_mae - lv_ctrl_mae:+.3f}")
print(f"  EV Task-2 (shock):    {lv_t2:.3f}")

# MAPE
lv_ev_mape = (lv_err[ev_test_mask] / (y_test[ev_test_mask] + 1e-6)).mean() * 100
lv_ctrl_mape = (lv_err[ctrl_test_mask] / (y_test[ctrl_test_mask] + 1e-6)).mean() * 100
print(f"  EV MAPE:              {lv_ev_mape:.2f}%")
print(f"  Ctrl MAPE:            {lv_ctrl_mape:.2f}%")
print(f"  MAPE gap:             {lv_ev_mape - lv_ctrl_mape:+.2f}pp")

# --- Baseline 2: HistMean ---
hm_pred = np.repeat(x_test.mean(axis=1, keepdims=True), 12, axis=1)
hm_err = np.abs(y_test - hm_pred)

hm_ev_mae = hm_err[ev_test_mask].mean()
hm_ctrl_mae = hm_err[ctrl_test_mask].mean()

print(f"\n--- HistMean ---")
print(f"  EV full-output MAE:   {hm_ev_mae:.3f}")
print(f"  Ctrl full-output MAE: {hm_ctrl_mae:.3f}")
print(f"  Gap (EV - Ctrl):      {hm_ev_mae - hm_ctrl_mae:+.3f}")

# --- Baseline 3: LinearDrift ---
t_in = np.arange(12, dtype=float)
t_out = np.arange(12, 24, dtype=float)
ld_pred = np.zeros_like(y_test)
for i in range(len(test_idx)):
    coeffs = np.polyfit(t_in, x_test[i], 1)
    ld_pred[i] = np.polyval(coeffs, t_out)
ld_err = np.abs(y_test - ld_pred)

ld_ev_mae = ld_err[ev_test_mask].mean()
ld_ctrl_mae = ld_err[ctrl_test_mask].mean()

print(f"\n--- LinearDrift ---")
print(f"  EV full-output MAE:   {ld_ev_mae:.3f}")
print(f"  Ctrl full-output MAE: {ld_ctrl_mae:.3f}")
print(f"  Gap (EV - Ctrl):      {ld_ev_mae - ld_ctrl_mae:+.3f}")

# --- Baseline 4: EventAwareLast ---
# Compute shock profile from training set
train_evs = train_idx[is_ev_matched[:n_train] == 1]
shock_profile = np.zeros(12)
shock_counts = np.zeros(12)
for i in train_evs:
    m = ev_masks[i, 12:]
    for step in range(12):
        if m[step] > 0:
            shock_profile[step] += vol[i, 12 + step] - vol[i, 11]
            shock_counts[step] += 1

for step in range(12):
    if shock_counts[step] > 0:
        shock_profile[step] /= shock_counts[step]

print(f"\n  Shock profile (train): {shock_profile[:4].round(2)}")

# Predict
eal_pred = np.copy(lv_pred)
for i in range(len(test_idx)):
    if ev_test_mask[i]:
        # Use last-step EV flag
        last_ev = ev_masks[test_idx[i], 11]
        if last_ev > 0:
            for step in range(12):
                eal_pred[i, step] = x_test[i, -1] + shock_profile[step]

eal_err = np.abs(y_test - eal_pred)
eal_ev_mae = eal_err[ev_test_mask].mean()
eal_ctrl_mae = eal_err[ctrl_test_mask].mean()

shock_errors_eal = []
for i in range(len(test_idx)):
    if not ev_test_mask[i]:
        continue
    out_mask = masks_test[i]
    if out_mask.sum() == 0:
        continue
    shock_steps = out_mask > 0
    shock_errors_eal.append(np.abs(y_test[i, shock_steps] - eal_pred[i, shock_steps]).mean())

eal_t2 = np.mean(shock_errors_eal) if shock_errors_eal else float('nan')

print(f"\n--- EventAwareLast ---")
print(f"  EV full-output MAE:   {eal_ev_mae:.3f}")
print(f"  Ctrl full-output MAE: {eal_ctrl_mae:.3f}")
print(f"  Gap (EV - Ctrl):      {eal_ev_mae - eal_ctrl_mae:+.3f}")
print(f"  EV Task-2 (shock):    {eal_t2:.3f}")

# --- Summary ---
print(f"\n{'='*70}")
print("SUMMARY: IS EV HARDER AFTER PROPENSITY MATCHING?")
print(f"{'='*70}")

print(f"\n{'Baseline':20s} {'EV MAE':>10s} {'Ctrl MAE':>10s} {'Gap':>10s} {'EV harder?':>12s}")
print("-" * 65)
for name, ev_m, ctrl_m in [
    ("LastValue", lv_ev_mae, lv_ctrl_mae),
    ("HistMean", hm_ev_mae, hm_ctrl_mae),
    ("LinearDrift", ld_ev_mae, ld_ctrl_mae),
    ("EventAwareLast", eal_ev_mae, eal_ctrl_mae),
]:
    gap = ev_m - ctrl_m
    harder = "YES" if gap > 0 else "NO"
    print(f"{name:20s} {ev_m:>10.3f} {ctrl_m:>10.3f} {gap:>+10.3f} {harder:>12s}")

# --- Paired analysis within balanced test set ---
print(f"\n{'='*70}")
print("PAIRED ANALYSIS WITHIN BALANCED TEST SET")
print(f"{'='*70}")

# Per-sample MAE
lv_sample_mae = lv_err.mean(axis=1)
ev_maes = lv_sample_mae[ev_test_mask]
ctrl_maes = lv_sample_mae[ctrl_test_mask]

print(f"\nLastValue per-sample MAE distribution:")
print(f"  EV:   mean={ev_maes.mean():.3f}, median={np.median(ev_maes):.3f}, p75={np.percentile(ev_maes,75):.3f}")
print(f"  Ctrl: mean={ctrl_maes.mean():.3f}, median={np.median(ctrl_maes):.3f}, p75={np.percentile(ctrl_maes,75):.3f}")

# Welch t-test
t, p = stats.ttest_ind(ev_maes, ctrl_maes, equal_var=False)
d_cohen = (ev_maes.mean() - ctrl_maes.mean()) / np.sqrt((ev_maes.std()**2 + ctrl_maes.std()**2)/2)
print(f"  Welch t={t:.3f}, p={p:.2e}, Cohen's d={d_cohen:.4f}")

# --- Output volatility check (should be balanced now) ---
print(f"\nOutput volatility (balanced set):")
ev_out_std = vol[test_idx[ev_test_mask], 12:].std(axis=1)
ctrl_out_std = vol[test_idx[ctrl_test_mask], 12:].std(axis=1)
print(f"  EV out std:   {ev_out_std.mean():.3f}")
print(f"  Ctrl out std: {ctrl_out_std.mean():.3f}")

# --- Transition surprise (balanced) ---
ev_bound = np.abs(vol[test_idx[ev_test_mask], 12] - vol[test_idx[ev_test_mask], 11])
ctrl_bound = np.abs(vol[test_idx[ctrl_test_mask], 12] - vol[test_idx[ctrl_test_mask], 11])
ev_in_s = vol[test_idx[ev_test_mask], :12].std(axis=1)
ctrl_in_s = vol[test_idx[ctrl_test_mask], :12].std(axis=1)
ev_surp = ev_bound / (ev_in_s + 1e-6)
ctrl_surp = ctrl_bound / (ctrl_in_s + 1e-6)

print(f"\nTransition surprise (balanced):")
print(f"  EV:   {ev_surp.mean():.4f}")
print(f"  Ctrl: {ctrl_surp.mean():.4f}")
t_s, p_s = stats.ttest_ind(ev_surp, ctrl_surp)
print(f"  t={t_s:.3f}, p={p_s:.2e}")
