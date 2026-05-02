"""
Critical analysis: Is shock ACTUALLY harder to predict than non-shock
when controlling for volume level?

The per-step data shows shock MAE < non-shock MAE at every horizon.
But this could be because shock windows have lower volumes (lower target = lower MAE).
We need to check: at the SAME volume level, is shock harder?
"""
import numpy as np

d = np.load('data_preprocessing_v4/NYC_EMS_Traffic_V4.npz', allow_pickle=True)
vol = d['volume_sequences']     # (33076, 24)
ev_masks = d['ev_masks']        # (33076, 24)
is_ev = d['is_ev_event']        # (33076,)

# Stratified split (canonical)
np.random.seed(42)
ev_idx = np.where(is_ev == 1)[0]
ctrl_idx = np.where(is_ev == 0)[0]
np.random.shuffle(ev_idx)
np.random.shuffle(ctrl_idx)

def split(idx):
    n = len(idx)
    return idx[:int(0.7*n)], idx[int(0.7*n):int(0.85*n)], idx[int(0.85*n):]

_, _, ev_test = split(ev_idx)
_, _, ctrl_test = split(ctrl_idx)
test_idx = np.sort(np.concatenate([ev_test, ctrl_test]))

x_test = vol[test_idx, :12]   # input
y_test = vol[test_idx, 12:]   # output (ground truth)
is_ev_test = is_ev[test_idx]
masks_test = ev_masks[test_idx, 12:]  # output-window EV mask

# LastValue prediction (simplest baseline for clean analysis)
lv_pred = np.repeat(x_test[:, -1:], 12, axis=1)
errors = np.abs(y_test - lv_pred)

# --- Analysis 1: Raw comparison ---
print("=== RAW COMPARISON (Step 0) ===")
ev_samples = is_ev_test == 1
ctrl_samples = is_ev_test == 0

# Step 0 error for EV vs Control
ev_step0_err = errors[ev_samples, 0]
ctrl_step0_err = errors[ctrl_samples, 0]
print(f"EV samples step-0 MAE:   {ev_step0_err.mean():.2f} (mean target vol: {y_test[ev_samples, 0].mean():.2f})")
print(f"Ctrl samples step-0 MAE: {ctrl_step0_err.mean():.2f} (mean target vol: {y_test[ctrl_samples, 0].mean():.2f})")

# --- Analysis 2: Volume-matched comparison ---
print("\n=== VOLUME-BIN MATCHED COMPARISON ===")
# Bin by output step-0 volume (target), then compare error within bins
y0_all = y_test[:, 0]
bins = [0, 50, 100, 150, 200, 300, 9999]
bin_labels = ['0-50', '50-100', '100-150', '150-200', '200-300', '300+']

print(f"{'Vol Bin':>10s} | {'EV MAE':>8s} {'EV N':>6s} {'EV vol':>8s} | {'Ctrl MAE':>8s} {'Ctrl N':>6s} {'Ctrl vol':>8s} | {'Gap':>8s}")
print("-" * 85)

total_ev_harder = 0
total_ctrl_harder = 0
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    ev_mask = ev_samples & (y0_all >= lo) & (y0_all < hi)
    ctrl_mask = ctrl_samples & (y0_all >= lo) & (y0_all < hi)
    
    n_ev = ev_mask.sum()
    n_ctrl = ctrl_mask.sum()
    if n_ev < 10 or n_ctrl < 10:
        print(f"{bin_labels[i]:>10s} | {'N/A':>8s} {n_ev:>6d} {'':>8s} | {'N/A':>8s} {n_ctrl:>6d} {'':>8s} | {'N/A':>8s}")
        continue
    
    ev_mae = errors[ev_mask, 0].mean()
    ctrl_mae = errors[ctrl_mask, 0].mean()
    ev_vol = y_test[ev_mask, 0].mean()
    ctrl_vol = y_test[ctrl_mask, 0].mean()
    gap = ev_mae - ctrl_mae
    
    if gap > 0:
        total_ev_harder += 1
    else:
        total_ctrl_harder += 1
    
    print(f"{bin_labels[i]:>10s} | {ev_mae:>8.2f} {n_ev:>6d} {ev_vol:>8.1f} | {ctrl_mae:>8.2f} {n_ctrl:>6d} {ctrl_vol:>8.1f} | {gap:>+8.2f}")

print(f"\nEV harder in {total_ev_harder}/{total_ev_harder+total_ctrl_harder} volume bins")

# --- Analysis 3: Shock-step vs non-shock-step WITHIN EV samples ---
print("\n=== WITHIN EV SAMPLES: shock-active vs shock-inactive steps ===")
ev_test_errors = errors[ev_samples]
ev_test_masks = masks_test[ev_samples]
ev_test_targets = y_test[ev_samples]

shock_step_errors = ev_test_errors[ev_test_masks > 0]
nonshock_step_errors_in_ev = ev_test_errors[ev_test_masks == 0]
shock_step_targets = ev_test_targets[ev_test_masks > 0]
nonshock_step_targets_in_ev = ev_test_targets[ev_test_masks == 0]

print(f"Shock-active steps:   MAE={shock_step_errors.mean():.2f}, mean target={shock_step_targets.mean():.2f}, N={len(shock_step_errors)}")
print(f"Non-shock steps (EV): MAE={nonshock_step_errors_in_ev.mean():.2f}, mean target={nonshock_step_targets_in_ev.mean():.2f}, N={len(nonshock_step_errors_in_ev)}")

# MAPE comparison (scale-normalized)
shock_mape = (shock_step_errors / (shock_step_targets + 1e-6)).mean() * 100
nonshock_mape = (nonshock_step_errors_in_ev / (nonshock_step_targets_in_ev + 1e-6)).mean() * 100
print(f"\nShock-active MAPE:   {shock_mape:.2f}%")
print(f"Non-shock MAPE (EV): {nonshock_mape:.2f}%")

# --- Analysis 4: Overall MAPE / relative error ---
print("\n=== RELATIVE ERROR (MAPE) BY GROUP ===")
ev_mape_step0 = (errors[ev_samples, 0] / (y_test[ev_samples, 0] + 1e-6)).mean() * 100
ctrl_mape_step0 = (errors[ctrl_samples, 0] / (y_test[ctrl_samples, 0] + 1e-6)).mean() * 100
print(f"EV step-0 MAPE:   {ev_mape_step0:.2f}%")
print(f"Ctrl step-0 MAPE: {ctrl_mape_step0:.2f}%")
print(f"Gap: {ev_mape_step0 - ctrl_mape_step0:+.2f}pp")
