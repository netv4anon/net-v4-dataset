"""
MATCHED-PAIR ANALYSIS: Does EV make prediction genuinely harder
when controlling for segment, time-of-day, day-of-week, and volume?

Strategy:
1. Match each EV test sample to the nearest Control test sample by:
   - Same segment_id (exact match)
   - Same DOW (exact match)
   - Closest hour (within 2 hours)
   - Closest mean input volume (within 20%)
2. Compute paired error differences
3. If EV is genuinely harder => positive paired difference
4. Also test: transition surprise, volatility surprise, MAPE surprise

Additionally:
5. Test "relative forecast degradation" - how much does one-step-ahead
   error grow at the exact transition boundary?
6. Directional accuracy: does the model get the SIGN of the change wrong
   more often for EV?
"""
import numpy as np
import pandas as pd
from scipy import stats

# --- Load data ---
d = np.load('data_preprocessing_v4/NYC_EMS_Traffic_V4.npz', allow_pickle=True)
df = pd.read_csv('data_preprocessing_v4/NYC_EMS_Traffic_V4.csv', low_memory=False)

vol = d['volume_sequences']
ev_masks = d['ev_masks']
is_ev = d['is_ev_event']
seg_ids = d['segment_ids']

dt = pd.to_datetime(df['incident_datetime'])
dow = dt.dt.dayofweek.values
hour = dt.dt.hour.values

# --- Canonical split ---
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

print(f"EV test: {len(ev_test)}, Ctrl test: {len(ctrl_test)}")

# --- Build control lookup ---
ctrl_data = {}
for ci in ctrl_test:
    key = (str(seg_ids[ci]), int(dow[ci]))
    if key not in ctrl_data:
        ctrl_data[key] = []
    ctrl_data[key].append(ci)

# --- Match ---
matched_ev = []
matched_ctrl = []
match_quality = []

for ei in ev_test:
    seg = str(seg_ids[ei])
    d_dow = int(dow[ei])
    h = int(hour[ei])
    ev_mean_in = vol[ei, :12].mean()
    
    # Try exact segment + DOW
    key = (seg, d_dow)
    candidates = ctrl_data.get(key, [])
    
    if not candidates:
        # Relax: try same segment, any DOW
        for dw in range(7):
            candidates = ctrl_data.get((seg, dw), [])
            if candidates:
                break
    
    if not candidates:
        continue
    
    # Find closest by hour + volume
    best_ci = None
    best_score = float('inf')
    for ci in candidates:
        h_diff = abs(int(hour[ci]) - h)
        if h_diff > 12:
            h_diff = 24 - h_diff
        vol_diff = abs(vol[ci, :12].mean() - ev_mean_in) / (ev_mean_in + 1e-6)
        
        if h_diff <= 3 and vol_diff <= 0.3:  # within 3 hours, 30% volume
            score = h_diff + vol_diff * 10
            if score < best_score:
                best_score = score
                best_ci = ci
    
    if best_ci is not None:
        matched_ev.append(ei)
        matched_ctrl.append(best_ci)
        match_quality.append(best_score)

matched_ev = np.array(matched_ev)
matched_ctrl = np.array(matched_ctrl)
print(f"\nMatched pairs: {len(matched_ev)} / {len(ev_test)} ({100*len(matched_ev)/len(ev_test):.1f}%)")

# --- Compute errors for matched pairs ---
# LastValue baseline
lv_ev = np.abs(vol[matched_ev, 12:] - vol[matched_ev, 11:12])
lv_ctrl = np.abs(vol[matched_ctrl, 12:] - vol[matched_ctrl, 11:12])

# Per-sample MAE
ev_mae = lv_ev.mean(axis=1)
ctrl_mae = lv_ctrl.mean(axis=1)
paired_diff = ev_mae - ctrl_mae  # positive = EV harder

print(f"\n========== MATCHED-PAIR RESULTS (LastValue) ==========")
print(f"EV matched MAE:     {ev_mae.mean():.3f} +/- {ev_mae.std():.3f}")
print(f"Ctrl matched MAE:   {ctrl_mae.mean():.3f} +/- {ctrl_mae.std():.3f}")
print(f"Paired difference:  {paired_diff.mean():.3f} +/- {paired_diff.std():.3f}")
print(f"EV harder fraction: {(paired_diff > 0).mean()*100:.1f}%")

t_stat, p_val = stats.ttest_rel(ev_mae, ctrl_mae)
print(f"Paired t-test:      t={t_stat:.3f}, p={p_val:.2e}")
cohens_d = paired_diff.mean() / paired_diff.std()
print(f"Cohen's d:          {cohens_d:.4f}")

# --- MAPE matched comparison ---
ev_targets = vol[matched_ev, 12:]
ctrl_targets = vol[matched_ctrl, 12:]
ev_mape = (lv_ev / (ev_targets + 1e-6)).mean(axis=1) * 100
ctrl_mape = (lv_ctrl / (ctrl_targets + 1e-6)).mean(axis=1) * 100
mape_diff = ev_mape - ctrl_mape

print(f"\n========== MATCHED-PAIR MAPE ==========")
print(f"EV matched MAPE:    {ev_mape.mean():.2f}%")
print(f"Ctrl matched MAPE:  {ctrl_mape.mean():.2f}%")
print(f"MAPE difference:    {mape_diff.mean():.2f}pp")
print(f"EV higher MAPE:     {(mape_diff > 0).mean()*100:.1f}%")
t2, p2 = stats.ttest_rel(ev_mape, ctrl_mape)
print(f"Paired t-test:      t={t2:.3f}, p={p2:.2e}")

# --- Transition surprise ---
# How much does the value change at the input→output boundary 
# relative to the within-input volatility?
ev_in_std = vol[matched_ev, :12].std(axis=1)
ctrl_in_std = vol[matched_ctrl, :12].std(axis=1)

ev_boundary_change = np.abs(vol[matched_ev, 12] - vol[matched_ev, 11])
ctrl_boundary_change = np.abs(vol[matched_ctrl, 12] - vol[matched_ctrl, 11])

ev_surprise = ev_boundary_change / (ev_in_std + 1e-6)
ctrl_surprise = ctrl_boundary_change / (ctrl_in_std + 1e-6)
surprise_diff = ev_surprise - ctrl_surprise

print(f"\n========== TRANSITION SURPRISE (|boundary_delta| / input_std) ==========")
print(f"EV surprise:        {ev_surprise.mean():.3f}")
print(f"Ctrl surprise:      {ctrl_surprise.mean():.3f}")
print(f"Difference:         {surprise_diff.mean():.3f}")
print(f"EV more surprising: {(surprise_diff > 0).mean()*100:.1f}%")
t3, p3 = stats.ttest_rel(ev_surprise, ctrl_surprise)
print(f"Paired t-test:      t={t3:.3f}, p={p3:.2e}")

# --- Directional accuracy ---
# Does the model get the DIRECTION of change wrong more often for EV?
ev_actual_dir = np.sign(vol[matched_ev, 12] - vol[matched_ev, 11])
ctrl_actual_dir = np.sign(vol[matched_ctrl, 12] - vol[matched_ctrl, 11])
# LastValue always predicts "no change" (direction = 0)
ev_dir_wrong = (ev_actual_dir != 0).mean()  
ctrl_dir_wrong = (ctrl_actual_dir != 0).mean()

print(f"\n========== DIRECTIONAL CHANGE FREQUENCY ==========")
print(f"EV: non-zero direction at boundary: {ev_dir_wrong*100:.1f}%")
print(f"Ctrl: non-zero direction at boundary: {ctrl_dir_wrong*100:.1f}%")

ev_drop = (ev_actual_dir < 0).mean()
ctrl_drop = (ctrl_actual_dir < 0).mean()
print(f"EV: volume DROP at boundary: {ev_drop*100:.1f}%")
print(f"Ctrl: volume DROP at boundary: {ctrl_drop*100:.1f}%")

# --- Output volatility comparison ---
ev_out_std = vol[matched_ev, 12:].std(axis=1)
ctrl_out_std = vol[matched_ctrl, 12:].std(axis=1)
vol_diff = ev_out_std - ctrl_out_std

print(f"\n========== OUTPUT VOLATILITY (matched) ==========")
print(f"EV output std:      {ev_out_std.mean():.3f}")
print(f"Ctrl output std:    {ctrl_out_std.mean():.3f}")
print(f"Difference:         {vol_diff.mean():.3f}")
print(f"EV more volatile:   {(vol_diff > 0).mean()*100:.1f}%")
t4, p4 = stats.ttest_rel(ev_out_std, ctrl_out_std)
print(f"Paired t-test:      t={t4:.3f}, p={p4:.2e}")

# --- Relative volatility (normalized) ---
ev_rel_vol = ev_out_std / (ev_targets.mean(axis=1) + 1e-6)
ctrl_rel_vol = ctrl_out_std / (ctrl_targets.mean(axis=1) + 1e-6)
rel_diff = ev_rel_vol - ctrl_rel_vol

print(f"\n========== RELATIVE VOLATILITY (std/mean, matched) ==========")
print(f"EV CoV:             {ev_rel_vol.mean():.4f}")
print(f"Ctrl CoV:           {ctrl_rel_vol.mean():.4f}")
print(f"Difference:         {rel_diff.mean():.4f}")
print(f"EV higher CoV:      {(rel_diff > 0).mean()*100:.1f}%")
t5, p5 = stats.ttest_rel(ev_rel_vol, ctrl_rel_vol)
print(f"Paired t-test:      t={t5:.3f}, p={p5:.2e}")

# --- Match quality check ---
print(f"\n========== MATCH QUALITY ==========")
ev_match_vol = vol[matched_ev, :12].mean(axis=1)
ctrl_match_vol = vol[matched_ctrl, :12].mean(axis=1)
print(f"Mean input vol - EV:   {ev_match_vol.mean():.2f}")
print(f"Mean input vol - Ctrl: {ctrl_match_vol.mean():.2f}")
print(f"Vol ratio (EV/Ctrl):   {(ev_match_vol / (ctrl_match_vol + 1e-6)).mean():.3f}")

ev_match_h = hour[matched_ev]
ctrl_match_h = hour[matched_ctrl]
print(f"Mean hour - EV:   {ev_match_h.mean():.1f}")
print(f"Mean hour - Ctrl: {ctrl_match_h.mean():.1f}")
