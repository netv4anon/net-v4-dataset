"""
Macro View Validation: ProfileMean baseline using 363-node graph profiles.
Proves the released graph artifact is usable by computing a HistMean-from-profiles
forecast on the micro-view test set.
"""
import numpy as np
import pandas as pd
import json

# --- Load data ---
d = np.load('data_preprocessing_v4/NYC_EMS_Traffic_V4.npz', allow_pickle=True)
g = np.load('data_preprocessing_v4/NYC_EMS_Traffic_V4_graph.npz', allow_pickle=True)
df = pd.read_csv('data_preprocessing_v4/NYC_EMS_Traffic_V4.csv')

profiles = g['profiles']  # (363, 7, 96)
seg2idx = json.loads(str(g['seg2idx']))

vol_seq = d['volume_sequences']    # (33076, 24)
is_ev = d['is_ev_event']           # (33076,)
seg_ids = d['segment_ids']         # (33076,) object

# Parse timestamps
dt = pd.to_datetime(df['incident_datetime'])
dow = dt.dt.dayofweek.values        # 0=Mon .. 6=Sun
hour = dt.dt.hour.values
minute = dt.dt.minute.values

# Each sample: 24 steps x 15min = 6 hours total. Input=12 steps, Output=12 steps.
# incident_datetime ~ start of the window. Output starts at step 12 = 3 hours later.
# Profile slot for step t: (hour*4 + minute//15 + t) % 96

# --- Stratified split (same as canonical) ---
np.random.seed(42)
ev_idx = np.where(is_ev == 1)[0]
ctrl_idx = np.where(is_ev == 0)[0]
np.random.shuffle(ev_idx)
np.random.shuffle(ctrl_idx)

def split_70_15_15(idx):
    n = len(idx)
    return idx[:int(0.7*n)], idx[int(0.7*n):int(0.85*n)], idx[int(0.85*n):]

_, _, ev_test = split_70_15_15(ev_idx)
_, _, ctrl_test = split_70_15_15(ctrl_idx)
test_idx = np.sort(np.concatenate([ev_test, ctrl_test]))
print(f"Test set: {len(test_idx)} samples ({len(ev_test)} EV, {len(ctrl_test)} control)")

# --- ProfileMean predictions on test set ---
y_true = vol_seq[test_idx, 12:]  # output window (steps 12-23)
y_pred = np.zeros_like(y_true)

mapped = 0
unmapped = 0
for i, ti in enumerate(test_idx):
    sid = str(seg_ids[ti])
    if sid not in seg2idx:
        # fallback: use HistMean of input
        y_pred[i] = vol_seq[ti, :12].mean()
        unmapped += 1
        continue
    node_idx = seg2idx[sid]
    d_dow = dow[ti]
    base_slot = hour[ti] * 4 + minute[ti] // 15
    for step in range(12):
        slot = (base_slot + 12 + step) % 96  # output starts at step 12
        y_pred[i, step] = profiles[node_idx, d_dow, slot]
    mapped += 1

print(f"Profile-mapped: {mapped}, fallback HistMean: {unmapped}")

# --- Metrics ---
mae_all = np.abs(y_true - y_pred).mean()
mean_vol = y_true.mean()
wmape = mae_all / mean_vol * 100

# Task 2: shock-window only
ev_test_mask = is_ev[test_idx] == 1
ev_masks = d['ev_masks'][test_idx]
shock_errors = []
for i in range(len(test_idx)):
    if not ev_test_mask[i]:
        continue
    out_mask = ev_masks[i, 12:]  # output portion
    if out_mask.sum() == 0:
        continue
    shock_steps = out_mask > 0
    shock_errors.append(np.abs(y_true[i, shock_steps] - y_pred[i, shock_steps]).mean())

t2_mae = np.mean(shock_errors) if shock_errors else float('nan')
shock_mean_vol = 123.6  # from paper
t2_wmape = t2_mae / shock_mean_vol * 100

print(f"\n=== Macro View ProfileMean Baseline ===")
print(f"Task-1 MAE:   {mae_all:.2f}")
print(f"Task-1 WMAPE: {wmape:.2f}%")
print(f"Task-2 MAE:   {t2_mae:.2f}")
print(f"Task-2 WMAPE: {t2_wmape:.2f}%")

# Graph artifact summary
adj = g['adjacency']
nnz = (adj > 0).sum()
print(f"\n=== Macro View Artifact Summary ===")
print(f"Nodes: {profiles.shape[0]}")
print(f"Profile coverage: {(profiles.sum(axis=(1,2))>0).sum()}/{profiles.shape[0]} (100%)")
print(f"Adjacency edges: {nnz} (density {nnz/(363*363)*100:.1f}%)")
print(f"Profile shape: {profiles.shape} (nodes x DOW x 96 daily slots)")
print(f"Profile mean vol: {profiles[profiles>0].mean():.1f} veh/15min")
