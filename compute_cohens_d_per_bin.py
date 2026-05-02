import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats

d = np.load('data_preprocessing_v4/NYC_EMS_Traffic_V4.npz', allow_pickle=True)
vol = d['volume_sequences']
is_ev = d['is_ev_event']
seg_ids = d['segment_ids']

df = pd.read_csv('data_preprocessing_v4/NYC_EMS_Traffic_V4.csv', low_memory=False)
dt = pd.to_datetime(df['incident_datetime'])
dow = dt.dt.dayofweek.values
hour = dt.dt.hour.values

input_mean_vol = vol[:, :12].mean(axis=1)
output_mean_vol = vol[:, 12:].mean(axis=1)
hour_bin = (hour // 4).astype(int)
vol_quartiles = pd.qcut(input_mean_vol, q=4, labels=False, duplicates='drop')

def make_key(i):
    return (str(seg_ids[i]), int(dow[i]), int(hour_bin[i]), int(vol_quartiles[i]))

ev_all = np.where(is_ev == 1)[0]
ctrl_all = np.where(is_ev == 0)[0]
ev_strata = defaultdict(list)
ctrl_strata = defaultdict(list)
for i in ev_all: ev_strata[make_key(i)].append(i)
for i in ctrl_all: ctrl_strata[make_key(i)].append(i)

matched_ev_A, matched_ctrl_A = [], []
for key in ev_strata:
    if key in ctrl_strata:
        n_ev = len(ev_strata[key])
        n_ctrl = len(ctrl_strata[key])
        n_match = min(n_ev, n_ctrl)
        if n_match > 0:
            np.random.seed(42)
            matched_ev_A.extend(np.random.choice(ev_strata[key], n_match, replace=False))
            np.random.seed(42)
            matched_ctrl_A.extend(np.random.choice(ctrl_strata[key], n_match, replace=False))

matched_ev_A = np.array(matched_ev_A)
matched_ctrl_A = np.array(matched_ctrl_A)

all_matched = np.concatenate([matched_ev_A, matched_ctrl_A])
is_ev_matched = np.concatenate([np.ones(len(matched_ev_A)), np.zeros(len(matched_ctrl_A))])
np.random.seed(42)
perm = np.random.permutation(len(all_matched))
all_matched = all_matched[perm]
is_ev_matched = is_ev_matched[perm]

n = len(all_matched)
n_train = int(0.7 * n)
n_val = int(0.85 * n)
test_idx = all_matched[n_val:]
test_ev = is_ev_matched[n_val:]

x_test = vol[test_idx, :12]
y_test = vol[test_idx, 12:]
lv_pred = np.repeat(x_test[:, -1:], 12, axis=1)
lv_err = np.abs(y_test - lv_pred)
lv_sample_mae = lv_err.mean(axis=1)

out_vols = output_mean_vol[test_idx]
bins = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 9999)]

print("Cohen's d per volume bin on PROPENSITY MATCHED test set:")
for b_min, b_max in bins:
    mask_bin = (out_vols >= b_min) & (out_vols < b_max)
    idx_bin = np.where(mask_bin)[0]
    
    ev_mask_bin = test_ev[idx_bin] == 1
    ctrl_mask_bin = test_ev[idx_bin] == 0
    
    ev_maes = lv_sample_mae[idx_bin][ev_mask_bin]
    ctrl_maes = lv_sample_mae[idx_bin][ctrl_mask_bin]
    
    if len(ev_maes) == 0 or len(ctrl_maes) == 0:
        d_cohen = np.nan
        var_ev = 0
        var_ctrl = 0
    else:
        var_ev = ev_maes.var(ddof=1) if len(ev_maes) > 1 else 0
        var_ctrl = ctrl_maes.var(ddof=1) if len(ctrl_maes) > 1 else 0
        pooled_std = np.sqrt((var_ev + var_ctrl)/2)
        if pooled_std == 0: d_cohen = 0
        else: d_cohen = (ev_maes.mean() - ctrl_maes.mean()) / pooled_std
        
    print(f"Bin {b_min}-{b_max}: EV_N={len(ev_maes)}, Ctrl_N={len(ctrl_maes)}, Cohen's d={d_cohen:.2f}")

