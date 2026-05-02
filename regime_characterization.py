"""
Final diagnostic: What IS genuinely different about EV samples?
Focus on what the data DOES support rather than what it doesn't.
"""
import numpy as np
import pandas as pd
from scipy import stats

d = np.load('data_preprocessing_v4/NYC_EMS_Traffic_V4.npz', allow_pickle=True)
df = pd.read_csv('data_preprocessing_v4/NYC_EMS_Traffic_V4.csv', low_memory=False)
vol = d['volume_sequences']
ev_masks = d['ev_masks']
is_ev = d['is_ev_event']
seg_ids = d['segment_ids']

np.random.seed(42)
ev_idx = np.where(is_ev == 1)[0]
ctrl_idx = np.where(is_ev == 0)[0]
np.random.shuffle(ev_idx); np.random.shuffle(ctrl_idx)
def split(idx):
    n = len(idx)
    return idx[:int(0.7*n)], idx[int(0.7*n):int(0.85*n)], idx[int(0.85*n):]
ev_tr, _, ev_te = split(ev_idx)
ctrl_tr, _, ctrl_te = split(ctrl_idx)

# === 1. Why are EV samples systematically low-volume? ===
print("=== WHY EV = LOW VOLUME? ===")
dt = pd.to_datetime(df['incident_datetime'])
hour = dt.dt.hour.values

ev_hours = hour[ev_te]
ctrl_hours = hour[ctrl_te]
print(f"EV mean hour:   {ev_hours.mean():.1f}")
print(f"Ctrl mean hour: {ctrl_hours.mean():.1f}")

# Hour distribution
for h_range, label in [((0,6), "0-6am"), ((6,12), "6am-noon"), ((12,18), "noon-6pm"), ((18,24), "6pm-midnight")]:
    ev_frac = ((ev_hours >= h_range[0]) & (ev_hours < h_range[1])).mean() * 100
    ctrl_frac = ((ctrl_hours >= h_range[0]) & (ctrl_hours < h_range[1])).mean() * 100
    print(f"  {label:>12s}: EV={ev_frac:.1f}%, Ctrl={ctrl_frac:.1f}%")

# === 2. Are EV samples on different segments? ===
print(f"\n=== SEGMENT OVERLAP ===")
ev_segs = set(str(s) for s in seg_ids[ev_te])
ctrl_segs = set(str(s) for s in seg_ids[ctrl_te])
overlap = ev_segs & ctrl_segs
print(f"EV segments:    {len(ev_segs)}")
print(f"Ctrl segments:  {len(ctrl_segs)}")
print(f"Overlap:        {len(overlap)} ({100*len(overlap)/max(len(ev_segs),1):.1f}%)")

# For overlapping segments, compare volumes
if overlap:
    ev_seg_vols = []
    ctrl_seg_vols = []
    for seg in list(overlap)[:100]:
        ev_mask = np.array([str(seg_ids[i]) == seg for i in ev_te])
        ctrl_mask = np.array([str(seg_ids[i]) == seg for i in ctrl_te])
        if ev_mask.sum() > 0 and ctrl_mask.sum() > 0:
            ev_seg_vols.append(vol[ev_te[ev_mask], :12].mean())
            ctrl_seg_vols.append(vol[ctrl_te[ctrl_mask], :12].mean())
    ev_seg_vols = np.array(ev_seg_vols)
    ctrl_seg_vols = np.array(ctrl_seg_vols)
    print(f"Same-segment EV mean vol:   {ev_seg_vols.mean():.2f}")
    print(f"Same-segment Ctrl mean vol: {ctrl_seg_vols.mean():.2f}")
    print(f"Ratio: {(ev_seg_vols / (ctrl_seg_vols+1e-6)).mean():.3f}")

# === 3. The ONE thing that IS different: input→output SHAPE CHANGE ===
print(f"\n=== SHAPE CHANGE METRICS (all test samples) ===")

# Compute trend-break: fit linear trend to input, extrapolate, measure deviation
from numpy.polynomial import polynomial as P

def trend_break(samples):
    """How much does output deviate from input-extrapolated trend?"""
    breaks = []
    for i in range(len(samples)):
        x_in = vol[samples[i], :12]
        x_out = vol[samples[i], 12:]
        # Fit linear trend to input
        t_in = np.arange(12)
        t_out = np.arange(12, 24)
        coeffs = np.polyfit(t_in, x_in, 1)
        extrapolated = np.polyval(coeffs, t_out)
        deviation = np.abs(x_out - extrapolated).mean()
        breaks.append(deviation)
    return np.array(breaks)

ev_breaks = trend_break(ev_te)
ctrl_breaks = trend_break(ctrl_te)

print(f"EV trend-break MAE:   {ev_breaks.mean():.2f}")
print(f"Ctrl trend-break MAE: {ctrl_breaks.mean():.2f}")
print(f"Ratio (EV/Ctrl):      {ev_breaks.mean()/ctrl_breaks.mean():.3f}")

# Normalize by volume level
ev_vol_mean = vol[ev_te].mean(axis=1)
ctrl_vol_mean = vol[ctrl_te].mean(axis=1)
ev_rel_break = ev_breaks / (ev_vol_mean + 1e-6)
ctrl_rel_break = ctrl_breaks / (ctrl_vol_mean + 1e-6)

print(f"\nNormalized trend-break (/ mean vol):")
print(f"EV:   {ev_rel_break.mean():.4f}")
print(f"Ctrl: {ctrl_rel_break.mean():.4f}")
print(f"Ratio: {ev_rel_break.mean()/ctrl_rel_break.mean():.3f}")

# === 4. Autocorrelation structure ===
print(f"\n=== AUTOCORRELATION (lag-1 in output window) ===")
def lag1_autocorr(indices):
    acs = []
    for i in indices:
        x = vol[i, 12:]
        if x.std() < 1e-6:
            acs.append(1.0)
            continue
        ac = np.corrcoef(x[:-1], x[1:])[0,1]
        acs.append(ac if not np.isnan(ac) else 0)
    return np.array(acs)

ev_ac = lag1_autocorr(ev_te)
ctrl_ac = lag1_autocorr(ctrl_te)
print(f"EV lag-1 autocorr:   {ev_ac.mean():.4f}")
print(f"Ctrl lag-1 autocorr: {ctrl_ac.mean():.4f}")
print(f"Difference:          {ev_ac.mean()-ctrl_ac.mean():.4f}")
t, p = stats.ttest_ind(ev_ac, ctrl_ac)
print(f"t-test: t={t:.3f}, p={p:.2e}")

# === 5. Quantify the PRACTICAL regime difference ===
print(f"\n=== REGIME CHARACTERIZATION ===")
print(f"{'Metric':30s} {'EV':>10s} {'Ctrl':>10s} {'Ratio':>8s}")
print("-" * 62)

metrics = [
    ("Mean volume (all 24 steps)", vol[ev_te].mean(), vol[ctrl_te].mean()),
    ("Output std (per-sample avg)", vol[ev_te, 12:].std(axis=1).mean(), vol[ctrl_te, 12:].std(axis=1).mean()),
    ("Input std (per-sample avg)", vol[ev_te, :12].std(axis=1).mean(), vol[ctrl_te, :12].std(axis=1).mean()),
    ("Lag-1 autocorrelation", ev_ac.mean(), ctrl_ac.mean()),
    ("Trend-break (absolute)", ev_breaks.mean(), ctrl_breaks.mean()),
    ("Trend-break (relative)", ev_rel_break.mean(), ctrl_rel_break.mean()),
    ("Boundary |delta|", np.abs(vol[ev_te,12]-vol[ev_te,11]).mean(), np.abs(vol[ctrl_te,12]-vol[ctrl_te,11]).mean()),
]

for name, ev_v, ctrl_v in metrics:
    ratio = ev_v / (ctrl_v + 1e-10)
    print(f"{name:30s} {ev_v:>10.3f} {ctrl_v:>10.3f} {ratio:>8.3f}")
