#!/usr/bin/env python3
"""
qlearning_traffic_controller.py

Usage:
    python qlearning_traffic_controller.py --input /path/to/sumo.csv --out_plan /path/to/qlearning_signal_plan.csv --out_policy /path/to/qlearning_state_policy.csv

Dependencies:
    pip install pandas numpy
"""

import argparse
import pandas as pd
import numpy as np

EPS = 1e-9

def preprocess(df, bin_seconds=10):
    if "dateandtime" in df.columns:
        df["dateandtime"] = pd.to_datetime(df["dateandtime"], errors="coerce")
    else:
        raise ValueError("CSV missing 'dateandtime' column")
    df = df.dropna(subset=["dateandtime"]).copy()
    df["intersection_id"] = df.get("nextTLS", df.get("edge", df.index.astype(str))).astype(str)
    df["time_bin"] = (df["dateandtime"].astype("int64") // 10**9 // bin_seconds) * bin_seconds
    df["time_bin"] = pd.to_datetime(df["time_bin"], unit="s")
    for c in ["spd","displacement","turnAngle","tl_phase_duration"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def aggregate(df):
    grp = df.groupby(["intersection_id","time_bin"], as_index=False).agg(
        vehicle_count=("vehid","nunique"),
        avg_speed=("spd","mean"),
        tl_phase_duration=("tl_phase_duration","mean"),
    ).fillna(0)
    grp["current_green"] = grp["tl_phase_duration"].replace(0, pd.NA).fillna(10)
    return grp

def compute_congestion(grp):
    eps = 1e-9
    cnt = grp["vehicle_count"]
    spd = grp["avg_speed"]
    cnt_n = (cnt - cnt.min()) / (cnt.max() - cnt.min() + eps)
    spd_n = (spd - spd.min()) / (spd.max() - spd.min() + eps)
    grp["congestion_score"] = (cnt_n * (1 - spd_n)).fillna(0.0)
    grp["cnt_norm"] = cnt_n
    grp["spd_norm"] = spd_n
    def discretize(x):
        if x < 0.33: return 0
        if x < 0.67: return 1
        return 2
    grp["dens_lvl"] = grp["cnt_norm"].apply(discretize)
    grp["spd_lvl"] = grp["spd_norm"].apply(discretize)
    return grp

def train_qlearning(grp, actions=np.array([-5,0,5]), alpha=0.2, gamma=0.9, epsilon=0.2, passes=20):
    # Build sorted sequence per intersection (time order)
    seq = grp.sort_values(["intersection_id","time_bin"]).reset_index(drop=True).copy()
    # build next index mapping within each intersection: last maps to itself
    def next_idx_for_group(g):
        idxs = g.index.tolist()
        if len(idxs) == 1:
            return pd.Series([idxs[0]], index=idxs)
        nxt = idxs[1:] + [idxs[-1]]
        return pd.Series(nxt, index=idxs)
    seq["abs_next_idx"] = seq.groupby("intersection_id").apply(next_idx_for_group).reset_index(level=0,drop=True)
    # Q-table initialize
    states = [(d,s) for d in [0,1,2] for s in [0,1,2]]
    Q = {st: {int(a): 0.0 for a in actions} for st in states}
    rng = np.random.RandomState(0)
    for _ in range(passes):
        for i, row in seq.iterrows():
            s = (int(row["dens_lvl"]), int(row["spd_lvl"]))
            if rng.rand() < epsilon:
                a = int(rng.choice(actions))
            else:
                a = int(max(Q[s], key=Q[s].get))
            j = int(row["abs_next_idx"])
            next_row = seq.loc[j]
            r = -float(next_row["congestion_score"])
            s_next = (int(next_row["dens_lvl"]), int(next_row["spd_lvl"]))
            Q[s][a] += alpha * (r + gamma * max(Q[s_next].values()) - Q[s][a])
    # derive policy
    policy_rows = []
    for st in sorted(Q.keys()):
        best_a = int(max(Q[st], key=Q[st].get))
        policy_rows.append({"dens_lvl": st[0], "spd_lvl": st[1], "best_delta": best_a, "q_value": Q[st][best_a]})
    policy_df = pd.DataFrame(policy_rows).sort_values(["dens_lvl","spd_lvl"])
    return Q, policy_df, seq

def apply_policy_to_group(grp, Q):
    def pick_action(d,s):
        return int(max(Q[(int(d),int(s))], key=Q[(int(d),int(s))].get))
    grp = grp.copy()
    grp["qlearn_delta"] = grp.apply(lambda r: pick_action(r["dens_lvl"], r["spd_lvl"]), axis=1)
    grp["suggested_green_qlearn"] = (grp["current_green"] + grp["qlearn_delta"]).clip(lower=5)
    return grp

def main(input_path, output_plan_path, output_policy_path, bin_seconds=10):
    df = pd.read_csv(input_path)
    df = preprocess(df, bin_seconds=bin_seconds)
    grp = aggregate(df)
    grp = compute_congestion(grp)
    Q, policy_df, seq = train_qlearning(grp, passes=25)
    applied = apply_policy_to_group(grp, Q)
    out = applied[["intersection_id","time_bin","vehicle_count","avg_speed","congestion_score",
                   "current_green","qlearn_delta","suggested_green_qlearn"]]
    out.to_csv(output_plan_path, index=False)
    policy_df.to_csv(output_policy_path, index=False)
    print(f"Q-learning plan saved to: {output_plan_path}")
    print(f"Q-learning state policy saved to: {output_policy_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_plan", required=True)
    ap.add_argument("--out_policy", required=True)
    ap.add_argument("--bin", type=int, default=10)
    args = ap.parse_args()
    main(args.input, args.out_plan, args.out_policy, bin_seconds=args.bin)
