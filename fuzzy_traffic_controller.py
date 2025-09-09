#!/usr/bin/env python3
"""
fuzzy_traffic_controller.py

Usage:
    python fuzzy_traffic_controller.py --input /path/to/sumo.csv --output /path/to/fuzzy_signal_plan.csv

Dependencies:
    pip install pandas numpy
"""

import argparse
import pandas as pd
import numpy as np

EPS = 1e-9

def tri(x, a, b, c):
    return np.maximum(np.minimum((x - a) / (b - a + EPS), (c - x) / (c - b + EPS)), 0.0)

def fuzzify_density(d):
    return {
        "low": tri(d, 0.0, 0.0, 0.4),
        "med": tri(d, 0.2, 0.5, 0.8),
        "high": tri(d, 0.6, 1.0, 1.0),
    }

def fuzzify_speed(s):
    return {
        "low": tri(s, 0.0, 0.0, 0.4),
        "med": tri(s, 0.2, 0.5, 0.8),
        "high": tri(s, 0.6, 1.0, 1.0),
    }

OUTPUT_VALUES = {"dec": -5.0, "keep": 0.0, "inc": 5.0}

def fuzzy_controller(cnt_n, spd_n):
    D = fuzzify_density(cnt_n)
    S = fuzzify_speed(spd_n)
    rules = []
    rules.append(("inc", min(D["high"], S["low"])))
    rules.append(("inc", min(D["high"], S["med"])))
    rules.append(("inc", min(D["med"], S["low"])))
    rules.append(("keep", min(D["med"], S["med"])))
    rules.append(("dec", min(D["low"], S["high"])))
    rules.append(("keep", max(D["low"]*S["med"], D["med"]*S["high"])))
    num = 0.0
    den = 0.0
    for out, w in rules:
        num += OUTPUT_VALUES[out] * w
        den += w
    return num / (den + EPS)

def preprocess(df, bin_seconds=10):
    # Parse date
    if "dateandtime" in df.columns:
        df["dateandtime"] = pd.to_datetime(df["dateandtime"], errors="coerce")
    else:
        raise ValueError("CSV missing 'dateandtime' column")

    df = df.dropna(subset=["dateandtime"]).copy()
    df["intersection_id"] = df.get("nextTLS", df.get("edge", df.index.astype(str))).astype(str)
    # time bin
    df["time_bin"] = (df["dateandtime"].astype("int64") // 10**9 // bin_seconds) * bin_seconds
    df["time_bin"] = pd.to_datetime(df["time_bin"], unit="s")
    numeric_cols = ["spd", "displacement", "turnAngle", "tl_phase_duration", "tl_next_switch"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def aggregate(df):
    grp = df.groupby(["intersection_id", "time_bin"], as_index=False).agg(
        vehicle_count=("vehid", "nunique"),
        avg_speed=("spd", "mean"),
        med_speed=("spd", "median"),
        mean_displacement=("displacement", "mean"),
        mean_turnAngle=("turnAngle", "mean"),
        tl_phase_duration=("tl_phase_duration", "mean"),
    )
    grp.fillna(0, inplace=True)
    return grp

def compute_congestion_and_apply_fuzzy(grp):
    # Normalize vehicle_count and avg_speed to [0,1]
    eps = 1e-9
    cnt = grp["vehicle_count"]
    spd = grp["avg_speed"]
    cnt_n = (cnt - cnt.min()) / (cnt.max() - cnt.min() + eps)
    spd_n = (spd - spd.min()) / (spd.max() - spd.min() + eps)
    grp["congestion_score"] = (cnt_n * (1 - spd_n)).fillna(0.0)
    grp["current_green"] = grp["tl_phase_duration"].replace(0, pd.NA).fillna(10)

    grp["fuzzy_delta"] = [fuzzy_controller(c, s) for c, s in zip(cnt_n, spd_n)]
    grp["fuzzy_delta"] = grp["fuzzy_delta"].clip(-10, 10)
    grp["suggested_green_fuzzy"] = (grp["current_green"] + grp["fuzzy_delta"]).clip(lower=5)
    return grp

def main(input_path, output_path, bin_seconds=10):
    df = pd.read_csv(input_path)
    df = preprocess(df, bin_seconds=bin_seconds)
    grp = aggregate(df)
    grp = compute_congestion_and_apply_fuzzy(grp)
    out = grp[["intersection_id","time_bin","vehicle_count","avg_speed","congestion_score",
               "current_green","fuzzy_delta","suggested_green_fuzzy"]]
    out.to_csv(output_path, index=False)
    print(f"Fuzzy planner saved to: {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--bin", type=int, default=10, help="aggregation bin size in seconds (default 10)")
    args = ap.parse_args()
    main(args.input, args.output, bin_seconds=args.bin)
