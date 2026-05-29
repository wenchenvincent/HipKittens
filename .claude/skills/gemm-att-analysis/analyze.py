#!/usr/bin/env python3
"""
Per-cluster cycle / stall / barrier-wait / XDL-utilization analysis from
an ATT (rocprofv3 --att) capture of a HipKittens GEMM kernel.

Usage:
    python3 analyze.py <stats_csv> [--waves N] [--mfma-cycles N]
                                   [--mfmas-per-cluster N]
                                   [--att-out-dir PATH]

The CSV must be the rocprofv3 `stats_ui_output_*.csv`. The analyzer:
  1. auto-detects the loop body from the Hitcount distribution
  2. segments instructions between consecutive `s_barrier`s into clusters
  3. normalizes per-cluster Latency / Stall to cyc/wave/iter
  4. estimates steady-state XDL utilization
  5. cross-checks `sum(CSV.Latency) ≈ sum(wave.duration)` from the JSONs

Read alongside `SKILL.md` for what the numbers mean.
"""
import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict


def classify(ins: str) -> str:
    if "v_mfma" in ins:
        return "mfma"
    if "buffer_load" in ins:
        return "gload"
    if "ds_read" in ins:
        return "dsread"
    if "ds_write" in ins:
        return "dswrite"
    if "s_waitcnt" in ins:
        return "waitcnt"
    if "s_barrier" in ins:
        return "barrier"
    if "s_setprio" in ins:
        return "setprio"
    return "other"


def load_rows(csv_path):
    rows = []
    with open(csv_path) as f:
        for x in csv.DictReader(f):
            try:
                rows.append(
                    (
                        int(x["Vaddr"]),
                        x["Instruction"].strip().strip(";").strip(),
                        int(x["Hitcount"]),
                        int(x["Latency"]),
                        int(x["Stall"]),
                    )
                )
            except (ValueError, KeyError):
                continue
    return rows


def detect_loop_hits(rows):
    """Loop body = the hitcount that accounts for the most total executions
    (hitcount × num_instructions). This robustly picks the loop body whether
    it has many instructions per iter (BF16 ping-pong: 4064 hits × 125 instrs
    = 508K dominates 32 × 736 = 23K) or many iterations on fewer instructions
    (FP8 4-wave: 148 hits × 4761 instrs = 705K dominates 2048 × 23 = 47K).
    Requires ≥ 5 instructions to filter outliers."""
    hc = Counter(h for _, _, h, _, _ in rows)
    candidates = [(h, hc[h]) for h in hc if h > 0 and hc[h] >= 5]
    if not candidates:
        return 0
    return max(candidates, key=lambda x: x[0] * x[1])[0]


def segment_loop(rows, loop_hit):
    """Return ordered (lo_addr_excl, hi_addr_incl, [rows]) segments split by
    s_barrier rows. Each segment ends at (and includes) a barrier row."""
    body = sorted([r for r in rows if r[2] == loop_hit], key=lambda r: r[0])
    barr_addrs = [v for v, i, _, _, _ in body if "s_barrier" in i]
    if not barr_addrs:
        return []
    bounds = [body[0][0] - 1] + barr_addrs
    segs = []
    for k in range(len(barr_addrs)):
        lo, hi = bounds[k], bounds[k + 1]
        segs.append((lo, hi, [r for r in body if lo < r[0] <= hi]))
    return segs


def summarize_segment(seg, waves):
    """Aggregate one cluster's rows; normalize to cyc/wave/iter."""
    _, _, rows = seg
    if not rows:
        return None
    hits = rows[0][2]  # uniform across loop body
    iters = hits // waves if waves else 1
    per_class_lat = defaultdict(int)
    per_class_stall = defaultdict(int)
    seg_lat = seg_stall = 0
    barr_lat = barr_stall = 0
    for _, ins, _, lat, stall in rows:
        cl = classify(ins)
        per_class_lat[cl] += lat
        per_class_stall[cl] += stall
        seg_lat += lat
        seg_stall += stall
        if cl == "barrier":
            barr_lat += lat
            barr_stall += stall
    denom = waves * iters
    return {
        "iters": iters,
        "instrs": len(rows),
        "addrs": (rows[0][0], rows[-1][0]),
        "total_cyc_per_iter": seg_lat / denom,
        "total_stall_per_iter": seg_stall / denom,
        "barrier_wait_per_iter": barr_stall / denom,
        "by_class_cyc": {
            cl: per_class_lat[cl] / denom for cl in per_class_lat if per_class_lat[cl]
        },
    }


def find_att_dir(csv_path, override):
    if override:
        return override
    # CSV typically sits in the same directory as the per-wave JSONs' parent
    d = os.path.dirname(os.path.abspath(csv_path)) or "."
    return d


def duration_ratio(csv_path, att_dir):
    """Empirical: sum(CSV.Latency) / sum(wave.duration). ~0.93–1.00 if column maps correctly."""
    lat = 0
    with open(csv_path) as f:
        for x in csv.DictReader(f):
            try:
                lat += int(x["Latency"])
            except (ValueError, KeyError):
                pass
    wave_files = glob.glob(os.path.join(att_dir, "ui_output_*", "se*_sm*_sl*_wv*.json"))
    durs = []
    for w in wave_files:
        try:
            with open(w) as f:
                d = json.load(f)
                if isinstance(d, dict) and "duration" in d:
                    durs.append(int(d["duration"]))
        except (ValueError, OSError, json.JSONDecodeError):
            pass
    if not durs:
        return None, len(wave_files)
    return lat / sum(durs), len(wave_files)


def render(report, args):
    print(f"Loop body autodetect: hitcount={report['loop_hit']}, "
          f"iterations={report['iters']} (assuming {args.waves} waves)")
    print(f"  (override --waves if the assumed wave count is wrong)\n")

    print(f"{'cluster':>8} {'addrs':>14} {'instrs':>6} {'total':>9} "
          f"{'stall':>9} {'barr_wait':>9}  per-class cyc/iter")
    for i, s in enumerate(report["segments"]):
        if s is None:
            continue
        cls_str = " ".join(
            f"{n}={s['by_class_cyc'].get(n, 0):.0f}"
            for n in ("mfma", "dsread", "gload", "waitcnt")
            if s["by_class_cyc"].get(n, 0)
        )
        addrs = f"{s['addrs'][0]}-{s['addrs'][1]}"
        print(
            f"  seg{i}  {addrs:>14} {s['instrs']:6} "
            f"{s['total_cyc_per_iter']:9.1f} {s['total_stall_per_iter']:9.1f} "
            f"{s['barrier_wait_per_iter']:9.1f}  {cls_str}"
        )

    # XDL utilization estimate
    mma_segs = [s for s in report["segments"]
                if s and s["by_class_cyc"].get("mfma", 0) > 0]
    if mma_segs:
        n_mma = len(mma_segs)
        xdl_busy = n_mma * 2 * args.mfmas_per_cluster * args.mfma_cycles  # 2 halves
        wall_clock = sum(s["total_cyc_per_iter"] for s in report["segments"] if s)
        util = xdl_busy / wall_clock if wall_clock else 0
        print(
            f"\nSteady-state XDL utilization estimate:\n"
            f"  MMA clusters: {n_mma}  (both halves contribute under ping-pong)\n"
            f"  XDL busy / K-step: {n_mma} clusters × 2 halves × "
            f"{args.mfmas_per_cluster} mfmas × {args.mfma_cycles} cyc = {xdl_busy} cyc\n"
            f"  K-step wall-clock (sum of cluster cyc/wave/iter): {wall_clock:.0f} cyc\n"
            f"  ⇒ XDL utilization ≈ {util * 100:.1f}%"
        )

    if report["lat_dur_ratio"] is not None:
        r = report["lat_dur_ratio"]
        flag = "ok" if 0.90 <= r <= 1.02 else "OUTSIDE 0.90-1.02 — check column semantics"
        print(
            f"\nSanity check: sum(CSV.Latency) / sum(wave.duration) = {r:.4f}  [{flag}]"
            f"  ({report['n_wave_files']} wave JSONs)"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="rocprofv3 stats_ui_output_*.csv")
    ap.add_argument("--waves", type=int, default=None,
                    help="number of waves traced (default: infer from hitcount/iter heuristic)")
    ap.add_argument("--mfma-cycles", type=int, default=32,
                    help="MFMA cycle count per issue (default 32 for v_mfma_f32_32x32x16_bf16)")
    ap.add_argument("--mfmas-per-cluster", type=int, default=16,
                    help="number of mfmas in one MMA cluster (default 16 for 32x16 HK GEMM)")
    ap.add_argument("--att-out-dir", default=None,
                    help="directory holding the per-wave ui_output_*/se*_wv*.json files "
                         "(default: same dir as the CSV)")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    loop_hit = detect_loop_hits(rows)
    if loop_hit == 0:
        print("No loop body detected (no positive Hitcount). Wrong CSV?")
        return 1

    # If waves not given, assume the loop body hit = waves * iters and the
    # iter count is the most common "fully traced" pass count. Reasonable
    # default for one-CU captures: 32 waves.
    if args.waves is None:
        args.waves = 32
    iters = loop_hit // args.waves

    segs_raw = segment_loop(rows, loop_hit)
    segs = [summarize_segment(s, args.waves) for s in segs_raw]
    att_dir = find_att_dir(args.csv, args.att_out_dir)
    ratio, n_waves = duration_ratio(args.csv, att_dir)

    report = {
        "loop_hit": loop_hit,
        "iters": iters,
        "segments": segs,
        "lat_dur_ratio": ratio,
        "n_wave_files": n_waves,
    }
    render(report, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
