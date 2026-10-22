"""STATS 逐轮更新的 DAUCE 检测适配器。

按声明列类型加载原始 CSV，使用与数据库相同的日期边界及批次顺序，
校准检测阈值并逐轮返回漂移决策。该模块只读数据，不操作 PostgreSQL。
"""
import os
import sys
import json
import time
import argparse
import warnings

import numpy as np
import pandas as pd

# drift/ lives at the DAUCE repo root (one level up from FactorJoin/).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
import drift.schema as dschema            # noqa: E402
from drift import build_detector          # noqa: E402

warnings.filterwarnings("ignore")

SPLIT_DATE = "2014-01-01 00:00:00"
START_DATE = "2010-07-19 00:00:00"
DATA_FOLDER = os.path.abspath(os.path.join(_HERE, "..", "data", "stats"))
OUT_FOLDER = os.path.join(_HERE, "dauce_offline")
ROUNDS = 7
TAU = 0.05
# Same 8 tables / order as gen_stats_light_schema; "tags" has no Date column and
# therefore no after-split delta in the e2e (never updated).
TABLES = ["badges", "votes", "postHistory", "posts", "users",
          "comments", "postLinks", "tags"]


def timestamp_transorform(time_string, start_date=START_DATE):
    """Byte-identical to Evaluation.updating.timestamp_transorform: int seconds
    since START_DATE. Replicated inline so this script stays free of any
    FactorJoin/torch import."""
    start = time.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    t = time.strptime(time_string, "%Y-%m-%d %H:%M:%S")
    return int(time.mktime(t)) - int(time.mktime(start))


def build_matrix(tn):
    """float64 matrix in declared column order + declared kinds. Row order == CSV
    file order so before/after split and iloc chunking match run_e2e_dauce."""
    key = "stats_" + tn
    names = dschema.column_names(key)
    if names is None:
        raise ValueError(f"no declared schema for {key}")
    df = pd.read_csv(os.path.join(DATA_FOLDER, f"{tn}.csv"))
    if set(df.columns) != set(names):
        raise ValueError(f"{tn}: CSV columns {list(df.columns)} != declared {names}")
    for c in names:
        if "Date" in c:
            df[c] = [timestamp_transorform(v) for v in df[c].values]
    M = df[names].to_numpy(dtype=np.float64)
    kinds = dschema.declared_kinds(key, n_cols=len(names))
    return names, kinds, M


def date_index(names):
    """First column whose name contains 'Date' (mirrors run_e2e_dauce.date_mask)."""
    for j, n in enumerate(names):
        if "Date" in n:
            return j
    return None


def split_and_chunk(M, names, rounds=ROUNDS, batch_divisor=0):
    """(reference=before-split, after, batches[rounds], batch_size).

    Two modes:
      * batch_divisor == 0 (default, mirrors run_e2e_dauce): batch = len(after)//
        rounds and the LAST round absorbs the remainder, so the `rounds` batches
        together insert the ENTIRE after-split delta.
      * batch_divisor == K > 0 (DECOUPLED): a fixed per-round batch
        B = min(len(after)//rounds, n_ref//K), uniform across rounds (no remainder
        absorption). The `rounds` batches then insert only rounds*B <= len(after)
        rows -- the update COUNT (rounds) is decoupled from the per-round DATA
        VOLUME (B). K >= 19 forces B <= n_ref/19, so the split-conformal threshold
        is EXACT (>=19 calibration blocks) even at rounds=7. The tail
        after[rounds*B:] is NOT inserted, so PG evolves only partway to the full DB
        (all arms share that evolution, so the latency comparison stays fair).

    Tables with no Date column (tags) get an empty delta -> no rounds.
    """
    dj = date_index(names)
    if dj is None:
        return M, M[:0], [], 0
    split_val = timestamp_transorform(SPLIT_DATE)
    mask = M[:, dj] >= split_val
    reference = M[~mask]
    after = M[mask]
    if batch_divisor and batch_divisor > 0:
        B = max(1, min(len(after) // rounds, len(reference) // batch_divisor))
        batches = [after[i * B:(i + 1) * B] for i in range(rounds)]
        return reference, after, batches, B
    ch = len(after) // rounds
    batches = [after[i * ch:] if i == rounds - 1 else after[i * ch:(i + 1) * ch]
               for i in range(rounds)]
    return reference, after, batches, ch


def run_detector(kind, reference, kinds, batches, ch, tau=TAU, repeats=10, seed=0):
    """Fit + conformal-calibrate + detect every round; return a serialisable dict."""
    det = build_detector(kind)
    det.fit_reference(reference, kinds)
    t0 = time.time()
    cal = det.calibrate(reference, kinds, batch_size=ch, tau=tau,
                        repeats=repeats, random_seed=seed)
    cal_s = time.time() - t0
    rounds_out = []
    for i, batch in enumerate(batches):
        if len(batch) == 0:
            rounds_out.append({"round": i + 1, "n_batch": 0, "is_drift": None})
            continue
        current = np.vstack((reference, batch))
        res = det.detect(reference, current)
        ex = {}
        for k, v in res.extras.items():
            ex[k] = float(v) if isinstance(v, (int, float, np.floating)) else v
        rounds_out.append({
            "round": i + 1,
            "n_batch": int(len(batch)),
            "distance": float(res.distance),
            "threshold": None if res.threshold is None else float(res.threshold),
            "is_drift": None if res.is_drift is None else bool(res.is_drift),
            "latency_s": float(res.latency_s),
            "extras": ex,
        })
    return {
        "detector": kind,
        "calibrate_s": cal_s,
        "threshold": float(cal.threshold),
        "strict_threshold": float(cal.strict_threshold),
        "n_blocks": int(cal.n_blocks),
        "block_size": int(cal.block_size),
        "repeats": int(cal.repeats),
        "tau": float(cal.tau),
        "exact": bool(cal.exact),
        "min_blocks_required": int(cal.min_blocks_required),
        "notes": list(cal.notes),
        "rounds": rounds_out,
    }


def _cell(is_drift):
    if is_drift is None:
        return "-"
    return "D" if is_drift else "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=",".join(TABLES))
    ap.add_argument("--detectors", default="dauce,legacy_kl")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--batch-divisor", type=int, default=0,
                    help="K>0 decouples per-round batch B=min(len(after)//rounds, "
                         "n_ref//K) from rounds; K>=19 -> exact conformal at any rounds")
    ap.add_argument("--out", default=os.path.join(OUT_FOLDER, "detection_rounds.json"))
    args = ap.parse_args()

    tables = [t for t in args.tables.split(",") if t]
    detectors = [d for d in args.detectors.split(",") if d]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    split_val = timestamp_transorform(SPLIT_DATE)
    print(f"[cfg] split={SPLIT_DATE} (val={split_val}) rounds={args.rounds} "
          f"tau={args.tau} repeats={args.repeats} detectors={detectors}", flush=True)
    print(f"[cfg] data={DATA_FOLDER}", flush=True)

    out = {"split_date": SPLIT_DATE, "split_val": split_val, "rounds": args.rounds,
           "tau": args.tau, "repeats": args.repeats, "detectors": detectors,
           "batch_divisor": args.batch_divisor, "tables": {}}

    for tn in tables:
        names, kinds, M = build_matrix(tn)
        reference, after, batches, ch = split_and_chunk(
            M, names, args.rounds, args.batch_divisor)
        n_ref, n_after = int(len(reference)), int(len(after))
        inserted = int(sum(len(b) for b in batches))
        kind_str = ",".join(f"{n}:{k.value[:3]}" for n, k in zip(names, kinds))
        print(f"\n=== {tn} === rows={len(M)} n_ref={n_ref} n_after={n_after} "
              f"batch={ch} inserted={inserted}"
              f"({100*inserted/n_after:.0f}% of after)" if n_after else
              f"\n=== {tn} === rows={len(M)} n_ref={n_ref} n_after={n_after} batch={ch}",
              flush=True)
        print(f"    cols=[{kind_str}]", flush=True)

        nref19 = n_ref // 19 if n_ref else 0
        tinfo = {
            "n_rows": int(len(M)), "n_ref": n_ref, "n_after": n_after, "chunk": ch,
            "batch_divisor": args.batch_divisor,
            "inserted_rows": inserted,
            "inserted_frac": (inserted / n_after) if n_after else 0.0,
            "columns": names, "kinds": [k.value for k in kinds],
            "batch_constraint": {
                "chunk": ch,
                "n_ref_over_19": nref19,
                "chunk_le_nref_over_19": bool(ch and nref19 and ch <= nref19),
                "max_blocks": (n_ref // ch) if ch else 0,
            },
            "detectors": {},
        }

        if ch == 0 or n_ref == 0:
            print("    no Date column / empty delta -> detection N/A "
                  "(matches e2e: table never updated)", flush=True)
            tinfo["skipped"] = "no delta (no Date column) or empty"
            out["tables"][tn] = tinfo
            continue

        for dkind in detectors:
            t0 = time.time()
            r = run_detector(dkind, reference, kinds, batches, ch,
                             tau=args.tau, repeats=args.repeats)
            tinfo["detectors"][dkind] = r
            print(f"    [{dkind}] calibrate {r['calibrate_s']:.1f}s "
                  f"thr={r['threshold']:.4e} strict={r['strict_threshold']:.4e} "
                  f"K={r['n_blocks']}/{r['min_blocks_required']} exact={r['exact']} "
                  f"total={time.time()-t0:.1f}s", flush=True)
            for ro in r["rounds"]:
                if ro["is_drift"] is None:
                    print(f"        r{ro['round']}: empty batch", flush=True)
                    continue
                ex = ro["extras"]
                exs = " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in ex.items())
                print(f"        r{ro['round']}: dist={ro['distance']:.4e} "
                      f"thr={ro['threshold']:.4e} drift={ro['is_drift']} "
                      f"n={ro['n_batch']} {exs}", flush=True)
        out["tables"][tn] = tinfo

    # ---------------- summary grid ---------------- #
    print("\n" + "=" * 72, flush=True)
    print("SUMMARY  (D=drift, .=no-drift, -=empty/no-delta)", flush=True)
    grid = {}
    for dkind in detectors:
        print(f"\n[{dkind}]", flush=True)
        print("table".ljust(14) + "".join(f"r{i+1}".rjust(5) for i in range(args.rounds))
              + "   n_drift   conformal", flush=True)
        tot_d = tot_c = 0
        for tn in tables:
            r = out["tables"][tn]["detectors"].get(dkind)
            if r is None:
                print(tn.ljust(14) + "  (no delta)", flush=True)
                grid.setdefault(dkind, {})[tn] = None
                continue
            cells = "".join(_cell(ro["is_drift"]).rjust(5) for ro in r["rounds"])
            nd = sum(1 for ro in r["rounds"] if ro["is_drift"] is True)
            nc = sum(1 for ro in r["rounds"] if ro["is_drift"] is not None)
            tot_d += nd
            tot_c += nc
            grid.setdefault(dkind, {})[tn] = [ro["is_drift"] for ro in r["rounds"]]
            print(tn.ljust(14) + cells + f"   {nd}/{nc}".rjust(8)
                  + f"     exact={r['exact']} K={r['n_blocks']}", flush=True)
        print(f"  TOTAL drift cells: {tot_d}/{tot_c}", flush=True)

    # ---------------- differentiation verdict ---------------- #
    if "dauce" in detectors and "legacy_kl" in detectors:
        print("\n" + "=" * 72, flush=True)
        print("DIFFERENTIATION (dauce vs legacy_kl, identical harness)", flush=True)
        diff_cells = 0
        for tn in tables:
            gd = grid.get("dauce", {}).get(tn)
            gl = grid.get("legacy_kl", {}).get(tn)
            if not gd or not gl:
                continue
            row = [f"r{i+1}:{'D' if gd[i] else '.'}/{'D' if gl[i] else '.'}"
                   for i in range(len(gd)) if gd[i] != gl[i]]
            if row:
                diff_cells += len(row)
                print(f"  {tn}: differ at " + ", ".join(row), flush=True)
        if diff_cells == 0:
            print("  NO differentiation: dauce and legacy_kl agree on every cell.",
                  flush=True)
        else:
            print(f"  dauce differs from legacy_kl on {diff_cells} table-round cell(s).",
                  flush=True)

    # ---------------- conformal batch-constraint recap ---------------- #
    print("\n" + "=" * 72, flush=True)
    print("CONFORMAL BATCH CONSTRAINT  (exact needs chunk <= n_ref/19, i.e. K>=19)",
          flush=True)
    for tn in tables:
        ti = out["tables"][tn]
        bc = ti["batch_constraint"]
        if ti.get("skipped"):
            print(f"  {tn:<12} skipped ({ti['skipped']})", flush=True)
            continue
        r = ti["detectors"].get(detectors[0])
        print(f"  {tn:<12} chunk={bc['chunk']:<6} n_ref/19={bc['n_ref_over_19']:<6} "
              f"max_blocks={bc['max_blocks']:<4} chunk<=n_ref/19={bc['chunk_le_nref_over_19']} "
              f"-> exact={r['exact']} K={r['n_blocks']}", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[out] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
