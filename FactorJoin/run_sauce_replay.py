"""在相同逐轮数据批次上运行 DAUCE 门控更新并测量查询时延。

复用已完成的 Always/None 实验输入，重新装载独立 PostgreSQL 数据库，
运行时生成 DAUCE 检测决策。数据库重载前检查上一阶段已完成。
"""
import os
import sys
import json
import time
import copy
import shutil
import pickle
import argparse
import subprocess
import warnings

sys.path.append("./")
warnings.filterwarnings("ignore")

import run_e2e_dauce as E      # reuse all helpers/constants (import is __main__-guarded)
import dauce_adapter as da     # DauceDetector split-conformal decision engine

log = E.log
MODEL_ROOT = E.MODEL_ROOT
PG_FOLDER = E.PG_FOLDER
LATENCY_FOLDER = E.LATENCY_FOLDER
RESULTS_FOLDER = E.RESULTS_FOLDER
CONTAINER_RESULTS = E.CONTAINER_RESULTS
MODEL_NAME = E.MODEL_NAME
SPLIT_DATE = E.SPLIT_DATE
ARM_TAG = "sauce_dauce"
CKPT_TAG = "sauce_replay"


# ----------------------------------------------------------------------------
# safety guard: never reset PG while the full run is alive / incomplete
# ----------------------------------------------------------------------------
def guard_pg_free(rounds):
    out = subprocess.run(["pgrep", "-af", "run_e2e_dauce.py"],
                         capture_output=True, text=True).stdout.strip()
    if out:
        log("ABORT: run_e2e_dauce.py is still alive (PG in use). Wait for it to "
            "finish before replaying (this script TRUNCATEs PG).\n" + out)
        sys.exit(3)
    native_r = f"{LATENCY_FOLDER}exec_time_PG_native_naive_naive_iter{rounds}.npy"
    if not os.path.exists(native_r):
        log(f"ABORT: full-run round-{rounds} native latency missing ({native_r}); "
            f"the full run is not complete -> refusing to reset PG.")
        sys.exit(3)
    log(f"  safety guard OK: no run_e2e_dauce alive; full-run round-{rounds} "
        f"native latency present -> PG is free to reset.")


# ----------------------------------------------------------------------------
# DauceDetector per-table per-round decisions (offline, no PG)
# ----------------------------------------------------------------------------
def compute_dauce_decisions(rounds, tau, repeats, batch_divisor):
    """{table: [bool x rounds]} drift verdicts on the same batches the e2e inserts.

    Uses dauce_adapter's coupled (batch_divisor=0) chunking, which is row-identical
    to run_e2e_dauce's per-round db_slice (same CSV, same Date>=split mask, same
    len(after)//rounds chunking). tags has no Date column -> no delta -> never fires.
    """
    decisions, detail = {}, {}
    for tn in da.TABLES:
        names, kinds, M = da.build_matrix(tn)
        ref, after, batches, ch = da.split_and_chunk(M, names, rounds, batch_divisor)
        if not batches:
            decisions[tn] = [False] * rounds
            detail[tn] = {"skipped": "no delta (no Date column)"}
            log(f"  [dauce] {tn:12s}: no delta -> never update")
            continue
        res = da.run_detector("dauce", ref, kinds, batches, ch,
                              tau=tau, repeats=repeats, seed=0)
        decisions[tn] = [bool(r.get("is_drift")) for r in res["rounds"]]
        detail[tn] = res
        nd = sum(decisions[tn])
        log(f"  [dauce] {tn:12s}: batch={ch:6d} drift {nd}/{rounds} "
            f"thr={res['threshold']:.3e} exact={res['exact']} n_blocks={res['n_blocks']}")
    return decisions, detail


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--smoke", action="store_true", help="仅运行前 K 条查询")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--batch-divisor", type=int, default=0,
                    help="0=coupled batch, reuses the full-run PG evolution "
                         "(RECOMMENDED, fair vs always/native). K>0 shrinks the "
                         "per-round batch to n_ref//K -> changes PG data, so "
                         "always/native are NO LONGER comparable (standalone use).")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip_align_gate", action="store_true")
    ap.add_argument("--skip_safety_guard", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("PG_USER", "postgres")
    os.environ.setdefault("PG_PASSWORD", "postgres")
    os.environ.setdefault("PG_HOST", "localhost")
    os.environ.setdefault("PG_PORT", "5432")
    for d in [MODEL_ROOT, PG_FOLDER, LATENCY_FOLDER, RESULTS_FOLDER,
              PG_FOLDER + "/update", PG_FOLDER + "/before"]:
        os.makedirs(d, exist_ok=True)

    log(f"=== SAUCE-ONLY REPLAY (DauceDetector gate) rounds={args.rounds} "
        f"tau={args.tau} repeats={args.repeats} batch_divisor={args.batch_divisor} "
        f"resume={args.resume} ===")
    if args.batch_divisor:
        log("  WARNING: batch_divisor>0 changes the PG data evolution; the full "
            "run's always/native arms are NOT comparable. Use 0 for the fair replay.")

    if not args.skip_safety_guard:
        log("--- Stage 0: safety guard ---")
        guard_pg_free(args.rounds)

    schema, origin, processed = E.load_data()
    split_val = E.timestamp_transorform(SPLIT_DATE)
    query_file, sub_query_file, n_q, n_sp = E.prepare_workload(args.smoke, args.k)
    log(f"  workload: {n_q} queries, {n_sp} sub-plans; query_file={query_file}")

    # ---- Stage D: DauceDetector decisions (offline, before any PG mutation) ----
    log("--- Stage D: DauceDetector split-conformal decisions ---")
    decisions, detail = compute_dauce_decisions(
        args.rounds, args.tau, args.repeats, args.batch_divisor)
    json.dump({"rounds": args.rounds, "tau": args.tau, "repeats": args.repeats,
               "batch_divisor": args.batch_divisor, "decisions": decisions,
               "detail": detail},
              open(f"{PG_FOLDER}/{CKPT_TAG}_decisions.json", "w"),
              indent=2, default=str)
    tot = sum(sum(v) for v in decisions.values())
    denom = args.rounds * 7
    log(f"  DAUCE gate: {tot} table-round updates flagged "
        f"(always baseline = {denom}); saving {100 * (denom - tot) / denom:.0f}%")
    fired_tables = [tn for tn in decisions if any(decisions[tn])]
    never_tables = [tn for tn in decisions if not any(decisions[tn])]
    log(f"  tables that EVER fire: {fired_tables}")
    log(f"  tables that NEVER fire: {never_tables}")

    # ---- resume / checkpoint ----
    ck = E.load_ckpt(CKPT_TAG) if args.resume else None
    start_round = 0
    if ck and ck.get("n_subplans") == n_sp and ck.get("rounds") == args.rounds:
        start_round = ck["last_completed_round"] + 1
        log(f"  RESUME from round {start_round} (last_completed={ck['last_completed_round']})")

    sauce_folder = f"{MODEL_ROOT}/{ARM_TAG}"
    base_folder = f"{MODEL_ROOT}/base"

    if start_round == 0:
        log("--- Stage A: reset PG to before-split (TRUNCATE + reload) ---")
        before_counts = E.init_pg_before(schema, origin, processed, split_val)
        log("--- Stage B: seed sauce_dauce lineage from before-split base ---")
        if not os.path.exists(os.path.join(base_folder, MODEL_NAME)):
            log("  base model missing -> training before-split base via train_base()")
            E.train_base(schema, processed, split_val)
        if os.path.exists(sauce_folder):
            shutil.rmtree(sauce_folder)
        shutil.copytree(base_folder, sauce_folder)
        log(f"  seeded {sauce_folder} from {base_folder}")
        E.save_ckpt(CKPT_TAG, {"tag": CKPT_TAG, "rounds": args.rounds,
                               "n_queries": n_q, "n_subplans": n_sp,
                               "last_completed_round": -1,
                               "before_counts": before_counts,
                               "decisions": decisions})

    # ---- Stage C: injection alignment gate (cheap EXPLAIN-based) ----
    if start_round == 0 and not args.skip_align_gate:
        log("--- Stage C: injection alignment gate ---")
        if not E.verify_smoke_alignment(query_file, n_q, n_sp):
            log("ALIGNMENT GATE FAILED -- aborting before any latency run.")
            sys.exit(2)

    # ---- load sauce_dauce model + after-split delta (coupled chunks) ----
    sauce_model = pickle.load(open(os.path.join(sauce_folder, MODEL_NAME), "rb"))
    after_data, after_data_processed = E.get_update_data_by_date(
        schema, origin, processed_data=processed, time_date=SPLIT_DATE)
    chunk = dict()
    for t in schema.tables:
        tn = t.table_name
        chunk[tn] = (len(after_data[tn]) // args.rounds) if after_data[tn] is not None else 0
    log("  after-split delta sizes: " +
        ", ".join(f"{tn}={0 if after_data[tn] is None else len(after_data[tn])}"
                  for tn in chunk))

    # ---- round loop (sauce_dauce arm ONLY) ----
    for i in range(start_round, args.rounds):
        rt0 = time.time()
        log(f"================ REPLAY ROUND {i + 1}/{args.rounds} ================")

        # 1) insert this round's delta into PG (coupled, identical to full run) + ANALYZE
        update_data = dict()
        for t in schema.tables:
            tn = t.table_name
            if after_data[tn] is None:
                update_data[tn] = None
                continue
            ch = chunk[tn]
            if i < args.rounds - 1:
                db_slice = after_data[tn].iloc[i * ch:(i + 1) * ch]
                md_slice = after_data_processed[tn].iloc[i * ch:(i + 1) * ch]
            else:
                db_slice = after_data[tn].iloc[i * ch:]
                md_slice = after_data_processed[tn].iloc[i * ch:]
            update_data[tn] = md_slice
            if len(db_slice) > 0:
                dpath = f"{PG_FOLDER}/update/{tn}_delta{i + 1}.csv"
                E.to_int64(db_slice).to_csv(dpath, index=False)
                E.update_pg("stats", tn, dpath)
        E.analyze_all(schema)
        log(f"  round {i + 1}: delta inserted + ANALYZE; PG rows={E.pg_row_counts(schema)}")

        # 2) update sauce_dauce ONLY on DauceDetector-flagged tables (per-table gate)
        sauce_update, fired = dict(), []
        for t in schema.tables:
            tn = t.table_name
            if decisions[tn][i]:
                sauce_update[tn] = update_data[tn]
                fired.append(tn)
            else:
                sauce_update[tn] = None
        t0 = time.time()
        if fired:
            E.update_model("stats", sauce_model, sauce_folder,
                           os.path.join(sauce_folder, MODEL_NAME),
                           copy.deepcopy(sauce_update))
            log(f"  sauce_dauce DRIFT {fired} -> update took {time.time() - t0:.1f}s")
        else:
            log(f"  sauce_dauce no-drift -> model unchanged ({time.time() - t0:.1f}s)")

        # 3) generate est_sauce_dauce.txt (host results/ == container /home/...)
        est_path = RESULTS_FOLDER + f"est_{ARM_TAG}.txt"
        ns = E.gen_estimates(os.path.join(sauce_folder, MODEL_NAME), sub_query_file, est_path)
        log(f"  estimates written: sauce_dauce={ns} lines")

        # 4) measure ONLY the sauce_dauce arm (fresh connection streaming its estimates)
        cont_est = CONTAINER_RESULTS + f"/est_{ARM_TAG}.txt"
        t0 = time.time()
        E.send_query(dataset="stats", method_name=cont_est, query_file=query_file,
                     save_folder=LATENCY_FOLDER, iteration=i + 1,
                     test_type="factorjoin", update_type=ARM_TAG)
        log(f"  arm {ARM_TAG} round {i + 1} done in {time.time() - t0:.1f}s")

        # 5) checkpoint
        ck = E.load_ckpt(CKPT_TAG) or {}
        ck.update({"tag": CKPT_TAG, "rounds": args.rounds, "n_queries": n_q,
                   "n_subplans": n_sp, "last_completed_round": i,
                   "decisions": decisions})
        ck.setdefault("round_wallclock", {})[str(i + 1)] = round(time.time() - rt0, 1)
        E.save_ckpt(CKPT_TAG, ck)
        log(f"  round {i + 1} wallclock {time.time() - rt0:.1f}s; checkpoint saved")

    log(f"=== SAUCE-ONLY REPLAY COMPLETE: {args.rounds} rounds, {ARM_TAG} arm ===")
    log(f"latency npy: exec_time_est_{ARM_TAG}_{ARM_TAG}_factorjoin_iter{{1..{args.rounds}}}.npy")
    log("compare vs reused full-run arms: est_always_always_factorjoin / PG_native_naive_naive")


if __name__ == "__main__":
    main()
