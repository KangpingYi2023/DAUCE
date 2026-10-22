"""Figure 8 的 PostgreSQL 与 FactorJoin 逐轮更新实验。

以 2014-01-01 为界初始化数据库和基模型，逐轮插入其余数据并 ANALYZE。
--skip_sauce 仅运行 Always/None；DAUCE 臂由 run_sauce_replay.py 执行。
每个实验臂在同一连接中依序消费子计划估计文件，保存查询时延数组。
"""
import os
import sys
import json
import time
import copy
import shutil
import pickle
import argparse
import warnings

sys.path.append("./")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from Schemas.stats.schema import gen_stats_light_schema
from Evaluation.training import train_one_stats
from Evaluation.updating import timestamp_transorform
from update_experiment import (read_table_csv, get_update_data_by_date,
                               update_pg, drifts_detection, update_model)
from send_query import send_query, connect_pg

# ----------------------------------------------------------------------------
# constants / paths
# ----------------------------------------------------------------------------
SPLIT_DATE = "2014-01-01 00:00:00"
DATA_FOLDER = "../data/stats"
WORKLOADS = "../workloads/stats_CEB"
RUN_DIR = os.environ.get("FACTORJOIN_RUN_DIR")
MODEL_ROOT = os.path.join(RUN_DIR, "checkpoints") if RUN_DIR else "./checkpoints/e2e"
PG_FOLDER = os.path.join(RUN_DIR, "pg") if RUN_DIR else "./pg"
LATENCY_FOLDER = (os.path.join(RUN_DIR, "latency") if RUN_DIR else "./latency") + "/"
RESULTS_FOLDER = (os.path.join(RUN_DIR, "results") if RUN_DIR else "./results") + "/"
# in-container absolute path where PG (ml_joinest_fname) reads estimate files;
# the repository root is mounted at /home/DAUCE inside the PG container, so
# host <repo>/FactorJoin/results == container /home/DAUCE/FactorJoin/results
CONTAINER_RESULTS = os.environ.get("CONTAINER_RESULTS", "/home/DAUCE/FactorJoin/results")
N_DIM_DIST = 2
BIN_SIZE = 200
BUCKET_METHOD = "greedy"
SEED = 0
MODEL_NAME = f"model_stats_{BUCKET_METHOD}_{BIN_SIZE}.pkl"


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# data loading (replicates read_data_and_init_db WITHOUT init_pg)
# ----------------------------------------------------------------------------
def load_data():
    schema = gen_stats_light_schema(DATA_FOLDER + "/{}.csv")
    origin, processed = dict(), dict()
    for t in schema.tables:
        tn = t.table_name
        df, raw = read_table_csv(t, db_name="stats")
        origin[tn] = raw
        for c in df.columns:
            if "Date" in c and df[c].values.dtype == "object":
                df[c] = [timestamp_transorform(v) for v in df[c].values]
        processed[tn] = df
        log(f"  loaded {tn}: {len(raw)} rows")
    return schema, origin, processed


def date_mask(processed_t, split_val):
    """Boolean mask (Date >= split) using the first Date column; None if no Date."""
    for c in processed_t.columns:
        if "Date" in c:
            return processed_t[c].values >= split_val
    return None


def to_int64(df):
    """float64 -> nullable Int64 so CSV has no '1.0' and NaN -> empty (NULL)."""
    df = df.copy()
    for c in df.columns:
        if df[c].values.dtype == "float64":
            df[c] = df[c].astype("Int64")
    return df


# ----------------------------------------------------------------------------
# before-split PG initialisation (fixes init_pg's FULL-data PK clash)
# ----------------------------------------------------------------------------
def init_pg_before(schema, origin, processed, split_val):
    os.makedirs(PG_FOLDER + "/before", exist_ok=True)
    conn = connect_pg("stats")
    cur = conn.cursor()
    counts = dict()
    for t in schema.tables:
        tn = t.table_name
        raw = origin[tn]
        m = date_mask(processed[tn], split_val)
        before_raw = raw if m is None else raw.loc[~m]
        before_raw = to_int64(before_raw)
        csv_path = f"{PG_FOLDER}/before/{tn}.csv"
        before_raw.to_csv(csv_path, index=False)
        cur.execute(f"TRUNCATE TABLE {tn};")
        with open(csv_path, "r") as f:
            cur.copy_expert(f"COPY {tn} FROM STDIN WITH CSV HEADER", f)
        counts[tn] = len(before_raw)
        log(f"  init before {tn}: {len(before_raw)} rows")
    conn.commit()
    for t in schema.tables:
        cur.execute(f"ANALYZE {t.table_name};")
    conn.commit()
    cur.close()
    conn.close()
    return counts


def analyze_all(schema):
    conn = connect_pg("stats")
    cur = conn.cursor()
    for t in schema.tables:
        cur.execute(f"ANALYZE {t.table_name};")
    conn.commit()
    cur.close()
    conn.close()


def pg_row_counts(schema):
    conn = connect_pg("stats")
    cur = conn.cursor()
    out = dict()
    for t in schema.tables:
        cur.execute(f"SELECT count(*) FROM {t.table_name};")
        out[t.table_name] = cur.fetchone()[0]
    cur.close()
    conn.close()
    return out


# ----------------------------------------------------------------------------
# estimate-file generation (self-contained, integer output floored at 1)
# ----------------------------------------------------------------------------
def gen_estimates(model_path, sub_query_file, out_path):
    with open(model_path, "rb") as f:
        be = pickle.load(f)
    for t in be.bns:
        be.bns[t].init_inference_method()
    queries = open(sub_query_file).readlines()
    n = 0
    with open(out_path, "w") as fo:
        for qs in queries:
            q = qs.split("||")[0][:-1]          # strip trailing ';' (and ||N)
            r = be.get_cardinality_bound_one(q)
            r = 1 if (r is None or r < 1) else int(round(float(r)))
            fo.write(str(r) + "\n")
            n += 1
    return n


# ----------------------------------------------------------------------------
# workload preparation (smoke = first K queries of the verified-aligned full set)
# ----------------------------------------------------------------------------
def prepare_workload(smoke, k):
    qf_full = WORKLOADS + "/stats_CEB.sql"
    sqf_full = WORKLOADS + "/sub_plan_queries/stats_CEB_sub_queries.sql"
    if not smoke:
        n_q = sum(1 for _ in open(qf_full))
        n_sp = sum(1 for _ in open(sqf_full))
        return qf_full, sqf_full, n_q, n_sp
    qlines = open(qf_full).readlines()[:k]
    sq_all = open(sqf_full).readlines()
    sqlines = [l for l in sq_all if int(l.rstrip("\n").split("||")[-1]) < k]
    os.makedirs(PG_FOLDER, exist_ok=True)
    qf = PG_FOLDER + "/smoke_queries.sql"
    sqf = PG_FOLDER + "/smoke_sub_queries.sql"
    open(qf, "w").writelines(qlines)
    open(sqf, "w").writelines(sqlines)
    return qf, sqf, len(qlines), len(sqlines)


def verify_smoke_alignment(query_file, n_queries, n_subplans):
    """Marker gate: PG must consume exactly n_subplans lines over the K queries.

    Uses a marker file (value 7000000+i). After EXPLAIN-ing the K queries in one
    connection the cursor sits at n_subplans; a final probe of query 0 (exactly 1
    sub-plan) then reveals the cursor position via its injected join rows.
    """
    import psycopg2
    import re
    marker_path = os.path.abspath(RESULTS_FOLDER + "/smoke_marker.txt")
    container_marker = CONTAINER_RESULTS + "/smoke_marker.txt"
    with open(marker_path, "w") as f:
        for i in range(n_subplans + 50):
            f.write(str(7000000 + i) + "\n")
    conn = psycopg2.connect(database="stats", user=os.environ["PG_USER"],
                            password=os.environ["PG_PASSWORD"],
                            host=os.environ["PG_HOST"], port=int(os.environ["PG_PORT"]))
    conn.set_client_encoding("UTF8")
    cur = conn.cursor()
    cur.execute("SET statement_timeout=0; commit;")
    cur.execute("SET ml_joinest_enabled=true; SET join_est_no=0;")
    cur.execute(f"SET ml_joinest_fname='{container_marker}';")
    cur.execute("SET enable_indexscan=on; SET enable_bitmapscan=on; "
                "SET enable_material=off; SET enable_hashjoin=on;")
    qs = open(query_file).readlines()

    def top_join_rows(sql):
        cur.execute("EXPLAIN " + sql)
        for r in cur.fetchall():
            if ("Join" in r[0] or "Nested Loop" in r[0] or "Merge" in r[0]) and "rows=" in r[0]:
                m = re.search(r"rows=(\d+)", r[0])
                if m:
                    return int(m.group(1))
        return None

    for i in range(n_queries):
        top_join_rows(qs[i].split("||")[-1].strip())
    probe = top_join_rows(qs[0].split("||")[-1].strip())
    cur.close()
    conn.close()
    consumed = (probe - 7000000) if probe is not None else -1
    ok = (consumed == n_subplans)
    log(f"  ALIGNMENT GATE: PG consumed {consumed} sub-plans, expected {n_subplans} "
        f"-> {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------
# training the before-split base model + seeding the two lineages
# ----------------------------------------------------------------------------
def train_base(schema, processed, split_val, skip_sauce=False):
    before_processed = dict()
    for t in schema.tables:
        tn = t.table_name
        m = date_mask(processed[tn], split_val)
        before_processed[tn] = processed[tn] if m is None else processed[tn].loc[~m]
    base = MODEL_ROOT + "/base"
    os.makedirs(base, exist_ok=True)
    log("  training BEFORE-split base model (validate=False, save_bucket_bins=True)")
    t0 = time.time()
    train_one_stats("stats", DATA_FOLDER + "/{}.csv", base, n_dim_dist=N_DIM_DIST,
                    n_bins=BIN_SIZE, bucket_method=BUCKET_METHOD,
                    save_bucket_bins=True, seed=SEED, validate=False,
                    actual_data=before_processed)
    log(f"  base training took {time.time() - t0:.1f}s")
    # seed the two independent lineages (model + buckets) from base
    for lin in (["always"] if skip_sauce else ["always", "sauce"]):
        dst = f"{MODEL_ROOT}/{lin}"
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(base, dst)
    return before_processed


def load_lineage(lin):
    folder = f"{MODEL_ROOT}/{lin}"
    with open(os.path.join(folder, MODEL_NAME), "rb") as f:
        model = pickle.load(f)
    return folder, model


# ----------------------------------------------------------------------------
# checkpoint
# ----------------------------------------------------------------------------
def ckpt_path(tag):
    return f"{PG_FOLDER}/e2e_checkpoint_{tag}.json"


def load_ckpt(tag):
    p = ckpt_path(tag)
    return json.load(open(p)) if os.path.exists(p) else None


def save_ckpt(tag, d):
    os.makedirs(PG_FOLDER, exist_ok=True)
    json.dump(d, open(ckpt_path(tag), "w"), indent=2)


# ----------------------------------------------------------------------------
# main orchestration
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full 146-query workload, 7 rounds")
    ap.add_argument("--smoke", action="store_true", help="first K queries, few rounds (pipeline validation)")
    ap.add_argument("--k", type=int, default=8, help="smoke: number of leading queries")
    ap.add_argument("--rounds", type=int, default=None, help="override number of update rounds")
    ap.add_argument("--resume", action="store_true", help="resume from checkpoint")
    ap.add_argument("--skip_align_gate", action="store_true")
    ap.add_argument("--skip_sauce", action="store_true",
                    help="skip the legacy KL-gated sauce lineage (always + native arms only)")
    args = ap.parse_args()

    os.environ.setdefault("PG_USER", "postgres")
    os.environ.setdefault("PG_PASSWORD", "postgres")
    os.environ.setdefault("PG_HOST", "localhost")
    os.environ.setdefault("PG_PORT", "5432")
    for d in [MODEL_ROOT, PG_FOLDER, LATENCY_FOLDER, RESULTS_FOLDER, PG_FOLDER + "/update",
              PG_FOLDER + "/pool", PG_FOLDER + "/before"]:
        os.makedirs(d, exist_ok=True)

    smoke = args.smoke or not args.full
    tag = "smoke" if smoke else "full"
    rounds = args.rounds if args.rounds is not None else (2 if smoke else 7)
    log(f"=== DAUCE e2e [{tag}] rounds={rounds} smoke={smoke} k={args.k} resume={args.resume} ===")

    schema, origin, processed = load_data()
    split_val = timestamp_transorform(SPLIT_DATE)
    query_file, sub_query_file, n_q, n_sp = prepare_workload(smoke, args.k)
    log(f"  workload: {n_q} queries, {n_sp} sub-plans; query_file={query_file}")

    ck = load_ckpt(tag) if args.resume else None
    start_round = 0
    if ck and ck.get("n_subplans") == n_sp and ck.get("rounds") == rounds:
        start_round = ck["last_completed_round"] + 1
        log(f"  RESUME from round {start_round} (checkpoint last_completed={ck['last_completed_round']})")

    # ---- before-split init + base training (skipped on resume) ----
    always_folder = MODEL_ROOT + "/always"
    sauce_folder = MODEL_ROOT + "/sauce"
    sauce_raw_path = sauce_folder + "/sauce_raw.pkl"
    pool_path = PG_FOLDER + "/pool/pool_data.pkl"

    if start_round == 0:
        log("--- Stage A: before-split PG init ---")
        before_counts = init_pg_before(schema, origin, processed, split_val)
        log("--- Stage B: train before-split base + seed lineages ---")
        before_processed = train_base(schema, processed, split_val, args.skip_sauce)
        # after-split delta (origin form for CSV, processed form for model update)
        after_data, after_data_processed = get_update_data_by_date(
            schema, origin, processed_data=processed, time_date=SPLIT_DATE)
        if not args.skip_sauce:
            pickle.dump(before_processed, open(sauce_raw_path, "wb"), pickle.HIGHEST_PROTOCOL)
        if os.path.exists(pool_path):
            os.remove(pool_path)
        save_ckpt(tag, {"tag": tag, "rounds": rounds, "n_queries": n_q, "n_subplans": n_sp,
                        "last_completed_round": -1, "before_counts": before_counts,
                        "smoke": smoke, "k": args.k})
    else:
        after_data, after_data_processed = get_update_data_by_date(
            schema, origin, processed_data=processed, time_date=SPLIT_DATE)

    # ---- smoke alignment gate ----
    if smoke and start_round == 0 and not args.skip_align_gate:
        log("--- Stage C: smoke alignment gate ---")
        if not verify_smoke_alignment(query_file, n_q, n_sp):
            log("ALIGNMENT GATE FAILED -- aborting before any latency run.")
            sys.exit(2)

    # ---- load lineages ----
    always_model = pickle.load(open(os.path.join(always_folder, MODEL_NAME), "rb"))
    if not args.skip_sauce:
        sauce_model = pickle.load(open(os.path.join(sauce_folder, MODEL_NAME), "rb"))
        sauce_raw = pickle.load(open(sauce_raw_path, "rb"))

    # chunk sizes for the after-split delta
    chunk = dict()
    for t in schema.tables:
        tn = t.table_name
        chunk[tn] = (len(after_data[tn]) // rounds) if after_data[tn] is not None else 0
    log("  after-split delta sizes: " +
        ", ".join(f"{tn}={0 if after_data[tn] is None else len(after_data[tn])}"
                  for tn in chunk))

    # ---- round loop ----
    for i in range(start_round, rounds):
        log(f"================ ROUND {i + 1}/{rounds} ================")
        rt0 = time.time()
        # 1) insert this round's delta into PG (all tables) + ANALYZE
        update_data = dict()
        for t in schema.tables:
            tn = t.table_name
            if after_data[tn] is None:
                update_data[tn] = None
                continue
            ch = chunk[tn]
            if i < rounds - 1:
                db_slice = after_data[tn].iloc[i * ch:(i + 1) * ch]
                md_slice = after_data_processed[tn].iloc[i * ch:(i + 1) * ch]
            else:
                db_slice = after_data[tn].iloc[i * ch:]
                md_slice = after_data_processed[tn].iloc[i * ch:]
            update_data[tn] = md_slice
            if len(db_slice) > 0:
                dpath = f"{PG_FOLDER}/update/{tn}_delta{i + 1}.csv"
                to_int64(db_slice).to_csv(dpath, index=False)
                update_pg("stats", tn, dpath)
        analyze_all(schema)
        log(f"  round {i + 1}: delta inserted + ANALYZE; PG rows={pg_row_counts(schema)}")

        # 2) update ALWAYS lineage (every round)
        # update_model 会原地改写连接键，每个实验臂必须使用独立深拷贝。
        t0 = time.time()
        update_model("stats", always_model, always_folder,
                     os.path.join(always_folder, MODEL_NAME), copy.deepcopy(update_data))
        log(f"  always update took {time.time() - t0:.1f}s")

        # 3) update SAUCE lineage (drift-gated)
        if not args.skip_sauce:
            t0 = time.time()
            need_update, sauce_update_data, sauce_raw = drifts_detection(
                schema.tables, copy.deepcopy(update_data), sauce_raw, pool_path=pool_path)
            if need_update:
                update_model("stats", sauce_model, sauce_folder,
                             os.path.join(sauce_folder, MODEL_NAME), sauce_update_data)
                log(f"  sauce DRIFT -> update took {time.time() - t0:.1f}s")
            else:
                log(f"  sauce no-drift -> model unchanged ({time.time() - t0:.1f}s)")
            pickle.dump(sauce_raw, open(sauce_raw_path, "wb"), pickle.HIGHEST_PROTOCOL)

        # 4) generate integer estimate files (host results/ == container /home/...)
        est_always = RESULTS_FOLDER + "est_always.txt"
        na = gen_estimates(os.path.join(always_folder, MODEL_NAME), sub_query_file, est_always)
        if not args.skip_sauce:
            est_sauce = RESULTS_FOLDER + "est_sauce.txt"
            ns = gen_estimates(os.path.join(sauce_folder, MODEL_NAME), sub_query_file, est_sauce)
            log(f"  estimates written: always={na} sauce={ns} lines")
        else:
            log(f"  estimates written: always={na} lines")

        # 5) arms -- each a fresh connection streaming its estimate file
        cont_always = CONTAINER_RESULTS + "/est_always.txt"
        cont_sauce = CONTAINER_RESULTS + "/est_sauce.txt"
        arms = [
            ("always", cont_always, "factorjoin", "always"),
            ("sauce", cont_sauce, "factorjoin", "sauce"),
            ("native", "PG_native.txt", "naive", "naive"),
        ]
        if args.skip_sauce:
            arms = [a for a in arms if a[0] != "sauce"]
        for arm, mname, ttype, utype in arms:
            t0 = time.time()
            send_query(dataset="stats", method_name=mname, query_file=query_file,
                       save_folder=LATENCY_FOLDER, iteration=i + 1,
                       test_type=ttype, update_type=utype)
            log(f"  arm {arm:7s} round {i + 1} done in {time.time() - t0:.1f}s")

        # 6) checkpoint
        ck = load_ckpt(tag) or {}
        ck.update({"tag": tag, "rounds": rounds, "n_queries": n_q, "n_subplans": n_sp,
                   "last_completed_round": i, "smoke": smoke, "k": args.k})
        ck.setdefault("round_wallclock", {})[str(i + 1)] = round(time.time() - rt0, 1)
        save_ckpt(tag, ck)
        log(f"  round {i + 1} wallclock {time.time() - rt0:.1f}s; checkpoint saved")

    log(f"=== DAUCE e2e [{tag}] COMPLETE: {rounds} rounds x {2 if args.skip_sauce else 3} arms ===")
    log(f"latency npy in {LATENCY_FOLDER} (exec_time_*_iter{{1..{rounds}}}.npy)")


if __name__ == "__main__":
    main()
