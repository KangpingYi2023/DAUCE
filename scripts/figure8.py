#!/usr/bin/env python3
"""Figure 8：独立 PostgreSQL 容器、None/Always/DAUCE 三臂和 CSV 聚合。"""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLES = ["users", "posts", "postLinks", "postHistory", "comments", "votes", "badges", "tags"]
ARMS = [("None", "PG_native_naive_naive"), ("Always", "est_always_always_factorjoin"),
        ("DAUCE", "est_sauce_dauce_sauce_dauce_factorjoin")]


def run(command, logfile, env=None, cwd=ROOT):
    print("运行：", " ".join(map(str, command)), flush=True)
    with logfile.open("w") as out:
        result = subprocess.run(list(map(str, command)), cwd=cwd, env=env,
                                stdout=out, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("运行失败（{}），日志：{}".format(result.returncode, logfile))


def aggregate(dest, rounds, expected_queries):
    with (dest / "figure8.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["round", "method", "queries", "execution_time_s", "planning_time_s"])
        for iteration in range(1, rounds + 1):
            for method, tag in ARMS:
                values = [np.load(dest / "latency" / ("{}_{}_iter{}.npy".format(kind, tag, iteration)))
                          for kind in ["exec_time", "plan_time"]]
                if any(v.size != expected_queries or not np.all(np.isfinite(v)) for v in values):
                    raise RuntimeError("时延数组不完整：{} / {}".format(method, iteration))
                writer.writerow([iteration, method, expected_queries,
                                 float(values[0].sum() / 1000), float(values[1].sum() / 1000)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="2 轮更新，每轮前 1 条查询")
    parser.add_argument("--port", type=int, default=5433, help="独立 PG 容器的宿主端口")
    parser.add_argument("--keep-container", action="store_true", help="完成后保留测试容器")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print("构建 docker/pg，启动独立 PG；Always/None（跳过旧 SAUCE），DAUCE 重放；聚合 figure8.csv")
        return
    for table in TABLES:
        path = ROOT / "data/stats" / (table + ".csv")
        if not path.is_file():
            raise FileNotFoundError("请放置数据：" + str(path))
    with (ROOT / ".reproduce.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dest = ROOT / "outputs" / ("figure8_{}{}".format("smoke_" if args.smoke else "", time.time_ns()))
        dest.mkdir(parents=True)
        (dest / "results").mkdir()
        container = "dauce-figure8-" + str(os.getpid())
        env = os.environ.copy()
        env.update(PG_HOST="127.0.0.1", PG_PORT=str(args.port), PG_USER="postgres",
                   PG_PASSWORD=os.environ.get("PG_PASSWORD", "postgres"),
                   PYTHONUNBUFFERED="1", FACTORJOIN_RUN_DIR=str(dest),
                   CONTAINER_RESULTS="/est")
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env.setdefault(key, "4")
        run(["docker", "build", "-t", "dauce-pg:revision", ROOT / "docker/pg"], dest / "docker-build.log")
        started = False
        try:
            run(["docker", "run", "-d"] + ([] if args.keep_container else ["--rm"]) +
                            ["--name", container, "--shm-size=16g",
                 "-p", "127.0.0.1:{}:5432".format(args.port),
                 "-v", str(ROOT) + ":/home/DAUCE:ro",
                 "-v", str(dest / "results") + ":/est:ro", "dauce-pg:revision"], dest / "docker-run.log")
            started = True
            run(["docker", "exec", "-e", "PG_PASSWORD", container, "bash", "/opt/stats/init_stats_db.sh"],
                dest / "docker-init.log", env)
            rounds = 2 if args.smoke else 7
            command = [sys.executable, "run_e2e_dauce.py", "--skip_sauce", "--rounds", rounds]
            command += ["--smoke", "--k", "1"] if args.smoke else ["--full"]
            run(command, dest / "always-none.log", env, ROOT / "FactorJoin")
            command = [sys.executable, "run_sauce_replay.py", "--rounds", rounds,
                       "--tau", "0.05", "--repeats", "10"]
            if args.smoke:
                command += ["--smoke", "--k", "1"]
            run(command, dest / "dauce.log", env, ROOT / "FactorJoin")
            queries = 1 if args.smoke else len((ROOT / "workloads/stats_CEB/stats_CEB.sql").read_text().splitlines())
            aggregate(dest, rounds, queries)
            print("完成：", dest / "figure8.csv", flush=True)
        finally:
            if started:
                with (dest / "postgres.log").open("w") as output:
                    subprocess.run(["docker", "logs", container], stdout=output,
                                   stderr=subprocess.STDOUT, check=False)
            if started and not args.keep_container:
                subprocess.run(["docker", "stop", container], stdout=subprocess.DEVNULL, check=False)
            elif started:
                print("容器：", container, flush=True)


if __name__ == "__main__":
    main()
