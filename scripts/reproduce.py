#!/usr/bin/env python3
"""Table 4–7 的输入检查、顺序运行和 CSV 聚合。"""
import argparse
import csv
import fcntl
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_DIRS = {m: ROOT / "models" / (m + "_origin_models")
              for m in ("naru", "transformer", "face")}


def dataset_file(dataset):
    tail = "_int.npy" if dataset == "census" else ".npy"
    return ROOT / "data" / dataset / (dataset + tail)


def checkpoint(dataset, model):
    pattern = dataset + "*.t" if model == "face" else "origin-" + dataset + "-*.pt"
    matches = list(MODEL_DIRS[model].glob(pattern))
    if len(matches) != 1:
        raise RuntimeError("需且仅需一份检查点：{} / {}".format(MODEL_DIRS[model], pattern))
    return matches[0]


def initialize():
    for name in ("query", "FACE/train", "FACE/evaluate/oracle",
                 "end2end/communicate", "end2end/experiment-records",
                 "end2end/parsed-records", "end2end/model-evaluation",
                 "end2end/end2end-evaluations", "end2end/end2end-bootstrap",
                 "models/checkpoints"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    for ds in ("census", "forest", "bjaq", "power"):
        (ROOT / "data" / ds / "end2end").mkdir(parents=True, exist_ok=True)
    for model in MODEL_DIRS:
        (ROOT / "models" / "end2end" / model).mkdir(parents=True, exist_ok=True)


def run_command(command, logfile):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "FACE/torchquadMy"), str(ROOT),
                                          env.get("PYTHONPATH", "")])
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(key, "4")
    with logfile.open("w") as output:
        output.write(" ".join(map(str, command)) + "\n")
        output.flush()
        result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                                stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("运行失败（退出码 {}），日志：{}".format(result.returncode, logfile))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("没有有效结果：" + str(path))
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)


def selected(values, requested, smoke=False):
    if requested:
        if requested not in values:
            raise ValueError("无效选项：{}（可选 {}）".format(requested, values))
        return [requested]
    return values[:1] if smoke else values


def table4_cells(config, args):
    for ds in selected(config["datasets"], args.dataset, args.smoke):
        scenarios = selected(config["scenarios"], args.drift)
        methods = selected(["DDUp", "DAUCE"], args.method)
        models = selected(config["ddup_models"][ds], args.model, args.smoke)
        for scenario in scenarios:
            if "DAUCE" in methods:
                yield dict(dataset=ds, model="naru", scenario=scenario, method="DAUCE")
            if "DDUp" in methods:
                for model in models:
                    yield dict(dataset=ds, model=model, scenario=scenario, method="DDUp")


def single_cells(config, args):
    values = [selected(config["datasets"], args.dataset, args.smoke),
              selected(config["models"], args.model, args.smoke),
              selected(config["methods"], args.method),
              selected(config["scenarios"], args.drift),
              config["query_seeds"][:1] if args.smoke else config["query_seeds"]]
    for ds, model, method, scenario, seed in itertools.product(*values):
        yield dict(dataset=ds, model=model, method=method, scenario=scenario, seed=seed)


def run_table4(config, args, cells, dest):
    for idx, cell in enumerate(cells):
        print("[{}/{}] {}".format(idx + 1, len(cells), cell), flush=True)
        command = [sys.executable, ROOT / "experiments/origproto/run_origproto.py",
                   "--dataset", cell["dataset"], "--model", cell["model"],
                   "--data_update", cell["scenario"], "--drift_test",
                   "sauce" if cell["method"] == "DAUCE" else "ddup",
                   "--skew_ratio", config["skew_ratio"], "--random_seed", config["random_seed"],
                   "--bootstrap", config["bootstrap"], "--sample_size", config["sample_size"],
                   "--output-dir", dest]
        if cell["scenario"] == "valueskew":
            opts = config["column_settings"][cell["dataset"]]
            command += ["--joint_col", "--value_skew_ncols", opts["ncols"], "--col_mix", opts["mix"]]
        run_command(command, dest / ("cell_{}.log".format(idx)))
    rows = [json.loads(line) for line in (dest / "results.jsonl").read_text().splitlines()]
    write_csv(dest / "table4.csv", [dict(dataset=r["dataset"], scenario=r["data_update"],
               method="DAUCE" if r["drift_test"] == "sauce" else "DDUp", model=r["model"],
               statistic=r["statistic"], threshold=r["threshold"], detected=r["detected"])
               for r in rows])


def run_single(config, args, cells, dest):
    from experiments.aggregate_e2e import parse_record
    rows = []
    for idx, cell in enumerate(cells):
        print("[{}/{}] {}".format(idx + 1, len(cells), cell), flush=True)
        ds, model = cell["dataset"], cell["model"]
        size = config.get("update_sizes", {}).get(ds)
        if size is None:
            data = np.load(dataset_file(ds), mmap_mode="r", allow_pickle=True)
            size = int(len(data) * config["update_fraction"])
        workload_count = 3 if args.smoke else config["num_workload"]
        num_queries = args.queries or (5 if args.smoke else config["num_queries"])
        epochs = args.epochs or (1 if args.smoke else config["epochs"])
        updater = "update" if cell["method"] == "DDUp" else "adapt"
        detector = ("ddup" if updater == "update" else "sauce") if args.target == "table7" else "always"
        tag = dest.name + "_" + str(idx)
        namespace = dict(dataset=ds, model=model, model_update=updater, drift_test=detector,
                         update_size=size, data_update=cell["scenario"], num_workload=workload_count,
                         query_seed=cell["seed"], num_queries=num_queries)
        params = dict(config["params"], **{model + ".epochs": epochs, "experiment.tag": tag})
        if args.smoke and model == "face":
            params.update({"face.map_samples": 20000, "face.map_iterations": 4,
                           "face.query_samples": 800, "face.query_iterations": 2})
        grid = {"args_combination": {k: [v] for k, v in namespace.items()},
                "params_combination": {k: [v] for k, v in params.items()}}
        cfg_path = dest / ("cell_{}.yaml".format(idx))
        cfg_path.write_text(yaml.safe_dump(grid, sort_keys=False))
        run_command([sys.executable, ROOT / "end2end/multi_experiments.py", "--config", cfg_path],
                    dest / ("cell_{}.log".format(idx)))
        records = []
        for path in (ROOT / "end2end/experiment-records").glob("*.txt"):
            row = parse_record(path)
            if row.get("experiment.tag") == tag:
                records.append((path, row))
        if len(records) != 1:
            raise RuntimeError("实验记录数量不符：{}，期望 1".format(len(records)))
        path, row = records[0]
        if row.get("num_query_workloads") != (workload_count + 1) // 2 or not row.get("qerr_mean"):
            raise RuntimeError("查询结果不完整：" + str(path))
        text = path.read_text()
        offline = sum(map(float, re.findall(r"DAUCE offline time = ([\d.]+)", text)))
        if args.target == "table7" and updater == "adapt" and "DAUCE offline time" not in text:
            raise RuntimeError("缺少 DAUCE offline 计时：" + str(path))
        total = row["total_time_s"] - offline
        rows.append(dict(dataset=ds, model=model, method=cell["method"], scenario=cell["scenario"],
                         query_seed=cell["seed"], qerr_mean=row["qerr_mean"],
                         qerr_p95=row["qerr_p95"], qerr_max=row["qerr_max"],
                         n_drift=row["n_drift"], n_data_updates=row["n_data_updates"],
                         update_time_s=row["model_update_time_s"], offline_time_s=offline,
                         total_time_s=total))
        shutil.copy2(path, dest / ("record_{}.txt".format(idx)))
        write_csv(dest / "runs.csv", rows)
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("dataset", "model", "method", "scenario"))
        groups.setdefault(key, []).append(row)
    summary = []
    for key, group in groups.items():
        row = dict(zip(("dataset", "model", "method", "scenario"), key))
        for metric in ("qerr_mean", "qerr_p95", "qerr_max", "n_drift", "n_data_updates",
                       "update_time_s", "offline_time_s", "total_time_s"):
            row[metric] = float(np.mean([r[metric] for r in group]))
        summary.append(row)
    write_csv(dest / (args.target + ".csv"), summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=["table4", "table5", "table6", "table7"])
    parser.add_argument("--smoke", action="store_true", help="少量查询、1 epoch 和首个数据集/模型/seed")
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--method", choices=["DDUp", "DAUCE"])
    parser.add_argument("--drift", choices=["sample", "tupleskew", "valueskew"])
    parser.add_argument("--queries", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dry-run", action="store_true", help="只打印实验矩阵")
    args = parser.parse_args()
    os.chdir(ROOT)
    config = yaml.safe_load((ROOT / "experiments/configs" / (args.target + ".yaml")).read_text())
    cells = list(table4_cells(config, args) if args.target == "table4" else single_cells(config, args))
    if args.dry_run:
        print(json.dumps(cells, indent=2, ensure_ascii=False))
        print("实验数量：", len(cells))
        return
    for cell in cells:
        if not dataset_file(cell["dataset"]).is_file():
            raise FileNotFoundError("请放置数据：" + str(dataset_file(cell["dataset"])))
        if args.target != "table4" and cell["dataset"] in ("census", "forest"):
            csv_path = ROOT / "data" / cell["dataset"] / (cell["dataset"] + ".csv")
            if not csv_path.is_file():
                raise FileNotFoundError("基线查询还需要原始 CSV：" + str(csv_path))
        if args.target != "table4" or cell["method"] == "DDUp":
            checkpoint(cell["dataset"], cell["model"])
    lockpath = ROOT / ".reproduce.lock"
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize()
        run_id = "{}_{}{}_{}".format(args.target, "smoke_" if args.smoke else "",
                                     time.strftime("%Y%m%d_%H%M%S"), time.time_ns())
        dest = ROOT / "outputs" / run_id
        dest.mkdir(parents=True)
        print("输出目录：", dest, flush=True)
        if args.target == "table4":
            run_table4(config, args, cells, dest)
        else:
            run_single(config, args, cells, dest)
        print("完成：", dest / (args.target + ".csv"), flush=True)


if __name__ == "__main__":
    main()
