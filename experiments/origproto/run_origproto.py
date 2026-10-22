#!/usr/bin/env python
"""Table IV 的漂移检测入口。

为每个实验指定独立上下文、骨干检查点和漂移参数，调用
Naru.eval_model.test_for_drift，保存检测统计量、阈值和日志。
"""
import argparse
import contextlib
import glob
import io
import json
import os
import re
import sys
import time

# --- locate the DAUCE root (this file lives in <root>/experiments/origproto/) --
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Naru"))

import eval_model  # noqa: E402  (needs Naru/ + root on sys.path)  # isort:skip

MODEL_DIRS = {
    "naru": "naru_origin_models",
    "transformer": "transformer_origin_models",
    "face": "face_origin_models",
}

# Frozen pre-revision SAUCE thresholds, mirrored from eval_model.py sauce_legacy.
FROZEN_THRESHOLD = {
    "census": 6.26e-6,
    "bjaq": 2.78e-5,
    "forest": 1.73e-6,
    "power": 8.03e-6,
    "bjaq_gaussian": 1.90e-5,
    "census_gaussian": 7.41e-5,
}

CTX_DIR = os.path.join(ROOT, "experiments/origproto/ctx")
RAW_DIR = os.path.join(ROOT, "experiments/origproto/raw")
RESULTS_JSONL = os.path.join(ROOT, "experiments/origproto/results.jsonl")

_LEGACY_VALUE_SKEW = False
_VALUE_SKEW_NCOLS = None  # None => keep current sampler (2 cols); int => force that many
_JOINT_COL = False  # Option-1: joint-PRESERVING column drift (over-represent real rows)
_COL_MIX = 1.0  # joint-col only: fraction of batch drawn from the biased pool (1.0=strongest)


def _legacy_value_skew_sample(self, data):
    """可指定偏斜列数量的独立列值采样。"""
    import numpy as np
    from update_utils.end2end_utils.json_communicator import JsonCommunicator
    self.get_update_size(data)
    if self.skew_ratio is None:
        self.skew_ratio = float(JsonCommunicator().get('data_updater.skew_ratio'))
    n_rows, n_cols = data.shape
    max_n_cols = n_cols // 3
    if not self.random_seed == "auto":
        np.random.seed(int(self.random_seed))
    update_n_cols = _VALUE_SKEW_NCOLS if _VALUE_SKEW_NCOLS else max_n_cols
    update_n_cols = max(1, min(int(update_n_cols), n_cols))
    update_col_idx = np.random.choice(range(n_cols), size=update_n_cols, replace=False)
    sample_idx = np.random.choice(n_rows, size=self.update_size, replace=True)
    sample_idx = np.sort(sample_idx)
    samples = data[sample_idx]
    for update_col in update_col_idx:
        candidate_values = np.random.choice(
            data[:, update_col], size=round(self.skew_ratio * n_rows), replace=False)
        samples[:, update_col] = np.random.choice(
            candidate_values, size=self.update_size, replace=True)
    return samples


def _joint_preserving_value_skew_sample(self, data):
    """整行重采样：对目标列值池内的真实行增加采样权重。"""
    import numpy as np
    from update_utils.end2end_utils.json_communicator import JsonCommunicator
    self.get_update_size(data)
    if self.skew_ratio is None:
        self.skew_ratio = float(JsonCommunicator().get('data_updater.skew_ratio'))
    n_rows, n_cols = data.shape
    if not self.random_seed == "auto":
        np.random.seed(int(self.random_seed))
    update_n_cols = _VALUE_SKEW_NCOLS if _VALUE_SKEW_NCOLS else 1
    update_n_cols = max(1, min(int(update_n_cols), n_cols))
    update_col_idx = np.random.choice(range(n_cols), size=update_n_cols, replace=False)

    # 对目标列的候选行取并集，始终保留原始行内的列组合。
    row_mask = np.zeros(n_rows, dtype=bool)
    for update_col in update_col_idx:
        col = data[:, update_col]
        uniq = np.unique(col)
        pool = max(1, int(round(self.skew_ratio * len(uniq))))
        cand_vals = np.random.choice(uniq, size=pool, replace=False)
        row_mask |= np.isin(col, cand_vals)
    cand_rows = np.nonzero(row_mask)[0]
    if len(cand_rows) == 0:
        cand_rows = np.arange(n_rows)
    # _COL_MIX 指定偏斜池的采样比例，其余部分均匀采样。
    n_biased = max(0, min(int(round(_COL_MIX * self.update_size)), self.update_size))
    n_uniform = self.update_size - n_biased
    parts = []
    if n_biased > 0:
        parts.append(np.random.choice(cand_rows, size=n_biased, replace=True))
    if n_uniform > 0:
        parts.append(np.random.choice(n_rows, size=n_uniform, replace=True))
    sample_idx = np.sort(np.concatenate(parts).astype(int))
    return data[sample_idx]


def _apply_value_skew_override():
    from end2end import data_updater
    if _JOINT_COL:
        data_updater.ValueSkewSampler.sample = _joint_preserving_value_skew_sample
    elif _LEGACY_VALUE_SKEW:
        data_updater.ValueSkewSampler.sample = _legacy_value_skew_sample


def resolve_pre_model(model, dataset):
    """Exact checkpoint for (model, dataset); asserts a unique match."""
    subdir = MODEL_DIRS[model]
    pattern = "{}*.t".format(dataset) if model == "face" else "origin-{}-*.pt".format(dataset)
    pat = os.path.join(ROOT, "models", subdir, pattern)
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError("no checkpoint matched {}".format(pat))
    if len(files) > 1:
        # 多份检查点中优先选择唯一的 seed0。
        seed0 = [f for f in files if "seed0" in os.path.basename(f)]
        if len(seed0) == 1:
            return seed0[0]
        raise RuntimeError("ambiguous checkpoints for {}: {}".format(model, files))
    return files[0]


def _parse_float(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def run_one(dataset, model, data_update, drift_test, random_seed, skew_ratio,
            bootstrap=1000, sample_size=500, tag="", quiet=True):
    """Run one (dataset, model, data_update, drift_test, seed, skew) cell.

    Returns a dict with the parsed statistic / threshold / drift verdict and the
    path to the raw stdout log. Never raises for the harmless post-print
    ``numpy.bool_`` crash on the sauce_legacy path.
    """
    os.makedirs(CTX_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    key = "{}_{}_{}_{}_skew{}_seed{}_b{}_s{}{}".format(
        dataset, data_update, drift_test, model,
        skew_ratio, random_seed, bootstrap, sample_size, tag)
    ctx_path = os.path.join(CTX_DIR, key + ".json")
    with open(ctx_path, "w") as f:
        json.dump({"params": {"data_updater": {"skew_ratio": skew_ratio}}}, f)
    os.environ["DAUCE_CONTEXT"] = ctx_path
    _apply_value_skew_override()

    argv = ["--dataset", dataset, "--model", model, "--eval_type", "drift",
            "--data_update", data_update, "--drift_test", drift_test,
            "--random_seed", str(random_seed)]
    args = eval_model.create_parser().parse_args(argv)
    # MakeMade/MakeTransformer read the MODULE-GLOBAL args (e.g. args.inv_order),
    # which main() normally sets. We bypass main(), so set it here too.
    eval_model.args = args

    pre_model = resolve_pre_model(model, dataset) if drift_test == "ddup" else None

    buf = io.StringIO()
    err = None
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            eval_model.test_for_drift(
                args=args, pre_model=pre_model, end2end=False, seed=0,
                bootstrap=bootstrap, sample_size=sample_size, data_type="raw")
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    elapsed = time.time() - t0
    out = buf.getvalue()

    log_path = os.path.join(RAW_DIR, key + ".log")
    with open(log_path, "w") as f:
        f.write(out)
        if err:
            f.write("\n### exception: " + err + "\n")

    res = {
        "key": key, "dataset": dataset, "model": model,
        "data_update": data_update, "drift_test": drift_test,
        "random_seed": random_seed, "skew_ratio": skew_ratio,
        "bootstrap": bootstrap, "sample_size": sample_size,
        "pre_model": os.path.relpath(pre_model, ROOT) if pre_model else None,
        "elapsed_s": round(elapsed, 2), "log": os.path.relpath(log_path, ROOT),
        "err": err,
    }
    res["distance"] = _parse_float(r"Distance score:\s*([0-9.eE+-]+)", out)
    res["stat"] = _parse_float(r"test statistic =\s*([0-9.eE+-]+)", out)
    res["ddup_threshold"] = _parse_float(r"detection threshold =\s*([0-9.eE+-]+)", out)
    if model == "face" and drift_test == "ddup":
        res["stat"] = _parse_float(r"Mean loss reduction:\s*([0-9.eE+-]+)", out)
        res["ddup_threshold"] = _parse_float(r"2 \* std:\s*([0-9.eE+-]+)", out)
    res["ddup_mean"] = _parse_float(r"detection mean =\s*([0-9.eE+-]+)", out)
    res["new_ll"] = _parse_float(r"new data average log likelihood =\s*([0-9.eE+-]+)", out)
    res["dauce_threshold"] = _parse_float(r"DAUCE calibration:\s*delta=([0-9.eE+-]+)", out)
    m = re.search(r"Drift detection:\s*(\w+)", out)
    res["drift"] = m.group(1) if m else None
    if drift_test == "sauce_legacy":
        res["frozen_threshold"] = FROZEN_THRESHOLD.get(dataset)

    # The headline statistic per detector + its threshold + detected?.
    if drift_test == "ddup":
        res["statistic"] = res["stat"]
        res["threshold"] = res["ddup_threshold"]
    elif drift_test == "sauce":
        res["statistic"] = res["distance"]
        res["threshold"] = res["dauce_threshold"]
    else:  # sauce_legacy
        res["statistic"] = res["distance"]
        res["threshold"] = res.get("frozen_threshold")
    if res["statistic"] is not None and res["threshold"] is not None:
        res["detected"] = bool(res["statistic"] > res["threshold"])

    with open(RESULTS_JSONL, "a") as f:
        f.write(json.dumps(res) + "\n")

    if not quiet:
        print(json.dumps(res, indent=2))
    if err or res.get("statistic") is None or res.get("threshold") is None:
        raise RuntimeError("检测未完成：{}；日志：{}".format(err or key, log_path))
    return res


def _fmt(res):
    s = res.get("statistic")
    t = res.get("threshold")
    d = res.get("detected")
    return "statistic={} threshold={} detected={} ({:.1f}s){}".format(
        "{:.6e}".format(s) if isinstance(s, float) else s,
        "{:.6e}".format(t) if isinstance(t, float) else t,
        d, res.get("elapsed_s", 0.0),
        "  ERR=" + res["err"] if res.get("err") and res.get("statistic") is None else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="census")
    p.add_argument("--model", default="naru")
    p.add_argument("--data_update", default="valueskew")
    p.add_argument("--drift_test", default="ddup")
    p.add_argument("--skew_ratio", type=float, default=1e-3)
    p.add_argument("--random_seed", type=int, default=1234)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--sample_size", type=int, default=500)
    p.add_argument("--tag", default="")
    p.add_argument("--output-dir", help="本次运行输出目录")
    p.add_argument("--legacy_value_skew", action="store_true",
                   help="valueskew uses n_cols//3 drift columns (original SAUCE-era)")
    p.add_argument("--value_skew_ncols", type=int, default=0,
                   help="force valueskew drift-column count (0 => n_cols//3 when legacy)")
    p.add_argument("--joint_col", action="store_true",
                   help="Option-1: valueskew = joint-PRESERVING column drift "
                        "(over-represent real rows in a narrow per-column value pool)")
    p.add_argument("--col_mix", type=float, default=1.0,
                   help="joint-col only: fraction of batch from the biased pool "
                        "(1.0 strongest; smaller => weaker column drift)")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--skew_list", default="")
    p.add_argument("--seed_list", default="")
    p.add_argument("--drift_list", default="")
    a = p.parse_args()
    global CTX_DIR, RAW_DIR, RESULTS_JSONL
    if a.output_dir:
        dest = os.path.abspath(a.output_dir)
        CTX_DIR = os.path.join(dest, "ctx")
        RAW_DIR = os.path.join(dest, "raw")
        RESULTS_JSONL = os.path.join(dest, "results.jsonl")

    global _LEGACY_VALUE_SKEW, _VALUE_SKEW_NCOLS, _JOINT_COL, _COL_MIX
    _JOINT_COL = a.joint_col
    _COL_MIX = a.col_mix
    _LEGACY_VALUE_SKEW = a.legacy_value_skew or (a.value_skew_ncols > 0)
    _VALUE_SKEW_NCOLS = a.value_skew_ncols if a.value_skew_ncols > 0 else None
    if a.joint_col:
        a.tag = a.tag or "_jcol{}{}".format(
            a.value_skew_ncols or 1, "" if a.col_mix >= 1.0 else "_m{}".format(a.col_mix))
    elif a.legacy_value_skew or a.value_skew_ncols > 0:
        a.tag = a.tag or "_vsk{}".format(a.value_skew_ncols or "legacy")

    if not a.sweep:
        res = run_one(a.dataset, a.model, a.data_update, a.drift_test,
                      a.random_seed, a.skew_ratio, a.bootstrap, a.sample_size,
                      a.tag, quiet=False)
        print("==>", _fmt(res))
        return

    skews = [float(x) for x in a.skew_list.split(",") if x] or [a.skew_ratio]
    seeds = [int(x) for x in a.seed_list.split(",") if x] or [a.random_seed]
    dts = [x for x in a.drift_list.split(",") if x] or [a.drift_test]
    for dt in dts:
        for sk in skews:
            for sd in seeds:
                res = run_one(a.dataset, a.model, a.data_update, dt, sd, sk,
                              a.bootstrap, a.sample_size, a.tag)
                print("[{}/{}/skew={}/seed={}] {}".format(
                    a.dataset, dt, sk, sd, _fmt(res)), flush=True)


if __name__ == "__main__":
    main()
