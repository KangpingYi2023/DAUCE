"""Aggregate end2end experiment records (E4/E5/E6) into summary CSVs.

Every sub-experiment of end2end/experiment.py writes one record file into
end2end/experiment-records/. Each record embeds its CLI arguments
(``Input arguments = Namespace(...)``), the parameter store snapshot
(``JSON-passed parameters = {...}``), per-query q-errors
(``ReportEsts: [...]``, one line per query workload), and a final
``Experiment Summary`` line. This script turns those text records into one
tidy CSV per experiment tag (the ``experiment.tag`` parameter).

Usage (from the DAUCE root):
    python experiments/aggregate_e2e.py --tag E4 --out experiments/results/E4_framework.csv
"""
import argparse
import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("./")
from update_utils import path_util

NAMESPACE_KEYS = [
    "dataset", "model", "model_update", "drift_test", "data_update",
    "update_size", "num_workload", "query_seed", "num_queries",
]
# Param-store keys surfaced as their own columns (E5 switches + E6 sweeps).
PARAM_KEYS = [
    "random_seed", "data_updater.skew_ratio", "data_updater.n_drift_cols",
    "naru.epochs", "transformer.epochs",
    "adapt.use_ema", "adapt.use_caching", "adapt.use_weighted_replay",
    "adapt.use_cosine_decay", "adapt.omega", "adapt.ema_beta",
    "adapt.ema_per_epoch", "adapt.transfer_ratio", "drift.weight", "drift.tau",
    # Anchor-strength knobs (design doc A3/A5) and the freeze switch (A14).
    "adapt.kd_weight", "adapt.ema_halflife_epochs", "adapt.freeze",
    # Common-random-numbers seeding for progressive sampling. Unset means the
    # q-error of a run is one unseeded draw, which is why two records sharing
    # every other key can still disagree.
    "eval.crn_seed",
]

RE_NAMESPACE = re.compile(r"Input arguments = Namespace\((.*)\)\s*$", re.M)
RE_PARAMS = re.compile(r"JSON-passed parameters = (\{.*\})\s*$", re.M)
RE_REPORT = re.compile(r"ReportEsts: \[([^\]]*)\]")
RE_SUMMARY = re.compile(
    r"#data-update-times=(\d+), #drift=(\d+) \| total-time=([\d.]+)"
)
RE_UPDATE_TIME = re.compile(r"Model-update-time: ([\d.]+)")
RE_DISTANCE = re.compile(r"Distance score: ([\d.eE+-]+)")
# Per-epoch loss decomposition (design doc A1). Values may be 'nan' when the
# arm has no replay set, so they are captured loosely and converted in Python.
RE_LOSSCOMP = re.compile(
    r"\[loss-comp\] mode=(\S+) ep=(\d+) s=(\S+) new=(\S+) replay=(\S+) "
    r"kd=(\S+) loss=(\S+)"
)
# Effective teacher configuration as actually applied (design doc A5). This is
# the proof-the-knob-is-live column: the layered-freeze bug showed that a
# configured switch can silently do nothing.
RE_EMA = re.compile(
    r"\[ema\] pool=(\d+) S=(\d+) steps/epoch refresh=(\S+) beta=(\S+) "
    r"half_life=(\S+) epochs theta0_residual_after_1epoch=(\S+)"
)
# Anchor staleness probe (design doc A4).
RE_STALE = re.compile(
    r"\[stale\] event=(\d+) pool=(\d+) old_rows=(\d+) nll_theta0_new=(\S+) "
    r"nll_theta0_old=(\S+) stale=(\S+) stale_per_col=(\S+)"
)
# Proof that the layered freeze was actually applied (design doc A14). Absent
# line == full-network training. Same lesson as the `requires_grad ==` typo:
# record the effect, not just the configuration.
RE_FREEZE = re.compile(r"\[freeze\] \w+Task froze (\d+)/(\d+) param tensors")
# Proof that the eval-side sampler was seeded. Absent line == every q-error in
# this record is one random draw from an unseeded global torch RNG.
RE_CRN = re.compile(r"\[crn\] progressive sampling seeded per query, base=(\d+)")


def _to_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def _losscomp_events(text: str):
    """Group the per-epoch component lines into update events.

    A new event starts whenever ``ep`` returns to 0, which avoids depending on
    the configured epoch count (smoke runs use 5, the real runs 40).
    """
    events, cur = [], []
    for m in RE_LOSSCOMP.finditer(text):
        ep = int(m.group(2))
        if ep == 0 and cur:
            events.append(cur)
            cur = []
        cur.append({
            "ep": ep,
            "new": _to_float(m.group(4)),
            "replay": _to_float(m.group(5)),
            "kd": _to_float(m.group(6)),
            "loss": _to_float(m.group(7)),
        })
    if cur:
        events.append(cur)
    return events


def _parse_namespace(text: str) -> dict:
    m = RE_NAMESPACE.search(text)
    if not m:
        return {}
    try:
        call = ast.parse(f"Namespace({m.group(1)})", mode="eval").body
        return {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    except (SyntaxError, ValueError):
        return {}


def _parse_params(text: str) -> dict:
    m = RE_PARAMS.search(text)
    if not m:
        return {}
    try:
        # params_as_str is a Python repr of the nested params subtree
        # (single quotes), not strict JSON.
        return ast.literal_eval(m.group(1))
    except (SyntaxError, ValueError):
        return {}


def _nested_get(d: dict, dotted_key: str):
    """The store keeps dotted keys as nested dicts ('adapt.omega' ->
    {'adapt': {'omega': ...}}); walk down to fetch."""
    node = d
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def parse_record(path: Path) -> dict:
    text = path.read_text(errors="replace")

    row = {"record": path.name}
    ns = _parse_namespace(text)
    for key in NAMESPACE_KEYS:
        row[key] = ns.get(key)
    row["sseed"] = ns.get("random_seed")

    params = _parse_params(text)
    row["experiment.tag"] = _nested_get(params, "experiment.tag")
    for key in PARAM_KEYS:
        row[key] = _nested_get(params, key)

    # q-errors: the first ReportEsts line is the untouched baseline model;
    # everything after it is measured under drift + updates.
    errs_all = RE_REPORT.findall(text)
    if errs_all:
        first = np.asarray(ast.literal_eval("[" + errs_all[0] + "]"), dtype=float)
        rest = np.concatenate(
            [np.asarray(ast.literal_eval("[" + e + "]"), dtype=float)
             for e in errs_all[1:]]
        ) if len(errs_all) > 1 else np.array([])
        row["baseline_qerr_median"] = float(np.median(first))
        if rest.size:
            row["qerr_median"] = float(np.median(rest))
            row["qerr_p90"] = float(np.quantile(rest, 0.90))
            row["qerr_p95"] = float(np.quantile(rest, 0.95))
            row["qerr_max"] = float(np.max(rest))
            row["qerr_mean"] = float(np.mean(rest))
            row["num_query_workloads"] = len(errs_all)

    m = RE_SUMMARY.search(text)
    if m:
        row["n_data_updates"] = int(m.group(1))
        row["n_drift"] = int(m.group(2))
        row["total_time_s"] = float(m.group(3))

    times = RE_UPDATE_TIME.findall(text)
    row["model_update_time_s"] = float(np.sum(np.asarray(times, dtype=float))) if times else 0.0

    distances = RE_DISTANCE.findall(text)
    if distances:
        row["distance_mean"] = float(np.mean(np.asarray(distances, dtype=float)))

    # --- A14: did the layered freeze actually run? -------------------------
    m_frz = RE_FREEZE.search(text)
    row["freeze_active"] = bool(m_frz)
    if m_frz:
        row["frozen_param_tensors"] = f"{m_frz.group(1)}/{m_frz.group(2)}"

    # --- was the eval sampler seeded, i.e. is this q-error reproducible? ----
    m_crn = RE_CRN.search(text)
    row["crn_active"] = bool(m_crn)
    if m_crn:
        row["crn_base"] = int(m_crn.group(1))

    # --- A1: is the KD anchor carrying any signal? -------------------------
    events = _losscomp_events(text)
    if events:
        ev1, evn = events[0], events[-1]
        row["n_update_events_logged"] = len(events)
        row["epochs_per_event"] = len(ev1)
        row["kd_ev1_ep0"] = ev1[0]["kd"]
        row["kd_ev1_last"] = ev1[-1]["kd"]
        row["kd_evN_last"] = evn[-1]["kd"]
        row["replay_ev1_ep0"] = ev1[0]["replay"]
        row["replay_ev1_last"] = ev1[-1]["replay"]
        row["new_ev1_last"] = ev1[-1]["new"]
        row["loss_ev1_last"] = ev1[-1]["loss"]
        # Collapse ratio: KD at the end of the first event relative to its
        # value at the very first epoch. ~1 means the anchor held, ~0 means the
        # teacher caught up with the student and KD stopped constraining it.
        base = ev1[0]["kd"]
        row["kd_collapse_ratio"] = (
            evn[-1]["kd"] / base if base and base == base and base > 0 else float("nan")
        )

    # --- A5: what teacher was actually used? -------------------------------
    m_ema = RE_EMA.search(text)
    if m_ema:
        row["ema_pool"] = int(m_ema.group(1))
        row["ema_steps_per_epoch"] = int(m_ema.group(2))
        row["ema_refresh"] = m_ema.group(3)
        row["ema_beta_eff"] = _to_float(m_ema.group(4))
        row["ema_half_life_epochs"] = _to_float(m_ema.group(5))
        row["ema_theta0_residual_1ep"] = _to_float(m_ema.group(6))

    # --- A4: how stale is the event-start teacher? -------------------------
    stale_rows = [m for m in RE_STALE.finditer(text)]
    if stale_rows:
        per_col = np.asarray(
            [_to_float(m.group(7)) for m in stale_rows], dtype=float
        )
        raw = np.asarray([_to_float(m.group(6)) for m in stale_rows], dtype=float)
        row["stale_per_col_mean"] = float(np.nanmean(per_col))
        row["stale_mean"] = float(np.nanmean(raw))
        row["nll_theta0_new_mean"] = float(
            np.nanmean([_to_float(m.group(4)) for m in stale_rows])
        )
        row["nll_theta0_old_mean"] = float(
            np.nanmean([_to_float(m.group(5)) for m in stale_rows])
        )

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="./end2end/experiment-records")
    parser.add_argument("--tag", required=True,
                        help="experiment.tag value(s), comma-separated "
                             "(e.g. E4 or E5_full,E5_no_ema)")
    parser.add_argument("--out", required=True, help="output CSV path")
    args = parser.parse_args()

    records_dir = path_util.get_absolute_path(args.records)
    rows = [parse_record(p) for p in sorted(Path(records_dir).glob("*.txt"))]
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit(f"no records found in {records_dir}")

    tags = [t.strip() for t in args.tag.split(",") if t.strip()]
    sel = df[df["experiment.tag"].isin(tags)].copy()
    if sel.empty:
        sys.exit(f"no records tagged {tags} "
                 f"(tags seen: {sorted(df['experiment.tag'].dropna().unique())})")

    out_path = Path(path_util.get_absolute_path(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out_path, index=False)
    print(f"wrote {len(sel)} rows -> {out_path}")

    group_keys = [c for c in ["experiment.tag", "dataset", "model",
                               "model_update",
                               "drift_test",
                               # drift magnitude: only vary in the E4mag sweep, so
                               # the nunique guard keeps them out of E4/E5/E6
                               "update_size", "data_updater.skew_ratio",
                               "data_updater.n_drift_cols",
                               "adapt.omega", "adapt.ema_beta",
                               "adapt.transfer_ratio", "drift.weight",
                               "drift.tau", "naru.epochs",
                               "adapt.use_ema", "adapt.use_caching",
                               "adapt.use_weighted_replay",
                               "adapt.use_cosine_decay", "adapt.ema_per_epoch",
                               # anchor knobs and the proof they took effect
                               "adapt.kd_weight", "adapt.ema_halflife_epochs",
                               "adapt.freeze", "freeze_active",
                               "eval.crn_seed", "crn_active",
                               "ema_beta_eff", "ema_half_life_epochs"]
                  if c in sel and sel[c].nunique(dropna=True) > 1]
    cols = group_keys + ["qerr_median", "qerr_p95", "qerr_max", "n_drift",
                         "model_update_time_s"]
    print(sel[cols].to_string(index=False))

    # Mechanism view: the point of A1/A4/A5 is that the KD anchor becomes
    # observable, so print it next to the q-error whenever it was logged.
    mech = [c for c in ["dataset", "query_seed", "adapt.kd_weight",
                        "adapt.ema_halflife_epochs", "adapt.freeze",
                        "freeze_active", "ema_beta_eff",
                        "ema_steps_per_epoch", "ema_theta0_residual_1ep",
                        "kd_ev1_ep0", "kd_ev1_last", "kd_evN_last",
                        "kd_collapse_ratio", "replay_ev1_last", "new_ev1_last",
                        "loss_ev1_last", "stale_per_col_mean",
                        "qerr_median", "qerr_p95", "baseline_qerr_median"]
            if c in sel and sel[c].notna().any()]
    if len(mech) > 2:
        print("\n=== mechanism view (A1/A4/A5) ===")
        print(sel[mech].to_string(index=False))


if __name__ == "__main__":
    main()
