"""Evaluate estimators (Naru or others) on queries."""
import argparse
import collections
import glob
import os
import pickle
import shutil
import re
import copy
import time
import sys
import pytz
from pathlib import Path
from datetime import datetime

sys.path.append("./")
from typing import List

import numpy as np
import pandas as pd
import torch

import common
import estimators as estimators_lib
import made
import transformer
from FACE.data import table_sample
from FACE.data.table_sample import MyFlowModel
from FactorJoin.send_query import send_query
from workloads.parse_query import parse_query
from end2end import data_updater
from sqlParser import Parser
from update_utils import dataset_util, log_util
from update_utils.arg_util import add_common_arguments, ArgType
from update_utils.end2end_utils import communicator
from update_utils.end2end_utils.json_communicator import JsonCommunicator
from update_utils.path_util import get_absolute_path, convert_path_to_linux_style
from update_utils.torch_util import get_torch_device
from update_utils.dataset_util import CsvDatasetLoader
from update_utils.db_utils import update_pg


# For inference speed.
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

DEVICE = get_torch_device()


def create_parser():
    parser = argparse.ArgumentParser()

    # global paramaters
    common_args: List[ArgType] = [
        ArgType.DATA_UPDATE,
        ArgType.DATASET,
        ArgType.DRIFT_TEST,
        ArgType.END2END,
        ArgType.MODEL_UPDATE,
    ]
    add_common_arguments(parser, arg_types=common_args)

    # parser.add_argument("--dataset", type=str, default="census", help="Dataset.")

    parser.add_argument("--random_seed", type=int, default=1234, help="Random seed")
    
    parser.add_argument("--num_workload", type=int, default=5, help="number of workloads")

    parser.add_argument(
        "--eval_type",
        type=str,
        choices=["estimate", "drift", "first_estimate"],
        default="estimate",
        help="Model evaluation type, estimate or drift",
    )

    parser.add_argument(
        "--inference-opts",
        action="store_true",
        help="Tracing optimization for better latency.",
    )

    parser.add_argument(
        "--err-csv",
        type=str,
        default="results.csv",
        help="Save result csv to what path?",
    )
    parser.add_argument("--glob", type=str, help="Checkpoints to glob under models/.")
    parser.add_argument(
        "--blacklist", type=str, help="Remove some globbed checkpoint files."
    )
    parser.add_argument(
        "--psample",
        type=int,
        default=2000,
        help="# of progressive samples to use per query.",
    )
    parser.add_argument(
        "--column-masking",
        action="store_true",
        help="Turn on wildcard skipping.  Requires checkpoints be trained with "
        "column masking.",
    )
    parser.add_argument("--order", nargs="+", type=int, help="Use a specific order?")

    # MADE.
    parser.add_argument(
        "--fc-hiddens", type=int, default=256, help="Hidden units in FC."
    )
    parser.add_argument("--layers", type=int, default=4, help="# layers in FC.")
    parser.add_argument("--residual", action="store_true", help="ResMade?")
    parser.add_argument("--direct-io", action="store_true", help="Do direct IO?")
    parser.add_argument(
        "--inv-order",
        action="store_true",
        help="Set this flag iff using MADE and specifying --order. Flag --order"
        "lists natural indices, e.g., [0 2 1] means variable 2 appears second."
        "MADE, however, is implemented to take in an argument the inverse "
        "semantics (element i indicates the position of variable i).  Transformer"
        " does not have this issue and thus should not have this flag on.",
    )
    parser.add_argument(
        "--input-encoding",
        type=str,
        default="binary",
        help="Input encoding for MADE/ResMADE, {binary, one_hot, embed}.",
    )
    parser.add_argument(
        "--output-encoding",
        type=str,
        default="one_hot",
        help="Iutput encoding for MADE/ResMADE, {one_hot, embed}.  If embed, "
        "then input encoding should be set to embed as well.",
    )

    # Transformer.
    parser.add_argument(
        "--heads",
        type=int,
        default=0,
        help="Transformer: num heads.  A non-zero value turns on Transformer"
        " (otherwise MADE/ResMADE).",
    )
    parser.add_argument(
        "--blocks", type=int, default=2, help="Transformer: num blocks."
    )
    parser.add_argument("--dmodel", type=int, default=32, help="Transformer: d_model.")
    parser.add_argument("--dff", type=int, default=128, help="Transformer: d_ff.")
    parser.add_argument(
        "--transformer-act", type=str, default="gelu", help="Transformer activation."
    )

    # Estimators to enable.
    parser.add_argument(
        "--run-sampling", action="store_true", help="Run a materialized sampler?"
    )
    parser.add_argument(
        "--run-maxdiff", action="store_true", help="Run the MaxDiff histogram?"
    )
    parser.add_argument(
        "--run-bn", action="store_true", help="Run Bayes nets? If enabled, run BN only."
    )

    # Bayes nets.
    parser.add_argument(
        "--bn-samples", type=int, default=200, help="# samples for each BN inference."
    )
    parser.add_argument(
        "--bn-root", type=int, default=0, help="Root variable index for chow liu tree."
    )
    # Maxdiff
    parser.add_argument(
        "--maxdiff-limit",
        type=int,
        default=30000,
        help="Maximum number of partitions of the Maxdiff histogram.",
    )

    return parser


args = None  # module-level global; populated by main(). Kept as a module global
# (rather than parsed at import time) so this file can be imported without
# consuming sys.argv, while the many helper functions that reference `args`
# continue to work unchanged once main() has set it.


def Entropy(name, data, bases=None):
    import scipy.stats

    # s = 'Entropy of {}:'.format(name)
    ret = []
    for base in bases:
        assert base == 2 or base == "e" or base is None
        e = scipy.stats.entropy(data, base=base if base != "e" else None)
        ret.append(e)
        unit = "nats" if (base == "e" or base is None) else "bits"
        # s += ' {:.4f} {}'.format(e, unit)
    # print(s)
    return ret


def RunEpoch(
    split,
    model,
    opt,
    scheduler,
    train_data,
    val_data=None,
    batch_size=100,
    upto=None,
    epoch_num=None,
    verbose=False,
    log_every=10,
    return_losses=False,
    table_bits=None,
):
    torch.set_grad_enabled(split == "train")
    model.train() if split == "train" else model.eval()
    dataset = train_data if split == "train" else val_data
    losses = []

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == "train")
    )

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, "orderings"):
        nsamples = len(model.orderings)

    for step, xb in enumerate(loader):
        if upto and step >= upto:
            break

        xb = xb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.
        xbhat = None
        model_logits = []
        num_orders_to_forward = 1
        if split == "test" and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, "update_masks"):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_out = model(xb)
            model_logits.append(model_out)
            if xbhat is None:
                xbhat = torch.zeros_like(model_out)
            xbhat += model_out

        loss = model.nll(xbhat, xb)
        losses+=loss.tolist()
        loss=loss.mean()

        if step % log_every == 11110:
            if split == "train":
                print(
                    "Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr".format(
                        epoch_num,
                        step,
                        split,
                        loss.item() / np.log(2) - table_bits,
                        loss.item() / np.log(2),
                        table_bits,
                        opt.param_groups[0]["lr"],
                    )
                )
            else:
                print(
                    "Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits".format(
                        epoch_num, step, split, loss.item(), loss.item() / np.log(2)
                    )
                )

        if split == "train":
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose:
            print("%s epoch average loss: %f" % (split, np.mean(losses)))

    if return_losses:
        return losses
    return np.mean(losses)


def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the "true" ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


def MakeTable(step=None):
    # Load dataset
    if args.eval_type == "first_estimate":
        table = dataset_util.DatasetLoader.load_dataset(dataset=args.dataset)
    else:
        table, _ = dataset_util.DatasetLoader.load_permuted_dataset(
            dataset=args.dataset, permute=False
        )

    oracle_est = estimators_lib.Oracle(table)
    if args.run_bn:
        return table, common.TableDataset(table), oracle_est
    return table, None, oracle_est


def ErrorMetric(est_card, card):
    if card == 0 and est_card != 0:
        return est_card
    if card != 0 and est_card == 0:
        return card
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)


_WORKLOAD_SHAPE = None


def _workload_shape():
    """Filter count range and operator mix, overridable from the config.

    The workload shape used to be hard-coded per dataset, which makes the
    ``workload-agnostic`` claim untestable: there was no way to ask for a
    different shape. Read once per process and cached, because the context store
    re-reads its file on every lookup and this sits inside the per-query loop.

    ``None`` everywhere keeps the historical hard-coded shape, so nothing already
    on disk changes meaning.
    """
    global _WORKLOAD_SHAPE
    if _WORKLOAD_SHAPE is None:
        try:
            jc = JsonCommunicator()
            shape = {
                "min_filters": jc.get_with_default("workload.min_filters", None),
                "max_filters": jc.get_with_default("workload.max_filters", None),
                "op_weights": jc.get_with_default("workload.op_weights", None),
            }
        except Exception:
            # Non-end2end entry points have no experiment context to read from.
            shape = {"min_filters": None, "max_filters": None, "op_weights": None}
        if (shape["min_filters"] is None) != (shape["max_filters"] is None):
            raise ValueError(
                "workload.min_filters and workload.max_filters must be set together"
            )
        if shape["op_weights"] is not None:
            weights = [float(w) for w in shape["op_weights"]]
            if len(weights) != 3 or abs(sum(weights) - 1.0) > 1e-9:
                raise ValueError(
                    "workload.op_weights must be 3 probabilities over ['<=','>=','='] "
                    "summing to 1, got {}".format(shape["op_weights"])
                )
            shape["op_weights"] = weights
        _WORKLOAD_SHAPE = shape
        if shape["min_filters"] is not None or shape["op_weights"] is not None:
            print("[workload] shape override {}".format(shape))
    return _WORKLOAD_SHAPE


def SampleTupleThenRandom(
    all_cols, num_filters, rng, table, return_col_idx=False, query_num=0
):
    s = table.data.iloc[rng.randint(0, table.cardinality)]
    vals = s.values

    if args.dataset in ["dmv", "dmv-tiny"]:
        # Giant hack for DMV.
        vals[6] = vals[6].to_datetime64()

    idxs = rng.choice(len(all_cols), replace=False, size=num_filters)
    cols = np.take(all_cols, idxs)
    vals = vals[idxs]

    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.

    op_weights = _workload_shape()["op_weights"]
    if op_weights is None:
        ops = rng.choice(["<=", ">=", "="], size=num_filters)
    else:
        # A weighted draw does not consume the same randoms as an unweighted one,
        # so the unweighted call has to stay reachable verbatim; otherwise merely
        # declaring the uniform mix explicitly would move every historical number.
        ops = rng.choice(["<=", ">=", "="], size=num_filters, p=op_weights)
    ops_all_eqs = ["="] * num_filters

    sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)

    """
    if num_filters == len(all_cols):
        if return_col_idx:
            return np.arange(len(all_cols)), ops, vals
        return all_cols, ops, vals


    if return_col_idx:
        return idxs, ops, vals
    """

    return cols, ops, vals


def GenerateQuery(
    all_cols,
    rng,
    table,
    return_col_idx=False,
    previous_queries=False,
    i=0,
    loaded_file=None,
    query_num=0,
):
    """Generate a random query."""
    if not previous_queries:
        shape = _workload_shape()
        if shape["min_filters"] is not None:
            num_filters = rng.randint(
                int(shape["min_filters"]), int(shape["max_filters"])
            )
        elif "census" in args.dataset:
            num_filters = rng.randint(5, 12)
        elif "forest" in args.dataset:
            num_filters = rng.randint(3, 9)
        elif "bjaq" in args.dataset:
            num_filters = rng.randint(2, 4)  
        elif "power" in args.dataset:
            num_filters = rng.randint(3, 6)  
        else:
            return
        cols, ops, vals = SampleTupleThenRandom(
            all_cols,
            num_filters,
            rng,
            table,
            return_col_idx=return_col_idx,
            query_num=query_num,
        )
        relative_path = "./query/previous_queries.csv"
        absolute_path = get_absolute_path(relative_path)
        pq = open(absolute_path, "a+")
        line = ""
        for i, c in enumerate(cols):
            line = line + c.name + "," + ops[i] + "," + str(vals[i]) + ","
        pq.write(line[:-1])
        # pq.write(','.join([c.name for c in cols]))
        # pq.write('|')
        # pq.write(','.join(ops))
        # pq.write('|')
        # pq.write(','.join([str(v) for v in vals]))
        pq.write("\n")
        pq.close()

        return cols, ops, vals

    else:
        cols = ops = vals = []
        q = ""
        if loaded_file:
            q = loaded_file[i]
        else:
            with open("previous_queries.csv", "r") as f:
                for cnt, line in enumerate(f):
                    if cnt == i:
                        q = line

        splits = q.split("\n")[0].split(",")

        atts = []
        ops = []
        vals = []
        for i in range(0, len(splits), 3):
            atts.append(splits[i])
            ops.append(splits[i + 1])
            vals.append(splits[i + 2])

        # atts = atts.split(',')
        # ops = ops.split(',')
        # vals = vals.split(',')
        for att in atts:
            for col in all_cols:
                if att == col.name:
                    cols.append(col)

        """ for DMV dataset                    
        for idx, val in enumerate(vals):
            try:
                vals[idx] = pd.to_datetime(val).to_datetime64()
            except:
                val = val.split('\n')[0]
                try:
                    vals[idx] = float(val)
                except:
                    vals[idx] = val
        """
        for idx, val in enumerate(vals):
            val = val.split("\n")[0]
            try:
                vals[idx] = float(val)
            except:
                vals[idx] = val

        return cols, ops, vals


def ReadMyQuery(all_cols, query_file, query_idx):
    query = ""
    with open(query_file, "r") as file:
        for i, line in enumerate(file):
            if i == query_idx:
                qs = line.split(";")
                for itm in qs:
                    if itm.lower().startswith("select"):
                        query = itm
                    elif len(itm) > 1:
                        print("query not recognized \n {}".format(itm))

    table_cols = [c.name for c in all_cols]
    parser = Parser()
    succ, conditions = parser.parse(query, table_cols)

    cols = []
    ops = []
    vals = []
    if succ:
        for key in conditions.keys():
            selected_col = None
            for c in all_cols:
                if c.name == key:
                    selected_col = c
            if conditions[key].lb is not None:
                cols.append(selected_col)
                ops.append(">=")
                vals.append(conditions[key].lb)

            if conditions[key].ub is not None:
                cols.append(selected_col)
                ops.append("<=")
                vals.append(conditions[key].ub)

            if conditions[key].equalities:
                for itm in conditions[key].equalities:
                    cols.append(selected_col)
                    ops.append("=")
                    vals.append(float(itm))
    else:
        print("query not supported!")
        return

    return cols, np.array(ops), np.array(vals)


def ReadLatestQueries(file, query_idx, generated_query):
    with open(file, "r") as f:
        for i, line in enumerate(f):
            if i == query_idx:
                c, o, v = line.split("|")
                cols = c.split(",")
                ops = o.split(",")
                vals = v.split(",")
                for idx, val in enumerate(vals):
                    try:
                        vals[idx] = float(val)
                    except:
                        if len(val) > 20:
                            val = pd.to_datetime(val).to_datetime64()
                        vals[idx] = val

                gcols, gops, gvals = generated_query
                for i, c in enumerate(gcols):
                    gcols[i].name = cols[i]
                for i, op in enumerate(gops):
                    gops[i] = ops[i]
                for i, val in enumerate(gvals):
                    gvals[i] = vals[i]
                return gcols, gops, gvals


def Query(
    estimators, do_print=True, oracle_card=None, query=None, table=None, oracle_est=None
):
    assert query is not None
    cols, ops, vals = query

    ### Actually estimate the query.

    def pprint(*args, **kwargs):
        if do_print:
            print(*args, **kwargs)

    # Actual.
    card = oracle_est.Query(cols, ops, vals) if oracle_card is None else oracle_card

    if card == 0:
        return

    # pprint("Q(", end="")
    # for c, o, v in zip(cols, ops, vals):
    #     pprint("{} {} {}, ".format(c.name, o, v), end="")

    # pprint("): ", end="")

    # pprint("actual {} ({:.3f}%) ".format(card, card / table.cardinality * 100), end=" ")

    for est in estimators:
        est_card = est.Query(cols, ops, vals)
        err = ErrorMetric(est_card, card)
        est.AddError(err, est_card, card)
        # pprint("{} {} (err={:.3f}) ".format(est.name, est_card, err), end=" ")
    # pprint()


def ReportEsts(estimators: list) -> list:
    v = -1
    errs_list = []
    if not args.end2end:
        current_datetime = datetime.now(pytz.timezone("Asia/Shanghai"))
        formatted_datetime = current_datetime.strftime(
            "%y%m%d-%H%M%S"
        )  
        output_file_name = (
            f"{args.model}+"  
            f"{args.dataset}+"  
            # f"{args.data_update}+"  
            f"qseed{args.query_seed}+"  
            f"t{formatted_datetime}"  
            f".txt"
        )
        output_file_path=f"./end2end/model-evaluation/{output_file_name}"
        log_util.append_to_file(output_file_path, f"Model evaluation results for {args.dataset}+{args.model}+query_seed{args.query_seed}\n")
    for est in estimators:
        message="{} \tmax: {:.4f}\t99th: {:.4f}\t95th: {:.4f}\tmedian: {:.4f}\tmean: {:.4f}".format(
                    est.name,
                    np.max(est.errs),
                    np.quantile(est.errs, 0.99),
                    np.quantile(est.errs, 0.95),
                    np.quantile(est.errs, 0.5),
                    np.mean(est.errs),
                )
        v = max(v, np.max(est.errs))
        if args.end2end:
            print(message)
            print(f"ReportEsts: {est.errs}")

            random_seed=JsonCommunicator().get(f"random_seed")
            log_file_name=(
                f"{args.model}+"  
                f"{args.dataset}+"  
                f"num_workloads{args.num_workload}+"
                f"{args.data_update}+"  
                f"{args.model_update}+" 
                f"qseed{args.query_seed}+"  
                f"rseed{random_seed}" 
                f".txt"
            )
            log_file_path=f"./end2end/end2end-evaluations/{log_file_name}"
            log_util.append_to_file(log_file_path, f"{message}\n")
        else:
            log_util.append_to_file(output_file_path, f"{message}\n")
        errs_list.append(est.errs)
    return errs_list


def _crn_seed_from_context():
    """Read ``eval.crn_seed``; ``None`` keeps the historical unseeded behaviour.

    ProgressiveSampling draws with ``torch.multinomial`` from the GLOBAL torch RNG
    and nothing on this path ever seeds it, so evaluating the same checkpoint twice
    reports two different q-error vectors. Seeding per query turns an A/B comparison
    into a common-random-numbers one: the same uniform stream drives both arms, so
    what is left in the difference is the model's, not the sampler's.

    Left at ``None`` by default because a fixed draw is a *different* draw from the
    one any historical run happened to make, so enabling it silently would break
    bit-level comparability with results already on disk.
    """
    try:
        value = JsonCommunicator().get_with_default("eval.crn_seed", None)
    except Exception:
        # Non-end2end entry points have no experiment context to read from.
        return None
    if value is None:
        return None
    return int(value)


def RunN(
    table,
    cols,
    estimators,
    rng=None,
    num=50,
    query_loc="benchmark/alldata/00.sql",
    log_every=50,
    num_filters=2,
    oracle_cards=None,
    oracle_est=None,
    crn_seed=None,
):
    if rng is None:
        rng = np.random.RandomState(1234)

    if crn_seed is None:
        crn_seed = _crn_seed_from_context()
    if crn_seed is not None:
        print("[crn] progressive sampling seeded per query, base={}".format(crn_seed))

    # rng = np.random.RandomState()
    last_time = None

    pr_qs = []
    if os.path.isfile("previous_queries.csv"):
        with open("previous_queries.csv", "r") as f:
            for line in f:
                pr_qs.append(line)

    for i in range(num):
        do_print = False
        if i % log_every == 0:
            if last_time is not None:
                # print(
                #     "{:.1f} queries/sec".format(log_every / (time.time() - last_time))
                # )
                pass
            do_print = True
            # print("Query {}:".format(i), end=" ")
            last_time = time.time()

        query = GenerateQuery(
            cols,
            rng,
            table,
            previous_queries=False,
            i=i,
            loaded_file=pr_qs,
            query_num=i,
        )

        # query = ReadLatestQueries('last_queries.sql', i, query)
        # query = ReadMyQuery(cols, query_loc, i)

        if crn_seed is not None:
            # Reseed per query, not once for the loop. Should the two arms ever
            # consume a different number of randoms inside a single query, one
            # seeding at the top would desynchronise every query after it and
            # silently destroy the pairing this exists to create. Seeding here
            # rather than before GenerateQuery keeps nothing between the seed and
            # the multinomial draws able to consume the stream.
            torch.manual_seed(crn_seed + i)

        Query(
            estimators,
            do_print,
            oracle_card=oracle_cards[i]
            if oracle_cards is not None and i < len(oracle_cards)
            else None,
            query=query,
            table=table,
            oracle_est=oracle_est,
        )

    errs_list = ReportEsts(estimators)

    # print("max_err", max_err)

    return False


def RunNParallel(
    estimator_factory,
    parallelism=2,
    rng=None,
    num=20,
    num_filters=11,
    oracle_cards=None,
):
    """RunN in parallel with Ray.  Useful for slow estimators e.g., BN."""
    import ray

    ray.init(redis_password="xxx")

    @ray.remote
    class Worker(object):
        def __init__(self, i):
            self.estimators, self.table, self.oracle_est = estimator_factory()
            self.columns = np.asarray(self.table.columns)
            self.i = i

        def run_query(self, query, j):
            col_idxs, ops, vals = pickle.loads(query)
            Query(
                self.estimators,
                do_print=True,
                oracle_card=oracle_cards[j] if oracle_cards is not None else None,
                query=(self.columns[col_idxs], ops, vals),
                table=self.table,
                oracle_est=self.oracle_est,
            )

            print("=== Worker {}, Query {} ===".format(self.i, j))
            for est in self.estimators:
                est.report()

        def get_stats(self):
            return [e.get_stats() for e in self.estimators]

    print("Building estimators on {} workers".format(parallelism))
    workers = []
    for i in range(parallelism):
        workers.append(Worker.remote(i))

    print("Building estimators on driver")
    estimators, table, _ = estimator_factory()
    cols = table.columns

    if rng is None:
        rng = np.random.RandomState(1234)
    queries = []
    for i in range(num):
        col_idxs, ops, vals = GenerateQuery(cols, rng, table=table, return_col_idx=True)
        queries.append((col_idxs, ops, vals))

    cnts = 0
    for i in range(num):
        query = queries[i]
        print("Queueing execution of query", i)
        workers[i % parallelism].run_query.remote(pickle.dumps(query), i)

    print("Waiting for queries to finish")
    stats = ray.get([w.get_stats.remote() for w in workers])

    print("Merging and printing final results")
    for stat_set in stats:
        for e, s in zip(estimators, stat_set):
            e.merge_stats(s)
    time.sleep(1)

    print("=== Merged stats ===")
    for est in estimators:
        est.report()
    return estimators


def MakeBnEstimators():
    table, train_data, oracle_est = MakeTable()
    estimators = [
        estimators_lib.BayesianNetwork(
            train_data,
            args.bn_samples,
            "chow-liu",
            topological_sampling_order=True,
            root=args.bn_root,
            max_parents=2,
            use_pgm=False,
            discretize=100,
            discretize_method="equal_freq",
        )
    ]

    for est in estimators:
        est.name = str(est)
    return estimators, table, oracle_est


def MakeLargerMade(scale, cols_to_train, seed, fixed_ordering=None):
    if args.inv_order:
        print("Inverting order!")
        fixed_ordering = InvertOrder(fixed_ordering)

    model = made.MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] * args.layers
        if args.layers > 0
        else [512, 256, 512 + 64, 128 + 64, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
        embed_size=32,
        seed=seed,
        do_direct_io_connections=args.direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=args.residual,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
    ).to(DEVICE)

    return model


def MakeMade(scale, cols_to_train, seed, fixed_ordering=None):
    
    if args.inv_order:
        print("Inverting order!")
        fixed_ordering = InvertOrder(fixed_ordering)

    model = made.MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] * args.layers
        if args.layers > 0
        else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
        embed_size=32,
        seed=seed,
        do_direct_io_connections=args.direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=args.residual,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
    ).to(DEVICE)

    return model


def MakeTransformer(cols_to_train, fixed_ordering, seed=None):
    num_heads=4
    return transformer.Transformer(
        num_blocks=args.blocks,
        d_model=args.dmodel,
        d_ff=args.dff,
        num_heads=num_heads,
        nin=len(cols_to_train),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        use_positional_embs=True,
        activation=args.transformer_act,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
        seed=seed,
    ).to(DEVICE)


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print("Number of model parameters: {} (~= {:.1f}MB)".format(num_params, mb))
    print(model)
    return mb


def SaveEstimators(path, estimators, return_df=False):
    # name, query_dur_ms, errs, est_cards, true_cards
    results_list = []  

    for est in estimators:
        data = {
            "est": [est.name] * len(est.errs),
            "err": est.errs,
            "est_card": est.est_cards,
            "true_card": est.true_cards,
            "query_dur_ms": est.query_dur_ms,
        }
        results_list.append(pd.DataFrame(data))  

    results = pd.concat(results_list, ignore_index=True)  

    if return_df:
        return results

    results.to_csv(path, index=False)


def LoadOracleCardinalities():
    ORACLE_CARD_FILES = {"dmv": "datasets/dmv-2000queries-oracle-cards-seed1234.csv"}
    path = ORACLE_CARD_FILES.get(args.dataset, None)
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        assert len(df) == 2000, len(df)
        return df.values.reshape(-1)
    return None


def LoadModelByName(dataset, table, model_dir, columns):
    model_pattern = os.path.join(model_dir, "*{}_{}*MB-model*-data*-*epochs-seed*.pt".format(dataset, table))
    model_path = glob.glob(str(model_pattern))[0]

    reg_pattern = ".*/([\D]+)-.+model([\d\.]+)-data([\d\.]+).+seed([\d\.]+).*.pt"
    z = re.match(reg_pattern, convert_path_to_linux_style(model_path))
    assert z
    model_name = z.group(1)
    model_bits = float(z.group(2))
    data_bits = float(z.group(3))
    seed = int(z.group(4))
    bits_gap = model_bits - data_bits
    # table_cols = CsvDatasetLoader.DICT_FROM_DATASET_TO_COLS[f"{dataset}_{table}"]
    
    print(f"Loading model for {dataset}_{table}!")
    start_time=time.time()
    model = MakeMade(
        scale=256,
        cols_to_train=columns,
        seed = seed,
        fixed_ordering=None,
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    load_latency=time.time()-start_time
    print("Done, took {:.2f}s".format(load_latency))

    return model


def Model_Eval(args, end2end: bool = False):
    if not end2end:
        relative_model_paths = "./models/*{}*MB-model*-data*-*epochs-seed*.pt".format(
            args.dataset
        )
        absolute_model_paths = get_absolute_path(relative_model_paths)
        all_ckpts = glob.glob(str(absolute_model_paths))
        if args.blacklist:
            all_ckpts = [ckpt for ckpt in all_ckpts if args.blacklist not in ckpt]
    else:
        # end2end mode
        if args.dataset in ["stats"]:
            abs_model_dir = communicator.ModelPathCommunicator().get()
            # all_ckpts = [os.path.join(abs_model_dir, model_filename) for model_filename in os.listdir(abs_model_dir)]

        else:
            all_ckpts = [communicator.ModelPathCommunicator().get()]

    if args.dataset in ["stats"]:
        table_dict=dict()
        model_dict=dict()
        abs_dataset_dir = communicator.DatasetPathCommunicator().get()
        dataset_tables = [dataset for dataset in CsvDatasetLoader.DICT_FROM_DATASET_TO_COLS.keys() if args.dataset in dataset]

        for dataset_table in dataset_tables:
            dataset, table_name = dataset_table.split("_")
            table = dataset_util.DatasetLoader.load_dataset(dataset=dataset_table)
            table_dict[table_name] = table

            model_dict[table_name]=LoadModelByName(dataset, table_name, abs_model_dir, table.columns)

        psample=2048
        join_estimator=estimators_lib.JoinSampling(
            table_dict=table_dict,
            model_dict=model_dict,
            r=psample,
            device=DEVICE,
            estimate_type="progressive_sampling"
        )

        # query_path="./workloads/stats_CEB/stats_small.sql"
        # sub_query_path="./workloads/stats_CEB/sub_plan_queries/stats_CEB_sub_queries_small.sql"
        query_path="./workloads/stats_CEB/stats_light2.sql"
        sub_query_path="./workloads/stats_CEB/sub_plan_queries/stats_CEB_sub_queries_light2.sql"

        result_path="./end2end/pg/results/stats_sub_queries.txt"
        log_path = "./end2end/checkpoints/log.txt"
        latency_path="./end2end/pg/latency"
        with open(sub_query_path, "r") as f:
            queries = f.readlines()

        cards = []
        with open(log_path, "w") as log_file:
            for i, query_str in enumerate(queries):
                tables_all, table_predicates, equivalent_sets, true_card = parse_query(query_str)
                log_file.write(f"Query {i+1}:\n")
                for key in equivalent_sets:
                    log_file.write(f"{equivalent_sets[key]}\n")

                cardinality=join_estimator.JoinQuery(tables_all, table_predicates, equivalent_sets, log_file)
                cards.append(cardinality)
                # print(f"pred card: {cardinality}, true card: {true_card}. Q-error for query {i+1}: {qerror}")
                log_file.write(f"pred card for query {i+1}: {cardinality}.\n")
            # log_file.write(qerror_message)

        with open(result_path, "w") as result_file:
            for card in cards:
                result_file.write(f"{card}\n")

        postgre_workdir = os.environ.get("PG_WORKDIR", get_absolute_path("./FactorJoin/Postgres/data"))
        related_result_path = os.path.relpath(result_path, postgre_workdir)
        current_datetime = datetime.now(pytz.timezone("Asia/Shanghai"))
        formatted_datetime = current_datetime.strftime(
            "%y%m%d-%H%M%S"
        )  # Format date and time as 'yyMMdd-HHMMSS'
        if args.model_update == "none":
            test_type = "naive"
        else:
            test_type = "asm"
        send_query(dataset=dataset, method_name=related_result_path, query_file=query_path, save_folder=latency_path, test_type=test_type, update_type=args.model_update, iteration=formatted_datetime)
        
    else:
        selected_ckpts = all_ckpts
        oracle_cards = None  # LoadOracleCardinalities()
        print("ckpts", selected_ckpts)

        if not args.run_bn:
            # OK to load tables now
            table, train_data, oracle_est = MakeTable()
            cols_to_train = table.columns

        Ckpt = collections.namedtuple(
            "Ckpt", "epoch model_bits bits_gap path loaded_model seed model_name"
        )
        parsed_ckpts = []

        for s in selected_ckpts:
            if args.order is None:
                reg_pattern = (
                    ".*/([\D]+)-.+model([\d\.]+)-data([\d\.]+).+seed([\d\.]+).*.pt"
                )
            else:
                reg_pattern = ".+model([\d\.]+)-data([\d\.]+).+seed([\d\.]+)-order.*.pt"
            z = re.match(reg_pattern, convert_path_to_linux_style(s))
            assert z
            model_name = z.group(1)
            model_bits = float(z.group(2))
            data_bits = float(z.group(3))
            seed = int(z.group(4))
            bits_gap = model_bits - data_bits

            order = None
            if args.order is not None:
                order = list(args.order)

            assert args.model in ["transformer", "naru"], "Wrong Model!"
            if args.model == "transformer":
                model = MakeTransformer(
                    cols_to_train=table.columns, fixed_ordering=order, seed=seed
                )
            elif args.model == "naru":
                model = MakeMade(
                    scale=args.fc_hiddens,
                    cols_to_train=table.columns,
                    seed=seed,
                    fixed_ordering=order,
                )
                

            assert order is None or len(order) == model.nin, order
            # ReportModel(model)
            print("Loading ckpt:", s)
            model.load_state_dict(torch.load(s))
            model.eval()

            # print(s, bits_gap, seed)

            parsed_ckpts.append(
                Ckpt(
                    path=s,
                    epoch=None,
                    model_bits=model_bits,
                    bits_gap=bits_gap,
                    model_name=model_name,
                    loaded_model=model,
                    seed=seed,
                )
            )

        # Estimators to run.
        if args.run_bn:
            estimators = RunNParallel(
                estimator_factory=MakeBnEstimators,
                parallelism=50,
                rng=np.random.RandomState(1234),
                num=args.num_queries,
                num_filters=None,
                oracle_cards=oracle_cards,
            )
        else:
            estimators = [
                estimators_lib.ProgressiveSampling(
                    c.loaded_model,
                    table,
                    args.psample,
                    device=DEVICE,
                    shortcircuit=args.column_masking,
                )
                for c in parsed_ckpts
            ]
            for est, ckpt in zip(estimators, parsed_ckpts):
                est.name = "{}_{:.2f}".format(ckpt.model_name, ckpt.bits_gap)

            if args.inference_opts:
                print("Tracing forward_with_encoded_input()...")
                for est in estimators:
                    encoded_input = est.model.EncodeInput(
                        torch.zeros(args.psample, est.model.nin, device=DEVICE)
                    )

                    # NOTE: this line works with torch 1.0.1.post2 (but not 1.2).
                    # The 1.2 version changes the API to
                    # torch.jit.script(est.model) and requires an annotation --
                    # which was found to be slower.
                    est.traced_fwd = torch.jit.trace(
                        est.model.forward_with_encoded_input, encoded_input
                    )

            if args.run_sampling:
                SAMPLE_RATIO = {"dmv": [0.0013]}  # ~1.3MB.
                for p in SAMPLE_RATIO.get(args.dataset, [0.01]):
                    estimators.append(estimators_lib.Sampling(table, p=p))

            if args.run_maxdiff:
                estimators.append(
                    estimators_lib.MaxDiffHistogram(table, args.maxdiff_limit)
                )

            # Other estimators can be appended as well.

            random_seed = args.query_seed
            print(f"query_seed:{random_seed}")
            """
            if end2end:
                random_seed = (
                    communicator.RandomSeedCommunicator().get()
                )  
                communicator.RandomSeedCommunicator().update()  
            """
            if len(estimators):
                RunN(
                    table,
                    cols_to_train,
                    estimators,
                    rng=np.random.RandomState(random_seed),
                    num=args.num_queries,
                    log_every=1,
                    num_filters=None,
                    oracle_cards=oracle_cards,
                    oracle_est=oracle_est,
                )

        SaveEstimators(args.err_csv, estimators)
        print("...Done, result:", args.err_csv)


def MainAll(modelspath):
    models = np.sort(
        os.listdir(modelspath)
    )  # sorted(Path(modelspath).iterdir(), key=os.path.getmtime)
    models = [os.path.join(modelspath, i) for i in models]
    print(models)
    for step, path in enumerate(models):
        if step < 5:
            continue
        all_ckpts = glob.glob("./models/{}".format(args.glob))
        if args.blacklist:
            all_ckpts = [ckpt for ckpt in all_ckpts if args.blacklist not in ckpt]

        selected_ckpts = [str(path)]
        oracle_cards = None  # LoadOracleCardinalities()
        print("ckpts", selected_ckpts)

        if not args.run_bn:
            # OK to load tables now
            table, train_data, oracle_est = MakeTable(step=step)
            cols_to_train = table.columns

        Ckpt = collections.namedtuple(
            "Ckpt", "epoch model_bits bits_gap path loaded_model seed"
        )
        parsed_ckpts = []

        for s in selected_ckpts:
            if args.order is None:
                z = re.match(".+model([\d\.]+)-data([\d\.]+).+seed([\d\.]+).*.pt", s)
            else:
                z = re.match(
                    ".+model([\d\.]+)-data([\d\.]+).+seed([\d\.]+)-order.*.pt", s
                )
            assert z
            model_bits = float(z.group(1))
            data_bits = float(z.group(2))
            seed = int(z.group(3))
            bits_gap = model_bits - data_bits

            order = None
            if args.order is not None:
                order = list(args.order)

            if args.heads > 0:
                model = MakeTransformer(
                    cols_to_train=table.columns, fixed_ordering=order, seed=seed
                )
            else:
                if args.dataset in ["dmv-tiny", "dmv", "tpcds", "forest", "census"]:
                    model = MakeMade(
                        scale=args.fc_hiddens,
                        cols_to_train=table.columns,
                        seed=seed,
                        fixed_ordering=order,
                    )
                else:
                    assert False, args.dataset

            assert order is None or len(order) == model.nin, order
            ReportModel(model)
            print("Loading ckpt:", s)
            model.load_state_dict(torch.load(s))
            model.eval()

            print(s, bits_gap, seed)

            parsed_ckpts.append(
                Ckpt(
                    path=s,
                    epoch=None,
                    model_bits=model_bits,
                    bits_gap=bits_gap,
                    loaded_model=model,
                    seed=seed,
                )
            )

        # Estimators to run.
        if args.run_bn:
            estimators = RunNParallel(
                estimator_factory=MakeBnEstimators,
                parallelism=50,
                rng=np.random.RandomState(1234),
                num=args.num_queries,
                num_filters=None,
                oracle_cards=oracle_cards,
            )
        else:
            estimators = [
                estimators_lib.ProgressiveSampling(
                    c.loaded_model,
                    table,
                    args.psample,
                    device=DEVICE,
                    shortcircuit=args.column_masking,
                )
                for c in parsed_ckpts
            ]
            for est, ckpt in zip(estimators, parsed_ckpts):
                est.name = str(est) + "_{}_{:.3f}".format(ckpt.seed, ckpt.bits_gap)

            if args.inference_opts:
                print("Tracing forward_with_encoded_input()...")
                for est in estimators:
                    encoded_input = est.model.EncodeInput(
                        torch.zeros(args.psample, est.model.nin, device=DEVICE)
                    )

                    # NOTE: this line works with torch 1.0.1.post2 (but not 1.2).
                    # The 1.2 version changes the API to
                    # torch.jit.script(est.model) and requires an annotation --
                    # which was found to be slower.
                    est.traced_fwd = torch.jit.trace(
                        est.model.forward_with_encoded_input, encoded_input
                    )

            if args.run_sampling:
                SAMPLE_RATIO = {"dmv": [0.0013]}  # ~1.3MB.
                for p in SAMPLE_RATIO.get(args.dataset, [0.01]):
                    estimators.append(estimators_lib.Sampling(table, p=p))

            if args.run_maxdiff:
                estimators.append(
                    estimators_lib.MaxDiffHistogram(table, args.maxdiff_limit)
                )

            # Other estimators can be appended as well.

            if len(estimators):
                RunN(
                    table,
                    cols_to_train,
                    estimators,
                    rng=np.random.RandomState(1234),
                    num=args.num_queries,
                    query_loc="benchmark/newdata/{}.sql".format(str(step).zfill(2)),
                    log_every=1,
                    num_filters=None,
                    oracle_cards=oracle_cards,
                    oracle_est=oracle_est,
                )

        SaveEstimators("results/{}.csv".format(step), estimators)
        print("...Done, result:", args.err_csv)


def LLTest(modelspath, seed=0):
    torch.manual_seed(0)
    np.random.seed(0)

    # Load dataset
    table = dataset_util.DatasetLoader.load_dataset(dataset=args.dataset)

    table_bits = Entropy(
        table,
        table.data.fillna(value=0).groupby([c.name for c in table.columns]).size(),
        [2],
    )

    fixed_ordering = None

    if args.order is not None:
        print("Using passed-in order:", args.order)
        fixed_ordering = args.order

    print(table.data.info())

    table_main = table

    models = np.sort(os.listdir(modelspath))
    models = [os.path.join(modelspath, i) for i in models]
    print(models)

    results = {}
    for step, path in enumerate(models):
        if args.dataset in ["dmv-tiny", "dmv", "tpcds", "census", "forest"]:
            model = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )

            model.load_state_dict(torch.load(path))
            model.eval()
        else:
            assert False, args.dataset

        mb = ReportModel(model)

        bs = 2048
        log_every = 200

        train_data = common.TableDataset(table_main)
        landmarks = int(len(train_data.tuples) * 10 / 12) + np.linspace(
            0, int(len(train_data.tuples) * 0.2 * 10 / 12), 6, dtype=np.int
        )
        train_data.tuples = train_data.tuples[: landmarks[step]]
        if step == 0:
            rndindices1 = torch.randperm(landmarks[step])[:2000]
            train_data.tuples = train_data.tuples[rndindices1]

        if step > 0:
            rndindices1 = torch.randperm(landmarks[step - 1])[:2000]
            rndindices2 = torch.arange(landmarks[step - 1], landmarks[step] - 1)
            rndindices = torch.cat((rndindices1, rndindices2), 0)
            train_data.tuples = train_data.tuples[rndindices]

        # idx = int(len(train_data.tuples)*10/18)
        # print (train_data.tuples.shape)
        # train_data.tuples = train_data.tuples[idx:]
        # rndindices = torch.randperm(landmarks[step])[:1000000]
        # train_data.tuples = train_data.tuples[rndindices]
        # train_data.tuples = train_data.tuples[11591878:,:]
        print(train_data.tuples.shape)
        train_losses = []
        train_start = time.time()

        all_losses = RunEpoch(
            "test",
            model,
            train_data=train_data,
            val_data=train_data,
            opt=None,
            scheduler=None,
            batch_size=1024,
            log_every=500,
            table_bits=table_bits,
            return_losses=True,
        )

        model_nats = np.mean(all_losses)
        results[step] = model_nats
        print(path, model_nats)

    for key in results:
        print(results[key])


def ConceptDriftTest(seed=0):
    torch.manual_seed(0)
    np.random.seed(0)

    results = {}
    for i in range(0, 10):
        table = dataset_util.DatasetLoader.load_partly_permuted_dataset(
            dataset=args.dataset, num_of_sorted_cols=i
        )

        table_bits = Entropy(
            table,
            table.data.fillna(value=0).groupby([c.name for c in table.columns]).size(),
            [2],
        )

        fixed_ordering = None

        if args.order is not None:
            print("Using passed-in order:", args.order)
            fixed_ordering = args.order

        print(table.data.info())

        table_main = table

        if args.dataset in ["dmv-tiny", "dmv", "tpcds", "census", "forest"]:
            model = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )

            model.load_state_dict(torch.load(args.glob))
            model.eval()
        else:
            assert False, args.dataset

        mb = ReportModel(model)

        bs = 2048
        log_every = 200

        train_data = common.TableDataset(table_main)
        rndindices1 = torch.randperm(len(train_data.tuples))[:5000]
        train_data.tuples = train_data.tuples[rndindices1]

        print(train_data.tuples.shape)
        train_losses = []
        train_start = time.time()

        all_losses = RunEpoch(
            "test",
            model,
            train_data=train_data,
            val_data=train_data,
            opt=None,
            scheduler=None,
            batch_size=1024,
            log_every=500,
            table_bits=table_bits,
            return_losses=True,
        )

        model_nats = np.mean(all_losses)
        results[i] = model_nats

    for key in results.keys():
        print(key, results[key])


def test_for_drift(
    args: argparse.Namespace,
    pre_model: Path,
    end2end: bool = False,
    seed=0,
    bootstrap=1000,
    sample_size=500,
    data_type="raw",
):
    torch.manual_seed(0)
    np.random.seed(0)
    
    def dataset_init(dataset):
        if "census" in dataset:
            npy_tail="_int.npy"
        else:
            npy_tail=".npy"

        # Get the dataset path
        if "_" in dataset:
            dataset_name=dataset.split("_")[0]
            src_dataset_path = f"./data/{dataset_name}/{dataset}{npy_tail}"  # Original dataset path
            end2end_dataset_path = f"./data/{dataset_name}/end2end/{dataset}{npy_tail}"  # End2end dataset path
        else:
            src_dataset_path = f"./data/{dataset}/{dataset}{npy_tail}"  # Original dataset path
            end2end_dataset_path = f"./data/{dataset}/end2end/{dataset}{npy_tail}"  # End2end dataset path
        abs_src_dataset_path = get_absolute_path(src_dataset_path)
        abs_end2end_dataset_path = get_absolute_path(end2end_dataset_path)

        # Copy the original dataset to the end2end folder, overwrite if it exists
        shutil.copy2(src=abs_src_dataset_path, dst=abs_end2end_dataset_path)

        return abs_end2end_dataset_path

    def offline_phase(table, model, table_bits, simulations, sample_size, update_size):
        t1 = time.time()
        train_data = common.TableDataset(table)
        train_data.tuples = train_data.tuples[: -update_size]  

        avg_ll = []
        std_ll=[]
        np.random.seed(1)
        for i in range(simulations):
            idx = np.random.randint(train_data.size(), size=sample_size)
            # if i % 100 ==0:
            #     print(f"idx size: {idx.size}")
            idx_tensor = torch.from_numpy(idx)

            # sample=copy.deepcopy(train_data)
            # sample.tuples=sample.tuples[idx]
            sample = torch.index_select(train_data.tuples, 0, idx_tensor)
            if i % 100 ==0:
                print(f"sample size: {sample.size(0)}")
            # train_losses = []
            # train_start = time.time()

            all_losses = RunEpoch(
                "test",
                model,
                train_data=sample,
                val_data=sample,
                opt=None,
                scheduler=None,
                batch_size=1024,
                log_every=500,
                table_bits=table_bits,
                return_losses=True,
            )
            # if i % 100 ==0:
            #     print(f"all losses shape: {len(all_losses)}")
            avg_ll.append(np.mean(all_losses))
            # std_ll.append(np.std(all_losses))
            # print(f"epoch {i}, losses std: {np.std(all_losses)}")
        t2 = time.time()

        return np.mean(avg_ll), 2 * np.std(avg_ll), t2 - t1

    def online_phase(table, model, table_bits, mean, threshold, sample_size, update_size):
        t1 = time.time()
        train_data = common.TableDataset(table)

        train_data.tuples = train_data.tuples[ -update_size : ]  
        # idx = rndindices = torch.randperm(len(train_data.tuples))[:5000]
        idx = rndindices = torch.randperm(len(train_data.tuples))[:sample_size]

        sample_data=copy.deepcopy(train_data)
        sample_data.tuples=sample_data.tuples[idx]

        # train_losses = []
        # train_start = time.time()

        all_losses = RunEpoch(
            "test",
            model,
            train_data=sample_data,
            val_data=sample_data,
            opt=None,
            scheduler=None,
            batch_size=1024,
            log_every=500,
            table_bits=table_bits,
            return_losses=True,
        )
        model_nats_new = np.mean(all_losses)

        """
        test_loss = RunEpoch(
            "test",
            model,
            train_data=train_data,
            val_data=train_data,
            opt=None,
            scheduler=None,
            batch_size=1024,
            log_every=500,
            table_bits=table_bits,
            return_losses=False,
        )
        """

        stat = np.abs(mean - model_nats_new)
        # print (scipy.stats.norm.ppf(1-p, scale=variance))
        t2 = time.time()

        print("new data average log likelihood = {:.4f}".format(model_nats_new))
        # print("new whole data log likelihood = {:.4f}".format(test_loss))
        print("detection mean = {:.4f}".format(mean))
        print("detection threshold = {:.4f}".format(threshold))
        print("test statistic = {:.4f}".format(stat))
        if stat > threshold:
            return True, t2 - t1
        else:
            return False, t2 - t1

    def data_update():
        raw_data, sampled_data = None, None
        if not end2end:
            abs_dataset_path = dataset_init(args.dataset)
            random_seed=args.random_seed

            (
                raw_data,
                sampled_data,
            ) = data_updater.DataUpdater.update_dataset_from_file_to_file(
                data_update_method=args.data_update,
                update_fraction=0.2,  
                # update_size=args.update_size,  
                raw_dataset_path=abs_dataset_path,
                updated_dataset_path=abs_dataset_path,
                random_seed=random_seed,
            )
            
            table, _ = dataset_util.DatasetLoader.load_permuted_dataset(
                dataset=args.dataset, permute=False
            )

        else:
            abs_dataset_path = communicator.DatasetPathCommunicator().get()

            if os.path.isdir(abs_dataset_path):
                table_dirs = os.listdir(abs_dataset_path)
                white_list =[]
                if args.dataset == "stats":
                    white_list = ["badges", "comments", "postHistory", "posts", "tags", "users", "votes"]

                tables=dict()
                raw_datas=dict()
                sampled_datas=dict()
                for table_name in table_dirs:
                    abs_table_path = os.path.join(abs_dataset_path, table_name)

                    if table_name in white_list:
                        random_seed=JsonCommunicator().get(f"random_seed")
                        (
                            raw_data,
                            sampled_data,
                            delta_data_to_db,
                        ) = data_updater.DataUpdater.update_dataset_from_dir_to_dir(
                            table_name=table_name,
                            data_update_method=args.data_update,
                            update_fraction=0.04,  
                            # update_size=args.update_size,  
                            raw_dataset_dir=abs_table_path,
                            random_seed=random_seed,
                        )

                        table = dataset_util.NpyDatasetLoader.load_npy_dataset_from_path(
                            path=os.path.join(abs_table_path, f"{table_name}.npy"), name=table_name
                        )

                        # table = dataset_util.NpyDatasetLoader.load_npy_dataset_from_path(
                        #     path=f"./data/stats/numpy/{table_name}.npy", name=table_name
                        # )

                        # table = dataset_util.CsvDatasetLoader.load_csv_dataset(dataset_name=f"{args.dataset}_{table_name}")

                        tables[table_name]=table
                        raw_datas[table_name]=raw_data.astype(np.float32)
                        sampled_datas[table_name]=sampled_data.astype(np.float32)

                        # Save to delta file and insert in postgres
                        # delta_dataname = f"{args.dataset}_{table_name}"
                        # delta_cols = CsvDatasetLoader.DICT_FROM_DATASET_TO_COLS.get(delta_dataname, [])
                        # delta_data_to_db = pd.DataFrame(sampled_data, columns=delta_cols)

                        delta_filename = f"{table_name}_delta.csv"
                        delta_path = os.path.join(abs_dataset_path, "delta", delta_filename)
                        delta_data_to_db.to_csv(delta_path, index=False)
                        update_pg(args.dataset, table_name, delta_path)
                    
                return tables, raw_datas, sampled_datas
            else:
                random_seed=JsonCommunicator().get(f"random_seed")
                (
                    raw_data,
                    sampled_data,
                ) = data_updater.DataUpdater.update_dataset_from_file_to_file(
                    data_update_method=args.data_update,
                    update_fraction=0.2,  
                    update_size=args.update_size,  
                    raw_dataset_path=abs_dataset_path,
                    updated_dataset_path=abs_dataset_path,
                    random_seed=random_seed,
                )

                table = dataset_util.NpyDatasetLoader.load_npy_dataset_from_path(
                    path=abs_dataset_path
                )
                # save_path=f"./data/{args.dataset}/permuted_dataset.csv"
                # table.data.to_csv(save_path, index=False)
                
        return table, raw_data.astype(np.float32), sampled_data.astype(np.float32)

    def kl_divergence(mu1, sigma1, mu2, sigma2):
        kl = np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5
        print(kl)
        return kl

    def distribution_test(raw_data, sampled_data, sample_size, threshold=1):
        """Pre-revision statistic: mean per-column Gaussian KL, all columns numeric.

        Reached via --drift_test sauce_legacy. Superseded by drift.DauceDetector,
        which is type-aware, covers inter-column dependency drift, and calibrates
        its own threshold; kept so the two can be compared directly.
        """
        # old_sample = table_sample.sampling(raw_data, sample_size, replace=False)
        # new_sample = table_sample.sampling(sampled_data, sample_size, replace=True)
        full_data=np.vstack((raw_data, sampled_data))

        old_mean = np.nanmean(raw_data, axis=0)
        new_mean = np.nanmean(full_data, axis=0)
        old_std = np.nanstd(raw_data, axis=0)
        new_std = np.nanstd(full_data,axis=0)
        kl_score=kl_divergence(old_mean, old_std, new_mean, new_std)
        kl_score=np.mean(kl_score)
        print(f"Distance score: {kl_score}")

        return kl_score > threshold

    def dauce_distribution_test(table, raw_data, sampled_data, table_name=None):
        """Revised DAUCE detector: type-aware marginals plus top-k dependency.

        The threshold is calibrated at run time from the reference snapshot, so the
        hard-coded per-dataset constants this replaces -- and the q-error bootstrap
        that produced them -- are gone. Nothing here reads the model, the workload
        or the true cardinalities.

        The threshold depends only on the reference snapshot and the size of one
        update batch, so it is calibrated once per reference and cached on disk
        until the reference moves, i.e. until the next model update clears the
        buffer pool. Every event tests the whole pool against the cached threshold.
        """
        from drift.detectors import DauceDetector
        from drift.schema import canonical_dataset, infer_kinds

        reference = np.asarray(raw_data, dtype=np.float64)
        current = np.vstack((reference, np.asarray(sampled_data, dtype=np.float64)))
        # Calibrate at the size of one update batch, not of the accumulated pool.
        batch_size = int(dauce_unit_rows[table_name] if isinstance(dauce_unit_rows, dict)
                         else dauce_unit_rows)

        schema_name = table_name or args.dataset
        tau = float(JsonCommunicator().get_with_default("drift.tau", 0.05))
        repeats = int(JsonCommunicator().get_with_default("drift.calibration_repeats", 4))
        detector_params = dict(
            weight=float(JsonCommunicator().get_with_default("drift.weight", 0.5)),
            n_bins=int(JsonCommunicator().get_with_default("drift.n_bins", 16)),
            pair_aggregate=str(JsonCommunicator().get_with_default(
                "drift.pair_aggregate", "mean")),
        )

        # The reference grows whenever an accepted update advances the snapshot, so
        # its shape is a constant-time version key. Avoid hashing the full table on
        # the online path.
        _t_lookup = time.perf_counter()
        cache_key = (reference.shape, tau, repeats, sorted(detector_params.items()))
        cache_path = dauce_cache_path(table_name)
        detector = None
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as f:
                cached_key, cached_detector = pickle.load(f)
            if cached_key == cache_key:
                detector = cached_detector
        cache_hit = detector is not None
        _t_off = time.perf_counter()
        if not cache_hit:
            kinds = infer_kinds(reference, dataset=canonical_dataset(schema_name))
            detector = DauceDetector(**detector_params)
            detector.fit_reference(reference, kinds)
            detector.calibrate(reference, kinds, batch_size=batch_size, tau=tau,
                               repeats=repeats, random_seed=0)
            with open(cache_path, "wb") as f:
                pickle.dump((cache_key, detector), f, pickle.HIGHEST_PROTOCOL)
        _t_on = time.perf_counter()
        detection = detector.detect(reference, current)
        _t_end = time.perf_counter()

        label = f"[{table_name}] " if table_name else ""
        print(f"{label}DAUCE calibration: {detector._calibration.summary()} "
              f"({'cached' if cache_hit else 'recalibrated'})")
        # Offline = reference fit + conformal calibration, paid once per reference;
        # online = cache lookup + the test of the buffered rows.
        print(f"{label}DAUCE offline time = {_t_on - _t_off:.6f}\n"
              f"{label}DAUCE online time = {(_t_off - _t_lookup) + (_t_end - _t_on):.6f}")
        print(f"{label}Distance score: {detection.distance:.6e} "
              f"(D_marginal={detection.extras.get('d_marginal'):.6e}, "
              f"D_dependency={detection.extras.get('d_dependency'):.6e}, "
              f"pairs={detection.extras.get('n_pairs')})")
        return bool(detection.is_drift)

    # update data
    table, raw_data, sampled_data = data_update()
    print(f"Data updated!")
    # Rows of this update batch alone; the DAUCE threshold is calibrated at this size.
    dauce_unit_rows = ({name: np.asarray(part).shape[0] for name, part in sampled_data.items()}
                       if isinstance(sampled_data, dict) else np.asarray(sampled_data).shape[0])

    def dauce_cache_path(table_name=None):
        dataset_name = args.dataset.split("_")[0]
        if table_name is not None:
            return f"./data/{dataset_name}/end2end/{table_name}/{table_name}_dauce_cache.pkl"
        return f"./data/{dataset_name}/end2end/{args.dataset}_dauce_cache.pkl"

    def drift_detect(table, raw_data, sampled_data, sample_size, pre_model, table_name = None) -> bool:
        # Fixed trigger schedule: fire on every update without looking at the data.
        # The control group for "how much of the end-to-end accuracy comes from the
        # detector rather than the updater" -- it must short-circuit before any
        # statistic is computed, so its detection latency is zero by construction.
        if args.drift_test == "always":
            return True
        if args.drift_test == "ddup":
            table_bits = Entropy(
                table,
                table.data.fillna(value=0)
                .groupby([c.name for c in table.columns])
                .size(),
                [2],
            )

            fixed_ordering = None

            if args.order is not None:
                print("Using passed-in order:", args.order)
                fixed_ordering = args.order

            # print(table.data.info())

            # Size of the newly inserted batch, shared by all base models below
            update_size = sampled_data.shape[0]

            if args.model == "naru":
                model = MakeMade(
                    scale=args.fc_hiddens,
                    cols_to_train=table.columns,
                    seed=seed,
                    fixed_ordering=fixed_ordering,
                )

                # model.load_state_dict(torch.load(args.glob))
                if args.dataset in ["stats"]:
                    pre_model_dir = Path(pre_model)
                    matching_files = [file for file in pre_model_dir.rglob("*") if table_name in file.name]
                    assert len(matching_files)==1, "Too Many Models!"
                    pre_model = os.path.join(pre_model_dir, matching_files[0])

                model.load_state_dict(torch.load(pre_model))
                model.eval()

                # mb = ReportModel(model)

                mean, threshold, time_off = offline_phase(
                    table,
                    model,
                    table_bits,
                    simulations=bootstrap,
                    sample_size=sample_size,
                    update_size=update_size,
                )
                drift, time_on = online_phase(
                    table, model, table_bits, mean, threshold, sample_size=sample_size, update_size=update_size,
                )
                print("Offline time = {}\nOnline time = {}".format(time_off, time_on))
                print("Test result: {}".format(drift))

            elif args.model == "face":
                assert args.dataset not in ["stats"], f"dataset {args.dataset} is not supported!" 

                device = torch.device("cpu")
                model_name = "BJAQ" if args.dataset == "bjaq" else args.dataset
                
                model_config_path = f"./FACE/config/{args.dataset}.yaml"
                abs_model_config_path = get_absolute_path(model_config_path)
                my_flow_model = MyFlowModel(config_path=abs_model_config_path)
                flow = my_flow_model.load_model(device=device, model_path=pre_model)
                # flow = table_sample.load_model(DEVICE, model_path=pre_model, model_name=model_name)

                is_bjaq=(args.dataset=="bjaq")
                mean_reduction, threshold = table_sample.loss_test(
                    raw_data, sampled_data, sample_size=sample_size, flow=flow, is_bjaq=is_bjaq
                )

                # loss_test compares numpy arrays, so the comparison yields a
                # numpy.bool_. Cast to python bool here (the naru/transformer
                # branches already return python bool) so DriftCommunicator.set
                # accepts it.
                drift = bool(mean_reduction > threshold)
                # drift = False
            elif args.model=="transformer":
                assert args.dataset not in ["stats"], f"dataset {args.dataset} is not supported!" 

                model = MakeTransformer(
                    cols_to_train=table.columns, fixed_ordering=None, seed=seed
                )

                # model.load_state_dict(torch.load(args.glob))
                model.load_state_dict(torch.load(pre_model))
                model.eval()

                # mb = ReportModel(model)

                mean, threshold, time_off = offline_phase(
                    table,
                    model,
                    table_bits,
                    simulations=bootstrap,
                    sample_size=sample_size,
                    update_size=update_size,
                )
                drift, time_on = online_phase(
                    table, model, table_bits, mean, threshold, sample_size=sample_size, update_size=update_size,
                )
                print("Offline time = {}\nOnline time = {}".format(time_off, time_on))
                print("Test result: {}".format(drift))

            return drift

        # drift detection - dauce (revised): data-only, threshold calibrated at
        # run time from the reference snapshot.
        if args.drift_test == "sauce":
            return dauce_distribution_test(
                table=table,
                raw_data=raw_data,
                sampled_data=sampled_data,
                table_name=table_name,
            )

        # drift detection - pre-revision statistic, kept for comparison.
        # The thresholds below came from the q-error bootstrap that the revision
        # replaces, so they are frozen here rather than maintained.
        if args.drift_test == "sauce_legacy":
            thres = 0
            if args.dataset == "census":
                thres=6.26e-6
            elif args.dataset == "bjaq":
                thres=2.78e-5
            elif args.dataset == "forest":
                thres=1.73e-6
            elif args.dataset == "power":
                thres=8.03e-6
            elif args.dataset == "bjaq_gaussian":
                thres=1.90e-5
            elif args.dataset == "census_gaussian":
                thres=7.41e-5
            elif args.dataset == "stats":
                thres_dict={
                    "badges":       3.3667e-4,
                    "votes":        4.5573e-4,
                    "postHistory":  4.3794e-4,
                    "posts":        4.7584e-4,
                    "users":        0.0011,
                    "comments":     6.5434e-4,
                    "postLinks":    5.2538e-4,
                    "tags":         0.0010,
                }
                thres=thres_dict[table_name]
        

            return distribution_test(
                raw_data=raw_data,
                sampled_data=sampled_data,
                sample_size=5000, 
                threshold=thres
            )

        if args.drift_test == "none":
            return False

    if end2end:
        if args.dataset in ["stats"]:
            dataset_name=args.dataset
            
            previous_data = dict()
            new_data = dict()
            for table_name in table:
                pool_path=f"./data/{dataset_name}/end2end/{table_name}/{table_name}_pool.npy"
                if os.path.isfile(pool_path):
                    unlearned_data=np.load(pool_path).astype(np.float32)
                    unlearned_size=unlearned_data.shape[0]
                    previous_data[table_name]=raw_data[table_name][:-unlearned_size]
                    new_data[table_name]=np.vstack((unlearned_data,sampled_data))
                else:
                    previous_data[table_name]=raw_data[table_name]
                    new_data[table_name]=sampled_data[table_name]
                np.save(pool_path, new_data[table_name])
            
        else:
            if "_" in args.dataset:
                dataset_name=args.dataset.split("_")[0]
            else:
                dataset_name=args.dataset
            pool_path=f"./data/{dataset_name}/end2end/{args.dataset}_pool.npy"
            if os.path.isfile(pool_path):
                unlearned_data=np.load(pool_path).astype(np.float32)
                unlearned_size=unlearned_data.shape[0]
                previous_data=raw_data[:-unlearned_size]
                new_data=np.vstack((unlearned_data,sampled_data))
            else:
                previous_data=raw_data
                new_data=sampled_data
            np.save(pool_path, new_data)
        
    else:
        assert args.dataset not in ["stats"], f"dataset {args.dataset} is not supported!" 
        previous_data=raw_data
        new_data=sampled_data

    if args.dataset in ["stats"]:
        detection_start=time.time()
        drifts_dict = dict()
        for table_name in table:
            table_single = table[table_name]
            previous_data_single = previous_data[table_name]
            new_data_single = new_data[table_name]

            print(f"Table {table_name}, previous data size: {previous_data_single.shape}, new data size: {new_data_single.shape}")
            is_drift: bool = drift_detect(table_single, previous_data_single, new_data_single, sample_size, pre_model, table_name)
            drifts_dict[table_name] = is_drift
        detection_finish=time.time()
        time_overhead=detection_finish-detection_start

        if args.drift_test=="ddup":
            detection_type="DDUp"
        elif args.drift_test=="sauce":
            detection_type="DAUCE"
        elif args.drift_test=="sauce_legacy":
            detection_type="DAUCE-legacy"
        elif args.drift_test=="always":
            detection_type="Always"
        else:
            detection_type="None"
        print(f"{detection_type} Drift detection: {drifts_dict}")
        print("Detection latency: {:.4f}s".format(time_overhead))
        communicator.DriftCommunicator().set(is_drift=drifts_dict)
    else:
        detection_start=time.time()

        print(f"Previous data size: {previous_data.shape}, new data size: {new_data.shape}")
        is_drift: bool = drift_detect(table, previous_data, new_data, sample_size, pre_model)
        detection_finish=time.time()
        time_overhead=detection_finish-detection_start
        
        if args.drift_test=="ddup":
            detection_type="DDUp"
        elif args.drift_test=="sauce":
            detection_type="DAUCE"
        elif args.drift_test=="sauce_legacy":
            detection_type="DAUCE-legacy"
        elif args.drift_test=="always":
            detection_type="Always"
        else:
            detection_type="None"
        print(f"{detection_type} Drift detection: {is_drift}")
        print("Detection latency: {:.4f}s".format(time_overhead))
        communicator.DriftCommunicator().set(is_drift=is_drift)  
  

def main(argv=None):
    global args
    args = create_parser().parse_args(argv)
    is_end2end: bool = args.end2end

    # query workload
    if args.eval_type == "estimate" or args.eval_type == "first_estimate":
        Model_Eval(args=args, end2end=is_end2end)

    if args.eval_type == "drift":
        # Only the loss-based DDUp test needs a model; the DAUCE detector reads
        # table statistics and nothing else, which is what makes it strictly
        # model-agnostic. Requiring a checkpoint for it would also mean that a
        # failed glob silently turns the whole run into a no-op.
        needs_model = args.drift_test == "ddup"

        if not needs_model:
            test_for_drift(
                args=args, end2end=is_end2end, pre_model=None, data_type="raw"
            )
            return

        if not is_end2end:
            relative_model_paths = (
                "./models/*/origin-{}-*MB-model*-data*-*epochs-seed*.pt".format(
                    args.dataset
                )
            )
            absolute_model_paths = get_absolute_path(relative_model_paths)
            model_paths_str: List[str] = glob.glob(str(absolute_model_paths))
            model_paths: List[Path] = [Path(i) for i in model_paths_str] 
            print("Count of model paths =", len(model_paths))
            assert model_paths, (
                f"--drift_test ddup needs a checkpoint but no file matched "
                f"{absolute_model_paths}"
            )

        else:
            model_paths: List[Path] = [communicator.ModelPathCommunicator().get()]

        for model_path in model_paths:
            test_for_drift(
                args=args, end2end=is_end2end, pre_model=model_path, data_type="raw"
            )


if __name__ == "__main__":
    main()
    # table, train_data, oracle_est = MakeTable()
    # num_samples=1000
    # JS=estimators_lib.JoinSampling(join_iter=train_data, table=table, num_samples=num_samples)
    # print("Success!")

