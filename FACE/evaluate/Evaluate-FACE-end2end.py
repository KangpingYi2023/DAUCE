import torch
import json
import numpy as np
import torch
import os, sys
import time
import datetime
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import csv
import pickle
import pytz
import nflows.nn as nn_

from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from tqdm import tqdm
from torchquad import enable_cuda, VEGAS
from time import sleep
from torch import optim
from torch.utils.data import Dataset
from nflows import transforms
from nflows import distributions
from nflows import utils as nflows_utils
from nflows import flows
from torchquad import BatchMulVEGAS
from datetime import date

import argparse
from pathlib import Path
from typing import List

from torchquad import set_up_backend

sys.path.append("./")
from FACE.data.table_sample import MyFlowModel

from update_utils import log_util
from update_utils.arg_util import ArgType, add_common_arguments
from update_utils.end2end_utils import communicator
from update_utils.model_util import save_torch_model
from update_utils.path_util import get_absolute_path
from update_utils.torch_util import get_torch_device
from update_utils.end2end_utils.json_communicator import JsonCommunicator

import FACE.FACE_utils.dataUtils as faceDataUtils

DEVICE = str(get_torch_device())

""" Change it to the project root path """
PROJECT_PATH = "./FACE/"
# os.environ['VISIBLE_CUDA_DEVICES'] = '1'

# GPU_ID = 1
# dataset_name = "BJAQ"

# """ network parameters """
# hidden_features = 56
# num_flow_steps = 6
flow_id = 2
# features = 5
# REUSE_FROM_FILE = False
# REUSE_FILE_PATH = PROJECT_PATH + "train/"



# """ detailed network parameters"""
# anneal_learning_rate = True
# base_transform_type = "rq-coupling"
# dropout_probability = 0
# grad_norm_clip_value = 5.0
# linear_transform_type = "lu"
# num_bins = 8
# num_training_steps = 400000
# num_transform_blocks = 2

# tail_bound = 3
# use_batch_norm = False


def create_parser():
    parser = argparse.ArgumentParser()

    common_args: List[ArgType] = [
        ArgType.DATASET,
        ArgType.END2END,
        ArgType.MODEL_UPDATE,
        ArgType.DATA_UPDATE,
    ]
    add_common_arguments(parser, arg_types=common_args)

    parser.add_argument("--num_workload", type=int, default=5, help="number of workloads")

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["origin", "update", "adaptST", "adaptTA", "finetn", "retrain"],
        default="origin",
        help="model incremental type",
    )

    return parser


args = create_parser().parse_args()

""" query settings"""
query_seed = args.query_seed
QUERY_CNT = args.num_queries
seed = 1638128
torch.manual_seed(seed)
np.random.seed(seed)


def evaluation_budget(key, default):
    value = JsonCommunicator().get("face." + key) if args.end2end else None
    return int(value) if value is not None else default


class HiddenPrints:
    def __init__(self, activated=True):
        self.activated = activated
        self.original_stdout = None

    def open(self):
        """no output"""
        sys.stdout.close()
        sys.stdout = self.original_stdout

    def close(self):
        """output"""
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __enter__(self):
        if self.activated:
            self.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.activated:
            self.open()

""" q-error """


def ErrorMetric(est_card, card):
    if isinstance(est_card, torch.FloatTensor) or isinstance(est_card, torch.IntTensor):
        est_card = est_card.cpu().detach().numpy()
    if isinstance(est_card, torch.Tensor):
        est_card = est_card.cpu().detach().numpy()
    est_card = np.float32(est_card)
    card = np.float32(card)
    if card == 0 and est_card != 0:
        return est_card
    if card != 0 and est_card == 0:
        return card
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)


def BatchErrorMetrix(est_list, oracle_list):
    ret = np.zeros(len(est_list))
    ID = 0
    for est, real in zip(est_list, oracle_list):
        ret[ID] = ErrorMetric(est, real)
        ID = ID + 1
    return ret


f_batch_time = 0


def f_batch(inp):
    global f_batch_time
    with torch.no_grad():
        inp = inp.to(DEVICE)

        # print("【Example input】", inp[0, :])
        # print("inp shape ", inp.shape)
        st = time.time()
        prob_list = model.log_prob(inp)
        prob_list = torch.exp(prob_list)
        # print("【max_prob】 ", prob_list.max())
        # print("【median_prob】 ", prob_list.median())
        en = time.time()
        f_batch_time += en - st

        return prob_list


""" set GPU first """
# os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(GPU_ID)
# assert torch.cuda.is_available()
# device = torch.device("cuda")
# device = torch.device('cpu')
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.manual_seed(seed)
np.random.seed(seed)

torch.set_default_tensor_type("torch.cuda.FloatTensor")

DEVICENAME = torch.cuda.get_device_name(DEVICE)
print("DEVICE NAME\n", DEVICENAME)

# distribution = distributions.StandardNormal((features,))
# transform = create_transform()
# flow = flows.Flow(transform, distribution).to(device)
model_name = args.dataset

def get_model_path(is_end2end: bool, flow_id=None):

    if not is_end2end:
        if args.model_type=="origin" or args.model_type=="retrain":
            tail=".t"
        else:
            tail=".pt"
        p = f"./models/face-{args.model_type}-{model_name}-id{flow_id}-best-val"+tail
        path = Path(p)

    else:
        path: Path = communicator.ModelPathCommunicator().get()

    return path

# flow.load_state_dict(torch.load(path))

# flow.cuda()
# flow.eval()

model_config_path = f"./FACE/config/{args.dataset}.yaml"
abs_model_config_path = get_absolute_path(model_config_path)
my_flow_model = MyFlowModel(config_path=abs_model_config_path)
if args.end2end:
    my_flow_model.learning_rate = float(JsonCommunicator().get('face.learning_rate'))

path = get_model_path(is_end2end=args.end2end, flow_id=my_flow_model.flow_id)
model = my_flow_model.load_model(device=DEVICE, model_path=path)

n_params = nflows_utils.get_num_parameters(model)
print("There are {} trainable parameters in this model.".format(n_params))
print("Parameters total size is {} MB".format(n_params * 4 / 1024 / 1024))

data, n, dim = faceDataUtils.LoadTable(
    dataset_name=args.dataset, is_end2end=args.end2end
)
DW = faceDataUtils.DataWrapper(data, args.dataset)
rng = np.random.RandomState(query_seed)
queries = DW.generateNQuery(QUERY_CNT, rng)
DW.getAndSaveOracle(queries, query_seed)

""" Load oracle_cards"""
oracle_cards = faceDataUtils.LoadOracleCardinalities(args.dataset, query_seed)

legal_lists = DW.getLegalRangeNQuery(queries)
legal_tensors = torch.Tensor(legal_lists)
legal_tensors = legal_tensors.to(DEVICE)

# from torchquad import set_up_backend

set_up_backend("torch", data_type="float32")

# 可选：从文件复用积分映射。
# pickle.dump(target_map, f)
# if my_flow_model.REUSE_FROM_FILE:
#     f = open(REUSE_FILE_PATH + "{}.pickle".format(dataset_name), "rb")
#     target_map = pickle.load(f)

z = DW.getLegalRangeQuery([[], [], []])
z = torch.Tensor(z)
print(z.shape)
full_integration_domain = torch.Tensor(z)
full_integration_domain = full_integration_domain.to(DEVICE)
# print(f"full_integration_domain: {full_integration_domain.device}")

domain_starts = full_integration_domain[:, 0]
domain_sizes = full_integration_domain[:, 1] - domain_starts
domain_volume = torch.prod(domain_sizes)

domain_starts=domain_starts.to(DEVICE)
domain_sizes=domain_sizes.to(DEVICE)
domain_volume=domain_volume.to(DEVICE)

if not my_flow_model.REUSE_FROM_FILE:
    vegas = VEGAS()
    # vegas=vegas.to(DEVICE)
    bigN = evaluation_budget("map_samples", 1000000 * 40)

    st = time.time()
    result = vegas.integrate(
        f_batch,
        dim=my_flow_model.features,
        N=bigN,
        integration_domain=full_integration_domain,
        use_warmup=True,
        use_grid_improve=True,
        max_iterations=evaluation_budget("map_iterations", 40),
    )

    en = time.time()
    print("Took ", en - st)
    print(result)
    result = result * DW.n

    print("result is ", result)

if not my_flow_model.REUSE_FROM_FILE:
    target_map = vegas.map

    f = open(my_flow_model.REUSE_FILE_PATH + "{}.pickle".format(my_flow_model.dataset_name), "wb")
    pickle.dump(target_map, f)
    f.close()


def getResult(n, N, num_iterations=3, alpha=0.5, beta=0.5):
    global f_batch_time
    """ n: batch size """
    z = BatchMulVEGAS()
    DIM = my_flow_model.features
    full_integration_domain = torch.Tensor(DIM * [[0, 1]])

    start_id = 0
    end_id = 0

    f_batch_time = 0
    st = time.time()
    results = []
    with torch.no_grad():
        while start_id < QUERY_CNT:
            end_id = end_id + n
            if end_id > QUERY_CNT:
                end_id = QUERY_CNT
            z.setValues(
                f_batch,
                dim=DIM,
                alpha=alpha,
                beta=beta,
                N=N,
                n=end_id - start_id,
                iterations=num_iterations,
                integration_domains=legal_tensors[start_id:end_id],
                rng=None,
                seed=1234,
                reuse_sample_points=True,
                target_map=target_map,
                target_domain_starts=domain_starts,
                target_domain_sizes=domain_sizes,
            )
            start_id = start_id + n
            results.append(z.integrate())

    en = time.time()
    total_time = en - st
    return total_time, results


def testHyper(n, N, num_iterations, alpha, beta):
    print("Enter testHyper")
    with HiddenPrints():
        total_time, result_origin = getResult(
            n=n, N=N, num_iterations=num_iterations, alpha=alpha, beta=beta
        )


        result_origin = torch.cat(tuple(result_origin))
        FULL_SIZE = torch.Tensor([DW.n])
        result = result_origin * FULL_SIZE
        result = result.to("cpu")

        n_ = QUERY_CNT
        oracle_list = oracle_cards.copy()

        err_list = BatchErrorMetrix(result.int(), oracle_list)

        total_query_time = total_time
        avg_per_query_time = 1000.0 * (total_query_time / n_)
        avg_f_batch_time = 1000.0 * f_batch_time / n_
        avg_vegas_time = avg_per_query_time - avg_f_batch_time

    if args.end2end:
        err_list1 = err_list.tolist()
        print(f"ReportEsts: {err_list1}")

    print(
        "********** total_n=[{}] batchn=[{}]  N=[{}]  nitr=[{}]  alpha=[{}]  beta=[{}] ******".format(
            n_, n, N, num_iterations, alpha, beta
        )
    )
    print("@ Average per query          [{}] ms".format(avg_per_query_time))
    print(" --  Average per query NF    [{}] ms".format(avg_f_batch_time))
    print(" --  Average per query vegas [{}] ms".format(avg_vegas_time))
    p50 = np.percentile(err_list, 50)
    mean = np.mean(err_list)
    p95 = np.percentile(err_list, 95)
    p99 = np.percentile(err_list, 99)
    pmax = np.max(err_list)
    message="Mean [{:.3f}]  Median [{:.3f}]  95th [{:.3f}]  99th [{:.3f}]  max [{:.3f}]".format(mean, p50, p95, p99, pmax)
    print(message)
    if args.end2end:
        log_file_name=(
            "face+"  
            f"{args.dataset}+" 
            f"num_workloads{args.num_workload}+"
            f"{args.data_update}+"  
            f"{args.model_update}+" 
            f"qseed{args.query_seed}+" 
            f".txt"
        )
        log_file_path=f"./end2end/end2end-evaluations/{log_file_name}"
        log_util.append_to_file(log_file_path, f"{message}\n")
    else:
        output_file_name = (
            "face+"  
            f"{args.dataset}+" 
            f"{args.data_update}+"  
            f"qseed{args.query_seed}"  
            f".txt"
        )
        output_file_path=f"./end2end/model-evaluation/{output_file_name}"
        if not os.path.isfile(output_file_path):
            log_util.append_to_file(output_file_path, f"Model evaluation results for {args.dataset}+face+query_seed{args.query_seed}\n")
        log_util.append_to_file(output_file_path, f"{message}\n")

    return mean, p50, p95, p99, pmax


alpha = 0.4
beta = 0.2

p50s = []
p95s = []
p99s = []
pmaxs = []

mean, p50, p95, p99, pmax = testHyper(
    100, evaluation_budget("query_samples", 3200),
    evaluation_budget("query_iterations", 4), alpha, beta
)

