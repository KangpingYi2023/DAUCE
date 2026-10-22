import argparse
import os
import sys
import time
import yaml

import nflows.nn as nn_
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows import distributions
from nflows import flows
from nflows import transforms
from nflows import utils

sys.path.append("./")
from update_utils import arg_util
from update_utils.path_util import get_absolute_path
from update_utils.end2end_utils.json_communicator import JsonCommunicator

PROJECT_PATH = "./FACE/"
GPU_ID = 1
seed = 1638128
tail_bound = 3
use_batch_norm = False


class MyFlowModel:
    def __init__(self, config_path):
        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)
        self.dataset_name = config["dataset_name"]
        self.hidden_features = config["hidden_features"]
        self.num_flow_steps = config["num_flow_steps"]
        self.flow_id = config["flow_id"]
        self.features = config["features"]
        self.REUSE_FROM_FILE = config["reuse_from_file"]
        self.REUSE_FILE_PATH = config["reuse_file_path"] if config["reuse_file_path"] is not None else "./FACE/train/"
        self.query_seed = config["query_seed"]
        self.QUERY_CNT = config["query_cnt"]
        self.anneal_learning_rate = config["anneal_learning_rate"]
        self.base_transform_type = config["base_transform_type"]
        self.dropout_probability = config["dropout_probability"]
        self.grad_norm_clip_value = config["grad_norm_clip_value"]
        self.linear_transform_type = config["linear_transform_type"]
        self.num_bins = config["num_bins"]
        self.num_training_steps = config["num_training_steps"]
        self.num_transform_blocks = config["num_transform_blocks"]
        self.tail_bound = config["tail_bound"]
        self.use_batch_norm = config["use_batch_norm"]

        self.train_batch_size = config["train_batch_size"]
        self.learning_rate = config["learning_rate"]
        # self.learning_rate = float(JsonCommunicator().get('face.learning_rate'))
        self.monitor_interval = config["monitor_interval"]
        self.anneal_learning_rate = config["anneal_learning_rate"]

    def create_linear_transform(self):
        if self.linear_transform_type == "permutation":
            return transforms.RandomPermutation(features=self.features)
        elif self.linear_transform_type == "lu":
            return transforms.CompositeTransform(
                [
                    transforms.RandomPermutation(features=self.features),
                    transforms.LULinear(self.features, identity_init=True),
                ]
            )
        elif self.linear_transform_type == "svd":
            return transforms.CompositeTransform(
                [
                    transforms.RandomPermutation(features=self.features),
                    transforms.SVDLinear(
                        self.features, num_householder=10, identity_init=True
                    ),
                ]
            )
        else:
            raise ValueError

    def create_base_transform(self, i):
        return transforms.coupling.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(self.features, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.nets.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=self.hidden_features,
                context_features=None,
                num_blocks=self.num_transform_blocks,
                activation=F.relu,
                dropout_probability=self.dropout_probability,
                use_batch_norm=self.use_batch_norm,
            ),
            num_bins=self.num_bins,
            tails="linear",
            tail_bound=self.tail_bound,
            apply_unconditional_transform=True,
        )

    def create_transform(self):
        transform = transforms.CompositeTransform(
            [
                transforms.CompositeTransform(
                    [self.create_linear_transform(), self.create_base_transform(i)]
                )
                for i in range(self.num_flow_steps)
            ]
            + [self.create_linear_transform()]
        )
        return transform

    def load_model(self, device, model_path=None):
        print("Load model - START")
        distribution = distributions.StandardNormal((self.features,))
        transform = self.create_transform()
        flow = flows.Flow(transform, distribution)

        model_name = "BJAQ" if self.dataset_name == "bjaq" else self.dataset_name
        if not model_path:
            model_path = os.path.join(
                PROJECT_PATH + "train/models/{}".format(model_name),
                "{}-id{}-best-val.t".format(model_name, self.flow_id),
            )

        print("Load model from:", model_path)

        flow.load_state_dict(torch.load(model_path))

        flow.to(device)
        # flow.eval()

        n_params = utils.get_num_parameters(flow)
        print("There are {} trainable parameters in this model.".format(n_params))
        print("Parameters total size is {} MB".format(n_params * 4 / 1024 / 1024))

        print("Load model - END")
        return flow

    def create_model(self, device):
        distribution = distributions.StandardNormal((self.features,))
        transform = self.create_transform()
        flow = flows.Flow(transform, distribution).to(device)

        return flow


def normalized(data):
    max = np.max(data, keepdims=True)
    min = np.min(data, keepdims=False)
    _range = max - min
    for i in range(_range.shape[1]):
        if _range[0][i] == 0:
            _range[0][i] = 1
    norm_data = (data - min) / _range
    norm_data[norm_data == 0] = np.finfo(float).eps
    return norm_data

def sampling(data: np.ndarray, size: int, replace: bool):
    if not replace and size > data.shape[0]:
        raise ValueError(
            "Size cannot be greater than the number of rows in data when replace is False"
        )

    sample_idx = np.random.choice(range(data.shape[0]), size=size, replace=replace)
    sample_idx = np.sort(sample_idx)
    sample = data[sample_idx]

    return sample


def permute(data, size):
    print("Permute - START")
    n_rows, n_cols = data.shape

    samples = np.empty(shape=(size, n_cols))

    for i in range(size):
        if i % 100 == 0:
            print("Permute - {}/{}".format(i, size))
        idxs = np.random.choice(range(n_rows), n_cols, replace=False)
        for j, idx in enumerate(idxs):
            samples[i, j] = data[idx, j]

    print("Permute - END")
    return samples.astype(np.float32)


def permute_optimized(data, size):
    print("Permute - START")
    n_rows, n_cols = data.shape
    samples = np.zeros((size, n_cols))

    for col in range(n_cols):
        samples[:, col] = np.random.choice(data[:, col], size, replace=False)

    print("Permute - END")
    return samples.astype(np.float32)


def single_sampling(data, size):
    idx = np.random.randint(data.shape[0])
    sample_idx = [idx] * size
    sample = data[sample_idx]
    return sample


def data_update_(data, size):
    update_data = sampling(data, size, replace=True)
    return update_data


def js_div(p_output, q_output, get_softmax=False):
    """
    Function that measures JS divergence between target and output logits:
    """

    KLDivLoss = nn.KLDivLoss(reduction="batchmean")
    p_output = torch.from_numpy(p_output)
    q_output = torch.from_numpy(q_output)
    # print("q_output shape: {}".format(q_output.shape))
    if get_softmax:
        p_output = F.softmax(p_output, dim=0)
        q_output = F.softmax(q_output, dim=0)
    log_mean_output = ((p_output + q_output) / 2).log()
    # print("P_output shape: {}".format(p_output.shape))
    # print("data sample: {}".format(q_output[:5]))
    return (
        KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)
    ) / 2


def loss(sample, flow, debug=False):
    sample_tensor=torch.from_numpy(sample)
    if debug:
        print(sample_tensor)
    log_density = flow.log_prob(sample_tensor)
    # print(log_density)
    loss = -torch.mean(log_density)
    std = torch.std(log_density)
    return loss.item(), std.item()

def conca_and_save(save_file, old_data, update_data):
    new_data = np.concatenate((old_data, update_data), axis=0)
    print("new data shape: {}".format(new_data.shape))
    np.save(save_file, new_data)

def BJAQ_normalized(data):
    n, dim = data.shape
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    s[s==0]=1

    for i in range(dim):
        data[:,i] = (data[:,i] - mu[i])/s[i]

    return data

def loss_test(data, update_data, sample_size, flow, bootstrap=50, is_bjaq=True):
    loss_start_time = time.time()

    if is_bjaq:
        data=BJAQ_normalized(data)
        update_data=BJAQ_normalized(update_data)
    
    # ratio=update_data.shape[0]/data.shape[0]
    # new_sample1=sampling(data, round((1-ratio)*sample_size), replace=False)
    # new_sample2 = sampling(update_data, round(ratio* sample_size), replace=False)
    # new_sample=np.concatenate((new_sample1, new_sample2), axis=0)
    losses=[]
    np.random.seed(1)
    for i in range(bootstrap):
        old_sample = sampling(data, sample_size, replace=False)
        old_mean, _ = loss(old_sample, flow=flow)
        losses.append(old_mean)
    avg_loss=np.mean(losses)
    print(f"old_mean_loss:{avg_loss}")
    threshold = 2 * np.std(losses)
    
    new_sample = sampling(update_data, sample_size, replace=True)
    new_mean, _ = loss(new_sample, flow=flow)
    print(f"new_mean_loss:{new_mean}")
    mean_reduction = abs(new_mean - avg_loss)

    loss_end_time = time.time()
    loss_running_time = loss_end_time - loss_start_time
    print("Mean loss reduction: {:.4f}".format(mean_reduction))
    print("2 * std: {:.4f}".format(threshold))

    return mean_reduction, threshold


def JS_test(
    data: np.ndarray,
    update_data: np.ndarray,
    sample_size: int,
    epoch: int = 32,
    threshold: float = 0.3,
) -> bool:
    # assert update_type in ["sample", "single", "permute"], "Update type error!"
    js_start_time = time.time()
    js_divergence = []
    rows, cols=data.shape
    update_size=data.shape[0]
    table_unique_values=[]
    for col in range(cols):
        col_unique_values=np.unique(data[:,col])
        col_unique_values=np.sort(col_unique_values)
        table_unique_values.append(col_unique_values)
    print("Unique value conut Done!")
             
    for step in range(epoch):
        old_sample = sampling(data, sample_size, replace=False)
        new_sample = sampling(update_data, sample_size, replace=False)

        # ratio=update_data.shape[0]/data.shape[0]
        # new_sample1=sampling(data, round((1-ratio)*sample_size), replace=False)
        # new_sample2 = sampling(update_data, round(ratio* sample_size), replace=False)
        # new_sample=np.concatenate((new_sample1, new_sample2), axis=0)

        # old_sample_norm = normalized(old_sample)
        # new_sample_norm = normalized(new_sample)
        # JS_diver = js_div(old_sample_norm, new_sample_norm, get_softmax=True)

        js_value_cols=[]
        for col in range(cols):
            unique_values=table_unique_values[col]
            old_pros = np.zeros_like(unique_values, dtype=np.float32)
            new_pros = np.zeros_like(unique_values, dtype=np.float32)

            for i, value in enumerate(unique_values):
                old_pros[i]=np.sum(old_sample==value)/sample_size
                new_pros[i]=np.sum(new_sample==value)/sample_size
                old_pros[old_pros==0]=np.finfo(float).eps
                new_pros[new_pros==0]=np.finfo(float).eps
            # if col==0:
                # print(old_pros)
                # print(new_pros)
            js_value_col=js_div(old_pros, new_pros)
            # print(f"JS divergence for col-{col}: {js_value_col}")
            js_value_cols.append(js_value_col)
        js_value=np.max(js_value_cols)
        # print(f"{step+1}th js calculate Done!")
        # JS_diver = js_div(old_sample, new_sample, get_softmax=True)
        js_divergence.append(js_value)
        # print("Epoch {} JS divergence: {:.4f}".format(i + 1, JS_diver))

    # js_divergence = np.array(js_divergence).astype(np.float32)
    js_mean = np.mean(js_divergence)
    dis_score = 1/(np.exp(-js_mean))

    js_end_time = time.time()
    js_running_time = js_end_time - js_start_time
    print("Mean JS divergence: {}".format(dis_score))
    print("JS devergence running time: {:.4f}s".format(js_running_time))

    return dis_score > threshold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bjaq", help="Dataset.")
    parser.add_argument(
        "--run",
        help="running type (init, update, default: update)",
        type=str,
        default="update",
    )
    parser.add_argument(
        "--update",
        help="data update type (sample, single, permute, default: sample) ",
        type=str,
        default="sample",
    )
    parser.add_argument(
        "--init_size",
        help="initial data size when run==init, default: 200000",
        type=int,
        default=200000,
    )
    parser.add_argument(
        "--update_size",
        help="update insert size when run==update, default: 10000",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--sample_size",
        help="sample size for update data when run==update, default: 10000",
        type=int,
        default=20000,
    )
    args = parser.parse_args()
    assert args.run in ["init", "update"], "Running Type Error!"
    assert args.update in ["sample", "single", "permute"], "Update Type Error!"
    assert (
        args.update_size >= args.sample_size
    ), "Error! Update Size Must Be Greater Than Sample Size!"
    return args


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cuda")

    args = parse_args()
    arg_util.validate_argument(arg_util.ArgType.DATASET, args.dataset)
    ini_data_size = args.init_size
    dataset_name = args.dataset
    if dataset_name in ["census", "forest", "bjaq", "power"]:
        raw_file_path = f"./data/{dataset_name}/{dataset_name}.npy"
        sampled_file_path = (
            f"./data/{dataset_name}/sampled/{dataset_name}-sample{ini_data_size}.npy"
        )
        config_path = f"./FACE/config/{dataset_name}.yaml"
    else:
        return
    raw_file_path = get_absolute_path(raw_file_path)
    sampled_file_path = get_absolute_path(sampled_file_path)

    if args.run == "init":
        raw_data = np.load(raw_file_path, allow_pickle=True)
        ini_data = sampling(raw_data, ini_data_size, replace=False)
        print(ini_data.shape)
        os.makedirs(os.path.dirname(sampled_file_path), exist_ok=True)
        np.save(sampled_file_path, ini_data)
        print(sampled_file_path, "saved")
        return

    flow_reader = MyFlowModel(config_path=config_path)
    flow = flow_reader.load_model(device)

    if args.run == "update":
        update_size = args.update_size
        sample_size = args.sample_size
        data = np.load(sampled_file_path).astype(np.float32)
        # data=np.load(root_file).astype(np.float32)

        if args.update == "permute":
            update_data = permute_optimized(data, update_size)
        elif args.update == "sample":
            update_data = sampling(data, update_size, replace=True)
        else:
            update_data = single_sampling(data, update_size)

        # update_data = np.concatenate((data, update_data), axis=0)

        mean_reduction, threshold = loss_test(data, update_data, sample_size, flow=flow)

        # print("sample dtype: {}".format(old_sample.dtype))
        is_drift = JS_test(data, update_data, sample_size)
        conca_and_save(sampled_file_path, data, update_data)

    # # print("data sample: {}".format(input[:5]))


if __name__ == "__main__":
    main()
