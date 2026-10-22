import argparse
import sys
import torch
import copy
import time
import numpy as np
import os
from pathlib import Path
from typing import List
from itertools import cycle
from torch.utils.data import Dataset, DataLoader

sys.path.append("./")
from Naru import common
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from FACE.data.table_sample import MyFlowModel
from update_utils import dataset_util
from update_utils.arg_util import add_common_arguments, ArgType
from update_utils.end2end_utils import communicator
from update_utils.model_util import save_torch_model
from update_utils.path_util import get_absolute_path
from update_utils.torch_util import get_torch_device
from update_utils.end2end_utils.json_communicator import JsonCommunicator
from update_utils.EMA import EMA

DEVICE = get_torch_device()

def KD_loss(pm_hat, nm_hat):
    loss = nn.MSELoss()
    output = loss(pm_hat, nm_hat)
    return output

class FaceDataset(Dataset):
    def __init__(self, path=None, dataset=None, data=None, frac=None):
        # path = os.path.join(data_PATH, '{}.npy'.format(dataset_name))

        if path:
            self.data = np.load(path).astype(np.float32)
            self.n, self.dim = self.data.shape
            if not dataset=="power":
                self._add_noise_and_normalize()
        
        else:
            self.data = data
            self.n, self.dim = self.data.shape
        
        # self._report_data()
        
    def _add_noise_and_normalize(self):
        rng = np.random.RandomState(1234)
        noise = rng.rand(self.n, self.dim)
        noise[:,-1] *= 0.1
        self.data += noise

        mu = self.data.mean(axis=0)
        s = self.data.std(axis=0)       
        for i in range(self.dim):
            self.data[:,i] = (self.data[:,i] - mu[i])/s[i]

    def _report_data(self):
        print(f"mu: {self.mu}")
        print(f"s: {self.s}")

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n
    
    def size(self):
        return self.n
    
class data_prefetcher():
    def __init__(self, loader):
        st = time.time()
        self.loader = iter(loader)

        self.origin_loader = iter(loader)
        # print('Generate loader took', time.time() - st)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = self.next_batch.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    common_args: List[ArgType] = [
        ArgType.DATASET,
        ArgType.END2END,
        ArgType.MODEL_UPDATE,
        ArgType.DATA_UPDATE,
    ]
    add_common_arguments(parser, arg_types=common_args)

    parser.add_argument("--bs", type=int, default=1024, help="Batch size.")
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train for."
    )

    return parser.parse_args(argv)

# Module-level globals populated by main() rather than at import time, so this
# file can be imported without consuming sys.argv. END2END_PATH depends on
# args.dataset and is therefore also deferred to main().
args = None
lr_bjaq=1e-3
END2END_PATH = None

def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
        # names = name.split(".")
        # if int(names[1]) in range(args.layers * 2):
        #     print(name)
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    # print("Number of model parameters: {} (~= {:.1f}MB)".format(num_params, mb))
    # print(model)
    return mb

def TransferDataPrepare(train_data, split_indices: List[int]):
    train_size=args.update_size*0.2

    factor = args.update_size/train_data.size()
    # factor = 0.5
    new_size = round(train_size*factor)
    old_size = round(train_size*(1-factor))

    transfer_data = copy.deepcopy(train_data)
    tuples1 = transfer_data.tuples[: -args.update_size]
    rndindices = torch.randperm(len(tuples1))[:old_size]
    transfer_data.tuples = tuples1[rndindices]  

    tuples1 = train_data.tuples[-args.update_size :]
    rndindices = torch.randperm(len(tuples1))[:new_size]
    transfer_data.tuples = torch.cat([transfer_data.tuples, tuples1[rndindices]], dim=0)

    return transfer_data

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

    for step, xb in enumerate(loader):
        if upto and step >= upto:
            break

        xb = xb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.
        log_density = model.log_prob(xb)
        loss = -torch.mean(log_density)
        losses.append(loss.item())

        if split == "train":
            opt.zero_grad()
            loss.backward()
            opt.step()

    if split == "train":
        x = "uncomment"
        scheduler.step()
    if return_losses:
        return losses
    return np.mean(losses)

def RunUpdateEpoch(
    model,
    pmodel,
    opt,
    scheduler,
    train_data,
    transfer_data,
    omega=0.0001,
    val_data=None,
    batch_size=100,
    upto=None,
    epoch_num=None,
    verbose=False,
    log_every=10,
    return_losses=False,
    # table_bits=None,
):
    model.train()
    pmodel.eval()
    dataset = train_data
    losses = []

    # print(train_data.size())
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size * 2, shuffle=True
    )
    trloader = torch.utils.data.DataLoader(
        transfer_data, batch_size=batch_size, shuffle=True
    )

    for step, (trb, xb) in enumerate(zip(cycle(trloader), loader)):
        xb = xb.to(DEVICE).to(torch.float32)
        trb = trb.to(DEVICE).to(torch.float32)

        nll1 = model.log_prob(xb)
        nll2 = model.log_prob(trb)
        nll3 = pmodel.log_prob(trb)
        pred1 = torch.exp(nll2)
        pred2 = torch.exp(nll3)

        loss1 = -torch.mean(nll1)
        loss2 = -torch.mean(nll2)
        loss2_tmp = -torch.mean(nll2)
        loss3 = -torch.mean(nll3)
        # loss3 = model.semiparam_kd_loss(trbhat, pmtrbhat,loss2)
        # loss3 = KD_loss(loss3, loss2_tmp)
        loss3=KD_loss(pred1, pred2)

        loss = (1 - omega) * (loss3 / 2 + loss2 / 2) + omega * loss1
        # loss = (1-omega)*loss3.mean() + omega*loss1

        losses.append(loss.item())

        if step % log_every == 0:
            if epoch_num % 10 == 0:
                print(
                    "Epoch {} Iter {}, loss {:.3f}, lr {:.5f}".format(
                        epoch_num,
                        step,
                        loss.item() / np.log(2),
                        opt.param_groups[0]["lr"],
                    )
                )

        opt.zero_grad()
        loss.backward()
        opt.step()
    scheduler.step()

    if return_losses:
        return losses
    return np.mean(losses)

def RunRetrainEpoch(
    model,
    pmodel,
    opt,
    scheduler,
    train_data,
    transfer_data,
    omega=0.0001,
    val_data=None,
    batch_size=100,
    upto=None,
    epoch_num=None,
    verbose=False,
    log_every=10,
    return_losses=False,
    # table_bits=None,
):
    model.train()
    pmodel.eval()
    dataset = train_data
    losses = []

    # print(train_data.size())
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size * 2, shuffle=True
    )
    trloader = torch.utils.data.DataLoader(
        transfer_data, batch_size=batch_size, shuffle=True
    )

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, "orderings"):
        nsamples = len(model.orderings)

    for step, (trb, xb) in enumerate(zip(trloader, cycle(loader))):
        if upto and step >= upto:
            break

        xb = xb.to(DEVICE).to(torch.float32)
        trb = trb.to(DEVICE).to(torch.float32)

        num_orders_to_forward = 1

        if hasattr(model, "update_masks"):
            # We want to update_masks even for first ever batch.
            model.update_masks()

        # xbhat = model(xb)
        # trbhat = model(trb)
        # pmtrbhat = pmodel(trb)

        if num_orders_to_forward == 1:
            nll1 = model.log_prob(xb)
            nll2 = model.log_prob(trb)
            nll3 = pmodel.log_prob(trb)
            loss1 = -torch.mean(nll1)
            loss2 = -torch.mean(nll2)
            loss3 = -torch.mean(nll3)
            # loss3 = model.semiparam_kd_loss(trbhat, pmtrbhat,loss2)
            # loss3 = model.kd_loss(trbhat, pmtrbhat)

            # loss = (1 - omega) * (loss3 / 2 + loss2 / 2) + omega * loss1
            # loss = (1-omega)*loss3.mean() + omega*loss1
            loss=loss1

        losses.append(loss.item())

        if step % log_every == 0:
            if epoch_num % 10 == 0:
                print(
                    "Epoch {} Iter {}, loss {:.3f}, lr {:.5f}".format(
                        epoch_num,
                        step,
                        loss.item() / np.log(2),
                        opt.param_groups[0]["lr"],
                    )
                )

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        if verbose:
            print("epoch average loss: %f" % (np.mean(losses)))

    if return_losses:
        return losses
    return np.mean(losses)

def NewTransferDataPrepare(train_data):
    train_size=args.update_size*0.2

    if args.end2end:
        factor_config = JsonCommunicator().get('face.factor')
    else:
        factor_config="auto"

    if factor_config == "auto":
        factor = args.update_size/train_data.shape[0]
    elif factor_config == "reverse":
        factor = 1 - args.update_size/train_data.shape[0]
    else:
        factor = float(factor_config)
    # factor = 0.5
    new_size = round(train_size*factor)
    old_size = round(train_size*(1-factor))

    transfer_data = copy.deepcopy(train_data)
    tuples1 = transfer_data[: -args.update_size]
    rndindices = torch.randperm(len(tuples1))[:old_size]
    transfer_data = tuples1[rndindices]  

    tuples1 = train_data[-args.update_size :]
    rndindices = torch.randperm(len(tuples1))[:new_size]
    transfer_data = np.vstack((transfer_data, tuples1[rndindices]))

    return FaceDataset(data=transfer_data)

def Adapt(
    prev_model_path: Path, new_model_path: Path, end2end: bool, freeze: bool = False, omega=0.1,
):
    st_time = time.time()


    model_config_path = f"./FACE/config/{args.dataset}.yaml"
    abs_model_config_path = get_absolute_path(model_config_path)
    my_flow_model = MyFlowModel(config_path=abs_model_config_path)
    model = my_flow_model.load_model(device=DEVICE, model_path=prev_model_path)
    pmodel = my_flow_model.load_model(device=DEVICE, model_path=prev_model_path)
    pmodel.eval()
    model_size = ReportModel(model)

    if freeze:
        # TODO: parameter freezing
        pass
    mb = ReportModel(model)

    if not args.end2end:
        path = END2END_PATH
        raw_dataset = FaceDataset(path=path, dataset=args.dataset)
    else:
        abs_dataset_path = communicator.DatasetPathCommunicator().get()
        raw_dataset = FaceDataset(path=abs_dataset_path, dataset=args.dataset)
        # raw_data=np.load(abs_dataset_path)
        # split_indices = communicator.SplitIndicesCommunicator().get()

    # train_data=raw_data[:-args.update_size]
    # train_data=copy.deepcopy(raw_dataset.data[-args.update_size:])
    train_data=copy.deepcopy(raw_dataset.data[-args.update_size:])
    train_dataset=FaceDataset(data=train_data)

    # train_dataset=raw_dataset
    # train_dataset.data=train_dataset.data[:-args.update_size]
    
    bs=4000 if args.dataset=="bjaq" else 8000
    train_loader = DataLoader(
        train_dataset,
        batch_size = bs*2,
        shuffle=True,
    )

    transfer_dataset = NewTransferDataPrepare(raw_dataset.data)
    transfer_loader = DataLoader(
        dataset=transfer_dataset,
        batch_size=bs,
        shuffle=True,
    )
    test_batch=8000
    test_loader=DataLoader(
        dataset=raw_dataset,
        batch_size=test_batch,
        shuffle=False,
    )

    # train_loader = list(train_loader)
    # transfer_loader = list(transfer_loader)  

    print(
        "train data shape: {}, transfer data shape:{}".format(
            train_dataset.size(), transfer_dataset.size()
        )
    )

    lr=lr_bjaq
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if my_flow_model.anneal_learning_rate:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ema=EMA(0.999)

    for epoch_num in range(args.epochs):
        model.train()
        losses=[]
        for step, (trb, xb) in enumerate(zip(cycle(transfer_loader), train_loader)):
            xb = xb.to(DEVICE).to(torch.float32)
            trb = trb.to(DEVICE).to(torch.float32)

            nll1 = model.log_prob(xb)
            nll2 = model.log_prob(trb)
            nll3 = pmodel.log_prob(trb)
            pred1 = torch.exp(nll2)
            pred2 = torch.exp(nll3)

            loss1 = -torch.mean(nll1)
            loss2 = -torch.mean(nll2)
            loss2_tmp = -torch.mean(nll2)
            loss3 = -torch.mean(nll3)
            # loss3 = model.semiparam_kd_loss(trbhat, pmtrbhat,loss2)
            # loss3 = KD_loss(loss3, loss2_tmp)
            loss3=KD_loss(pred1, pred2)

            loss = (1 - omega) * (loss3 / 2 + loss2 / 2) + omega * loss1
            # loss = loss2
            # loss = (1-omega)*loss3.mean() + omega*loss1

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            if my_flow_model.grad_norm_clip_value is not None:
                clip_grad_norm_(model.parameters(), my_flow_model.grad_norm_clip_value)
            optimizer.step()
            ema.update_model_average(pmodel, model)
        
        scheduler.step()

        if (epoch_num+1) % 10 == 0:
            # test_data=torch.from_numpy(raw_dataset.data).to(DEVICE).to(torch.float32)
            all_losses=[]
            for step, test_data in enumerate(test_loader):
                test_data=test_data.to(DEVICE).to(torch.float32)
                loss = model.log_prob(test_data)
                loss = -torch.mean(loss)
                all_losses.append(loss.item())
            print(
                "Epoch {}, loss {:.3f}, lr {:.5f}".format(
                    (epoch_num+1),
                    np.mean(all_losses),
                    optimizer.param_groups[0]["lr"],
                )
            )
        

    print("Training done; evaluating likelihood on full data:")

    # model_nats = loss
    # model_bits = model_nats / np.log(2)
    # model.model_bits = model_bits
    
    PATH = f"./models/face-adaptST-{args.dataset}-id{my_flow_model.flow_id}-best-val.pt"

    PATH_TA = f"./models/face-adaptTA-{args.dataset}-id{my_flow_model.flow_id}-best-val.pt"

    if not end2end:
        save_torch_model(model, PATH)
        save_torch_model(model, PATH_TA)
        # pre_model = PATH
    else:
        save_torch_model(model, new_model_path)

    torch.cuda.empty_cache()

def Update(
    prev_model_path: Path, new_model_path: Path, end2end: bool, freeze: bool = False
):
    torch.manual_seed(0)
    np.random.seed(0)

    if not args.end2end:
        path = END2END_PATH
        raw_dataset = FaceDataset(path=path, dataset=args.dataset)
    else:
        abs_dataset_path = communicator.DatasetPathCommunicator().get()
        raw_dataset = FaceDataset(path=abs_dataset_path, dataset=args.dataset)
        # raw_data=np.load(abs_dataset_path)
        split_indices = communicator.SplitIndicesCommunicator().get()

    # print("data info: {}".format(table.data.info())
    # print("split index: {}".format(split_indices))
    
    # prepare dataset
    # train_data=raw_data[:-args.update_size]
    # train_data=copy.deepcopy(raw_dataset.data[-args.update_size:])
    train_data=copy.deepcopy(raw_dataset.data[-args.update_size:])
    train_dataset=FaceDataset(data=train_data)

    transfer_dataset = NewTransferDataPrepare(raw_dataset.data)


    # train_loader = list(train_loader)
    # transfer_loader = list(transfer_loader)  

    print(
        "train data shape: {}, transfer data shape:{}".format(
            train_dataset.size(), transfer_dataset.size()
        )
    )


    model_config_path = f"./FACE/config/{args.dataset}.yaml"
    abs_model_config_path = get_absolute_path(model_config_path)
    my_flow_model = MyFlowModel(config_path=abs_model_config_path)
    model = my_flow_model.load_model(device=DEVICE, model_path=prev_model_path)
    pmodel = my_flow_model.load_model(device=DEVICE, model_path=prev_model_path)
    pmodel.eval()
    if freeze:
        pass
    mb = ReportModel(model)

    if False:  # not isinstance(model, transformer.Transformer):
        print("Applying InitWeight()")
        model.apply(InitWeight)

    lr=lr_bjaq
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bs = 4000 if args.dataset=="bjaq" else 8000
    if my_flow_model.anneal_learning_rate:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, 0)
    log_every = 200

    train_loader = DataLoader(
        train_dataset,
        batch_size = my_flow_model.train_batch_size*2,
        shuffle=True,
    )
    transfer_loader = DataLoader(
        dataset=transfer_dataset,
        batch_size=my_flow_model.train_batch_size,
        shuffle=True,
    )
    test_batch=1024
    test_loader=DataLoader(
        dataset=raw_dataset,
        batch_size=test_batch,
        shuffle=False,
    )

    train_losses = []
    train_start = time.time()
    for epoch in range(args.epochs):
        mean_epoch_train_loss = RunUpdateEpoch(
            model,
            pmodel,
            opt,
            scheduler=lr_scheduler,
            train_data=train_dataset,
            transfer_data=transfer_dataset,
            val_data=train_data,
            omega=0.1,
            batch_size=bs,
            epoch_num=epoch,
            log_every=log_every,
            # table_bits=table_bits,
        )

        if (epoch + 1) % 10 == 0:
            print(
                "epoch {} train loss {:.4f} nats / {:.4f} bits".format(
                    (epoch+1), mean_epoch_train_loss, mean_epoch_train_loss / np.log(2)
                )
            )
            since_start = time.time() - train_start
            print("time since start: {:.1f} secs".format(since_start))

            check_point_path = "./models/checkpoints/update__epoch_{}.pt".format(epoch)
            absolute_checkpoint_path = get_absolute_path(check_point_path)
            torch.save(model.state_dict(), absolute_checkpoint_path)
        train_losses.append(mean_epoch_train_loss)

    print("Training done; evaluating likelihood on full data:")
    all_losses = RunEpoch(
        "test",
        model,
        train_data=raw_dataset,
        val_data=raw_dataset,
        opt=None,
        scheduler=lr_scheduler,
        batch_size=8000,
        log_every=500,
        return_losses=True,
    )
    model_nats = np.mean(all_losses)
    model_bits = model_nats / np.log(2)
    model.model_bits = model_bits

    PATH = f"./models/face-update-{args.dataset}-id{my_flow_model.flow_id}-best-val.pt"

    if not end2end:
        save_torch_model(model, PATH)
        pre_model = PATH
    else:
        save_torch_model(model, new_model_path)

    torch.cuda.empty_cache()

def FineTune(prev_model_path: Path, new_model_path: Path, end2end: bool):
    # print("FineTune not implemented")
    # pass

    st_time = time.time()

    model_config_path = f"./FACE/config/{args.dataset}.yaml"
    abs_model_config_path = get_absolute_path(model_config_path)
    my_flow_model = MyFlowModel(config_path=abs_model_config_path)
    model = my_flow_model.load_model(device=DEVICE, model_path=prev_model_path)
    model_size = ReportModel(model)

    mb = ReportModel(model)

    # prepare datasets
    if not args.end2end:
        path = END2END_PATH
        raw_dataset = FaceDataset(path=path, dataset=args.dataset)
    else:
        abs_dataset_path = communicator.DatasetPathCommunicator().get()
        raw_dataset = FaceDataset(path=abs_dataset_path, dataset=args.dataset)
        # raw_data=np.load(abs_dataset_path)
        split_indices = communicator.SplitIndicesCommunicator().get()

    # train_data=raw_data[:-args.update_size]
    # train_data=copy.deepcopy(raw_dataset.data[-args.update_size:])
    train_data=copy.deepcopy(raw_dataset.data[-args.update_size:])
    train_dataset=FaceDataset(data=train_data)

    # train_dataset=raw_dataset
    # train_dataset.data=train_dataset.data[:-args.update_size]

    train_loader = DataLoader(
        train_dataset,
        batch_size = my_flow_model.train_batch_size*2,
        shuffle=True,
    )

    test_batch=1024
    test_loader=DataLoader(
        dataset=raw_dataset,
        batch_size=test_batch,
        shuffle=False,
    )

    # train_loader = list(train_loader)
    # transfer_loader = list(transfer_loader)
    features = train_dataset.dim
    

    print("train data shape: {}".format(train_dataset.size()))

    lr=lr_bjaq
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if my_flow_model.anneal_learning_rate:
        decay_rate = 1
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=decay_rate
        )

    for epoch_num in range(args.epochs):
        model.train()
        losses=[]
        log_every=200
        for step, xb in enumerate(train_loader):
            xb = xb.to(DEVICE).to(torch.float32)

            num_orders_to_forward = 1
            if num_orders_to_forward == 1:
                # loss function of face
                nll1 = model.log_prob(xb)
                loss = -torch.mean(nll1)               

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            if my_flow_model.grad_norm_clip_value is not None:
                clip_grad_norm_(model.parameters(), my_flow_model.grad_norm_clip_value)
            optimizer.step()
        
        scheduler.step()

        if (epoch_num+1) % 10 == 0:
            # test_data=torch.from_numpy(raw_dataset.data).to(DEVICE).to(torch.float32)
            all_losses=[]
            for step, test_data in enumerate(test_loader):
                test_data=test_data.to(DEVICE).to(torch.float32)
                loss = model.log_prob(test_data)
                loss = -torch.mean(loss)
                all_losses.append(loss.item())
            print(
                "Epoch {}, loss {:.3f}, lr {:.5f}".format(
                    (epoch_num+1),
                    np.mean(all_losses),
                    optimizer.param_groups[0]["lr"],
                )
            )
        

    print("Training done; evaluating likelihood on full data:")

    model_nats = loss
    model_bits = model_nats / np.log(2)
    model.model_bits = model_bits
    
    PATH = f"./models/face-finetn-{args.dataset}-id{my_flow_model.flow_id}-best-val.pt"

    if not end2end:
        save_torch_model(model, PATH)
        # pre_model = PATH
    else:
        save_torch_model(model, new_model_path)

def main(argv=None):
    global args, END2END_PATH
    args = parse_args(argv)
    END2END_PATH = f"./data/{args.dataset}/end2end/{args.dataset}.npy"
    end2end: bool = args.end2end
    model_update: str = args.model_update

    # model path
    if not end2end:
        if args.dataset=="bjaq":
            origin_model_path: str = "./models/face-origin-bjaq-id2-best-val.t"
            updated_model_path: str ="./models/face-updated-bjaq-id2-best-val.t"
        elif args.dataset=="power":
            origin_model_path: str = "./models/face-origin-power-id1-best-val.t"
            updated_model_path: str ="./models/face-updated-power-id1-best-val.t"
        prev_model_path: Path = get_absolute_path(origin_model_path)
        new_model_path: Path = get_absolute_path(updated_model_path)

        Update(prev_model_path=prev_model_path, new_model_path=new_model_path, end2end=end2end)
        # FineTune(prev_model_path=prev_model_path, new_model_path=new_model_path, end2end=end2end)
        # Adapt(prev_model_path=prev_model_path, new_model_path=new_model_path, end2end=end2end)

    # end2end
    else:
        prev_model_path: Path = (
            communicator.ModelPathCommunicator().get()
        )  
        new_model_path: Path = prev_model_path 

        # data pool
        pool_path=f"./data/{args.dataset}/end2end/{args.dataset}_pool.npy"
        unlearned_data=np.load(pool_path)
        update_size=unlearned_data.shape[0]
        args.update_size=update_size
        print(f"update_size: {update_size}")

        # remove after model update
        os.remove(pool_path)

        if model_update == "update":
            print("UpdateTask - START")
            Update(
                prev_model_path=prev_model_path,
                new_model_path=new_model_path,
                end2end=end2end,
            )
            print("UpdateTask - END")

        elif model_update == "adapt":
            print("AdaptTask - START")
            Adapt(
                prev_model_path=prev_model_path,
                new_model_path=new_model_path,
                end2end=end2end,
            )
            # Adapt(prev_model_path=prev_model_path, new_model_path=new_model_path)
            print("AdaptTask - END")

        elif model_update == "finetune":
            print("FineTuneTask - START")
            FineTune(prev_model_path=prev_model_path, new_model_path=new_model_path, end2end=end2end)
            print("FineTuneTask - END")

        else:
            raise ValueError("model update method not supported")


if __name__ == "__main__":
    main()
