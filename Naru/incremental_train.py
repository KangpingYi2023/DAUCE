"""Model training."""
import argparse
import copy
import glob
import os
import time
import sys
import collections

sys.path.append("./")
from itertools import cycle
from typing import List

import higher
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import common
import made
import transformer
from update_utils import dataset_util
from update_utils.arg_util import add_common_arguments, ArgType
from update_utils.end2end_utils import communicator
from update_utils.end2end_utils.json_communicator import JsonCommunicator
from update_utils.model_util import save_torch_model
from update_utils.path_util import get_absolute_path
from update_utils.torch_util import get_torch_device
from update_utils.EMA import EMA


DEVICE = get_torch_device(extra=True)

LR_transformer=3e-3
LR_Naru=4e-3


def _get_ablation_flag(key: str, default: bool) -> bool:
    """Read a boolean ablation switch from the shared parameter store.

    Tolerates string values ("false"/"0") in case a config round-trip
    serialised the flag as text.
    """
    value = JsonCommunicator().get_with_default(key, default)
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off")
    return bool(value)


def _get_optional_float(key: str):
    """Read an optional numeric knob; ``None`` when it was never set.

    ``None`` means "keep the historical behaviour", which lets a new knob be
    introduced without silently changing the default of existing configs.
    """
    value = JsonCommunicator().get_with_default(key, None)
    if value is None:
        return None
    return float(value)


def create_parser():
    parser = argparse.ArgumentParser()

    common_args: List[ArgType] = [
        ArgType.DATA_UPDATE,
        # ArgType.DATASET,
        ArgType.END2END,
        ArgType.MODEL_UPDATE,
    ]
    add_common_arguments(parser, arg_types=common_args)

    # Training.
    parser.add_argument(
                "--dataset",
                type=str,
                choices=["bjaq", "census", "forest", "power", "census_gaussian", "bjaq_gaussian",
                         "stats","stats_users", "stats_badges","stats_comments","stats_postHistory","postLinks","stats_posts","stats_tags","stats_votes", 
                        ],
                required=True,
                help="choose datasets: bjaq, census, forest, power",
            )
    parser.add_argument("--num-gpus", type=int, default=0, help="#gpus.")
    parser.add_argument("--bs", type=int, default=512, help="Batch size.")
    parser.add_argument(
        "--warmups",
        type=int,
        default=0,
        help="Learning rate warmup steps.  Crucial for Transformer.",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train for."
    )
    parser.add_argument("--constant-lr", type=float, default=None, help="Constant LR?")
    parser.add_argument(
        "--column-masking",
        action="store_true",
        help="Column masking training, which permits wildcard skipping"
        " at querying time.",
    )

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
        help="Set this flag iff using MADE and specifying --order. Flag --order "
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

    # Ordering.
    parser.add_argument(
        "--num-orderings", type=int, default=1, help="Number of orderings."
    )
    parser.add_argument(
        "--order",
        nargs="+",
        type=int,
        required=False,
        help="Use a specific ordering.  "
        "Format: e.g., [0 2 1] means variable 2 appears second.",
    )
    return parser


args = None  # module-level global; populated by main(). Kept as a module global
# (rather than parsed at import time) so this file can be imported without
# consuming sys.argv, while the many helper functions that reference `args`
# continue to work unchanged once main() has set it.


def Entropy(name, data, bases=None):
    import scipy.stats

    # s = "Entropy of {}:".format(name)
    ret = []
    for base in bases:
        assert base == 2 or base == "e" or base is None
        e = scipy.stats.entropy(data, base=base if base != "e" else None)
        ret.append(e)
        unit = "nats" if (base == "e" or base is None) else "bits"
        # s += " {:.4f} {}".format(e, unit)
    # print(s)
    return ret


def KD_loss(pm_hat, nm_hat):
    loss = nn.MSELoss()
    output = loss(pm_hat, nm_hat)
    return output


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
        if split == "train":
            base_lr = 8e-4
            for param_group in opt.param_groups:
                if args.constant_lr:
                    lr = args.constant_lr
                    param_group["lr"] = lr
                elif args.warmups:
                    t = args.warmups
                    d_model = model.embed_size
                    global_steps = len(loader) * epoch_num + step + 1
                    lr = (d_model**-0.5) * min(
                        (global_steps**-0.5), global_steps * (t**-1.5)
                    )
                    param_group["lr"] = lr

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

        if model.input_bins is None:
            # NOTE: we have to view() it in this order due to the mask
            # construction within MADE.  The masks there on the output unit
            # determine which unit sees what input vars.
            xbhat = xbhat.view(-1, model.nout // model.nin, model.nin)
            # Equivalent to:
            loss = F.cross_entropy(xbhat, xb.long(), reduction="none").sum(-1).mean()
            # loss = loss
        else:
            if num_orders_to_forward == 1:
                loss = model.nll(xbhat, xb).mean()
                loss = loss
            else:
                # Average across orderings & then across minibatch.
                #
                #   p(x) = 1/N sum_i p_i(x)
                #   log(p(x)) = log(1/N) + log(sum_i p_i(x))
                #             = log(1/N) + logsumexp ( log p_i(x) )
                #             = log(1/N) + logsumexp ( - nll_i (x) )
                #
                # Used only at test time.
                logps = []  # [batch size, num orders]
                assert len(model_logits) == num_orders_to_forward, len(model_logits)
                for logits in model_logits:
                    # Note the minus.
                    logps.append(-model.nll(logits, xb))
                logps = torch.stack(logps, dim=1)
                logps = logps.logsumexp(dim=1) + torch.log(
                    torch.tensor(1.0 / nsamples, device=logps.device)
                )
                loss = (-logps).mean()
                # loss = loss

        losses.append(loss.item())

        if step % log_every == 0:
            if split == "train":
                if epoch_num % 10 == 0:
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

    if split == "train":
        x = "uncomment"
        scheduler.step()
    if return_losses:
        return losses
    return np.mean(losses)


def RunUpdateEpoch(
    split,
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
    table_bits=None,
):
    torch.set_grad_enabled(split == "train")
    model.train() if split == "train" else model.eval()
    pmodel.eval()
    dataset = train_data if split == "train" else val_data
    losses = []
    # Per-component loss accumulators (design doc A1). The composite loss alone
    # cannot show whether the KD anchor carries any signal, so the three terms
    # are recorded separately and printed once per training epoch.
    comp_new, comp_replay, comp_kd = [], [], []

    # print(train_data.size())
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size * 2, shuffle=(split == "train")
    )
    trloader = torch.utils.data.DataLoader(
        transfer_data, batch_size=batch_size, shuffle=(split == "train")
    )

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, "orderings"):
        nsamples = len(model.orderings)

    for step, (trb, xb) in enumerate(zip(cycle(trloader), loader)):
        if upto and step >= upto:
            break

        xb = xb.to(DEVICE).to(torch.float32)
        trb = trb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.
        xbhat = None
        trbhat = None
        pmtrbhat = None
        model_logits = []
        model_tr_logits = []
        pmodel_logits = []
        num_orders_to_forward = 1
        if split == "test" and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, "update_masks"):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_out = model(xb)
            model_tr_out = model(trb)
            # Teacher forward is inference-only (design doc A2): building a graph
            # here just accumulates unused grads into pmodel, and it is the cost
            # bottleneck for the second teacher forward added by the anchor work.
            with torch.no_grad():
                pmodel_out = pmodel(trb)
            model_logits.append(model_out)
            model_tr_logits.append(model_tr_out)
            pmodel_logits.append(pmodel_out)
            if xbhat is None:
                xbhat = torch.zeros_like(model_out)
                trbhat = torch.zeros_like(model_tr_out)
                pmtrbhat = torch.zeros_like(pmodel_out)
            xbhat += model_out
            trbhat += model_tr_out
            pmtrbhat += pmodel_out

        if num_orders_to_forward == 1:
            nll1 = model.nll(xbhat, xb)
            nll2 = model.nll(trbhat, trb)
            loss1 = nll1.mean()
            loss2 = nll2.mean()
            # loss3 = model.semiparam_kd_loss(trbhat, pmtrbhat,loss2)
            # loss3 = model.kd_loss(trbhat, pmtrbhat).mean()
            loss3 = KD_loss(trbhat, pmtrbhat).mean()
        
            loss = (1 - omega) * (loss3 / 2 + loss2 / 2) + omega * loss1
            # loss = (1-omega)*loss3.mean() + omega*loss1
            comp_new.append(loss1.item())
            comp_replay.append(loss2.item())
            comp_kd.append(loss3.item())
        
        losses.append(loss.item())

        if step % log_every == 0:
            if split == "train":
                if epoch_num % 10 == 0:
                    print(
                        "Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr".format(
                            epoch_num,
                            step,
                            split,
                            loss.item() / np.log(2) - table_bits[0],
                            loss.item() / np.log(2),
                            table_bits[0],
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

    if split == "train":
        scheduler.step()
        PrintLossComponents(
            "update", epoch_num, comp_new, comp_replay, comp_kd, losses
        )
    if return_losses:
        return losses
    return np.mean(losses)


def PrintLossComponents(
    tag, epoch_num, comp_new, comp_replay, comp_kd, losses, s="const0.5"
):
    """Emit one machine-parsable line with the mean of each loss term (A1).

    ``s`` is the anchor strength the composite loss actually used: the legacy
    composition ``(1-omega)*(KD/2 + replay/2) + omega*new`` is exactly the
    budget-normalised form evaluated at s = 0.5, hence the default tag.
    """
    if not comp_new:
        return

    def _mean(xs):
        return float(np.mean(xs)) if len(xs) else float("nan")

    print(
        "[loss-comp] mode={} ep={} s={} new={:.4f} replay={:.4f} kd={:.4f} loss={:.4f}".format(
            tag,
            epoch_num,
            s,
            _mean(comp_new),
            _mean(comp_replay),
            _mean(comp_kd),
            _mean(losses),
        )
    )


def RunAdaptEpoch(
    split,
    model,
    pmodel,
    opt,
    scheduler,
    train_data,
    ema: EMA,
    transfer_data,
    omega=0.0001,
    val_data=None,
    batch_size=100,
    upto=None,
    epoch_num=None,
    verbose=False,
    log_every=10,
    return_losses=False,
    table_bits=None,
    replay: bool = True,
    ema_per_epoch: bool = False,
    kd_weight=None,
):
    torch.set_grad_enabled(split == "train")
    model.train() if split == "train" else model.eval()
    pmodel.eval()
    dataset = train_data if split == "train" else val_data
    losses = []
    # Per-component loss accumulators (design doc A1); see PrintLossComponents.
    comp_new, comp_replay, comp_kd = [], [], []

    # print(train_data.size())
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size * 2, shuffle=(split == "train")
    )
    trloader = None
    if replay:
        trloader = torch.utils.data.DataLoader(
            transfer_data, batch_size=batch_size, shuffle=(split == "train")
        )

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, "orderings"):
        nsamples = len(model.orderings)

    step_source = zip(cycle(trloader), loader) if replay else ((None, xb) for xb in loader)
    for step, (trb, xb) in enumerate(step_source):
        """
        if split == 'train':
            base_lr = 8e-4
            for param_group in opt.param_groups:
                if args.constant_lr:
                    lr = args.constant_lr
                elif args.warmups:
                    t = args.warmups
                    d_model = model.embed_size
                    global_steps = len(loader) * epoch_num + step + 1
                    lr = (d_model**-0.5) * min(
                        (global_steps**-.5), global_steps * (t**-1.5))
                else:
                    lr = 1e-2

                param_group['lr'] = lr
        """
        if upto and step >= upto:
            break

        xb = xb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.
        xbhat = None
        trbhat = None
        pmtrbhat = None
        model_logits = []
        model_tr_logits = []
        pmodel_logits = []
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

            if replay:
                trb = trb.to(DEVICE).to(torch.float32)
                model_tr_out = model(trb)
                # Teacher forward is inference-only (design doc A2).
                with torch.no_grad():
                    pmodel_out = pmodel(trb)
                model_tr_logits.append(model_tr_out)
                pmodel_logits.append(pmodel_out)
                if trbhat is None:
                    trbhat = torch.zeros_like(model_tr_out)
                    pmtrbhat = torch.zeros_like(pmodel_out)
                trbhat += model_tr_out
                pmtrbhat += pmodel_out

        if num_orders_to_forward == 1:
            nll1 = model.nll(xbhat, xb)
            loss1 = nll1.mean()
            if replay:
                nll2 = model.nll(trbhat, trb)
                loss2 = nll2.mean()
                # loss3 = model.semiparam_kd_loss(trbhat, pmtrbhat,loss2)
                # loss3 = model.kd_loss(trbhat, pmtrbhat).mean()
                loss3 = KD_loss(trbhat, pmtrbhat).mean()

                if kd_weight is None:
                    # Legacy DAUCE composition: KD and replay-NLL split the
                    # (1 - omega) budget 50/50, i.e. a constant anchor strength
                    # s = 0.5 in the budget-normalised form below.
                    loss = (1 - omega) * (loss3 / 2 + loss2 / 2) + omega * loss1
                    # loss = (1-omega)*loss3.mean() + omega*loss1
                else:
                    # Budget-normalised form (design doc A3/A6): whatever the
                    # anchor does not consume flows back to weighted replay
                    # instead of being spent on a collapsed KD target. Setting
                    # kd_weight=0 gives the KD-off diagnostic arm, which
                    # separates "the anchor vanished" from "the lagging teacher
                    # drags the student".
                    loss = omega * loss1 + (1 - omega) * (
                        kd_weight * loss3 + (1 - kd_weight) * loss2
                    )
                comp_replay.append(loss2.item())
                comp_kd.append(loss3.item())
            else:
                # w/o weighted replay: no transfer set, pure fit on the new
                # batch -- the catastrophic-forgetting baseline.
                loss = loss1
            comp_new.append(loss1.item())

        losses.append(loss.item())

        if step % log_every == 0:
            if split == "train":
                if epoch_num % 10 == 0:
                    print(
                        "Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr".format(
                            epoch_num,
                            step,
                            split,
                            loss.item() / np.log(2) - table_bits[0],
                            loss.item() / np.log(2),
                            table_bits[0],
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
            if ema is not None and not ema_per_epoch:
                ema.update_model_average(pmodel, model)

        if verbose:
            print("%s epoch average loss: %f" % (split, np.mean(losses)))

    if split == "train":
        # Per-epoch teacher refresh. beta=0.8 has a half-life of ~3 updates, so
        # applying it per optimizer step lets the teacher collapse onto the student
        # within the first epoch once the update batch spans many steps (bjaq ~45
        # steps/epoch, forest ~69), which erases the pre-drift distillation anchor.
        if ema is not None and ema_per_epoch:
            ema.update_model_average(pmodel, model)
        if scheduler is not None:
            scheduler.step()
        s_tag = "const0.5" if kd_weight is None else "{:.3f}".format(kd_weight)
        PrintLossComponents(
            "adapt", epoch_num, comp_new, comp_replay, comp_kd, losses, s=s_tag
        )
    if return_losses:
        return losses
    return np.mean(losses)


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


def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the 'true' ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


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
    NUM_HEADS=4
    return transformer.Transformer(
        num_blocks=args.blocks,
        d_model=args.dmodel,
        d_ff=args.dff,
        num_heads=NUM_HEADS,
        nin=len(cols_to_train),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        use_positional_embs=True,
        activation=args.transformer_act,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
        seed=seed,
    ).to(DEVICE)


def InitWeight(m):
    if type(m) == made.MaskedLinear or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight, std=0.02)


def GetDatasetPath(end2end: bool = False):
    if not end2end:
        table, split_indices = dataset_util.DatasetLoader.load_permuted_dataset(
            dataset=args.dataset, permute=False
        )
    else:
        dataset_name = args.dataset.split("_")[0]
        if dataset_name in ["stats"]:
            abs_dataset_dir = communicator.DatasetPathCommunicator().get()
            table_name = args.dataset.split("_")[1]
            dataset_filename = f"{table_name}.npy"
            abs_dataset_path = os.path.join(abs_dataset_dir, dataset_filename)
        else:
            abs_dataset_path = communicator.DatasetPathCommunicator().get()
        
        table = dataset_util.NpyDatasetLoader.load_npy_dataset_from_path(
            path=abs_dataset_path
        )
        split_indices = communicator.SplitIndicesCommunicator().get()

    return table, split_indices


def AdaptTask(
    pre_model, new_model_path=None, seed=0, end2end: bool = False, freeze=False
):
    torch.manual_seed(0)
    np.random.seed(0)

    table, split_indices = GetDatasetPath(end2end)

    table_bits = Entropy(
        table,
        table.data.fillna(value=0).groupby([c.name for c in table.columns]).size(),
        [2],
    )

    fixed_ordering = None

    if args.order is not None:
        print("Using passed-in order:", args.order)
        fixed_ordering = args.order

    # print("data info: {}".format(table.data.info()))

    # Progressive-distillation ablation switches (defaults keep the full DAUCE
    # behaviour): EMA teacher update, cosine LR decay, weighted replay and their
    # hyper-parameters (omega, EMA beta, replay sampling ratio).
    use_ema = _get_ablation_flag("adapt.use_ema", True)
    use_cosine = _get_ablation_flag("adapt.use_cosine_decay", True)
    use_replay = _get_ablation_flag("adapt.use_weighted_replay", True)
    omega = float(JsonCommunicator().get_with_default("adapt.omega", 0.1))
    ema_beta = float(JsonCommunicator().get_with_default("adapt.ema_beta", 0.8))
    ema_per_epoch = _get_ablation_flag("adapt.ema_per_epoch", False)
    # Design doc A3: explicit constant anchor strength. None keeps the legacy
    # (1-omega)*(KD/2 + replay/2) composition; 0.0 is the KD-off arm that
    # separates "anchor vanished" from "lagging teacher drags the student".
    kd_weight = _get_optional_float("adapt.kd_weight")
    # Design doc A5: teacher half-life expressed in epochs. None keeps the
    # historical per-step beta so earlier results stay reproducible.
    ema_halflife_epochs = _get_optional_float("adapt.ema_halflife_epochs")
    stale_log = _get_ablation_flag("adapt.stale_log", True)
    transfer_ratio = float(
        JsonCommunicator().get_with_default("adapt.transfer_ratio", 0.2)
    )
    print(
        f"AdaptTask ablation config: use_ema={use_ema}, use_cosine={use_cosine}, "
        f"use_replay={use_replay}, omega={omega}, ema_beta={ema_beta}, "
        f"ema_per_epoch={ema_per_epoch}, transfer_ratio={transfer_ratio}, "
        f"kd_weight={kd_weight}, ema_halflife_epochs={ema_halflife_epochs}, "
        f"stale_log={stale_log}"
    )

    """
    Loop over update batches (split indexes) and create new models base on previous model
    """

    # print("split index: {}".format(split_indices))
    for update_step in range(len(split_indices) - 1):
        table_main = table

        if args.model == "transformer":
            model = MakeTransformer(
                cols_to_train=table.columns, fixed_ordering=fixed_ordering, seed=seed
            )

            pmodel = MakeTransformer(
                cols_to_train=table.columns, fixed_ordering=fixed_ordering, seed=seed
            )

            model.load_state_dict(torch.load(pre_model))
            pmodel.load_state_dict(torch.load(pre_model))
            pmodel.eval()
        else:
            model = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )
            pmodel = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )
            pmodel.load_state_dict(torch.load(pre_model))
            model.load_state_dict(torch.load(pre_model))
            pmodel.eval()
            if freeze:
                # Layered freeze: fix the bottom (layers-1) FC blocks so only the
                # top hidden + output layer adapt to the EMA teacher. NOTE: this
                # was `p.requires_grad == False` (a comparison whose result was
                # discarded), so the freeze never took effect and adapt trained
                # the whole net. Cast to a real assignment and log how many
                # tensors were frozen so a rerun can confirm it is active.
                frozen = 0
                for name, p in model.named_parameters():
                    names = name.split(".")
                    if int(names[1]) in range((args.layers - 1) * 2):
                        p.requires_grad = False
                        frozen += 1
                total = sum(1 for _ in model.named_parameters())
                print(
                    f"[freeze] AdaptTask froze {frozen}/{total} param tensors "
                    f"(bottom {args.layers - 1} of {args.layers} FC layers fixed)"
                )

        mb = ReportModel(model)

        if False:  # not isinstance(model, transformer.Transformer):
            print("Applying InitWeight()")
            model.apply(InitWeight)

        if args.model == "transformer":
            opt = torch.optim.Adam(
                list(model.parameters()),
                LR_transformer,
                betas=(0.9, 0.98),
                eps=1e-9,
            )

            decay_rate = 0.98
        else:
            opt = torch.optim.Adam(list(model.parameters()), lr=LR_Naru)
            decay_rate = 0.98
        if use_cosine:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, 0)
        else:
            lr_scheduler = None
        bs = args.bs
        log_every = 200

        train_data = common.TableDataset(table_main)
        print("dataset size: {}".format(train_data.size()))

        transfer_data = TransferDataPrepare(
            train_data=train_data,
            split_indices=split_indices,
            update_step=update_step,
            transfer_ratio=transfer_ratio,
        )

        train_data.tuples = train_data.tuples[-args.update_size: ]

        print(
            "train data shape: {}, transfer data shape:{}".format(
                train_data.tuples.shape, transfer_data.tuples.shape
            )
        )

        # Design doc A5: beta is applied once per teacher refresh, but the number
        # of refreshes per epoch grows with the update pool (census ~4 steps,
        # bjaq ~45, forest ~69), so a single beta silently encodes three
        # different algorithms -- on a large pool the teacher forgets the
        # pre-drift model inside the first epoch. Supplying a half-life in epochs
        # converts it to the matching per-refresh beta, which makes the tracking
        # time scale pool-independent.
        pool = int(train_data.tuples.shape[0])
        steps_per_epoch = max(1, int(np.ceil(pool / float(bs * 2))))
        refreshes_per_epoch = 1 if ema_per_epoch else steps_per_epoch
        if use_ema:
            if ema_halflife_epochs is not None and ema_halflife_epochs > 0:
                ema_beta_eff = 0.5 ** (
                    1.0 / (ema_halflife_epochs * refreshes_per_epoch)
                )
                print(
                    "[ema] halflife override: beta={:.6f} (h={} epochs, S={} "
                    "steps/epoch, refresh={}; configured ema_beta={} ignored)".format(
                        ema_beta_eff,
                        ema_halflife_epochs,
                        steps_per_epoch,
                        "per-epoch" if ema_per_epoch else "per-step",
                        ema_beta,
                    )
                )
            else:
                ema_beta_eff = ema_beta
            ema = EMA(ema_beta_eff)
            if 0.0 < ema_beta_eff < 1.0:
                half_life_epochs = (
                    float(np.log(0.5) / np.log(ema_beta_eff)) / refreshes_per_epoch
                )
                # Share of theta_0 still present in the teacher after one epoch.
                residual = ema_beta_eff ** refreshes_per_epoch
            else:
                half_life_epochs = float("nan")
                residual = float("nan")
            print(
                "[ema] pool={} S={} steps/epoch refresh={} beta={:.6f} "
                "half_life={:.3f} epochs theta0_residual_after_1epoch={:.3e}".format(
                    pool,
                    steps_per_epoch,
                    "per-epoch" if ema_per_epoch else "per-step",
                    ema_beta_eff,
                    half_life_epochs,
                    residual,
                )
            )
        else:
            ema = None
            print(
                "[ema] disabled (adapt.use_ema=false) pool={} S={} steps/epoch".format(
                    pool, steps_per_epoch
                )
            )

        # Design doc A4: record how stale the event-start teacher already is.
        # pmodel still holds theta_0 here (no EMA refresh has run yet), so this
        # probe must stay ahead of the training loop.
        if stale_log:
            old_rows = getattr(transfer_data, "old_size", None)
            old_part = transfer_data.tuples[:old_rows] if old_rows else None
            st = ProbeAnchorNll(pmodel, train_data.tuples, old_part)
            n_cols = int(train_data.tuples.shape[1])
            stale = st["new"] - st["old"]
            print(
                "[stale] event={} pool={} old_rows={} nll_theta0_new={:.4f} "
                "nll_theta0_old={:.4f} stale={:.4f} stale_per_col={:.6f}".format(
                    update_step + 1,
                    pool,
                    0 if old_part is None else int(old_part.shape[0]),
                    st["new"],
                    st["old"],
                    stale,
                    stale / max(n_cols, 1),
                )
            )

        train_losses = []
        train_start = time.time()
        for epoch in range(args.epochs):
            mean_epoch_train_loss = RunAdaptEpoch(
                "train",
                model,
                pmodel,
                opt,
                scheduler=lr_scheduler,
                train_data=train_data,
                transfer_data=transfer_data,
                val_data=train_data,
                omega=omega,
                batch_size=bs,
                epoch_num=epoch,
                log_every=log_every,
                table_bits=table_bits,
                ema=ema,
                replay=use_replay,
                ema_per_epoch=ema_per_epoch,
                kd_weight=kd_weight,
            )

            if (epoch + 1) % 10 == 0:
                print(
                    "epoch {} train loss {:.4f} nats / {:.4f} bits".format(
                        epoch+1, mean_epoch_train_loss, mean_epoch_train_loss / np.log(2)
                    )
                )
                since_start = time.time() - train_start
                print("time since start: {:.1f} secs".format(since_start))

                check_point_path = (
                    "./models/checkpoints/update_batch_{}_epoch_{}.pt".format(
                        update_step + 1, epoch
                    )
                )
                absolute_checkpoint_path = get_absolute_path(check_point_path)
                torch.save(model.state_dict(), absolute_checkpoint_path)
            train_losses.append(mean_epoch_train_loss)

        print("Training done; evaluating likelihood on full data:")
        all_losses = RunEpoch(
            "test",
            model,
            train_data=train_data,
            val_data=train_data,
            opt=None,
            scheduler=lr_scheduler,
            batch_size=1024,
            log_every=500,
            table_bits=table_bits,
            return_losses=True,
        )
        model_nats = np.mean(all_losses)
        model_bits = model_nats / np.log(2)
        model.model_bits = model_bits

        if fixed_ordering is None:
            if seed is not None:
                PATH = "./models/adapt{}-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}.pt".format(
                    "T" if freeze else "F",
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits[0],
                    args.epochs,
                    seed,
                )
            else:
                PATH = "./models/adapt-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}-{}.pt".format(
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits[0],
                    args.epochs,
                    seed,
                    time.time(),
                )
        else:
            annot = ""
            if args.inv_order:
                annot = "-invOrder"

            PATH = "./models/adapt-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}-order{}{}.pt".format(
                args.dataset,
                mb,
                model.model_bits,
                table_bits[0],
                args.epochs,
                seed,
                "_".join(map(str, fixed_ordering)),
                annot,
            )
        PPATH= "./models/adaptTA-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}.pt".format(
            args.dataset,
            mb,
            model.model_bits,
            table_bits[0],
            args.epochs,
            seed,
        )
        if not end2end:
            save_torch_model(model, PATH)
            save_torch_model(pmodel, PPATH)
            pre_model = PATH
        else:
            save_torch_model(model, new_model_path)


def ProbeAnchorNll(pmodel, new_tuples, old_tuples, batch_size=1024):
    """Mean NLL of the event-start teacher on the pool and on the replay-old part.

    Their gap (``stale``) is the candidate signal for adapting the anchor
    strength (design doc A4/A10): it measures how much of the incoming batch the
    pre-drift model can no longer explain, relative to the rows sampled from
    before the drift. Inference only -- no graph, no parameter update.

    The RNG state is saved and restored so that inserting this probe cannot
    perturb the training that follows it: without that guarantee the legacy arm
    would no longer reproduce the pre-instrumentation runs bit-for-bit, and
    every new number would be incomparable with E4ema/E4probe2.
    """
    pmodel.eval()
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        out = {}
        for name, tuples in (("new", new_tuples), ("old", old_tuples)):
            if tuples is None or len(tuples) == 0:
                out[name] = float("nan")
                continue
            total, count = 0.0, 0
            with torch.no_grad():
                for i in range(0, len(tuples), batch_size):
                    xb = tuples[i : i + batch_size].to(DEVICE).to(torch.float32)
                    if hasattr(pmodel, "update_masks"):
                        pmodel.update_masks()
                    nll = pmodel.nll(pmodel(xb), xb)
                    total += float(nll.sum())
                    count += int(nll.numel())
            out[name] = total / max(count, 1)
    finally:
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    return out


def TransferDataPrepare(train_data, split_indices: List[int], update_step, transfer_ratio: float = 0.2):
    # print(f"Update size:{args.update_size}")
    transfer_size=round(args.update_size*transfer_ratio)

    # weighted sampled transfer set construction
    factor=args.update_size/(train_data.size()-args.update_size)
    # factor=0.7
    old_size=round((1-factor)*transfer_size)
    new_size=round(factor*transfer_size)

    transfer_data = copy.deepcopy(train_data)
    tuples1 = transfer_data.tuples[: -args.update_size]
    rndindices = torch.randperm(len(tuples1))[:old_size]
    transfer_data.tuples = tuples1[rndindices]  

    tuples1 = train_data.tuples[-args.update_size :]
    rndindices = torch.randperm(len(tuples1))[:new_size]
    transfer_data.tuples = torch.cat(
        [transfer_data.tuples, tuples1[rndindices]], dim=0
    )
    # Remember the split so the stale probe (design doc A4) can address the
    # replay-old rows without re-deriving the sampling arithmetic.
    transfer_data.old_size = old_size

    return transfer_data


def UpdateTask(
    pre_model, new_model_path=None, seed=0, freeze=False, end2end: bool = False
):
    torch.manual_seed(0)
    np.random.seed(0)

    table, split_indices = GetDatasetPath(end2end)

    table_bits = Entropy(
        table,
        table.data.fillna(value=0).groupby([c.name for c in table.columns]).size(),
        [2],
    )

    fixed_ordering = None

    if args.order is not None:
        print("Using passed-in order:", args.order)
        fixed_ordering = args.order

    # print("data info: {}".format(table.data.info()))

    """
    Loop over update batches (split indexes) and create new models base on previous model
    """

    # print("split index: {}".format(split_indices))
    for update_step in range(len(split_indices) - 1):
        table_main = table

        if args.model == "transformer":
            model = MakeTransformer(
                cols_to_train=table.columns, fixed_ordering=fixed_ordering, seed=seed
            )

            pmodel = MakeTransformer(
                cols_to_train=table.columns, fixed_ordering=fixed_ordering, seed=seed
            )

            model.load_state_dict(torch.load(pre_model))
            pmodel.load_state_dict(torch.load(pre_model))
            pmodel.eval()
        else:
            model = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )
            pmodel = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )
            pmodel.load_state_dict(torch.load(pre_model))
            model.load_state_dict(torch.load(pre_model))
            pmodel.eval()
            if freeze:
                # Same layered-freeze fix as AdaptTask: `== False` was a
                # discarded comparison, so the freeze never took effect.
                frozen = 0
                for name, p in model.named_parameters():
                    names = name.split(".")
                    if int(names[1]) in range((args.layers - 1) * 2):
                        p.requires_grad = False
                        frozen += 1
                total = sum(1 for _ in model.named_parameters())
                print(
                    f"[freeze] UpdateTask froze {frozen}/{total} param tensors "
                    f"(bottom {args.layers - 1} of {args.layers} FC layers fixed)"
                )

        mb = ReportModel(model)

        if False:  # not isinstance(model, transformer.Transformer):
            print("Applying InitWeight()")
            model.apply(InitWeight)

        if args.model == "transformer":
            opt = torch.optim.Adam(
                list(model.parameters()),
                LR_transformer,
                betas=(0.9, 0.98),
                eps=1e-9,
            )

            decay_rate = 0.98
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=opt, gamma=decay_rate
            )
        else:
            opt = torch.optim.Adam(list(model.parameters()), lr=LR_Naru)
            decay_rate = 0.98
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=opt, gamma=decay_rate
            )
        bs = args.bs
        log_every = 200

        train_data = common.TableDataset(table_main)
        print("dataset size: {}".format(train_data.size()))

        transfer_data = TransferDataPrepare(
            train_data=train_data, split_indices=split_indices, update_step=update_step
        )

        train_data.tuples = train_data.tuples[ -args.update_size: ]

        print(
            "train data shape: {}, transfer data shape:{}".format(
                train_data.tuples.shape, transfer_data.tuples.shape
            )
        )
        train_losses = []
        train_start = time.time()
        for epoch in range(args.epochs):
            mean_epoch_train_loss = RunUpdateEpoch(
                "train",
                model,
                pmodel,
                opt,
                scheduler=lr_scheduler,
                train_data=train_data,
                transfer_data=transfer_data,
                val_data=train_data,
                omega=0.1,
                batch_size=bs,
                epoch_num=epoch,
                log_every=log_every,
                table_bits=table_bits,
            )

            if (epoch + 1) % 10 == 0:
                print(
                    "epoch {} train loss {:.4f} nats / {:.4f} bits".format(
                        epoch, mean_epoch_train_loss, mean_epoch_train_loss / np.log(2)
                    )
                )
                since_start = time.time() - train_start
                print("time since start: {:.1f} secs".format(since_start))

                check_point_path = (
                    "./models/checkpoints/update_batch_{}_epoch_{}.pt".format(
                        update_step + 1, epoch
                    )
                )
                absolute_checkpoint_path = get_absolute_path(check_point_path)
                torch.save(model.state_dict(), absolute_checkpoint_path)
            train_losses.append(mean_epoch_train_loss)

        print("Training done; evaluating likelihood on full data:")
        all_losses = RunEpoch(
            "test",
            model,
            train_data=train_data,
            val_data=train_data,
            opt=None,
            scheduler=lr_scheduler,
            batch_size=1024,
            log_every=500,
            table_bits=table_bits,
            return_losses=True,
        )
        model_nats = np.mean(all_losses)
        model_bits = model_nats / np.log(2)
        model.model_bits = model_bits

        if fixed_ordering is None:
            if seed is not None:
                PATH = "./models/update{}-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}.pt".format(
                    "T" if freeze else "F",
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits[0],
                    args.epochs,
                    seed,
                )
            else:
                PATH = "./models/update-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}-{}.pt".format(
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits[0],
                    args.epochs,
                    seed,
                    time.time(),
                )
        else:
            annot = ""
            if args.inv_order:
                annot = "-invOrder"

            PATH = "./models/update-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}-order{}{}.pt".format(
                args.dataset,
                mb,
                model.model_bits,
                table_bits[0],
                args.epochs,
                seed,
                "_".join(map(str, fixed_ordering)),
                annot,
            )
        if not end2end:
            save_torch_model(model, PATH)
            pre_model = PATH
        else:
            save_torch_model(model, new_model_path)


def FineTuneTask(pre_model, new_model_path=None, seed=0, end2end: bool = False):
    torch.manual_seed(0)
    np.random.seed(0)

    table, split_indices = GetDatasetPath(end2end)

    table_bits = Entropy(
        table,
        table.data.fillna(value=0).groupby([c.name for c in table.columns]).size(),
        [2],
    )
    table_bits = table_bits[0]
    fixed_ordering = None

    if args.order is not None:
        print("Using passed-in order:", args.order)
        fixed_ordering = args.order

    # print(table.data.info())

    """
    Loop over update batches (split indexes) and create new models base on previous model
    """

    for update_step in range(len(split_indices) - 1):
        table_main = table

        if args.model == "transformer":
            model = MakeTransformer(
                cols_to_train=table.columns, fixed_ordering=fixed_ordering, seed=seed
            )

            model.load_state_dict(torch.load(pre_model))
            model.train()
        else:
            model = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )
            model.load_state_dict(torch.load(pre_model))
            model.train()

        mb = ReportModel(model)

        if isinstance(model, transformer.Transformer):
            opt = torch.optim.Adam(
                list(model.parameters()),
                LR_transformer,
                betas=(0.9, 0.98),
                eps=1e-9,
            )
            decay_rate = 1
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=opt, gamma=decay_rate
            )
        else:
            opt = torch.optim.Adam(list(model.parameters()), lr=LR_Naru)
            decay_rate = 1
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=opt, gamma=decay_rate
            )

        bs = args.bs
        log_every = 200

        train_data = common.TableDataset(table_main)

        train_data.tuples = train_data.tuples[-args.update_size:]

        train_losses = []
        train_start = time.time()
        print(
            "finetuning, initial lr = {}, update batch {}".format(
                opt.param_groups[0]["lr"], update_step
            )
        )
        for epoch in range(args.epochs):
            # print(train_data.tuples.shape)
            mean_epoch_train_loss = RunEpoch(
                "train",
                model,
                opt,
                scheduler=lr_scheduler,
                train_data=train_data,
                val_data=train_data,
                batch_size=bs,
                epoch_num=epoch,
                log_every=log_every,
                table_bits=table_bits,
            )

            if (epoch + 1) % 10 == 0:
                print(
                    "epoch {} train loss {:.4f} nats / {:.4f} bits".format(
                        epoch, mean_epoch_train_loss, mean_epoch_train_loss / np.log(2)
                    )
                )
                since_start = time.time() - train_start
                print("time since start: {:.1f} secs".format(since_start))

            train_losses.append(mean_epoch_train_loss)

        print("Training done; evaluating likelihood on full data:")
        all_losses = RunEpoch(
            "test",
            model,
            train_data=train_data,
            val_data=train_data,
            opt=None,
            scheduler=lr_scheduler,
            batch_size=1024,
            log_every=500,
            table_bits=table_bits,
            return_losses=True,
        )
        model_nats = np.mean(all_losses)
        model_bits = model_nats / np.log(2)
        model.model_bits = model_bits

        if fixed_ordering is None:
            if seed is not None:
                PATH = "./models/FineTn-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}.pt".format(
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits,
                    args.epochs,
                    seed,
                )
            else:
                PATH = "./models/FineTn-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}-{}.pt".format(
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits,
                    args.epochs,
                    seed,
                    time.time(),
                )
        else:
            annot = ""
            if args.inv_order:
                annot = "-invOrder"

            PATH = "./models/FineTn-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}-order{}{}.pt".format(
                args.dataset,
                mb,
                model.model_bits,
                table_bits[0],
                args.epochs,
                seed,
                "_".join(map(str, fixed_ordering)),
                annot,
            )
        if not end2end:
            save_torch_model(model, PATH)
            pre_model = PATH
        else:
            save_torch_model(model, new_model_path)


def RetrainTask(pre_model, new_model_path=None, seed=0, end2end: bool = False):
    """Baseline: discard the previous model and retrain from scratch.

    On every trigger the model is rebuilt with random initialization and
    trained on the full updated table (original rows + all appended batches),
    i.e. the standard "full retrain" baseline for Table V/VI.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    table, split_indices = GetDatasetPath(end2end)

    table_bits = Entropy(
        table,
        table.data.fillna(value=0).groupby([c.name for c in table.columns]).size(),
        [2],
    )[0]

    # Keep the natural column ordering so the from-scratch model stays
    # parameter-compatible with the pretrained checkpoint it replaces.
    fixed_ordering = None
    if args.order is not None:
        print("Using passed-in order:", args.order)
        fixed_ordering = args.order
    else:
        fixed_ordering = list(range(len(table.columns)))

    train_data = common.TableDataset(table)

    for update_step in range(len(split_indices) - 1):
        if args.model == "transformer":
            model = MakeTransformer(
                cols_to_train=table.columns, fixed_ordering=fixed_ordering, seed=seed
            )
        else:
            model = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )
        model.train()

        mb = ReportModel(model)

        if isinstance(model, transformer.Transformer):
            opt = torch.optim.Adam(
                list(model.parameters()),
                LR_transformer,
                betas=(0.9, 0.98),
                eps=1e-9,
            )
        else:
            opt = torch.optim.Adam(list(model.parameters()), lr=LR_Naru)
        decay_rate = 1
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=opt, gamma=decay_rate
        )

        bs = args.bs
        log_every = 200

        train_losses = []
        train_start = time.time()
        print(
            "retraining from scratch on {} rows, initial lr = {}, update batch {}".format(
                train_data.tuples.shape[0], opt.param_groups[0]["lr"], update_step
            )
        )
        for epoch in range(args.epochs):
            mean_epoch_train_loss = RunEpoch(
                "train",
                model,
                opt,
                scheduler=lr_scheduler,
                train_data=train_data,
                val_data=train_data,
                batch_size=bs,
                epoch_num=epoch,
                log_every=log_every,
                table_bits=table_bits,
            )

            if (epoch + 1) % 10 == 0:
                print(
                    "epoch {} train loss {:.4f} nats / {:.4f} bits".format(
                        epoch, mean_epoch_train_loss, mean_epoch_train_loss / np.log(2)
                    )
                )
                since_start = time.time() - train_start
                print("time since start: {:.1f} secs".format(since_start))

            train_losses.append(mean_epoch_train_loss)

        model_bits = np.mean(train_losses) / np.log(2)
        model.model_bits = model_bits

        if not end2end:
            PATH = "./models/Retrain-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}-step{}.pt".format(
                args.dataset,
                mb,
                model.model_bits,
                table_bits,
                args.epochs,
                seed,
                update_step,
            )
            save_torch_model(model, PATH)
        else:
            save_torch_model(model, new_model_path)


def BayesCardExp(pre_model=None, seed=0):
    torch.manual_seed(0)
    np.random.seed(0)

    # Load dataset
    table, split_indices = dataset_util.DatasetLoader.load_permuted_dataset(
        dataset=args.dataset, permute=False
    )

    table_bits = Entropy(
        table,
        table.data.fillna(value=0).groupby([c.name for c in table.columns]).size(),
        [2],
    )[0]
    fixed_ordering = None

    if args.order is not None:
        print("Using passed-in order:", args.order)
        fixed_ordering = args.order

    print(table.data.info())

    if pre_model == None:
        if args.dataset in ["dmv-tiny", "dmv", "tpcds"]:
            model = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )
        else:
            assert False, args.dataset

        mb = ReportModel(model)

        if not isinstance(model, transformer.Transformer):
            print("Applying InitWeight()")
            model.apply(InitWeight)

        if isinstance(model, transformer.Transformer):
            opt = torch.optim.Adam(
                list(model.parameters()),
                LR_transformer,
                betas=(0.9, 0.98),
                eps=1e-9,
            )
        else:
            opt = torch.optim.Adam(list(model.parameters()), lr=LR_Naru)
            decay_rate = 0.96
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=opt, gamma=decay_rate
            )

        bs = args.bs
        log_every = 200

        train_data = common.TableDataset(table)
        split_idx = int(len(train_data.tuples) * 0.2)
        train_data.tuples = train_data.tuples[:split_idx, :]
        train_losses = []
        train_start = time.time()
        for epoch in range(args.epochs):
            mean_epoch_train_loss = RunEpoch(
                "train",
                model,
                opt,
                scheduler=lr_scheduler,
                train_data=train_data,
                val_data=train_data,
                batch_size=bs,
                epoch_num=epoch,
                log_every=log_every,
                table_bits=table_bits,
            )

            if epoch % 1 == 0:
                print(
                    "epoch {} train loss {:.4f} nats / {:.4f} bits".format(
                        epoch, mean_epoch_train_loss, mean_epoch_train_loss / np.log(2)
                    )
                )
                since_start = time.time() - train_start
                print("time since start: {:.1f} secs".format(since_start))

            train_losses.append(mean_epoch_train_loss)

        print("Training done; evaluating likelihood on full data:")
        all_losses = RunEpoch(
            "test",
            model,
            train_data=train_data,
            val_data=train_data,
            opt=None,
            scheduler=lr_scheduler,
            batch_size=1024,
            log_every=500,
            table_bits=table_bits,
            return_losses=True,
        )
        model_nats = np.mean(all_losses)
        model_bits = model_nats / np.log(2)
        model.model_bits = model_bits

        if fixed_ordering is None:
            if seed is not None:
                PATH = "./models/00{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}.pt".format(
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits,
                    model.name(),
                    args.epochs,
                    seed,
                )
            else:
                PATH = "./models/00{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-{}.pt".format(
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits,
                    model.name(),
                    args.epochs,
                    seed,
                    time.time(),
                )
        else:
            annot = ""
            if args.inv_order:
                annot = "-invOrder"

            PATH = "./models/00{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-order{}{}.pt".format(
                args.dataset,
                mb,
                model.model_bits,
                table_bits,
                model.name(),
                args.epochs,
                seed,
                "_".join(map(str, fixed_ordering)),
                annot,
            )
        save_torch_model(model, PATH)
        pre_model = PATH

    if args.dataset in ["dmv-tiny", "dmv", "tpcds"]:
        model = MakeMade(
            scale=args.fc_hiddens,
            cols_to_train=table.columns,
            seed=seed,
            fixed_ordering=fixed_ordering,
        )
        pmodel = MakeMade(
            scale=args.fc_hiddens,
            cols_to_train=table.columns,
            seed=seed,
            fixed_ordering=fixed_ordering,
        )
        pmodel.load_state_dict(torch.load(pre_model))
        pmodel.eval()
    else:
        assert False, args.dataset

    mb = ReportModel(model)

    if not isinstance(model, transformer.Transformer):
        print("Applying InitWeight()")
        model.apply(InitWeight)

    if isinstance(model, transformer.Transformer):
        opt = torch.optim.Adam(
            list(model.parameters()),
            LR_transformer,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
    else:
        opt = torch.optim.Adam(list(model.parameters()), lr=LR_Naru)
        decay_rate = 0.96
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=opt, gamma=decay_rate
        )
    bs = args.bs
    log_every = 200

    train_data = common.TableDataset(table)
    split_idx = int(len(train_data.tuples) * 0.2)

    transfer_data = copy.deepcopy(train_data)
    tuples1 = transfer_data.tuples[:split_idx]
    rndindices = torch.randperm(len(tuples1))[:20000]
    transfer_data.tuples = tuples1[rndindices]

    train_data.tuples = train_data.tuples[split_idx:]

    print(train_data.tuples.shape, transfer_data.tuples.shape)
    train_losses = []
    train_start = time.time()
    for epoch in range(args.epochs):
        mean_epoch_train_loss = RunUpdateEpoch(
            "train",
            model,
            pmodel,
            opt,
            scheduler=lr_scheduler,
            train_data=train_data,
            transfer_data=transfer_data,
            val_data=train_data,
            omega=0.1,
            batch_size=bs,
            epoch_num=epoch,
            log_every=log_every,
            table_bits=table_bits,
        )

        if epoch % 1 == 0:
            print(
                "epoch {} train loss {:.4f} nats / {:.4f} bits".format(
                    epoch, mean_epoch_train_loss, mean_epoch_train_loss / np.log(2)
                )
            )
            since_start = time.time() - train_start
            print("time since start: {:.1f} secs".format(since_start))

        train_losses.append(mean_epoch_train_loss)

    print("Training done; evaluating likelihood on full data:")
    all_losses = RunEpoch(
        "test",
        model,
        train_data=train_data,
        val_data=train_data,
        opt=None,
        scheduler=lr_scheduler,
        batch_size=1024,
        log_every=500,
        table_bits=table_bits,
        return_losses=True,
    )
    model_nats = np.mean(all_losses)
    model_bits = model_nats / np.log(2)
    model.model_bits = model_bits

    if fixed_ordering is None:
        if seed is not None:
            PATH = "models/01{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}.pt".format(
                args.dataset,
                mb,
                model.model_bits,
                table_bits,
                model.name(),
                args.epochs,
                seed,
            )
        else:
            PATH = "models/01{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-{}.pt".format(
                args.dataset,
                mb,
                model.model_bits,
                table_bits,
                model.name(),
                args.epochs,
                seed,
                time.time(),
            )
    else:
        annot = ""
        if args.inv_order:
            annot = "-invOrder"

        PATH = "models/01{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-order{}{}.pt".format(
            args.dataset,
            mb,
            model.model_bits,
            table_bits[0],
            model.name(),
            args.epochs,
            seed,
            "_".join(map(str, fixed_ordering)),
            annot,
        )
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    torch.save(model.state_dict(), PATH)
    print("Saved to:")
    print(PATH)
    pre_model = PATH


def main(argv=None):
    global args
    args = create_parser().parse_args(argv)
    if not args.end2end:
        relative_model_paths = (
            "./models/origin-{}*MB-model*-data*-*epochs-seed*.pt".format(args.dataset)
        )
        absolute_model_paths = get_absolute_path(relative_model_paths)
        model_paths = glob.glob(str(absolute_model_paths))
        for model_path in model_paths:
            AdaptTask(pre_model=model_path)
            UpdateTask(pre_model=model_path)
            # UpdateTask(pre_model=model_path, freeze=True)
            FineTuneTask(pre_model=model_path)
    else:
        if "_" in args.dataset:
            dataset_name, extend = args.dataset.split("_")
        else:
            dataset_name = args.dataset

        if dataset_name in ["stats"]:
            model_dir = communicator.ModelPathCommunicator().get()
            reg_pattren = f"*{args.dataset}*.pt"
            abs_model_pattern = os.path.join(model_dir, reg_pattren)
            model_paths = glob.glob(str(abs_model_pattern))
            assert len(model_paths)==1, "More than 1 model!"
            model_path = model_paths[0]
            new_model_path = model_path
            
            pool_path=f"./data/{dataset_name}/end2end/{extend}/{extend}_pool.npy"
        else:
            model_path = communicator.ModelPathCommunicator().get()
            new_model_path = model_path

            pool_path=f"./data/{dataset_name}/end2end/{args.dataset}_pool.npy"
        
        unlearned_data=np.load(pool_path)
        if args.model_update == "adapt":
            # w/o Caching ablation: drop everything except the most recent
            # update batch instead of replaying the whole accumulated pool.
            if not _get_ablation_flag("adapt.use_caching", True):
                batch_rows = args.update_size
                unlearned_data = unlearned_data[-batch_rows:]
                print(f"Caching disabled: keeping the last {unlearned_data.shape[0]} rows of the pool")
        update_size=unlearned_data.shape[0]
        args.update_size=update_size
        print(f"update_size: {update_size}")
        # clear pool after model update
        os.remove(pool_path)

        if args.model_update == "update":
            print("UpdateTask - START")
            UpdateTask(
                pre_model=model_path, new_model_path=new_model_path, end2end=True
            )
            print("UpdateTask - END")
        elif args.model_update == "adapt":
            print("AdaptTask - START")
            # Design doc A14: the layered freeze is a separate knob from the EMA
            # teacher and must not ride along with it. It was hardcoded True
            # here, so once the `requires_grad ==` typo was fixed it started
            # freezing the bottom layers of adapt only -- UpdateTask trains the
            # full net -- which confounds every adapt-vs-update comparison and
            # is why the E4emafix tails got worse. The default preserves the
            # current behaviour; set adapt.freeze=false to train the full net,
            # which is also what the pre-fix E4ema/E4probe2 numbers measured.
            AdaptTask(
                pre_model=model_path,
                new_model_path=new_model_path,
                end2end=True,
                freeze=_get_ablation_flag("adapt.freeze", True),
            )
            print("AdaptTask - END")
        elif args.model_update == "finetune":
            print("FineTuneTask - START")
            FineTuneTask(
                pre_model=model_path, new_model_path=new_model_path, end2end=True
            )
            print("FineTuneTask - END")
        elif args.model_update == "retrain":
            print("RetrainTask - START")
            RetrainTask(
                pre_model=model_path, new_model_path=new_model_path, end2end=True
            )
            print("RetrainTask - END")
        elif args.model_update == "none":
            print("No model update-CONTINUE")
        else:
            raise ValueError("model update method not supported")



if __name__ == "__main__":
    main()
