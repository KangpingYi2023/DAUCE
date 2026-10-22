"""Model training."""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common
import made
import transformer
import sys

sys.path.append("./")
from update_utils import dataset_util
from update_utils.model_util import save_torch_model
from update_utils.torch_util import get_torch_device


def create_parser():
    parser = argparse.ArgumentParser()

    # Training.
    parser.add_argument("--model", type=str, choices=["naru", "transformer"], default="naru", help="Training model")
    parser.add_argument("--training_type", type=str, choices=["train", "retrain"], default="train", help="Training type")
    parser.add_argument("--dataset", type=str, default="dmv-tiny", help="Dataset.")
    parser.add_argument("--num-gpus", type=int, default=0, help="#gpus.")
    parser.add_argument("--bs", type=int, default=1024, help="Batch size.")
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


args = create_parser().parse_args()

if args.model == "transformer":
    DEVICE = get_torch_device(extra=True)
else:
    DEVICE = get_torch_device(extra=False)


def Entropy(name, data, bases=None):
    import scipy.stats

    s = "Entropy of {}:".format(name)
    ret = []
    for base in bases:
        assert base == 2 or base == "e" or base is None
        e = scipy.stats.entropy(data, base=base if base != "e" else None)
        ret.append(e)
        unit = "nats" if (base == "e" or base is None) else "bits"
        s += " {:.4f} {}".format(e, unit)
    print(s)
    return ret


def RunEpoch(
    split,
    model,
    opt,
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
                elif args.warmups:
                    t = args.warmups
                    d_model = model.embed_size
                    global_steps = len(loader) * epoch_num + step + 1
                    lr = (d_model**-0.5) * min(
                        (global_steps**-0.5), global_steps * (t**-1.5)
                    )
                else:
                    lr = 1e-2

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
        else:
            if num_orders_to_forward == 1:
                loss = model.nll(xbhat, xb).mean()
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

        losses.append(loss.item())

        if step % log_every == 0:
            if split == "train":
                print(
                    "Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr".format(
                        epoch_num,
                        step,
                        split,
                        loss.item() / np.log(2) - table_bits,
                        loss.item() / np.log(2),
                        table_bits,
                        lr,
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
    print("MakeMade - START")
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

    print("MakeMade - END")
    return model


def MakeTransformer(cols_to_train, fixed_ordering, seed=None):
    HEAD=4
    return transformer.Transformer(
        num_blocks=args.blocks,
        d_model=args.dmodel,
        d_ff=args.dff,
        # num_heads=args.heads,
        num_heads=HEAD,
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


def TrainTask(dataset, table, seed=0):
    torch.manual_seed(0)
    np.random.seed(0)

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

    mb = ReportModel(model)

    if not isinstance(model, transformer.Transformer):
        print("Applying InitWeight()")
        model.apply(InitWeight)

    if isinstance(model, transformer.Transformer):
        opt = torch.optim.Adam(
            list(model.parameters()),
            2e-4,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
    else:
        opt = torch.optim.Adam(list(model.parameters()), 2e-4)

    bs = args.bs
    log_every = 200

    train_data = common.TableDataset(table)

    patience = 10
    best_loss = np.inf
    train_losses = []
    train_start = time.time()
    for epoch in range(args.epochs):
        mean_epoch_train_loss = RunEpoch(
            "train",
            model,
            opt,
            train_data=train_data,
            val_data=train_data,
            batch_size=bs,
            epoch_num=epoch,
            log_every=log_every,
            table_bits=table_bits,
        )

        if mean_epoch_train_loss >= best_loss:
            patience -= 1
            if patience == 0:
                print("Early Stop!")
                break
        else:
            patience = 10
            best_loss = mean_epoch_train_loss

        if epoch % 10 == 0:
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
        batch_size=1024,
        log_every=500,
        table_bits=table_bits,
        return_losses=True,
    )
    model_nats = np.mean(all_losses)
    model_bits = model_nats / np.log(2)
    model.model_bits = model_bits

    if args.training_type=="train":
        PATH = "./models/origin-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}.pt".format(
            dataset,
            mb,
            model.model_bits,
            table_bits,
            args.epochs,
            seed,
        )
    elif args.training_type=="retrain":
        PATH = "./models/retrain-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}.pt".format(
            dataset,
            mb,
            model.model_bits,
            table_bits,
            args.epochs,
            seed,
        )
    save_torch_model(model, PATH)


def main():
    # Load dataset
    if args.dataset == "stats":
        table_list=['badges', 'votes', 'postHistory', 'posts', 'users', 'comments', 'postLinks', 'tags']
        # table_list=['posts', 'users', 'comments', 'postLinks', 'tags']
        for table_name in table_list:
            dataset_name="stats_" + table_name
            table = dataset_util.DatasetLoader.load_dataset(dataset=dataset_name)
            TrainTask(dataset=dataset_name, table=table)
    elif args.dataset in ["power", "census", "bjaq", "census_gaussian", "bjaq_gaussian", "forest_gaussian"]:
        if args.training_type=="train":
            table = dataset_util.DatasetLoader.load_dataset(dataset=args.dataset)
        elif args.training_type=="retrain":
            table, _ = dataset_util.DatasetLoader.load_permuted_dataset(dataset=args.dataset)
        TrainTask(dataset=args.dataset, table=table)


if __name__ == "__main__":
    main()
