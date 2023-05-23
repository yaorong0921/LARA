import gc
import json
import os
import random
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch

from options.base_options import BaseOptions
from trainer import trainer
from utils import print_args
import torch_geometric.datasets
from torch_geometric.transforms import ToSparseTensor, ToUndirected
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from GraphSampling.LARA import Explainer, LARA



def load_data(dataset_name, to_sparse=True):
    if dataset_name in ["ogbn-products", "ogbn-papers100M", "ogbn-arxiv"]:
        root = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name
        )
        T = ToSparseTensor() if to_sparse else lambda x: x
        if to_sparse and dataset_name == "ogbn-arxiv":
            T = lambda x: ToSparseTensor()(ToUndirected()(x))
        dataset = PygNodePropPredDataset(name=dataset_name, root=root, transform=T)
        processed_dir = dataset.processed_dir
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name=dataset_name)
        data = dataset[0]
        split_masks = {}
        for split in ["train", "valid", "test"]:
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[split_idx[split]] = True
            data[f"{split}_mask"] = mask
            split_masks[f"{split}"] = data[f"{split}_mask"]

        x = data.x
        y = data.y = data.y.squeeze()

    elif dataset_name in ["Reddit", "Flickr", "AmazonProducts", "Yelp"]:
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name
        )
        T = ToSparseTensor() if to_sparse else lambda x: x
        dataset_class = getattr(torch_geometric.datasets, dataset_name)
        dataset = dataset_class(path, transform=T)
        processed_dir = dataset.processed_dir
        data = dataset[0]
        evaluator = None
        split_masks = {}
        split_masks["train"] = data.train_mask
        split_masks["valid"] = data.val_mask
        split_masks["test"] = data.test_mask
        x = data.x
        y = data.y
        # E = data.edge_index.shape[1]
        # N = data.train_mask.shape[0]
        # data.edge_idx = torch.arange(0, E)
        # data.node_idx = torch.arange(0, N)

    else:
        raise Exception(f"the dataset of {dataset} has not been implemented")
    return data, x, y, split_masks, evaluator, processed_dir

def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_num)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)


def main(args):
    list_test_acc = []
    list_valid_acc = []
    list_train_loss = []
    print_args(args)

    if args.debug_mem_speed:
        trnr = trainer(args)
        trnr.mem_speed_bench()

    set_seed(args)

    torch.cuda.empty_cache()
    trnr = trainer(args)

    trnr.load_trained_model()
    ori_pred, (train_loss, valid_acc, test_acc) = trnr.test_net()  ## ger original predictions
    list_test_acc.append(test_acc)
    list_valid_acc.append(valid_acc)
    list_train_loss.append(train_loss)

    ### get the trained GNN
    surrogate = deepcopy(trnr.model)
    del trnr
    torch.cuda.empty_cache()
    gc.collect()

    ## record training data
    print(
        "mean and std of test acc: {:.4f} {:.4f} ".format(
            np.mean(list_test_acc) * 100, np.std(list_test_acc) * 100
        )
    )

    print(
        "final mean and std of test acc: ",
        f"{np.mean(list_test_acc)*100:.4f} $\\pm$ {np.std(list_test_acc)*100:.4f}",
    )

    ###### load the data for explainer
    (data, x, y, split_masks, evaluator, processed_dir) = load_data(args.dataset, args.tosparse)

    # ## Set up explainer model: GCN
    device = torch.device('cuda:0')
    explainer = Explainer(args.num_feats, args.dim_hidden, num_layers=args.num_layers, dropout=args.dropout, src_pad_idx=-1, trg_pad_idx=-1)
    print(explainer)

    # explainer.to(device)
    surrogate.eval().to(device)
    print(surrogate)

    # Set up LARA
    lara = LARA(explainer, surrogate)

    # Train the LARA
    lara.train(
        data,
        ori_pred.cpu(),
        batch_size=args.batch_size,
        num_samples=64,
        max_epochs=1000,
        lr=1e-3,
        weight_decay=1e-5,
        verbose=True,
        training_seed=args.random_seed, 
        hops=1,
        args=args)


if __name__ == "__main__":
    args = BaseOptions().initialize()
    main(args)
