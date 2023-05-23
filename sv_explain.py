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
from GraphSampling.SVExplainer_test import FastGCNSHAPNet, FastSHAP_GNN

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

    # filedir = f"./logs/{args.dataset}"
    # if not os.path.exists(filedir):
    #     os.makedirs(filedir)
    # if not args.exp_name:
    #     filename = f"{args.type_model}.json"
    # else:
    #     filename = f"{args.exp_name}.json"
    # path_json = os.path.join(filedir, filename)

    # try:
    #     resume_seed = 0
    #     if os.path.exists(path_json):
    #         if args.resume:
    #             with open(path_json, "r") as f:
    #                 saved = json.load(f)
    #                 resume_seed = saved["seed"] + 1
    #                 list_test_acc = saved["test_acc"]
    #                 list_valid_acc = saved["val_acc"]
    #                 list_train_loss = saved["train_loss"]
    #         else:
    #             t = os.path.getmtime(path_json)
    #             tstr = datetime.fromtimestamp(t).strftime("%Y_%m_%d_%H_%M_%S")
    #             os.rename(
    #                 path_json, os.path.join(filedir, filename + "_" + tstr + ".json")
    #             )
    #     if resume_seed >= args.N_exp:
    #         print("Training already finished!")
    #         return
    # except:
    #     pass

    print_args(args)

    if args.debug_mem_speed:
        trnr = trainer(args)
        trnr.mem_speed_bench()

    # for seed in range(resume_seed, args.N_exp):
    #     print(f"seed (which_run) = <{seed}>")

    #     args.random_seed = seed
    #     set_seed(args)
    
    ### random seed
    # seed = 30
    # args.random_seed = seed
    set_seed(args)

    torch.cuda.empty_cache()
    trnr = trainer(args)
    # if args.type_model in [
    #     "SAdaGCN",
    #     "AdaGCN",
    #     "GBGCN",
    #     "AdaGCN_CandS",
    #     "AdaGCN_SLE",
    #     "EnGCN",
    # ]:
    #     train_loss, valid_acc, test_acc = trnr.train_ensembling(seed)
    # else:
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

    # try:
    #     to_save = dict(
    #         seed=seed,
    #         test_acc=list_test_acc,
    #         val_acc=list_valid_acc,
    #         train_loss=list_train_loss,
    #         mean_test_acc=np.mean(list_test_acc),
    #         std_test_acc=np.std(list_test_acc),
    #     )
    #     with open(path_json, "w") as f:
    #         json.dump(to_save, f)
    # except:
    #     pass
    print(
        "final mean and std of test acc: ",
        f"{np.mean(list_test_acc)*100:.4f} $\\pm$ {np.std(list_test_acc)*100:.4f}",
    )

    ###### load the data for explainer
    (data, x, y, split_masks, evaluator, processed_dir) = load_data(args.dataset, args.tosparse)

    # ## Set up explainer model: GCN
    device = torch.device('cuda:0')
    explainer = FastGCNSHAPNet(args.num_feats, args.dim_hidden, num_layers=args.num_layers, dropout=args.dropout, src_pad_idx=-1, trg_pad_idx=-1)
    print(explainer)

    # explainer.to(device)
    surrogate.eval().to(device)
    print(surrogate)

    # Set up FastSHAP wrapper
    fastshap = FastSHAP_GNN(explainer, surrogate, normalization=None) #'additive' , link=nn.Softmax(dim=-1)

    # Train the FastSHAP
    fastshap.train(
        data,
        ori_pred.cpu(),
        batch_size=args.batch_size,  #128
        num_samples=64,  #10
        max_epochs=1000,
        eff_lambda=0.0,
        paired_sampling=True,
        validation_samples=128,
        validation_seed=0, 
        lr=1e-3,
        weight_decay=1e-5,
        verbose=True,
        training_seed=args.random_seed, 
        hops=1,
        args=args)

if __name__ == "__main__":
    args = BaseOptions().initialize()
    main(args)
