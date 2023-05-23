import json
import time
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import GraphSAINTRandomWalkSampler as RWSampler
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from torch_geometric.utils import degree
import sys

from utils import GB, MB, compute_tensor_bytes, get_memory_usage

from pyrsistent import s
from sklearn import neighbors
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
import torch_geometric
from tqdm import tqdm
import pickle

import os

class Explainer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.0, src_pad_idx=-1, trg_pad_idx=-1, device='cuda:0', args=None):
        super(Explainer, self).__init__()
        self.input_dim = input_dim
        print('Explainer input_dim:', self.input_dim)
        self.hidden_dim = hidden_dim
        print('Explainer hidden_dim:', self.hidden_dim)
        self.num_layers = num_layers
        print('Explainer num_layers:', self.num_layers)

        self.args = args
        self.dropout = dropout
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(self.input_dim, self.hidden_dim))
        for layer in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
        print('len(self.convs):', len(self.convs))

        self.W = torch.nn.Linear(
            len(self.convs) * self.hidden_dim, self.hidden_dim)
        self.p = torch.nn.Linear(
            len(self.convs) * self.hidden_dim, self.hidden_dim)

    def forward(self, x, edge_index, src_idx, tgt_idx):
        x_all = []
        batch_size = src_idx.shape[0]


        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_all.append(x)
        x = torch.cat(x_all, dim=1)
        x_W = self.W(x)
        x_p = self.p(x)

        ## predict explanations using W and p
        sv_t = x_W[tgt_idx,:].view(batch_size, self.hidden_dim).unsqueeze(-1)
        sv_s = x_p[src_idx,:].view(batch_size, self.hidden_dim).unsqueeze(-2)
        sv = torch.bmm(sv_s, sv_t)

        return sv.view(batch_size, -1).squeeze(-1)

def calculate_coalition_gnn_ps(x, edge_index, ori_pred, perturb_node_index, imputer, null, num_sample, device, z_=None, hops=1, train=True, neighbor_index=None, subgraph_neighbor_index=None):
    if not train:
        ori_pred_cls = ori_pred.argmax(dim=1)[perturb_node_index]
        # Investigate k-hop subgraph of the node of interest (v)
        neighbor_index, _, _, edge_mask =\
            torch_geometric.utils.k_hop_subgraph(node_idx=perturb_node_index,
                                                num_hops=hops,
                                                edge_index=edge_index)
    else:
        ### get from the input
        neighbor_index = neighbor_index
        ori_pred_cls = ori_pred
    #### get outputs
    src_list = []
    tgt_list = []
    for idx in neighbor_index.tolist():
        src_list.append(idx)
        tgt_list.append(perturb_node_index)

    M = len(neighbor_index)
    if z_ == None:
        ### permutation
        queue = torch.arange(M)
        queue = queue[torch.randperm(M)]
    else:
        queue = torch.flip(z_, dims=(0,))
        #queue = z_
    S = torch.zeros_like(queue).type(torch.long)
    if train:
        ### go through the permutation index
        #N = num_sample if M > num_sample else M
        data_list = []
        output_list = []
        ###### sub sample the graph of this node
        edges, _ = torch_geometric.utils.subgraph(subgraph_neighbor_index, edge_index)
        feats = torch.zeros((subgraph_neighbor_index.size(0), x.size(1)))
        old2new = torch.zeros(x.size(0), dtype=torch.long)
        old2new[subgraph_neighbor_index] = torch.arange(subgraph_neighbor_index.size(0))
        for i in subgraph_neighbor_index:
            feats[old2new[i]] = deepcopy(x[i,:])   
        graph = old2new[edges]
        ################   
        for index in range(M):
            edge_index_new = deepcopy(graph)
            S[queue[index]] = 1
            data_cp = torch_geometric.data.Data(x=deepcopy(feats), edge_index=None)
            for k, idx in enumerate(neighbor_index):
                if S[k] == 0:
                    index_to_remove = (edge_index_new == old2new[idx]).nonzero(as_tuple=True)[1]
                    index_to_reserve = [i for i in range(edge_index_new.shape[1]) if i not in index_to_remove]
                    edge_index_new = edge_index_new[:, index_to_reserve]
            # if edge_index_new.shape[1] == 0:
            #     edge_index_new = torch.tensor([[old2new[perturb_node_index]], [old2new[perturb_node_index]]])
            data_cp.edge_index = edge_index_new
            data_list.append(data_cp)
        #### calculate the sv values for neighbors
        iter_batch_size = num_sample
        loader = torch_geometric.loader.DataLoader(data_list, batch_size=iter_batch_size, shuffle=False)
        imputer = imputer.to(device)
        imputer.eval()
        with torch.no_grad():
            for i, data_sample in enumerate(loader):
                bs = data_sample.x.shape[0] // data_cp.x.shape[0]
                cur_value = imputer(data_sample.x.to(device), data_sample.edge_index.to(device)).view(bs, data_cp.x.shape[0], -1)[:, old2new[perturb_node_index], ori_pred_cls]
                # torch.set_printoptions(profile="full")
                output_list.append(cur_value.clone().detach())
        cur_value_m = torch.cat(output_list, dim=0)

        baseline_value = null.to(device)

        v_s = torch.zeros_like(S).type(torch.float)
        for index in range(M):  
            ## queueu[index] is the index of the vector of the neighbor node list
            cur_value = deepcopy(cur_value_m[index]) - baseline_value
            ### put this value into output vector with the correpsonding index
            v_s[queue[index]] = cur_value
            baseline_value = deepcopy(cur_value_m[index])
        return ori_pred, v_s, src_list, tgt_list, S, queue
    else: 
        return ori_pred[perturb_node_index, ori_pred_cls], torch.ones(S.shape).type(torch.long), src_list, tgt_list, torch.ones(S.shape).type(torch.long), queue


class LARA:

    def __init__(self,
                 explainer,
                 imputer):
        # Set up explainer, imputer
        self.explainer = explainer
        self.imputer = imputer
        self.null = None

    def train(self,
              data,
              ori_pred,
              batch_size,
              num_samples,
              max_epochs,
              lr=2e-4,
              weight_decay=1e-4,
              training_seed=None,
              verbose=False, 
              hops=1,
              args=None):
        # Set up explainer model.
        explainer = self.explainer
        self.imputer = self.imputer.eval()
        explainer.train()
        device = next(self.imputer.parameters()).device
        explainer.to(device)

        # Null coalition.
        with torch.no_grad():
            zero_edge = torch.empty((2,0), dtype=torch.long)

            null = self.imputer(data.x.to(device), zero_edge.to(device))
            if len(null.shape) == 1:
                null = null.reshape(1, 1)
        self.null = null.detach().cpu()

        all_train_node = []
        all_train_node.extend(torch.nonzero(data.train_mask).squeeze().tolist())
        get_indices = torch.randperm(len(all_train_node))[:len(all_train_node)//args.num_splits]
        all_train_node = torch.tensor(all_train_node)[get_indices]
        all_train_node = all_train_node.tolist()
        all_test_node = torch.nonzero(data.test_mask).squeeze().tolist()
        test_node = all_test_node[:1000]
        val_node = all_test_node[1000:3000]

        saved_dir = './lara/%s_%s_%d'%(args.dataset, str(training_seed), args.num_splits)
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
        buffer_path = os.path.join(saved_dir, '%s_%s_%d.pkl'%(args.dataset, str(training_seed), args.num_splits))
        if os.path.exists(buffer_path):
            saved_buffer_path = buffer_path
        else:
            saved_buffer_path = None
        print("# Training nodes", len(all_train_node))
        buffer_loader = GraphSampler(data, self.imputer, batch_size, ori_pred, num_samples, './lara', device, hops, all_train_node, saved_buffer_path)

        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(saved_dir)
        loss_fn = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(explainer.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_list = []
        best_fidelity = 0.0
        best_model = None

####### get all samples for valiatiion
        data_sample_val_src = []
        data_sample_val_tgt = []
        data_sample_val_null = []

        ori_pred_cls = ori_pred.argmax(dim=1)

        val_node_new = []
        for j, perturb_node_index in enumerate(val_node):
            ori_pred_cls = ori_pred.argmax(dim=1)
            null = self.null[perturb_node_index][ori_pred_cls[perturb_node_index]]
            grand, values, perturb_input_src, perturb_input_tgt, S, z_ = \
                calculate_coalition_gnn_ps(data.x, data.edge_index, ori_pred, perturb_node_index, self.imputer, null, 5, device, z_=None, hops=hops, train=False)
            if len(perturb_input_src) > 500:
                continue
            elif len(perturb_input_src) < 2:
                continue
            val_node_new.append(perturb_node_index)
            data_sample_val_src.extend(perturb_input_src)
            data_sample_val_tgt.extend(perturb_input_tgt)
            data_sample_val_null.extend(null.repeat(len(perturb_input_src)))
        val_node = val_node_new
        print("# Val nodes", len(val_node))

        val_set = LARAData(data_sample_val_src, data_sample_val_tgt, data_sample_val_null)
        val_loader = DataLoader(val_set, batch_size=256, shuffle=False, pin_memory=True, \
                        drop_last=False)

####### For each node, perturb its direct neighbors
        for epoch in range(max_epochs):
            mean_sv_loss = 0
            M = 0
            minus_v = 0
            #########################

            buffer = buffer_loader.buffer
            # print(buffer[list(buffer.keys())[2]])
            #### save buffer
            if epoch%2 == 0:
                with open(buffer_path, 'wb') as f:
                    pickle.dump(buffer, f)

            explainer.train()
            for train_data, x_src, x_tgt, value, _ in tqdm(buffer_loader): 
                # Move to device.
                value = value.to(device)
                explainer.zero_grad()
                pred = explainer(train_data.x.to(device), train_data.edge_index.to(device), x_src, x_tgt) 
                minus_v += (value<0).sum() 
 
                loss = loss_fn(pred, value) 
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                mean_sv_loss += loss.item()
                M += 1
            explainer.eval()
            fidelity_list, _, _ = self.validate(val_loader, explainer, data, ori_pred, val_node)
            fidelity = fidelity_list[0]
            writer.add_scalar("Loss/train", mean_sv_loss/M, epoch)
            writer.add_scalar("Fidelity/validation", fidelity, epoch)

            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_model = deepcopy(explainer)
                state = {
                    'epoch': epoch,
                    'state_dict': best_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }
                savepath = os.path.join(saved_dir, '%s_%s_%dsplits_best.pth'%(args.dataset, str(training_seed), args.num_splits))
                torch.save(state, savepath)
            #######################
            if verbose:
                print('----- Epoch = {} -----'.format(epoch + 1))
                print('Train loss = {:.8f}'.format(mean_sv_loss/M))
                print('Best Fidelity = {:.8f}'.format(best_fidelity))
                print('') 
            writer.flush()
        # Copy best model.
        for param, best_param in zip(explainer.parameters(),
                                     best_model.parameters()):
            param.data = best_param.data
        explainer.eval()

    def validate(self, val_loader, explainer, data, ori_pred, test_node, sparsity=[0.6]):
        node_idx_list = []
        pred_sv_list = []
        src_idx_list = []
        with torch.no_grad():
            # Setup.
            device = next(explainer.parameters()).device
            N = 0
            minus_v = 0
            total_time = 0

            for x_src, x_tgt, _ in val_loader:
                # Evaluate explainer.
                start = time.time()
                pred = explainer(data.x.to(device), data.edge_index.to(device), x_src, x_tgt) #embedding_ori
                end = time.time()
                total_time += (end - start)
                minus_v += (pred<0).sum()
                
                node_idx_list.extend(x_tgt)
                src_idx_list.extend(x_src)
                pred_sv_list.extend(pred.detach())
            indices_src = torch.stack(src_idx_list, dim=0)
            indices_tgt = torch.stack(node_idx_list, dim=0)
            sv = torch.stack(pred_sv_list, dim=0)
            fidelity, fidelity_minus= calc_fidelity_pairwise(data, self.imputer, test_node, indices_tgt, indices_src, sv, ori_pred[test_node], sparsity=sparsity)
        print('Test Fidelity+: ', np.asarray(fidelity).mean())
        print('Test Fidelity-: ', np.asarray(fidelity_minus).mean())
        return fidelity, fidelity_minus, total_time

    def test(self,
              data,
              ori_pred,
              batch_size,
              training_seed=None,
              verbose=False, 
              hops=1,
              savepath=None,
              args=None):

        # Set up explainer model.
        explainer = self.explainer
        self.imputer = self.imputer.eval()

        explainer.train()
        device = next(self.imputer.parameters()).device
        explainer.to(device)

        # Null coalition.
        with torch.no_grad():
            zero_edge = torch.arange(data.x.shape[0]).unsqueeze(-1)
            zero_edge = zero_edge.repeat(1,2).t()

            null = self.imputer(data.x.to(device), zero_edge.to(device))
            if len(null.shape) == 1:
                null = null.reshape(1, 1)
        self.null = null.detach().cpu()
        test_node = torch.nonzero(data.test_mask).squeeze().tolist()
        test_node = test_node[:1000]

####### get all samples for valiatiion
        data_sample_val_src = []
        data_sample_val_tgt = []
        data_sample_val_null = []

        ori_pred_cls = ori_pred.argmax(dim=1)

        total_time = 0
        # test_node_new = []
        start = time.time()
        for j, perturb_node_index in enumerate(test_node):
            ori_pred_cls = ori_pred.argmax(dim=1)
            null = self.null[perturb_node_index][ori_pred_cls[perturb_node_index]]
            grand, values, perturb_input_src, perturb_input_tgt, S, z_ = \
                calculate_coalition_gnn_ps(data.x, data.edge_index, ori_pred, perturb_node_index, self.imputer, null, 5, device, z_=None, hops=hops, train=False)
            data_sample_val_src.extend(perturb_input_src)
            data_sample_val_tgt.extend(perturb_input_tgt)
            data_sample_val_null.extend(null.repeat(len(perturb_input_src)))            
        # test_node = deepcopy(test_node_new)
        end = time.time()
        total_time += (end - start)
        val_set = LARAData(data_sample_val_src, data_sample_val_tgt, data_sample_val_null)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True, \
                        drop_last=False)

        ### load train RTGraphX
        explainer.load_state_dict(torch.load(savepath)['state_dict'])
        explainer.eval()
        explainer.to(device)

        fidelity, fidelity_minus, run_time = self.validate(val_loader, explainer, data, ori_pred, test_node, sparsity=[0.75, 0.70, 0.65, 0.60, 0.55])
        throughput = len(test_node)/(total_time+run_time)
        print('Throughput: ', throughput)


def calc_fidelity_pairwise(data, model, test_idx, node_idx_list, src_idx_list, output_sv, prediction_origin, sparsity=[0.6]):
    fidelity_list = []
    fidelity_minus_list = []
    ### calculate the fidelity
    for p in sparsity:
        prediction = torch.zeros(prediction_origin.shape)
        prediction_minus = torch.zeros(prediction_origin.shape)
        count = 0
        count_minus = 0
        for j, node_idx in (enumerate(test_idx)):
            src_idx = (node_idx_list==node_idx).nonzero(as_tuple=True)[0] #indices_src[]
            if (src_idx.shape[0] < 2):
                prediction[j] = prediction_origin[j]
                prediction_minus[j] = prediction_origin[j]
                continue
            output_sv_src = output_sv[src_idx]
            indices = torch.argsort(output_sv_src, dim=-1, descending=True)
            indices_minus = torch.argsort(output_sv_src, dim=-1, descending=False)
            # torch.set_printoptions(profile="full")
            index_list = src_idx_list[src_idx]
            index_list_p = [index_list[i] for i in indices.squeeze().tolist()]
            index_list_minus = [index_list[i] for i in indices_minus.squeeze().tolist()]

            ###### sub sample the graph of this node
            neighbor_index, _, _, edge_mask =\
                torch_geometric.utils.k_hop_subgraph(node_idx=node_idx,
                                                        num_hops=2,
                                                        edge_index=data.edge_index)
            edges, _ = torch_geometric.utils.subgraph(neighbor_index, data.edge_index)
            x = torch.zeros((neighbor_index.size(0), data.x.size(1)))
            old2new = torch.zeros(data.x.size(0), dtype=torch.long)
            old2new[neighbor_index] = torch.arange(neighbor_index.size(0))
            for i in neighbor_index:
                x[old2new[i]] = deepcopy(data.x[i,:])   
            edges = old2new[edges]
            edge_minus = deepcopy(edges) 
            ################

            for t in range(int(len(index_list)*(1-p))):
                idx = index_list_p[t]
                # if idx == node_idx:
                #     continue
                index_to_remove = (edges == old2new[idx]).nonzero(as_tuple=True)[1]
                index_to_reserve = [i for i in range(edges.shape[1]) if i not in index_to_remove]
                edges = edges[:, index_to_reserve]
            if (edges.shape[1] < 1):
                prediction[j] = prediction_origin[j]
            else:
                count += 1
                with torch.no_grad():
                    prediction[j] = model(x.cuda(), edges.cuda())[old2new[node_idx]]

            for t in range(int(len(index_list)*p)):
                idx = index_list_minus[t]
                # if idx == node_idx:
                #     continue
                index_to_remove = (edge_minus == old2new[idx]).nonzero(as_tuple=True)[1]
                index_to_reserve = [i for i in range(edge_minus.shape[1]) if i not in index_to_remove]
                edge_minus = edge_minus[:, index_to_reserve]
            if (edge_minus.shape[1] < 1):
                prediction_minus[j] = prediction_origin[j]
            else:
                count_minus += 1
                with torch.no_grad():
                    prediction_minus[j] = model(x.cuda(), edge_minus.cuda())[old2new[node_idx]]

        #### calculate the fidelity
        print('Evaluating sparsity: %.2f'%p)
        true_pred_value, true_pred = prediction_origin.softmax(dim=1).max(dim=1)
        perturb_value = prediction.softmax(dim=1)[torch.arange(len(test_idx)),true_pred]
        fidelity_v = (torch.logit(true_pred_value.cpu()) - torch.logit(perturb_value)).sum()/count
        print('Fidelity+: ', fidelity_v)
        fidelity_list.append(fidelity_v)

        perturb_value = prediction_minus.softmax(dim=1)[torch.arange(len(test_idx)),true_pred]
        fidelity_minus = (torch.logit(true_pred_value.cpu()) - torch.logit(perturb_value)).sum()/count_minus
        print('Fidelity-: ', fidelity_minus)
        fidelity_minus_list.append(fidelity_minus)
    return fidelity_list, fidelity_minus_list


def collate_fn(data):
    input_src_list = []
    input_tgt_list = []
    output_list = []
    null_list = []
    S_list = []
    PAD_IDX = -1
    for i, (input_src, input_tgt, output, S, null) in enumerate(data): #value
        if len(input_src) == 0:
            input_src.append(-1)
            input_tgt.append(-1)
        input_src_list.append(torch.tensor(input_src))
        input_tgt_list.append(torch.tensor(input_tgt))
        output_list.append(output)
        S_list.append(S)
        null_list.append(null)
    
    source_batch = pad_sequence(input_src_list, 
                        padding_value=PAD_IDX,
                        batch_first=True)
    target_batch = pad_sequence(input_tgt_list, 
                        padding_value=PAD_IDX,
                        batch_first=True)

    output_t = torch.stack(output_list)
    S_t = pad_sequence(S_list, 
                        padding_value=PAD_IDX,
                        batch_first=True)
    null_t = torch.stack(null_list)
    batch = [source_batch, target_batch, output_t, S_t, null_t] #value_batch
    return batch


class LARAData(torch.utils.data.Dataset):
    def __init__(self, input_src_list, input_tgt_list, null_list):
        super(LARAData, self).__init__()
        self.input_src_list = input_src_list
        self.input_tgt_list = input_tgt_list
        self.null_list = null_list

    def __len__(self):
        return len(self.null_list)

    def __getitem__(self, idx):
        input_src = self.input_src_list[idx]
        input_tgt = self.input_tgt_list[idx]
        null = self.null_list[idx]
        return input_src, input_tgt, null

import os.path as osp
import os
from torch_sparse import SparseTensor
class GraphSampler(torch.utils.data.DataLoader):
    def __init__(self, data, model, batch_size: int, original_label: list = None, num_steps: int = 1,
                 save_dir: str = None, device: str = 'cuda:0', num_hops: int = 1, train_node_idx: list = None,  buffer_path: str = None,
                **kwargs):

        assert data.edge_index is not None
        assert not data.edge_index.is_cuda

        self.device = device

        self.model = model.eval().to(self.device)
        self.num_steps = num_steps
        self.__batch_size__ = batch_size
        self.save_dir = save_dir
        self.num_hops = num_hops
        self.train_node_idx = train_node_idx

        self.N = data.num_nodes
        self.E = data.edge_index.shape[1] 

        self.original_label = original_label.argmax(dim=-1)

        self.adj = SparseTensor(
            row=data.edge_index[0], col=data.edge_index[1],
            value=torch.arange(self.E, device=data.edge_index.device),
            sparse_sizes=(self.N, self.N))

        self.data = data

        #### init buffer
        self.__init_buffer__(buffer_path)

        super().__init__(self, batch_size=1, collate_fn=self.__collate__, num_workers=0, **kwargs) #


    def __null_calculation__(self):
        with torch.no_grad():
            #embedding_ori = self.imputer.embedding_tensor.squeeze(0).detach()
            zero_edge = torch.arange(self.data.x.shape[0]).unsqueeze(-1)
            zero_edge = zero_edge.repeat(1,2).t()
            # zeros = torch.zeros((data.x.shape), dtype=torch.float32,
            #                     device=device)
            null = self.model(self.data.x.to(self.device), zero_edge.to(self.device))
            self.null = null.detach().cpu()
            
    def __init_buffer__(self, buffer_path):
        if buffer_path == None:
            self.buffer = {}
            self.__null_calculation__()
            self.buffer['null'] = self.null
            for i, key in enumerate(self.train_node_idx): #range(self.N)
                if (i%10000 == 0):
                    print(i)
                item_dict = {}
                # ### get neighbors
                neighbor_index, _, _, edge_mask =\
                    torch_geometric.utils.k_hop_subgraph(node_idx=key,
                                                        num_hops=self.num_hops,
                                                        edge_index=self.data.edge_index)
                item_dict.update({'neighbor': neighbor_index})
                #item_dict.update({'pointer': 0})
                #### permutation init
                M = len(neighbor_index)
                ### permutation
                queue = torch.arange(M)
                queue = queue[torch.randperm(M)]
                S = torch.zeros_like(queue).type(torch.float)
                N = 0
                item_dict.update({'permu': queue})
                item_dict.update({'s': S})
                item_dict.update({'N': N})
                item_dict.update({'antithetical': True})
                #item_dict.update({'previous_s': self.null[key][self.original_label[key]].type(torch.float).item()})
                self.buffer[key] = item_dict
        else:
            with open(buffer_path, 'rb') as file:
                self.buffer = pickle.load(file)
                self.null = self.buffer['null']

    def __init_permu__(self, n, buffer, antithetical=False):
        neighbor_index = buffer[n]['neighbor']
        #### permutation init
        M = len(neighbor_index)
        ### permutation
        if antithetical:
            if buffer[n]['antithetical']:
                queue = deepcopy(buffer[n]['permu'])
                queue = torch.flip(queue, dims=(0,))
                buffer[n]['antithetical'] = False
            else:
                queue = torch.arange(M)
                queue = queue[torch.randperm(M)]
                buffer[n]['antithetical'] = True
        else:
            queue = torch.arange(M)
            queue = queue[torch.randperm(M)]            
        buffer[n]['permu'] = queue
        #buffer[n]['pointer'] = 0
        #buffer[n]['previous_s'] = self.null[n][self.original_label[n]].type(torch.float).item()
        return buffer

    def __len__(self):
        return self.num_steps
    
    def __getitem__(self, idx):
        data, src_list, tgt_list, s_list, tgt_old_list = self.__sample_nodes__() #

        return data, src_list, tgt_list, s_list, tgt_old_list

    def __get_subgraph__(self, idx, node_idx_final, data, src_list, tgt_list, s_list): #edge_removed
        init_node = []
        neighbors = self.buffer[idx]['neighbor']
        neighbor_index, _, _, edge_mask =\
            torch_geometric.utils.k_hop_subgraph(node_idx=idx,
                                                    num_hops=2,
                                                    edge_index=self.data.edge_index)
        #### if too many neighbors, will discard this node
        if len(neighbor_index) > 500 or len(neighbor_index) <2:
            node_idx_final = node_idx_final
            data = data
            src_list = src_list
            tgt_list = tgt_list
            s_list = s_list
            return False, data, node_idx_final, src_list, tgt_list, s_list
    
        edges, _ = torch_geometric.utils.subgraph(neighbor_index, self.data.edge_index)
        ### generate a graph for training
        init_node.extend(neighbor_index)
        # ### get new indices for the subgraph
        init_node = torch.stack(init_node).unique()
        num_reserve_node = data.num_nodes
        data.num_nodes += init_node.size(0)
        x_old = deepcopy(data.x)
        edge_old = deepcopy(data.edge_index)
        old2new = torch.zeros(self.data.x.size(0), dtype=torch.long)
        old2new[init_node] = torch.arange(num_reserve_node, (num_reserve_node + init_node.size(0)))
        data.x = torch.zeros((num_reserve_node + init_node.size(0), self.data.x.size(1)))
        data.x[:num_reserve_node,:] = x_old
        for i in init_node:
            data.x[old2new[i]] = deepcopy(self.data.x[i,:])
        edges = old2new[edges]
        data.edge_index = torch.concat((edge_old, edges), dim=1)
        node_idx_final.append(old2new[idx])

        ### get scores for this node
        for n_idx in neighbors:
            src_list.append(old2new[n_idx])
            tgt_list.append(old2new[idx])

        ###### update buffer
        z_ = self.buffer[idx]['permu']
        _, delta, _, _, _, _ =  calculate_coalition_gnn_ps(self.data.x, self.data.edge_index, self.original_label[idx], idx, self.model, self.null[idx][self.original_label[idx]], 64, self.device, z_=z_, hops=1, train=True, neighbor_index=neighbors, subgraph_neighbor_index=neighbor_index)
        N = self.buffer[idx]['N'] + 1
        s_new = self.buffer[idx]['s']*((N-1)/N) + delta*(1/N)
        self.buffer[idx]['s'] = s_new
        self.buffer[idx]['N'] += 1
        if self.buffer[idx]['antithetical']:
            self.buffer = self.__init_permu__(idx, self.buffer, antithetical=True)        
        s_list.extend(s_new)
        return True, data, node_idx_final, src_list, tgt_list, s_list

    def __sample_nodes__(self):
        self.model.to(self.device)
        # ### sample the subgraph
        node_sample = []
        node_idx_final = []
        src_list_res = []
        src_list_pre = []
        tgt_list_pre = []
        s_pre = []
        data = self.data.__class__()
        data.num_nodes = 0
        data.x = torch.empty((0, self.data.x.shape[1]), dtype=torch.float)
        data.edge_index = torch.empty((2,0), dtype=torch.long)

        indices = torch.randperm(len(self.train_node_idx))[:self.__batch_size__]
        for i in indices:
            idx = self.train_node_idx[i]
            #idx = torch.randint(0, len(self.train_node_idx), (1,), dtype=torch.long)
            success_flag, data, node_idx_final, src_list_pre, tgt_list_pre, s_pre = \
                     self.__get_subgraph__(idx, node_idx_final, data, src_list_pre, tgt_list_pre, s_pre)
            if success_flag:
                node_sample.append(torch.tensor(self.train_node_idx[i]))
        return data, src_list_pre, tgt_list_pre, s_pre, node_sample

    def __collate__(self, data_list):
        assert len(data_list) == 1
        data, src_list, tgt_list, s_list, tgt_old_batch= data_list[0]

        source_batch = torch.stack(src_list)
        target_batch = torch.stack(tgt_list)
        S_t = torch.stack(s_list)

        return data, source_batch, target_batch, S_t, tgt_old_batch

