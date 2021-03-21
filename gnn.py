import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from logger import Logger

##########################################################################################
# Adding other libraries to use
import numpy as np
from tqdm import tqdm
from collections import defaultdict
##########################################################################################

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, x, adj_t, split_edge, optimizer, batch_size):

    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(x, adj_t)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                 num_neg_samples=perm.size(0), method='dense')

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, x, adj_t, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(x, adj_t)

    pos_train_edge = split_edge['eval_train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results

##########################################################################################
# Create functions for calculating number of hops and cycles

def count_hop(k):
    # Counts k-hop neighborhood nodes
    hop_list = []
    hop = torch.sum(torch.matrix_power(adj_t.to_dense(), k), 1).numpy().astype(int)
    max_length = int(np.ceil(np.log2(max(hop)))) # 1,2-hop: 12,21
    for x in hop:
        hop_list.append(([0 for x in range(max_length)] + \
                         [int(y) for y in list(np.binary_repr(x))])[-max_length:])
    return hop_list

def count_cycle(n):
    # Counts cycles of size n
    cycle_list = []
    cycles = (torch.diag(adj_t.to_dense().matrix_power(n)).numpy()/2).astype(int)
    max_length = int(np.ceil(np.log2(max(cycles)))) # 2,3,5,7-cycle: 12,21,41,60
    for x in cycles:
        cycle_list.append(([0 for x in range(max_length)] + \
                           [int(y) for y in list(np.binary_repr(x))])[-max_length:])
    return cycle_list

##########################################################################################

def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--embedding_or_feature_type', type=str, default='xavier')
    # embedding_or_feature_type values:
    # 'xavier','he','uniform','hops','cycles','subgraphs','hops_cycles_subgraphs'
    args = parser.parse_args(args=[]) ### Added 'args=[]' so that it can run on Google Collab
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ddi',
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    adj_t = data.adj_t.to(device)

    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    if args.use_sage:
        model = SAGE(args.hidden_channels, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(args.hidden_channels, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    
    ##########################################################################################
    
    # Commenting out emb as we will build our own embedding or node features instead
    # emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
    
    # Configure embeddings/feature types
    
    emb = None
    features = None
    
    if args.embedding_or_feature_type in ['xavier','he','uniform']: # Create learnable embeddings
        emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
    
    elif args.embedding_or_feature_type == 'one_hot':  # Create static features
        node_one_hot = torch.zeros([data.num_nodes, args.hidden_channels])
        for i in range(data.num_nodes):
            node_one_hot[i,:] = torch.Tensor(([0 for x in range(args.hidden_channels)] + \
            [int(x) for x in np.binary_repr(i+1)])[-args.hidden_channels:]).reshape(1,args.hidden_channels)
        features = node_one_hot
        
    else: # Create static features
        combined = [] # where to save features to
        if args.embedding_or_feature_type == 'hops':
            one_hop = count_hop(1)
            two_hop = count_hop(2)
            for i in range(data.num_nodes): 
                combined.append(([0 for x in range(args.hidden_channels)] + \
                                 one_hop[i] + two_hop[i])[-args.hidden_channels:])

        elif args.embedding_or_feature_type == 'cycles':
            find2_cycle = count_cycle(2)
            find3_cycle = count_cycle(3)
            find5_cycle = count_cycle(5)
            find7_cycle = count_cycle(7)
            for i in range(data.num_nodes): 
                combined.append(([0 for x in range(args.hidden_channels)] + \
                                 find2_cycle[i] + find3_cycle[i] + find5_cycle[i] + \
                                 find7_cycle[i])[-args.hidden_channels:])

        elif args.embedding_or_feature_type in ['subgraphs','multifeature','xavier_multifeature']:

            # Calculate powers of the adjacency matrix
            adj_t_dense = data.adj_t.to_dense()
            num_powers = 2
            node_degrees = torch.zeros([data.num_nodes, num_powers])
            adj_power_list = []

            for i in tqdm(range(0,num_powers)):
                adj_power = torch.matrix_power(adj_t_dense, i+1)
                adj_power_list.append(adj_power)
                adj_power = (adj_power!=0).float()
                node_degrees[:,i] = torch.sum(adj_power,axis=0)

            # Count number of neighbors at k hops
            hop_nodes = [((adj_power.detach().cpu().numpy())!=0).astype(float) for adj_power in adj_power_list]

            # Count number of three-cycles per node
            three_cycles = (torch.diag(adj_t_dense.matrix_power(3)).numpy()/2).astype(int)

            # Create subgraph count dictionary
            getattr(tqdm, '_instances', {}).clear()
            subgraphs = defaultdict(dict)
            for node_idx in tqdm(range(data.num_nodes)):
                one_hop = list(*np.nonzero(hop_nodes[0][:,node_idx]))
                two_hop = list(*np.nonzero(hop_nodes[1][:,node_idx]))
                intersection = list(set(one_hop).intersection(set(two_hop)))
                subgraphs['num_neighbors'][node_idx] = len(one_hop)
                subgraphs['num_three_cycles'][node_idx] = three_cycles[node_idx]
                subgraphs['num_three_caret_ends'][node_idx] = len(two_hop) - len(intersection)
                subgraphs['num_three_caret_tops'][node_idx] = int(len(one_hop)*(len(one_hop)-1)/2) - \
                                                              three_cycles[node_idx]

            # Create binarized subgraph count dictionary
            getattr(tqdm, '_instances', {}).clear()
            subgraphs_binarized = defaultdict(list)
            for subgraph in ['num_neighbors','num_three_cycles','num_three_caret_ends','num_three_caret_tops']:
                max_length = int(np.ceil(np.log2(max([v for v in subgraphs[subgraph].values()]))))
                for node_idx in tqdm(range(data.num_nodes)):
                    subgraphs_binarized[subgraph].append(([0 for i in range(max_length)] + \
                    [int(y) for y in list(np.binary_repr(subgraphs[subgraph][node_idx]))])[-max_length:])

            if args.embedding_or_feature_type == 'subgraphs':
                # Combine all subgraph counts
                getattr(tqdm, '_instances', {}).clear()
                for node_idx in tqdm(range(data.num_nodes)):
                    combined.append(([0 for x in range(args.hidden_channels)] + \
                                     subgraphs_binarized['num_neighbors'][node_idx] + \
                                     subgraphs_binarized['num_three_cycles'][node_idx] + \
                                     subgraphs_binarized['num_three_caret_ends'][node_idx] + \
                                     subgraphs_binarized['num_three_caret_tops'][node_idx]
                                    )[-args.hidden_channels:])

            else # args.embedding_or_feature_type in ['multifeature', 'xavier_multifeature']
                getattr(tqdm, '_instances', {}).clear()
                one_hop = count_hop(1)
                two_hop = count_hop(2)
                find2_cycle = count_cycle(2)
                find3_cycle = count_cycle(3)
                find5_cycle = count_cycle(5)
                find7_cycle = count_cycle(7)
                for node_idx in tqdm(range(data.num_nodes)):
                    combined.append((one_hop[i] + two_hop[i] + \
                                     find2_cycle[i] + find3_cycle[i] + \
                                     find5_cycle[i] + find7_cycle[i] +\
                                     subgraphs_binarized['num_neighbors'][node_idx] + \
                                     subgraphs_binarized['num_three_cycles'][node_idx] + \
                                     subgraphs_binarized['num_three_caret_ends'][node_idx] + \
                                     subgraphs_binarized['num_three_caret_tops'][node_idx]
                                    )[-args.hidden_channels:])
            features = torch.from_numpy((np.array(combined))).float()
            
                if args.embedding_or_feature_type == 'xavier_multifeature':
                    features = torch.column_stack(
                        (torch.nn.init.xavier_uniform_(torch.zeros_like(features))[:, :24], \
                         features[:, 24:])
                    )
            
        else:
            raise ValueError('The embedding_or_feature_type argument value is not recognized.')
        
    ##########################################################################################
        
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ddi')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@20': Logger(args.runs, args),
        'Hits@30': Logger(args.runs, args),
    }

    for run in range(args.runs):
        ##########################################################################################
        # Specify embedding or feature type
        
        emb_parameters = []
        
        if args.embedding_or_feature_type in ['xavier','he','uniform']:
            
            if args.embedding_or_feature_type == 'xavier':
                torch.nn.init.xavier_uniform_(emb.weight)
            elif args.embedding_or_feature_type == 'he':
                torch.nn.init.kaiming_uniform_(emb.weight)
            else: # uniform
                torch.nn.init.uniform_(emb.weight)
                
            emb_parameters = list(emb.parameters())
            features = emb.weight
            
        else: # use calculated node features, as stored in the variable 'features'
            pass
            
        ##########################################################################################
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + emb_parameters + # Changed list(emb.parameters()) to emb_parameters
            list(predictor.parameters()), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, features, adj_t, split_edge, # Changed emb.weight to features
                         optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, features, adj_t, split_edge, # Changed emb.weight to features
                               evaluator, args.batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()