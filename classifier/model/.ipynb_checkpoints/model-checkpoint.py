import sys
import torch as th
import torch.nn.functional as F
import torch_geometric.nn as tgnn
import torch_geometric.utils as tgu
import math
from model.modules import dense_mincut_pool_adapted

# Dense/Diffpool Model
class GNN(th.nn.Module):
    def __init__(self, in_c, hidden_c, out_c):
        self.in_c = in_c
        self.hidden_c = hidden_c
        self.out_c = out_c
        super().__init__()

        self.conv = th.nn.ModuleList()
        self.norm = th.nn.ModuleList()

        self.conv.append(tgnn.DenseGraphConv(self.in_c, self.hidden_c))  # DenseGCNConv
        self.norm.append(tgnn.LayerNorm(self.hidden_c)) 
        # BatchNorm1d(self.hidden_c)) #tgnn.GraphNorm(self.hidden_c))# tgnn or th.nn layernorm??

        self.conv.append(tgnn.DenseGraphConv(self.hidden_c, self.hidden_c))
        self.norm.append(tgnn.LayerNorm(self.hidden_c))
        # BatchNorm1d(self.hidden_c)) #tgnn.GraphNorm(self.hidden_c))#

        self.conv.append(tgnn.DenseGraphConv(self.hidden_c, self.out_c))
        self.norm.append(tgnn.LayerNorm(self.out_c))
        # BatchNorm1d(self.out_c)) #tgnn.GraphNorm(self.out_c))#

    def forward(self, x, adj, mask=None):
        for step in range(len(self.conv)):
            x = self.conv[step](x, adj, mask)
            x = F.elu(self.norm[step](x))

        return x


# CG RNA Classifier Model using DMoN pooling
class DMoNCG(th.nn.Module):
    def __init__(self, num_node_feats):
        self.num_node_feats = num_node_feats
        super().__init__()
        
        self.pre = tgnn.Sequential(
            "x, edge_index",
            [(tgnn.TAGConv(self.num_node_feats, 64), "x, edge_index -> x"),
            #tgnn.norm.BatchNorm(64), #InstanceNorm(64), #GraphNorm(64), #DiffGroupNorm(64, 1), #
            th.nn.ELU(),
            (tgnn.TAGConv(64, 64), "x, edge_index -> x"),
            #tgnn.norm.BatchNorm(64), #InstanceNorm(64), #GraphNorm(64), #DiffGroupNorm(64, 1), #
            th.nn.ELU(),
            (tgnn.TAGConv(64, 64, bias=False), "x, edge_index -> x"),
            tgnn.norm.LayerNorm(64), #InstanceNorm(64), #GraphNorm(64), #DiffGroupNorm(64, 1), #BatchNorm(64), #
            th.nn.ELU()
            ])

        num_nodes = math.ceil(0.25 * 64)
        self.gcn1 = tgnn.Sequential(
                "x, adj",
                [(tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                th.nn.ELU(),
                (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                th.nn.ELU()
                ])  # GNN(self.num_node_feats, 64, 64)
        self.pool1 = tgnn.DMoNPooling([64, 64], num_nodes)

        num_nodes = math.ceil(0.25 * num_nodes)
        self.gcn2 = tgnn.Sequential(
                "x, adj",
                [(tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                th.nn.ELU(),
                (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                th.nn.ELU()
                ])  # GNN(64, 64, 64)
        self.pool2 = tgnn.DMoNPooling([64, 64], num_nodes)

        num_nodes = math.ceil(0.25 * num_nodes)
        self.gcn3 = tgnn.Sequential(
                "x, adj",
                [(tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                th.nn.ELU(),
                (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                th.nn.ELU()
                ])  # GNN(64, 64, 64)

        self.classify = th.nn.Sequential(
            th.nn.Linear(64, 128),
            th.nn.ELU(),
            th.nn.Linear(128, 128),
            th.nn.ELU(),
            th.nn.Linear(128, 1)
            )

    def forward(self, data):
        x = data.x
        batch = data.batch
        edge_index = data.edge_index

        x = self.pre(x, edge_index)

        x, mask = tgu.to_dense_batch(x, batch, max_num_nodes=64)
        adj = tgu.to_dense_adj(edge_index, batch, max_num_nodes=64)

        x = F.elu(self.gcn1(x, adj))
        _, x, adj, sp1, o1, c1 = self.pool1(x, adj, mask)

        x = F.elu(self.gcn2(x, adj))
        _, x, adj, sp2, o2, c2 = self.pool2(x, adj)

        x = F.elu(self.gcn3(x, adj))
        #_, x, adj, sp3, o3, c3 = self.pool3(x, adj)

        #x = F.elu(self.gcn4(x, adj))

        x = x.mean(dim=1)

        x = self.classify(x)
        x = th.flatten(x)

        return x, sp1 + sp2 + o1 + o2 + c1 + c2#).detach().item() #(sp1 + sp2 + sp3 + o1 + o2 + o3 + c1 + c2 + c3).detach().item()

# CG RNA Classifier Model using MinCut pooling
class MinCutCG(th.nn.Module):
    def __init__(self, num_node_feats):
        self.num_node_feats = num_node_feats
        super().__init__()

        self.pre = tgnn.Sequential(
            "x, edge_index",
            [(tgnn.TAGConv(self.num_node_feats, 64), "x, edge_index -> x"),
            #tgnn.norm.BatchNorm(64), #InstanceNorm(64), #GraphNorm(64), #DiffGroupNorm(64, 1), #
            th.nn.ELU(),
            (tgnn.TAGConv(64, 64), "x, edge_index -> x"),
            #tgnn.norm.BatchNorm(64), #InstanceNorm(64), #GraphNorm(64), #DiffGroupNorm(64, 1), #
            th.nn.ELU(),
            (tgnn.TAGConv(64, 64, bias=False), "x, edge_index -> x"),
            tgnn.norm.LayerNorm(64), #InstanceNorm(64), #GraphNorm(64), #DiffGroupNorm(64, 1), #BatchNorm(64), #
            th.nn.ELU()
            ])

        self.gcn1 = tgnn.Sequential(
            "x, adj",
            [(tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
            #tgnn.norm.BatchNorm(64), #InstanceNorm(64), #GraphNorm(64), #DiffGroupNorm(64, 1), #
            th.nn.ELU(),
            (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
            #tgnn.norm.BatchNorm(64), #InstanceNorm(64), #GraphNorm(64), #DiffGroupNorm(64, 1), #
            th.nn.ELU()
            ])
        num_nodes = 16
        self.pool1 = th.nn.Sequential(
            th.nn.Linear(64, 64),
            #tgnn.norm.BatchNorm(16), #InstanceNorm(64), #GraphNorm(64), #DiffGroupNorm(64, 1), #
            th.nn.ELU(),
            th.nn.Linear(64, num_nodes),
            #tgnn.norm.BatchNorm(num_nodes), #GraphNorm(num_nodes), #InstanceNorm(num_nodes), #
            th.nn.ELU()
            )

        self.gcn2 = tgnn.Sequential(
            "x, adj",
            [(tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
            #tgnn.norm.BatchNorm(16), #InstanceNorm(64), #GraphNorm(64), #DiffGroupNorm(64, 1), #
            th.nn.ELU(),
            (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
            #tgnn.norm.BatchNorm(16), #InstanceNorm(64), #GraphNorm(64), #DiffGroupNorm(64, 1), #
            th.nn.ELU()
            ])
        num_nodes = 4
        self.pool2 = th.nn.Sequential(
            th.nn.Linear(64, 64),
            #tgnn.norm.BatchNorm(16), #InstanceNorm(64), #GraphNorm(64), #DiffGroupNorm(64, 1), #
            th.nn.ELU(),
            th.nn.Linear(64, num_nodes),
            #tgnn.norm.BatchNorm(num_nodes), #GraphNorm(num_nodes), #InstanceNorm(num_nodes), #
            th.nn.ELU()
            )

        self.gcn3 = tgnn.Sequential(
            "x, adj",
            [(tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
            #tgnn.norm.BatchNorm(4), #InstanceNorm(64), #GraphNorm(64), #DiffGroupNorm(64, 1), #
            th.nn.ELU(),
            (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
            #tgnn.norm.BatchNorm(4), #InstanceNorm(64), #GraphNorm(64), #DiffGroupNorm(64, 1), #
            th.nn.ELU(),
            ])

        '''
        num_nodes = 2
        self.pool3 = th.nn.Sequential(
            th.nn.Linear(64, 64),
            th.nn.ELU(),
            th.nn.Linear(64, num_nodes),
            th.nn.ELU()
            )

        self.gcn4 = tgnn.Sequential(
            "x, adj",
            [(tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
            th.nn.ELU(),
            (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
            th.nn.ELU(),
            ])
        '''
        self.classify = th.nn.Sequential(
            th.nn.Linear(64, 64),
            th.nn.ELU(),
            th.nn.Linear(64, 64),
            th.nn.ELU(),
            th.nn.Linear(64, 1)
            )
        

    def forward(self, data):
        x = data.x
        batch = data.batch
        edge_index = data.edge_index

        x = self.pre(x, edge_index)

        x, mask = tgu.to_dense_batch(x, batch, max_num_nodes=64)
        adj = tgu.to_dense_adj(edge_index, batch, max_num_nodes=64)

        x = self.gcn1(x, adj)
        s = self.pool1(x)

        x, adj, mcl1, ol1 = dense_mincut_pool_adapted(x, adj, s, mask) #tgnn.dense_mincut_pool(x, adj, s, mask)

        x = self.gcn2(x, adj)
        s = self.pool2(x)

        x, adj, mcl2, ol2 = dense_mincut_pool_adapted(x, adj, s) #tgnn.dense_mincut_pool(x, adj, s)
        

        x = self.gcn3(x, adj)
        #s = self.pool3(x)

        #x, adj, mcl3, ol3 = tgnn.dense_mincut_pool(x, adj, s)
        

        #x = self.gcn4(x, adj)


        x = x.mean(dim=1) #sum(dim=1) # sum decreased acc

        x = self.classify(x)
        x = th.flatten(x)

        return x, mcl1 + mcl2 + ol1 + ol2#(mcl1 + mcl2 + mcl3 + ol1 + ol2 + ol3).detach().item()

# Coarse Grain RNA Classifier Model using differentiable pooling
class DiffCG(th.nn.Module):
    def __init__(self, num_node_feats):
        self.num_node_feats = num_node_feats
        super().__init__()

        num_nodes = math.ceil(0.25 * 64)
        self.gcn_pool1 = GNN(self.num_node_feats, 64, num_nodes)
        self.gcn_embed1 = GNN(self.num_node_feats, 64, 64)

        num_nodes = math.ceil(0.25 * num_nodes)
        self.gcn_pool2 = GNN(64, 64, num_nodes)
        self.gcn_embed2 = GNN(64, 64, 64)

        num_nodes = math.ceil(0.25 * num_nodes)
        self.gcn_pool3 = GNN(64, 64, num_nodes)
        self.gcn_embed3 = GNN(64, 64, 64)

        self.gcn_embed4 = GNN(64, 64, 64)

        self.classify = th.nn.Sequential(
            th.nn.Linear(64, 256),  # 512),
            th.nn.ELU(),
            th.nn.Linear(256, 256),  # (512, 512),
            th.nn.ELU(),
            th.nn.Linear(256, 256),  # (512, 512),
            th.nn.ELU(),
            th.nn.Linear(256, 256),  # (512, 512),
            th.nn.ELU(),
            th.nn.Linear(256, 1),  # (512, 1)
        )
        self.pos = th.nn.ReLU()  # th.nn.Softplus(threshold=1)

    def forward(self, data):
        x = data.x
        batch = data.batch
        edge_index = data.edge_index

        x, mask = tgu.to_dense_batch(x, batch, max_num_nodes=64)
        adj = tgu.to_dense_adj(edge_index, batch, max_num_nodes=64)

        s = self.gcn_pool1(x, adj)
        x = self.gcn_embed1(x, adj)

        x, adj, l1, e1 = tgnn.dense_diff_pool(x, adj, s, mask)
        l = l1
        e = e1

        s = self.gcn_pool2(x, adj)
        x = self.gcn_embed2(x, adj)

        x, adj, l2, e2 = tgnn.dense_diff_pool(x, adj, s)
        l += l2
        e += e2

        s = self.gcn_pool3(x, adj)
        x = self.gcn_embed3(x, adj)

        x, adj, l3, e3 = tgnn.dense_diff_pool(x, adj, s)
        l += l3
        e += e3

        x = self.gcn_embed4(x, adj)

        # x = tgnn.global_mean_pool(x, batch)
        x = x.mean(dim=1)

        x = self.classify(x)
        x = th.flatten(x)

        return x, l + e

# Coarse Grain RNA Classifier Model
class BuildGCN(th.nn.Module):
    def __init__(self, channels, layers):
        self.channels = channels
        self.layers = layers
        super().__init__()

        modules = []
        for i in range(self.layers): 
            modules.append((tgnn.GCN2Conv(self.channels, alpha=0.1), "x, x_0, edge_index -> x")) #(tgnn.GENConv(self.channels, self.channels, aggr='add', learn_t=True, learn_p=True), "x, edge_index -> x"))#
            modules.append(tgnn.norm.BatchNorm(self.channels)) #LayerNorm(64)) #
            modules.append(th.nn.ELU(inplace=True))

        self.gcn = tgnn.Sequential("x, x_0, edge_index", modules) #("x, edge_index", modules)#

    def forward(self, x, x_0, edge_index): #(self, x, edge_index):#
        x = self.gcn(x, x_0, edge_index) #(x, edge_index)#
        return x

class DeepCG(th.nn.Module):
    def __init__(self, num_node_feats, blocks):

        if len(blocks) == 4:
            self.blocks = blocks
        else:
            print("4 layer blocks need to be specified")
            sys.exit(1)

        self.num_node_feats = num_node_feats
        self.c = 0
        super().__init__()

        self.pre = th.nn.Sequential(
            th.nn.Linear(self.num_node_feats, 64),
            th.nn.ELU(inplace=True),
            th.nn.Linear(64, 64),
            th.nn.ELU(inplace=True)
        )

        self.conv = tgnn.Sequential("x0, x_0, edge_index, batch", [ #"x0, edge_index, batch", [#
            (BuildGCN(64, self.blocks[0]), "x0, x_0, edge_index -> x1"), #, 1 # "x0, edge_index -> x1"),#
            (lambda x1, x2: [x1, x2], 'x0, x1 -> xs'),
            (tgnn.JumpingKnowledge("lstm", 64, num_layers=2), 'xs -> x'),
            #(tgnn.global_mean_pool, 'xs, batch -> x'),

            (BuildGCN(64, self.blocks[1]), "x, x_0, edge_index -> x2"), #, sum(self.blocks[:1])-1 #"x, edge_index -> x2"),#
            (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
            (tgnn.JumpingKnowledge("lstm", 64, num_layers=2), 'xs -> x'),
            #(tgnn.global_mean_pool, 'xs, batch -> x'),

            (BuildGCN(64, self.blocks[2]), "x, x_0, edge_index -> x3"), #, sum(self.blocks[:2])-1 #"x, edge_index -> x3"),#
            (lambda x1, x2: [x1, x2], 'x2, x3 -> xs'),
            (tgnn.JumpingKnowledge("lstm", 64, num_layers=2), 'xs -> x'),
            #(tgnn.global_mean_pool, 'xs, batch -> x'),

            (BuildGCN(64, self.blocks[1]), "x, x_0, edge_index -> x4"), #, sum(self.blocks[:3])-1 #"x, edge_index -> x4"),#
            (lambda x0, x1, x2, x3, x4: [x0, x1, x2, x3, x4], 'x0, x1, x2, x3, x4 -> x'),
            (tgnn.JumpingKnowledge("lstm", 64, num_layers=2), 'x -> xs'),
            (tgnn.global_add_pool, 'xs, batch -> x'),
        ])

        self.classify = th.nn.Sequential(
            th.nn.Linear(64, 64),
            th.nn.ELU(),
            th.nn.Linear(64, 64),
            th.nn.ELU(),
            th.nn.Linear(64, 1)
            )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        #GCN2Conv
        x = x_0 = self.pre(x)#, edge_index)
        x = self.conv(x, x_0, edge_index, batch)

        #GENConv
        #x = self.pre(x)
        #x = self.conv(x, edge_index, batch)

        x = self.classify(x)

        x = th.flatten(x)
        return x, th.tensor(0)
