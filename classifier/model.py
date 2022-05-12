import torch as th
import torch.nn.functional as F
import torch_geometric.nn as tgnn
import math

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
        self.norm.append(
            tgnn.LayerNorm(self.hidden_c)
        )  # BatchNorm1d(self.hidden_c)) #tgnn.GraphNorm(self.hidden_c))# tgnn or th.nn layernorm??

        self.conv.append(tgnn.DenseGraphConv(self.hidden_c, self.hidden_c))
        self.norm.append(
            tgnn.LayerNorm(self.hidden_c)
        )  # BatchNorm1d(self.hidden_c)) #tgnn.GraphNorm(self.hidden_c))#

        self.conv.append(tgnn.DenseGraphConv(self.hidden_c, self.out_c))
        self.norm.append(
            tgnn.LayerNorm(self.out_c)
        )  # BatchNorm1d(self.out_c)) #tgnn.GraphNorm(self.out_c))#

    def forward(self, x, adj, mask=None):
        for step in range(len(self.conv)):
            x = self.conv[step](x, adj, mask)
            x = F.elu(self.norm[step](x))

        return x


# CG RNA Classifier Model using DMoN pooling
class DMoN_CG_Classifier(th.nn.Module):
    def __init__(self, num_node_feats):
        self.num_node_feats = num_node_feats
        super().__init__()

        num_nodes = math.ceil(0.25 * 64)
        self.gcn1 = tgnn.Sequential(
                "x, adj",
                [(tgnn.DenseGraphConv(self.num_node_feats, 64), "x, adj -> x"),
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
        '''
        self.pool3 = tgnn.DMoNPooling([64, 64], num_nodes)

        self.gcn4 = tgnn.Sequential(
                "x, adj",
                [(tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                th.nn.ELU(),
                (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                th.nn.ELU()
                ])  # tgnn.DenseGraphConv(64, 64) # GNN(64, 64, 64)
        '''
        self.classify = th.nn.Sequential(
            th.nn.Linear(64, 128),
            th.nn.ELU(),
            th.nn.Linear(128, 128),
            th.nn.ELU(),
            th.nn.Linear(128, 1)
            )
        self.pos = th.nn.ReLU()  # th.nn.Softplus(threshold=1)

    def forward(self, data, training=False):
        x = data.x
        adj = data.adj

        x = F.elu(self.gcn1(x, adj))
        _, x, adj, sp1, o1, c1 = self.pool1(x, adj)

        x = F.elu(self.gcn2(x, adj))
        _, x, adj, sp2, o2, c2 = self.pool2(x, adj)

        x = F.elu(self.gcn3(x, adj))
        #_, x, adj, sp3, o3, c3 = self.pool3(x, adj)

        #x = F.elu(self.gcn4(x, adj))

        x = x.mean(dim=1)

        x = self.classify(x)

        if training:
            return x, (sp1 + sp2 + o1 + o2 + c1 + c2).detach().item() #(sp1 + sp2 + sp3 + o1 + o2 + o3 + c1 + c2 + c3).detach().item()
        else:
            return self.pos(x), (sp1 + sp2 + o1 + o2 + c1 + c2).detach().item() #(sp1 + sp2 + sp3 + o1 + o2 + o3 + c1 + c2 + c3).detach().item() # should this be added when not training?


# CG RNA Classifier Model using MinCut pooling
"""
class MinCut_CG_Classifier(th.nn.Module):
    def __init__(self, num_node_feats):
        self.num_node_feats = num_node_feats
        super().__init__()

        self.gcn1 = tgnn.Sequential(
            "x, adj",
            [(tgnn.DenseGraphConv(self.num_node_feats, 64), "x, adj -> x"),
                th.nn.ELU(),
                # (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                # th.nn.ELU(),
                (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                th.nn.ELU(),
            ])
        num_nodes = 16 #math.ceil(0.25 * 64)
        self.pool1 = th.nn.Sequential(
            th.nn.Linear(64, 64),
            th.nn.ELU(),
            # th.nn.Linear(64, 64),
            # th.nn.ELU(),
            th.nn.Linear(64, num_nodes),
            th.nn.ELU()
            )

        self.gcn2 = tgnn.Sequential(
            "x, adj",
            [(tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                th.nn.ELU(),
                # (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                # th.nn.ELU(),
                (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                th.nn.ELU(),
            ])
        num_nodes = 4 #math.ceil(0.25 * num_nodes)
        self.pool2 = th.nn.Sequential(
            th.nn.Linear(64, 64),
            th.nn.ELU(),
            # th.nn.Linear(64, 64),
            # th.nn.ELU(),
            th.nn.Linear(64, num_nodes),
            th.nn.ELU())

        self.gcn3 = tgnn.Sequential(
            "x, adj",
            [(tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                th.nn.ELU(),
                # (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                # th.nn.ELU(),
                (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                th.nn.ELU(),
            ])
        '''
        num_nodes = math.ceil(0.25 * num_nodes)
        self.pool3 = th.nn.Sequential(
            th.nn.Linear(64, 64),
            th.nn.ELU(),
            # th.nn.Linear(64, 64),
            # th.nn.ELU(),
            th.nn.Linear(64, num_nodes),
            th.nn.ELU())

        self.gcn4 = tgnn.Sequential(
            "x, adj",
            [(tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                th.nn.ELU(),
                # (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                # th.nn.ELU(),
                (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
                th.nn.ELU(),
            ])
        '''
        self.classify = th.nn.Sequential(
            th.nn.Linear(64, 64),
            th.nn.ELU(),
            th.nn.Linear(64, 64),
            th.nn.ELU(),
            th.nn.Linear(64, 64),
            th.nn.ELU(),
            # th.nn.Linear(128, 128),
            # th.nn.ELU(),
            th.nn.Linear(64, 1)
        )
        self.pos = th.nn.ReLU()

    def forward(self, data, training=False):
        x = data.x
        adj = data.adj

        x = self.gcn1(x, adj)
        s = self.pool1(x)

        x, adj, mcl1, ol1 = tgnn.dense_mincut_pool(x, adj, s)

        x = self.gcn2(x, adj)
        s = self.pool2(x)

        x, adj, mcl2, ol2 = tgnn.dense_mincut_pool(x, adj, s)
        

        x = self.gcn3(x, adj)
        #s = self.pool3(x)

        #x, adj, mcl3, ol3 = tgnn.dense_mincut_pool(x, adj, s)
        

        #x = self.gcn4(x, adj)

        x = x.mean(dim=1)

        x = self.classify(x)

        return x, (mcl1 + mcl2 + ol1 + ol2).detach().item()# (mcl1 + mcl2 + mcl3 + ol1 + ol2 + ol3).detach().item()
        # if training:
        #    return x, (mcl + ol).item()
        # else:
        #    return self.pos(x), (mcl + ol).item()
"""

class MinCut_CG_Classifier(th.nn.Module):
    def __init__(self, num_node_feats):
        self.num_node_feats = num_node_feats
        super().__init__()

        self.gcn1 = tgnn.Sequential(
            "x, adj",
            [(tgnn.DenseGraphConv(self.num_node_feats, 64), "x, adj -> x"),
            th.nn.ELU(),
            (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
            th.nn.ELU(),
            ])
        num_nodes = 32 #math.ceil(0.25 * 64)
        self.pool1 = th.nn.Sequential(
            th.nn.Linear(64, 64),
            th.nn.ELU(),
            th.nn.Linear(64, num_nodes),
            th.nn.ELU()
            )

        self.gcn2 = tgnn.Sequential(
            "x, adj",
            [(tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
            th.nn.ELU(),
            (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
            th.nn.ELU(),
            ])
        num_nodes = 16
        self.pool2 = th.nn.Sequential(
            th.nn.Linear(64, 64),
            th.nn.ELU(),
            th.nn.Linear(64, num_nodes),
            th.nn.ELU())

        self.gcn3 = tgnn.Sequential(
            "x, adj",
            [(tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
            th.nn.ELU(),
            (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
            th.nn.ELU(),
            ])

        num_nodes = 4
        self.pool3 = th.nn.Sequential(
            th.nn.Linear(64, 64),
            th.nn.ELU(),
            th.nn.Linear(64, num_nodes),
            th.nn.ELU())

        self.gcn4 = tgnn.Sequential(
            "x, adj",
            [(tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
            th.nn.ELU(),
            (tgnn.DenseGraphConv(64, 64), "x, adj -> x"),
            th.nn.ELU(),
            ])

        self.classify = th.nn.Sequential(
            th.nn.Linear(64, 64),
            th.nn.ELU(),
            th.nn.Linear(64, 64),
            th.nn.ELU(),
            th.nn.Linear(64, 64),
            th.nn.ELU(),
            th.nn.Linear(64, 64),
            th.nn.ELU(),
            th.nn.Linear(64, 1)
        )
        self.pos = th.nn.ReLU()

    def forward(self, data, training=False):
        x = data.x
        adj = data.adj

        x = self.gcn1(x, adj)
        s = self.pool1(x)

        x, adj, mcl1, ol1 = tgnn.dense_mincut_pool(x, adj, s)

        x = self.gcn2(x, adj)
        s = self.pool2(x)

        x, adj, mcl2, ol2 = tgnn.dense_mincut_pool(x, adj, s)
        

        x = self.gcn3(x, adj)
        s = self.pool3(x)

        x, adj, mcl3, ol3 = tgnn.dense_mincut_pool(x, adj, s)
        

        x = self.gcn4(x, adj)

        x = x.mean(dim=1)

        x = self.classify(x)

        return x, (mcl1 + mcl2 + mcl3 + ol1 + ol2 + ol3).detach().item()#(mcl1 + mcl2 + mcl3 + mcl4 + ol1 + ol2 + ol3 + ol4).detach().item()
        # if training:
        #    return x, (mcl + ol).item()
        # else:
        #    return self.pos(x), (mcl + ol).item()

# Coarse Grain RNA Classifier Model using differentiable pooling
class Diff_CG_Classifier(th.nn.Module):
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

    def forward(self, data, training=False):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        adj = data.adj

        s = self.gcn_pool1(x, adj)
        x = self.gcn_embed1(x, adj)

        x, adj, l1, e1 = tgnn.dense_diff_pool(x, adj, s)
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

        # return x, l, e

        if training:
            return x, l + e
        else:
            return self.pos(x), l + e


# Coarse Grain RNA Classifier Model
class CG_Classifier(th.nn.Module):
    def __init__(self, num_node_feats):
        self.num_node_feats = num_node_feats
        self.c = 0
        super().__init__()

        self.conv1 = tgnn.TAGConv(self.num_node_feats, 64, K=2)
        self.norm1 = th.nn.LayerNorm(64)
        self.conv2 = tgnn.TAGConv(64, 64, K=2)
        self.norm2 = th.nn.LayerNorm(64)
        self.conv3 = tgnn.TAGConv(64, 64, K=2)
        self.norm3 = th.nn.LayerNorm(64)
        self.conv4 = tgnn.TAGConv(64, 64, K=2)
        self.norm4 = th.nn.LayerNorm(64)
        self.conv5 = tgnn.TAGConv(64, 32, K=2)
        self.norm5 = th.nn.LayerNorm(32)

        self.sage_conv1 = tgnn.SAGEConv(32, 32)
        self.norm6 = th.nn.LayerNorm(32)
        self.sage_conv2 = tgnn.SAGEConv(32, 32)
        self.norm7 = th.nn.LayerNorm(32)
        self.sage_conv3 = tgnn.SAGEConv(32, 32)
        self.norm8 = th.nn.LayerNorm(32)
        self.sage_conv4 = tgnn.SAGEConv(32, 32)
        self.norm9 = th.nn.LayerNorm(32)
        self.sage_conv5 = tgnn.SAGEConv(32, 32)
        self.norm10 = th.nn.LayerNorm(32)

        self.classify = th.nn.Sequential(
            th.nn.Linear(32, 256),  # 512),
            th.nn.ELU(),
            th.nn.Linear(256, 256),  # (512, 512),
            th.nn.ELU(),
            th.nn.Linear(256, 256),  # (512, 512),
            th.nn.ELU(),
            th.nn.Linear(256, 256),  # (512, 512),
            th.nn.ELU(),
            th.nn.Linear(256, 1),  # (512, 1)
        )

        self.pos = th.nn.ReLU()

    def forward(self, data, training=False):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.conv1(x, edge_index)
        x = F.elu(self.norm1(x))
        x = self.conv2(x, edge_index)
        x = F.elu(self.norm2(x))
        x = self.conv3(x, edge_index)
        x = F.elu(self.norm3(x))
        x = self.conv4(x, edge_index)
        x = F.elu(self.norm4(x))
        x = self.conv5(x, edge_index)
        x = F.elu(self.norm5(x))

        x = self.sage_conv1(x, edge_index)
        x = F.elu(self.norm6(x))
        x = self.sage_conv2(x, edge_index)
        x = F.elu(self.norm7(x))
        x = self.sage_conv3(x, edge_index)
        x = F.elu(self.norm8(x))
        x = self.sage_conv4(x, edge_index)
        x = F.elu(self.norm9(x))
        x = self.sage_conv5(x, edge_index)
        x = F.elu(self.norm10(x))

        x = tgnn.global_mean_pool(x, batch)  # self.readout(x, edge_index)

        x = self.classify(x)
        x = th.flatten(x)
        # return x

        if training:
            return x
        else:
            return self.pos(x)
