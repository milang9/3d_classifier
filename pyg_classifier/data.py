import numpy as np
import torch as th
import forgi.threedee.model.coarse_grain as ftmc
import os
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

#Graph Building
#load coarse grain file
def load_cg_file(file): 
    cg = ftmc.CoarseGrainRNA.from_bg_file(file) 
    c_dict = dict(cg.coords)
    t_dict = dict(cg.twists)
    coord_dict = {}
    twist_dict = {}
    for e in c_dict:
        a = th.from_numpy(c_dict[e][0])
        b = th.from_numpy(c_dict[e][1])
        coord_dict[e] = a, b
        if e in t_dict:
            c = th.from_numpy(t_dict[e][0])
            d = th.from_numpy(t_dict[e][1])
            twist_dict[e] = c, d
        
    # Get elements and neighbours:
    connections = {}
    for elem in cg.sorted_element_iterator():
        if elem not in connections:
            connections[elem] = cg.connections(elem)
    return coord_dict, twist_dict, connections

#create a dict with name and rmsd as labels
def get_rmsd_dict(rmsd_list):
    rmsd_dict = {}
    with open(rmsd_list, "r") as fh:
        for line in fh.readlines():
            name, rmsd = (line.rstrip()).split("\t")
            rmsd_dict[name] = float(rmsd)
    return rmsd_dict

def build_graph(coord_dict, twist_dict, connections, label):
    #dictionary to convert type
    type_transl = {
        "h": [1, 0, 0, 0, 0, 0],
        "i": [0, 1, 0, 0, 0, 0],
        "m": [0, 0, 1, 0, 0, 0],
        "s": [0, 0, 0, 1, 0, 0],
        "f": [0, 0, 0, 0, 1, 0],
        "t": [0, 0, 0, 0, 0, 1]
    } 

    #encode nodes numerically for edge index
    num_graph = {}
    for num, n in enumerate(sorted(connections)):
        num_graph[n] = num

    #build graph and edges
    u = []
    v = []
    for node in connections:
        for c in connections[node]:
            u.append(num_graph[node])
            v.append(num_graph[c])
    
    edge_index = th.tensor([u, v], dtype=th.long)

    x = []
    for elem in sorted(connections):
        a = np.array(type_transl[elem[0]])
        b = np.concatenate(coord_dict[elem])
        if elem in twist_dict:
            c = np.concatenate(twist_dict[elem]) 
        else:
            c = np.zeros(6)
        z = np.append(a, [b, c])
        x.append(z)
    x = np.array(x)
    x = th.tensor(x, dtype=th.float32)

    graph = Data(x=x, edge_index=edge_index, y=label)

    return graph

#Graph Dataset Class
class CGDataset(InMemoryDataset):
    def __init__(self, root, rmsd_list, transform=None, pre_transform=None):
        self.file_path = root
        self.rmsd_list = rmsd_list
        super(CGDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = th.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.file_path) if os.path.isfile(os.path.join(self.file_path, f))]
    
    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        self.graphs = []
        rmsd_dict = get_rmsd_dict(self.rmsd_list)
        
        files = []

        for file in self.raw_file_names:
            if file.endswith(".cg") and file in rmsd_dict.keys():
                files.append(file)

        for struc in files:
            coord_dict, twist_dict, connections = load_cg_file(os.path.join(self.file_path, struc))
            graph = build_graph(coord_dict, twist_dict, connections, rmsd_dict[struc])
            self.graphs.append(graph)

        if self.pre_filter is not None:
            self.graphs = [data for data in self.graphs if self.pre_filter(data)]

        if self.pre_transform is not None:
            self.graphs = [self.pre_transform(data) for data in self.graphs]

        data, slices = self.collate(self.graphs)
        th.save((data, slices), self.processed_paths[0])
        
    '''
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def len(self):
        return len(self.graphs)
    '''

