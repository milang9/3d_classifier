import numpy as np
import torch as th
import forgi.threedee.model.coarse_grain as ftmc
import os
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

#Graph Dataset Class
class CGDataset(InMemoryDataset):
    def __init__(self, root, rmsd_list, vectorize, k, transform=None, pre_transform=None):
        self.file_path = root
        self.rmsd_list = rmsd_list
        self.vectorize = vectorize
        self.k = k
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
        self.get_rmsd_dict()
        files = []

        for file in self.raw_file_names:
            if file.endswith(".cg") and file in self.rmsd_dict.keys():
                files.append(file)

        for struc in files:
            self.load_cg_file(os.path.join(self.file_path, struc))
            graph = self.build_graph(self.rmsd_dict[struc], struc)
            self.graphs.append(graph)

        if self.pre_filter is not None:
            self.graphs = [data for data in self.graphs if self.pre_filter(data)]

        if self.pre_transform is not None:
            self.graphs = [self.pre_transform(data) for data in self.graphs]

        data, slices = self.collate(self.graphs)
        th.save((data, slices), self.processed_paths[0])
    
    #Graph Building
    #load coarse grain file
    def load_cg_file(self, file):
        '''
        Load coarse grained structure.
        '''
        cg = ftmc.CoarseGrainRNA.from_bg_file(file) 
        c_dict = dict(cg.coords)
        t_dict = dict(cg.twists)
        self.coord_dict = {}
        self.twist_dict = {}
        for e in c_dict:
            a = np.array(c_dict[e][0])
            b = np.array(c_dict[e][1])
            self.coord_dict[e] = a, b
            if e in t_dict:
                c = np.array(t_dict[e][0])
                d = np.array(t_dict[e][1])
                self.twist_dict[e] = c, d
            
        # Get elements and neighbours:
        self.connections = {}
        for elem in cg.sorted_element_iterator():
            if elem not in self.connections:
                self.connections[elem] = cg.connections(elem)

    #create a dict with name and rmsd as labels
    def get_rmsd_dict(self):
        '''
        Load the file containing the RMSD for each structure into a dictionary.
        '''
        self.rmsd_dict = {}
        with open(self.rmsd_list, "r") as fh:
            for line in fh.readlines():
                name, rmsd = (line.rstrip()).split("\t")
                self.rmsd_dict[name] = float(rmsd)

    def coords_as_vectors(self):
        '''
        Transform Coordinates of a given Node into a Vector pointing towards the next element.
        '''
        self.vector_dict = {}
        for elem in self.coord_dict:
            vector = []
            if elem == "t0" or elem == "f0":
                for i in range(3):
                    v = self.coord_dict[elem][0][i] - self.coord_dict[elem][1][i]
                    vector.append(v)
            else:
                for i in range(3):
                    v = self.coord_dict[elem][1][i] - self.coord_dict[elem][0][i]
                    vector.append(v)
            
            self.vector_dict[elem] = np.array(vector)
    
    def n_neighbours(self):
        mp_dir = {}
        for elem in self.coord_dict:
            mp = (self.coord_dict[elem][0] + self.coord_dict[elem][1])/2
            mp_dir[elem] = mp

        #calculate distance from each midpoint to every other
        dist_dir = {}
        for a in mp_dir:
            helper_d = {}
            for b in mp_dir:
                if a != b:
                    dist = np.linalg.norm(mp_dir[b] - mp_dir[a])
                    helper_d[b] = dist
            if helper_d != {}:
                dist_dir[a] = helper_d

        #get the nearest k=5 elements
        n_dict = {}
        for f in dist_dir:
            n_list = []
            i = 0
            for n in  {k: v for k, v in sorted(dist_dir[f].items(), key=lambda item: item[1])}:
                n_list.append(n)
                i+=1
                if i == 3:
                    break
            n_dict[f] = n_list

        self.neighbour_dict = {}
        for elem in n_dict:
            v_arr = [mp_dir[elem]]
            for e in n_dict[elem]:
                v_arr.append(np.array(mp_dir[e]))
            self.neighbour_dict[elem] = np.concatenate(v_arr)

    def build_graph(self, label, name):
        '''
        Build the Graph from the coarse grained RNA structures given by forgi.
        Nodes are labeled in the scheme: [TYPE, COORDINATES/VECTOR, TWIST]
        '''

        #dictionary to convert type
        type_transl = {
            "h": np.array([1, 0, 0, 0, 0, 0]),
            "i": np.array([0, 1, 0, 0, 0, 0]),
            "m": np.array([0, 0, 1, 0, 0, 0]),
            "s": np.array([0, 0, 0, 1, 0, 0]),
            "f": np.array([0, 0, 0, 0, 1, 0]),
            "t": np.array([0, 0, 0, 0, 0, 1])
        } 

        #encode nodes numerically for edge index
        num_graph = {}
        for num, n in enumerate(sorted(self.connections)):
            num_graph[n] = num

        #build graph and edges
        u = []
        v = []
        for node in self.connections:
            for c in self.connections[node]:
                u.append(num_graph[node])
                v.append(num_graph[c])
        
        edge_index = th.tensor([u, v], dtype=th.long)

        if self.vectorize:
            self.coords_as_vectors()
        
        if self.k > 0:
            self.n_neighbours()

        x = []
        for elem in sorted(self.connections):
            a = type_transl[elem[0]]
            if self.vectorize and self.k == 0:
                b = self.vector_dict[elem]
            elif self.k > 0 and self.vectorize == False:
                b = self.neighbour_dict[elem]
            elif self.vectorize and self.k > 0:
                b = np.concatenate([self.vector_dict[elem], self.neighbour_dict[elem]])
            else:
                b = np.concatenate(self.coord_dict[elem])
            if elem in self.twist_dict:
                c = np.concatenate(self.twist_dict[elem]) 
            else:
                c = np.zeros(6)
            z = np.concatenate([a, b, c])
            x.append(z)
        
        x = np.array(x)
        x = th.tensor(x, dtype=th.float32)

        graph = Data(x=x, edge_index=edge_index, y=label, name=name)
        return graph
    
    '''
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def len(self):
        return len(self.graphs)
    '''