import os
import math
import numpy as np
import torch as th
import forgi.threedee.model.coarse_grain as ftmc
from torch_geometric.data import Data, InMemoryDataset, Dataset
import forgi.threedee.classification.aminor as ftca #forgi/threedee/classification/aminor.py:19 change "from sklearn.neighbors.kde import KernelDensity" to "from sklearn.neighbors import KernelDensity"
import forgi.threedee.utilities.vector as ftuv

def s0_dist(cg_d: dict):
    ideal_start = np.array([0, 0, 1])
    diff_start = cg_d["s0"][0] - ideal_start
    return diff_start

def s0_angle(cg_d: dict):
    s0_len = math.sqrt((cg_d["s0"][1][0] - cg_d["s0"][0][0])**2 + (cg_d["s0"][1][1] - cg_d["s0"][0][1])**2 + (cg_d["s0"][1][2] - cg_d["s0"][0][2])**2)
    A = np.array([0, s0_len, 1])
    B = np.array([0, 0, 1])
    ba = A - B
    s0_vec = cg_d["s0"][1] - cg_d["s0"][0]
    rot_m = ftuv.get_alignment_matrix(ba, s0_vec)
    return rot_m

#Graph Dataset Class
class CGDataset(InMemoryDataset): #Dataset):#
    def __init__(self, root, rmsd_list, vectorize, k, transform=None, pre_transform=None):
        self.file_path = root
        self.rmsd_list = rmsd_list
        self.vectorize = vectorize
        self.k = k
        super().__init__(root, transform, pre_transform) #super(CGDataset, self)
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

        for struc in self.raw_file_names: #files:
            if struc in self.rmsd_dict:
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
    def load_cg_file(self, file: str):
        '''
        Load coarse grained structure.
        '''
        cg = ftmc.CoarseGrainRNA.from_bg_file(file)

        y_e = cg.coords["s0"][1][1]
        i_start = np.array([0, 0, 1])
        i_end = np.array([0, y_e, 1])

        # check if end of s0 is equal to convention. if not rotate structure
        # check if start of s0 is equal to convention, if not translate structure
        if not np.array_equal(cg.coords["s0"][1], i_end) or not np.array_equal(cg.coords["s0"][0], i_start):
            rot_m = s0_angle(dict(cg.coords))
            cg.rotate_translate([0,0,0], rot_m)
            diff_start = s0_dist(dict(cg.coords))
            cg.rotate_translate(diff_start, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        self.coord_dict = dict(cg.coords)
        self.twist_dict = dict(cg.twists)
        
        # Get A-Minor interactions
        a_dict = {}
        for pair in ftca.all_interactions(cg):
            a_dict[pair[0]] = pair[1]

        # Get elements and neighbours:
        self.connections = {}
        for elem in cg.sorted_element_iterator():
            if elem not in self.connections:
                self.connections[elem] = cg.connections(elem)
        
        # add A-Minor interactions as edges
        for ami in a_dict:
            self.connections[ami].append(a_dict[ami])
            self.connections[a_dict[ami]].append(ami)

    # create a dict with name and rmsd as labels
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
            for i in range(3):
                v = self.coord_dict[elem][1][i] - self.coord_dict[elem][0][i]
                vector.append(v)
            
            self.vector_dict[elem] = np.array(vector)
    
    def n_neighbours(self):
        '''
        Calculates the middle point for each element. Returns for each element its center and the center points of the k nearest elements.
        '''
        #calculate the midpoint for each element
        mp_dir = {"null": [0, 0, 0]}
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

        #get the nearest k elements
        n_dict = {}
        for f in dist_dir:
            n_list = []
            for n in {k: v for k, v in sorted(dist_dir[f].items(), key=lambda item: item[1])}:
                while len(n_list) < self.k:
                    n_list.append(n)
                    if len(n_list) == len(dist_dir[f]):
                        n_list.append("null")
                    else:
                        n_list.append(n)
                    
                
            #if len(n_list) < self.k:
            #    while len(n_list) < self.k:
            #        n_list.append("null")
            n_dict[f] = n_list

        #add the nearest k midpoints as vector node features
        self.neighbour_dict = {}
        for elem in n_dict:
            v_arr = []
            for e in n_dict[elem]:
                vec = []
                for i in range(3):
                    p = mp_dir[e][i] - mp_dir[elem][i]
                    vec.append(p)
                v_arr.append(np.array(vec))#mp_dir[e]))
            self.neighbour_dict[elem] = np.concatenate(v_arr)

    def build_graph(self, label: float, name: str):
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
        data = th.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def len(self):
        return len(self.graphs)
    '''