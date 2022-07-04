import os
import math
import numpy as np
import torch as th
import forgi.threedee.model.coarse_grain as ftmc
from torch_geometric.data import Data, InMemoryDataset, Dataset
import forgi.threedee.classification.aminor as ftca #forgi/threedee/classification/aminor.py:19 change "from sklearn.neighbors.kde import KernelDensity" to "from sklearn.neighbors import KernelDensity"
import forgi.threedee.utilities.vector as ftuv

def elem_len(cg_d: dict, elem: str) -> float:
    return math.sqrt((cg_d[elem][1][0] - cg_d[elem][0][0])**2 + (cg_d[elem][1][1] - cg_d[elem][0][1])**2 + (cg_d[elem][1][2] - cg_d[elem][0][2])**2)

def s0_dist(cg_d: dict) -> np.ndarray:
    ideal_start = np.array([0, 0, 1])
    return cg_d["s0"][0] - ideal_start

def s0_angle(cg_d: dict) -> np.ndarray:
    s0_len = elem_len(cg_d, "s0")
    A = np.array([0, s0_len, 1])
    B = np.array([0, 0, 1])
    ba = A - B
    s0_vec = cg_d["s0"][1] - cg_d["s0"][0]
    return ftuv.get_alignment_matrix(ba, s0_vec)

def s1_angle(cg_d: dict) -> np.ndarray:
    vec1 = cg_d["s1"][0] - cg_d["s0"][1]
    n = np.array([1, 0, 0])
    proj_n = (np.dot(vec1, n) / np.linalg.norm(n)**2) * n
    vec2 = np.absolute(vec1 - proj_n)
    len_v1 = np.linalg.norm(vec1)
    len_v2 = np.linalg.norm(vec2)
    return np.arccos(np.dot(vec1, vec2)/(len_v1 * len_v2))

#Graph Dataset Class
class CGDataset(InMemoryDataset): #Dataset):
    def __init__(self, root, rmsd_list, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        self.file_path = root
        self.rmsd_list = rmsd_list
        self.rmsd_dict = {}
        self.pr_files = []
        super().__init__(root, transform, pre_transform, pre_filter)
        #InMemoryDataset
        print(self.processed_paths[0])
        #if os.path.exists(self.processed_paths[0]):
        self.data, self.slices = th.load(self.processed_paths[0])
        print(self.data, self.slices)
        #else:
        #    self.data = None
        #    self.slices = None 

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.file_path) if os.path.isfile(os.path.join(self.file_path, f))]
    
    @property
    def processed_file_names(self):
        #InMemoryDataset
        return ["data.pt"]
        #Dataset
        #return self.pr_files#[pr for pr in os.listdir(self.file_path) if os.path.isfile(os.path.join(self.processed_paths[0], pr))]# 

    def process(self):
        #InMemoryDataset
        self.graphs = []
        self.get_rmsd_dict()
        #Dataset
        #idx = 0
        for struc in self.raw_file_names:
            if struc in self.rmsd_dict:
                self.load_cg_file(os.path.join(self.file_path, struc))
                graph = self.build_graph(self.rmsd_dict[struc], struc)
                #InMemoryDataset
                self.graphs.append(graph)

                #Dataset
                '''
                if self.pre_filter is not None and not self.pre_filter(graph):
                    continue

                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)
                
                data_file = f"data_{idx}.pt"
                th.save(graph, os.path.join(self.processed_dir, data_file))#f'data_{idx}.pt'))
                self.pr_files.append(data_file)
                idx += 1
                '''

        #InMemoryDataset
        #'''
        if self.pre_filter is not None:
            self.graphs = [data for data in self.graphs if self.pre_filter(data)]

        if self.pre_transform is not None:
            self.graphs = [self.pre_transform(data) for data in self.graphs]

        data, slices = self.collate(self.graphs)
        th.save((data, slices), self.processed_paths[0])
        #'''
    #Dataset
    '''
    def get(self, idx):
        data = th.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def len(self):
        return len(self.processed_file_names)
    '''

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
        if not np.array_equal(cg.coords["s0"][1], i_end):
            rot_m = s0_angle(dict(cg.coords))
            cg.rotate_translate([0,0,0], rot_m)

        # rotate around y-axis, so that connecting element between s0 and s1 is on the yz-plane
        # rotation influences s0 --> move the structure up
        s1angle = s1_angle(dict(cg.coords)) 
        cg.rotate(s1angle, axis="y")
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
        with open(self.rmsd_list, "r") as fh:
            for line in fh.readlines():
                name, rmsd = (line.rstrip()).split("\t")
                self.rmsd_dict[name] = float(rmsd)

    def build_graph(self, label: float, name: str) -> Data:
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

        x = []
        for elem in sorted(self.connections):
            a = type_transl[elem[0]]
            b = np.concatenate(self.coord_dict[elem])
            if elem in self.twist_dict:
                c = np.concatenate(self.twist_dict[elem]) 
            else:
                c = np.zeros(6)
            z = np.concatenate([a, b, c])
            x.append(z)
        
        x = np.array(x)
        x = th.as_tensor(x, dtype=th.float32)

        graph = Data(x=x, edge_index=edge_index, y=label, name=name)
        return graph
    
class VectorCGDataset(InMemoryDataset): #Dataset):
    def __init__(self, root, rmsd_list, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        self.file_path = root
        self.rmsd_list = rmsd_list
        self.rmsd_dict = {}
        self.pr_files = []
        super().__init__(root, transform, pre_transform, pre_filter)
        #InMemoryDataset
        self.data, self.slices = th.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.file_path) if os.path.isfile(os.path.join(self.file_path, f))]
    
    @property
    def processed_file_names(self):
        #InMemoryDataset
        return ["data.pt"]
        #Dataset
        #return self.pr_files#[pr for pr in os.listdir(self.file_path) if os.path.isfile(os.path.join(self.processed_paths[0], pr))]# 

    def process(self):
        #InMemoryDataset
        self.graphs = []
        self.get_rmsd_dict()
        #Dataset
        #idx = 0
        for struc in self.raw_file_names:
            if struc in self.rmsd_dict:
                self.load_cg_file(os.path.join(self.file_path, struc))
                graph = self.build_graph(self.rmsd_dict[struc], struc)
                #InMemoryDataset
                self.graphs.append(graph)

                #Dataset
                '''
                if self.pre_filter is not None and not self.pre_filter(graph):
                    continue

                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)
                
                data_file = f"data_{idx}.pt"
                th.save(graph, os.path.join(self.processed_dir, data_file))#f'data_{idx}.pt'))
                self.pr_files.append(data_file)
                idx += 1
                '''

        #InMemoryDataset
        #'''
        if self.pre_filter is not None:
            self.graphs = [data for data in self.graphs if self.pre_filter(data)]

        if self.pre_transform is not None:
            self.graphs = [self.pre_transform(data) for data in self.graphs]

        data, slices = self.collate(self.graphs)
        th.save((data, slices), self.processed_paths[0])
        #'''
    #Dataset
    '''
    def get(self, idx):
        data = th.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def len(self):
        return len(self.processed_file_names)
    '''

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
        if not np.array_equal(cg.coords["s0"][1], i_end):
            rot_m = s0_angle(dict(cg.coords))
            cg.rotate_translate([0,0,0], rot_m)

        # rotate around y-axis, so that connecting element between s0 and s1 is on the yz-plane
        # rotation influences s0 --> move the structure up
        s1angle = s1_angle(dict(cg.coords)) 
        cg.rotate(s1angle, axis="y")
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

    def build_graph(self, label: float, name: str) -> Data:
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

        self.coords_as_vectors()

        x = []
        for elem in sorted(self.connections):
            a = type_transl[elem[0]]
            b = self.vector_dict[elem]
            if elem in self.twist_dict:
                c = np.concatenate(self.twist_dict[elem]) 
            else:
                c = np.zeros(6)
            z = np.concatenate([a, b, c])
            x.append(z)
        
        x = np.array(x)
        x = th.as_tensor(x, dtype=th.float32)

        graph = Data(x=x, edge_index=edge_index, y=label, name=name)
        return graph

class NeighbourCGDataset(InMemoryDataset): #Dataset):
    def __init__(self, root, rmsd_list, k, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        self.file_path = root
        self.rmsd_list = rmsd_list
        self.k = k
        self.rmsd_dict = {}
        self.pr_files = []
        super().__init__(root, transform, pre_transform, pre_filter)
        #InMemoryDataset
        self.data, self.slices = th.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.file_path) if os.path.isfile(os.path.join(self.file_path, f))]
    
    @property
    def processed_file_names(self):
        #InMemoryDataset
        return ["data.pt"]
        #Dataset
        #return self.pr_files#[pr for pr in os.listdir(self.file_path) if os.path.isfile(os.path.join(self.processed_paths[0], pr))]# 

    def process(self):
        #InMemoryDataset
        self.graphs = []
        self.get_rmsd_dict()
        #Dataset
        #idx = 0
        for struc in self.raw_file_names:
            if struc in self.rmsd_dict:
                self.load_cg_file(os.path.join(self.file_path, struc))
                graph = self.build_graph(self.rmsd_dict[struc], struc)
                #InMemoryDataset
                self.graphs.append(graph)

                #Dataset
                '''
                if self.pre_filter is not None and not self.pre_filter(graph):
                    continue

                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)
                
                data_file = f"data_{idx}.pt"
                th.save(graph, os.path.join(self.processed_dir, data_file))#f'data_{idx}.pt'))
                self.pr_files.append(data_file)
                idx += 1
                '''

        #InMemoryDataset
        #'''
        if self.pre_filter is not None:
            self.graphs = [data for data in self.graphs if self.pre_filter(data)]

        if self.pre_transform is not None:
            self.graphs = [self.pre_transform(data) for data in self.graphs]

        data, slices = self.collate(self.graphs)
        th.save((data, slices), self.processed_paths[0])
        #'''
    #Dataset
    '''
    def get(self, idx):
        data = th.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def len(self):
        return len(self.processed_file_names)
    '''

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
        if not np.array_equal(cg.coords["s0"][1], i_end):
            rot_m = s0_angle(dict(cg.coords))
            cg.rotate_translate([0,0,0], rot_m)

        # rotate around y-axis, so that connecting element between s0 and s1 is on the yz-plane
        # rotation influences s0 --> move the structure up
        s1angle = s1_angle(dict(cg.coords)) 
        cg.rotate(s1angle, axis="y")
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
        with open(self.rmsd_list, "r") as fh:
            for line in fh.readlines():
                name, rmsd = (line.rstrip()).split("\t")
                self.rmsd_dict[name] = float(rmsd)
    
    def k_neighbours(self):
        '''
        Calculates the distance between start- and endpoints of elements. Elements are sorted by the shortest distance.
        The set of k vectors pointing from the start and from the end of the current element to the nearest one are returned.
        '''

        #calculate distance between start- and endpoints of elements, sort by shortest
        dist_dir = {}
        for a in self.coord_dict:
            helper_dict = {}
            for b in self.coord_dict:
                if a != b:
                    start_dist = np.linalg.norm(self.coord_dict[a][0] - self.coord_dict[b][0])
                    end_dist = np.linalg.norm(self.coord_dict[a][1] - self.coord_dict[b][1])
                    helper_dict[b] = sorted([start_dist, end_dist])
            dist_dir[a] = helper_dict

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
            n_dict[f] = n_list

        #add the nearest k midpoints as vector node features
        self.neighbour_dict = {}
        for elem in n_dict:
            v_arr = []
            for e in n_dict[elem]:
                if elem == "null" or e == "null":
                    v_arr.append(np.array([0, 0, 0, 0, 0, 0]))
                else:
                    start_vec = []
                    end_vec = []
                    for i in range(3):
                        start_p = self.coord_dict[e][0][i] - self.coord_dict[elem][0][i]
                        end_p = self.coord_dict[e][1][i] - self.coord_dict[elem][1][i]
                        start_vec.append(start_p)
                        end_vec.append(end_p)
                    v_arr.append(np.array(start_vec + end_vec))
            self.neighbour_dict[elem] = np.concatenate(v_arr)

    def build_graph(self, label: float, name: str) -> Data:
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

        self.k_neighbours()

        x = []
        for elem in sorted(self.connections):
            a = type_transl[elem[0]]
            b = self.neighbour_dict[elem]
            if elem in self.twist_dict:
                c = np.concatenate(self.twist_dict[elem]) 
            else:
                c = np.zeros(6)
            z = np.concatenate([a, b, c])
            x.append(z)
        
        x = np.array(x)
        x = th.as_tensor(x, dtype=th.float32)

        graph = Data(x=x, edge_index=edge_index, y=label, name=name)
        return graph