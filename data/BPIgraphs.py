import torch
import torch.utils.data
import time

import dgl

from scipy import sparse as sp
import numpy as np
import networkx as nx
import hashlib


class GraphsDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, num_nodes):
        self.data_dir = data_dir
        self.num_nodes = num_nodes

        enriched_graphs = open(self.data_dir, 'r').read()
        graphs = enriched_graphs.split('XP\n')
        graphs.remove('')
        graphs_dict = {}
        self.data = []
        for graph in graphs:
            elements = graph.split('\n')
            nodes = []
            edges = []
            for i in range(0, len(elements)):
                if elements[i].startswith('v'):
                    nodes.append(elements[i])
                else:
                    edges.append(elements[i])
            graphs_dict['nodes'] = nodes
            graphs_dict['edges'] = edges
            self.data.append(graphs_dict)
            graphs_dict = {}

        self.subgraph_list = []

        # creating subgraphs
        for i in self.data:
            for j in range(0, len(i['nodes']) - 1):
                subgraph = []
                event_number = []
                # only creating prefix graphs of len > num_nodes
                if j >= self.num_nodes:
                    for k in range(0, j):
                        subgraph.append(i['nodes'][k])
                    for z in subgraph:
                        event_number.append(z.split(' ')[1])
                        label = i['nodes'][k + 1].split(' ')[3]

                    for w in (i['edges']):
                        if w != '':
                            if event_number.__contains__(w.split(' ')[1]) and event_number.__contains__(
                                    w.split(' ')[2]):
                                subgraph.append(w)
                    subgraph.append(label)
                    self.subgraph_list.append(subgraph)


        self.n_samples = len(self.subgraph_list)
        self.graph_lists = []
        self.graph_labels = []
        self._prepare()

    def _prepare(self):

        label_list = []

        for graph in self.subgraph_list:
            g = dgl.DGLGraph()
            number_of_nodes = 0
            features = []
            source = []
            destination = []
            for i in graph:
                if i.startswith('v'):
                    number_of_nodes = number_of_nodes + 1
                    feature = []
                    feature.append(float(i.split(' ')[4]))
                    feature.append(float(i.split(' ')[5]))
                    feature.append(float(i.split(' ')[6]))
                    features.append(feature)
                elif i.startswith('e'):
                    source.append(int(i.split(' ')[1][0]) - 1)
                    destination.append(int(i.split(' ')[2][0]) - 1)
                else:
                    label_list.append(i)
            g.add_nodes(number_of_nodes)
            if source != [] and destination != []:
                g.add_edges(source, destination)
            g.ndata['feat'] = torch.tensor(features)
            g.edata['feat'] = torch.ones(len(source))
            # g.ndata['feat'] = torch.zeros((number_of_nodes, 3))
            # for i in range(0, len(features)):
            #    g.nodes[i].data['feat'] += torch.tensor(features[i]).type(torch.LongTensor)
            self.graph_lists.append(g)

        lines = 0
        labels = []
        indices = []
        hot_labels = []
        with open('data/graphs/attributi.txt', 'r') as f:
            for line in f:
                # for one hot encoding
                if line.strip('\n') in set(label_list):
                    indices.append(lines)
                    lines = lines + 1
                    labels.append(line.strip('\n'))
            # for label in label_list:
            # for one hot encoding
            # hot = [0] * len(labels)
            # index = labels.index(label)
            # hot[index] = hot[index] + 1
            # tensor = torch.tensor(hot)
            self.label_dict = dict(zip(labels, indices))
            for label in label_list:
                label_value = self.label_dict[label]
                self.graph_labels.append(torch.tensor(label_value))

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


class GraphDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='Zinc'):
        t0 = time.time()
        self.name = name

        self.num_atom_type = 28  # known meta-info about the zinc dataset; can be calculated as well
        self.num_bond_type = 4  # known meta-info about the zinc dataset; can be calculated as well

        data_dir = './data/molecules'

        self.train = GraphsDGL(data_dir, 'train', num_graphs=10000)
        self.val = GraphsDGL(data_dir, 'val', num_graphs=1000)
        self.test = GraphsDGL(data_dir, 'test', num_graphs=1000)
        print("Time taken: {:.4f}s".format(time.time() - t0))


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in GraphsDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


def make_full_graph(g):
    """
        Converting the given graph to fully connected
        This function just makes full connections
        removes available edge features 
    """

    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))
    full_g.ndata['feat'] = g.ndata['feat']
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges()).long()

    try:
        full_g.ndata['lap_pos_enc'] = g.ndata['lap_pos_enc']
    except:
        pass

    try:
        full_g.ndata['wl_pos_enc'] = g.ndata['wl_pos_enc']
    except:
        pass

    return full_g


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()

    return g


def wl_positional_encoding(g):
    """
        WL-based absolute positional embedding 
        adapted from 
        
        "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
        Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
        https://github.com/jwzhanggy/Graph-Bert
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1

    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v + 1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1

    g.ndata['wl_pos_enc'] = torch.LongTensor(list(node_color_dict.values()))
    return g


class GraphsDataset(torch.utils.data.Dataset):

    def __init__(self, name, num_nodes):
        start = time.time()
        self.num_nodes = num_nodes

        print("Creating graph dataset...")
        self.name = name
        self.train = GraphsDGL('data/graphs/training.g', self.num_nodes)
        self.test = GraphsDGL('data/graphs/test.g', self.num_nodes)
        self.val = GraphsDGL('data/graphs/val.g', self.num_nodes)
        print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    def check_class_imbalance(self, graph_labels, num_classes):
        label_count = [0] * num_classes
        for label in graph_labels:
            label_index = int(label)
            label_count[label_index] = label_count[label_index] + 1
        return label_count

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(labels).unsqueeze(1)
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels

    def _add_self_loops(self):
        # function for adding self loops
        # this function will be called only if self_loop flag is True
        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _make_full_graph(self):
        # function for converting graphs to full graphs
        # this function will be called only if full_graph flag is True
        self.train.graph_lists = [make_full_graph(g) for g in self.train.graph_lists]
        self.val.graph_lists = [make_full_graph(g) for g in self.val.graph_lists]
        self.test.graph_lists = [make_full_graph(g) for g in self.test.graph_lists]

    def _add_laplacian_positional_encodings(self, pos_enc_dim):
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]

    def _add_wl_positional_encodings(self):
        # WL positional encoding from Graph-Bert, Zhang et al 2020.
        self.train.graph_lists = [wl_positional_encoding(g) for g in self.train.graph_lists]
        self.val.graph_lists = [wl_positional_encoding(g) for g in self.val.graph_lists]
        self.test.graph_lists = [wl_positional_encoding(g) for g in self.test.graph_lists]
