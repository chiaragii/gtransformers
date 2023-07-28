import dgl
import torch
import tensorflow as tf
import numpy as np

def load_graphs(path_to_graphs):
    enriched_graphs = open(path_to_graphs, 'r').read()
    graphs = enriched_graphs.split('XP\n')
    graphs.remove('')
    graphs_dict = {}
    graphs_list = []
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
        graphs_list.append(graphs_dict)
        graphs_dict = {}

    subgraph_list = []

    for i in graphs_list:
        for j in range(0, len(i['nodes']) - 1):
            subgraph = []
            event_number = []
            if j >= 1:
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
                subgraph_list.append(subgraph)

    return subgraph_list


def create_dgl_graphs(graph_list):
    dgl_graphs = []
    graph_labels = []
    for graph in graph_list:
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
                graph_labels.append(i)
        g.add_nodes(number_of_nodes)
        if source != [] and destination != []:
            g.add_edges(source, destination)
        g.ndata['features'] = torch.zeros((number_of_nodes, 3))
        for i in range(0, len(features)):
            g.nodes[i].data['features'] += torch.tensor(features[i])
        dgl_graphs.append(g)
    return dgl_graphs, graph_labels

def hot_encoding(graph_labels):
    lines = 0
    labels = []
    indices = []
    hot_labels = []
    with open('data/graphs/attributi.txt', 'r') as f:
        for line in f:
            indices.append(lines)
            lines = lines + 1
            labels.append(line.strip('\n'))
    for label in graph_labels:
        hot = [0] * len(labels)
        index = labels.index(label)
        hot[index] = hot[index]+1
        #tensor = torch.tensor(hot)
        hot_labels.append(hot)
    return hot_labels

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    labels = torch.tensor(np.array(labels)).unsqueeze(1)
    batched_graph = dgl.batch(graphs)
    return batched_graph, labels

def create_label_dict(tensor_labels):
    pass

print("Creating dataset..." )
data_dir = 'data/graphs/'
samples = []
subgraphs = load_graphs(data_dir + 'BPI12_graph.g')
dgl_subgraphs, graph_labels = create_dgl_graphs(subgraphs)
label_tensors = hot_encoding(graph_labels)
for i in range(0, len(dgl_subgraphs)):
    pair = []
    pair.append(dgl_subgraphs[i])
    pair.append(label_tensors[i])
    samples.append(pair)
batched_graphs, tensor_labels = collate(samples)
#dgl.save_graphs("Graphs.pkl", batched_graphs, {'labels': tensor_labels})
#print('ciao')
#save_graphs(data_dir+'graphs.pkl', dgl_subgraphs, {'labels': graph_labels})