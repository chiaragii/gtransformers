import dgl
import torch
from dgl import save_graphs


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


print("Creating dataset..." )
data_dir = 'data/graphs/'
subgraphs = load_graphs(data_dir + 'BPI12_graph.g')
dgl_subgraphs, graph_labels = create_dgl_graphs(subgraphs)

save_graphs(data_dir+'graphs.pkl', dgl_subgraphs, {'labels': graph_labels})
