from graph_al.acquisition.galaxy.graph import Node, Graph, MultiLinearGraph
import numpy as np

def create_linear_from_list(labels, name, n_order=1):
    """
    Construct a linear graph based on the sequential labels.
    :param labels: -1's and 1's
    :return: a linear graph
    """
    assert len(labels) >= 3, "Linear graph must have more than 3 nodes."
    nodes = []
    idxs = [0 for _ in labels]
    current_idx = 0
    for i, label in enumerate(labels):
        if label == -1:
            idxs[i] = current_idx
            current_idx += 1
    for i, label in enumerate(labels):
        if label == 1:
            idxs[i] = current_idx
            current_idx += 1
    for i, label in enumerate(labels):
        nodes.append(Node(idxs[i], label, i))
    for order in range(1, n_order + 1):
        for i in range(order):
            nodes[i].add_neighbors([nodes[i + order]])
            nodes[-i - 1].add_neighbors([nodes[-i - order - 1]])
        for i in range(order, len(nodes) - order):
            nodes[i].add_neighbors([nodes[i - order], nodes[i + order]])

    return Graph(nodes, name)


def create_rotating_linear_from_list(labels, name, num_rotation):
    """
    Construct a list of linear graph based on the sequential labels but permutated vertex labels.
    :param labels: -1's and 1's
    :return: a list of linear graphs
    """
    assert len(labels) >= 3, "Linear graph must have more than 3 nodes."
    graphs = []
    for graph_idx in range(num_rotation):
        nodes = []
        idxs = [0 for _ in labels]
        current_idx = (graph_idx * 27) % len(labels)
        for i, label in enumerate(labels):
            if label == -1:
                idxs[i] = current_idx
                current_idx = (current_idx + 1) % len(labels)
        for i, label in enumerate(labels):
            if label == 1:
                idxs[i] = current_idx
                current_idx = (current_idx + 1) % len(labels)
        assert current_idx == (graph_idx * 27) % len(labels)
        for i, label in enumerate(labels):
            nodes.append(Node(idxs[i], label, i))
        nodes[0].add_neighbors([nodes[1]])
        nodes[-1].add_neighbors([nodes[-2]])
        for i in range(1, len(nodes) - 1):
            nodes[i].add_neighbors([nodes[i - 1], nodes[i + 1]])
        graphs.append(Graph(nodes, name + ("_%d" % graph_idx)))
    return graphs

def create_linear_graphs(scores, labels, name, n_order=1):
    """
    Construct linear graphs based on the sorted scores along each class.
    :param labels: If K classes, each elements of labels takes 0, ..., K-1.
    :return: a MultiLinearGraph
    """
    most_confident = np.max(scores, axis=1).reshape((-1, 1))
    scores = scores - most_confident + 1e-8 * most_confident
    num_classes = int(np.max(labels)) + 1
    graphs = []
    for c in range(num_classes):
        sorted_idx = np.argsort(scores[:, c])
        nodes = []
        label_class = (labels == c).astype(float) * 2 - 1
        for idx in sorted_idx:
            nodes.append(Node(idx, label_class[idx], idx))
        for order in range(1, n_order + 1):
            for i in range(order):
                nodes[i].add_neighbors([nodes[i + order]])
                nodes[-i - 1].add_neighbors([nodes[-i - order - 1]])
            for i in range(order, len(nodes) - order):
                nodes[i].add_neighbors([nodes[i - order], nodes[i + order]])
        graphs.append(Graph(nodes, name + "class_%d" % c))
    return MultiLinearGraph(graphs, n_order)