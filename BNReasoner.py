from copy import deepcopy
from typing import Union
from BayesNet import BayesNet
import networkx as nx
import itertools


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # TODO: This is where your methods should go
    def check_d_separation(self, X: str, Y: str, Z: list):
        """
        Are X and Y d-seperated given Z?
        :param X: A start node
        :param Y: An end node
        :param Z: A list of given nodes
        :return: bool
        """
        net = self.bn
        for path_list in nx.all_shortest_paths(net.structure.to_undirected(), X, Y):
            path_length = len(path_list)
            print(path_list, path_length)

            # handle the edge case for very small paths first
            if path_length < 3:
                if path_length < 2:
                    return True
                elif path_length == 2:
                    if path_list[0] in Z or path_list[1] in Z:
                        return True
                    else:
                        return False

            d_separated = False
            for i in range(path_length - 2):
                # check triplets
                tr_1 = path_list[i]
                tr_2 = path_list[i + 1]
                tr_3 = path_list[i + 2]

                if tr_2 in net.get_children(tr_1) and tr_3 in net.get_children(tr_2):
                    # causal chain
                    if tr_2 in Z:
                        print('causal chain: d-separated')
                        d_separated = True
                        break

                if tr_1 in net.get_children(tr_2) and tr_3 in net.get_children(tr_2):
                    # common cause
                    if tr_2 in Z:
                        print('common cause: d-separated')
                        d_separated = True
                        break

                if tr_2 in net.get_children(tr_1) and tr_2 in net.get_children(tr_3):
                    # common effect
                    if tr_2 not in Z and not \
                            nx.descendants(net.structure, tr_2).intersection(Z):
                        # check also descendents of middle Node
                        print('common effect: d-separated')
                        d_separated = True
                        break

            if d_separated is False:
                print('Causal Path found')
                return False

        print('All possible paths are d-separated')
        return True

    def min_degree_heuristic(self, X: list):
        """
        Calculate the order in which the variables shall be eliminated.
        In that cast: order by amount of neighbors (lowest to highest amount)
        :param X: subset of Bayesian Network (Graph)
        :return:
        """
        graph = self.bn.get_interaction_graph()
        num_neighbours = {}
        for node in X:
            num_neighbours[node] = [graph.degree(node)]

        ordered_vars = []
        for k in sorted(num_neighbours, key=num_neighbours.get, reverse=False):
            ordered_vars.append(k)
        return ordered_vars

    def min_fill_heuristic(self, X):
        """
        Order the list of nodes by the amount of edges that has to be added
        between adjacent neighbours before removement
        :param X:
        :return:
        """
        graph = self.bn.get_interaction_graph()
        num_edges = {}
        for node in X:
            neighbors = graph.neighbors(node)
            combination_of_neighbors = itertools.combinations(neighbors, r=2)
            edges_to_add = 0
            for n_1, n_2 in combination_of_neighbors:
                if not graph.has_edge(n_1, n_2):
                    edges_to_add += 1
            num_edges[node] = edges_to_add

        ordered_vars = []
        for k in sorted(num_edges, key=num_edges.get, reverse=False):
            ordered_vars.append(k)
        return ordered_vars

    def prune_network(self, Q, E):
        node_pruned_network = self._node_pruning(Q, E)

    def _node_pruning(self, Q: list, E: list):
        """
        Given a list of evidence Q, perform node pruning
        :param Q: query , E: evidence
        :return:
        """
        net = deepcopy(self.bn)
        QE = Q + E

        leaf_nodes = [node for node in net.get_all_variables()
                      if len(net.get_children(node)) == 0]

        prunable_nodes = [node for node in leaf_nodes if node not in QE]

        pruned = [node for node in net.get_all_variables()
                  if node not in prunable_nodes]

        return pruned

    def _edge_pruning(self, Q: list, E: list):
        """
        Given a list of evidence Q, perform edge pruning
        :param Q: query, E: evidence
        :return:
        """
        net = deepcopy(self.bn)
        cpt = deepcopy(self.bn.get_all_cpts())

        for node in E:
            children = net.get_children(node)
            for child in children:
                net.structure.remove_edges_from([(node, child)])

        # TODO UPDATE CPT

        return net, cpt
