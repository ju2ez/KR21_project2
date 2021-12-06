from typing import Union
from BayesNet import BayesNet
import networkx as nx


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

    def min_degree_heuristic(self):
        pass
