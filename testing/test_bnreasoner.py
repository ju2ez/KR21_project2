import BayesNet
import BNReasoner
import networkx as nx

bayes_net = BayesNet.BayesNet()
bayes_net.load_from_bifxml('./dog_problem.BIFXML')
reasoner = BNReasoner.BNReasoner(bayes_net)


# If you want to see the plots for verification, uncomment this.
# bayes_net.draw_structure()
# nx.draw(bayes_net.get_interaction_graph(), with_labels=True)

def test_d_separation():
    # direct causal chain
    assert reasoner.check_d_separation('bowel-problem', 'dog-out', ['family-out']) is False
    assert reasoner.check_d_separation('bowel-problem', 'dog-out', ['dog-out']) is True
    # causal chain
    assert reasoner.check_d_separation('bowel-problem', 'hear-bark', ['family-out']) is False
    assert reasoner.check_d_separation('bowel-problem', 'hear-bark', ['dog-out']) is True

    # common cause
    assert reasoner.check_d_separation('light-on', 'dog-out', []) is False
    assert reasoner.check_d_separation('light-on', 'dog-out', ['hear-bark']) is False
    assert reasoner.check_d_separation('light-on', 'dog-out', ['family-out']) is True

    # common effect
    assert reasoner.check_d_separation('bowel-problem', 'family-out', []) is True
    assert reasoner.check_d_separation('bowel-problem', 'family-out', ['hear-bark']) is False


def test_min_degree_heuristic():
    """
    Test the min degree heuristic
    """
    X = ['hear-bark', 'dog-out', 'bowel-problem']
    X_2 = ['dog-out', 'bowel-problem', 'hear-bark']
    assert reasoner.min_degree_heuristic(X) == ['hear-bark', 'bowel-problem', 'dog-out']
    assert reasoner.min_degree_heuristic(X_2) == ['hear-bark', 'bowel-problem', 'dog-out']


def test_min_fill_heuristic():
    """
    Test the min fill heuristic
    """
    X_1 = ['light-on', 'family-out', 'bowel-problem']
    X_2 = ['dog-out', 'bowel-problem', 'hear-bark']

    exp_X_1 = ['light-on', 'bowel-problem', 'family-out']
    exp_X_2 = ['bowel-problem', 'hear-bark', 'dog-out']

    assert reasoner.min_fill_heuristic(X_1) == exp_X_1
    assert reasoner.min_fill_heuristic(X_2) == exp_X_2


def test_node_pruning():
    """
    Test node pruning
    """
    Q = ['light-on']
    exp_net = ['bowel-problem', 'family-out', 'dog-out', 'light-on']
    assert sorted(exp_net) == sorted(reasoner._node_pruning(Q))

