import pathpyG as pp
from pathpyG.visualisations.Project_JS.SGDStressFunctions import *

def test_shortest_paths_path_data():
    g = pp.Graph.from_edge_list([('a', 'b'), ('b','c'), ('b','d')])
    test_path_data = pp.PathData(g.mapping)
    test_path_data.append_walk(('a', 'b', 'c'))
    test_path_data.append_walk(('b', 'd'))
    test_path_data.append_walk(('b', 'a'))

    dist = shortest_paths_path_data(test_path_data)
    assert dist[0, 0] == 0
    assert dist[0, 1] == 1
    assert dist[0, 2] == 2
    assert dist[0, 3] == float('inf')
    assert dist[1, 0] == 1
    assert dist[1, 1] == 0
    assert dist[1, 2] == 1
    assert dist[1, 3] == 1
    assert dist[2, 0] == float('inf')
    assert dist[2, 1] == float('inf')
    assert dist[2, 2] == 0
    assert dist[2, 3] == float('inf')
    assert dist[3, 0] == float('inf')
    assert dist[3, 1] == float('inf')
    assert dist[3, 2] == float('inf')
    assert dist[3, 3] == 0

def test_stress_loss():
    s_l = stress_loss(torch.Tensor([[0, 1], [1, 1]]), torch.Tensor([[0, 3], [3, 0]]))
    assert round(s_l.item(), 2) == round(4/9, 2)

