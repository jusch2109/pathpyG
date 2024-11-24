import torch
import torch_geometric
import pathpyG as pp

def HotVis(data: pp.TemporalGraph|pp.PathData, orders: int, iterations: int, delta: int, 
           alpha: torch.Tensor | None = None, initial_positions: torch.Tensor | None = None, force: int = 1) -> torch.Tensor:
    
    if(isinstance(data, pp.TemporalGraph)):   
        mo_model = pp.MultiOrderModel.from_temporal_graph(data, delta=delta, max_order=orders)
    elif(isinstance(data, pp.PathData)):
        mo_model = pp.MultiOrderModel.from_PathData(data, max_order=orders)
    else:
        return
    

    if alpha is None:
        alpha = torch.ones(orders)
    if initial_positions is None:
        initial_positions = torch.rand((mo_model.layers[1].N, 2))*100
    A = torch.zeros((mo_model.layers[1].N, mo_model.layers[1].N))

    # iterate over higher orders
    for i in range(orders):
        # get higher order graph
        ho_graph = mo_model.layers[i+1]

        # for edge ((v_0, ..., v_{k-1}), (v_1,...,v_k)) get nodes v_0 and v_k
        nodes_start = ho_graph.data.node_sequence[:, 0][ho_graph.data.edge_index[0]]
        nodes_end = ho_graph.data.node_sequence[:, -1][ho_graph.data.edge_index[1]]
        # stack tensors for later use
        indices = torch.stack((nodes_start, nodes_end), dim=0)
        # get edge weights
        edge_weights = ho_graph['edge_weight']
        # remove duplicates while summing their weights up
        indices, edge_weights = torch_geometric.utils.coalesce(indices, edge_weights)
        # add weights to A
        A[indices[0], indices[1]] += alpha[i] * edge_weights

    positions = initial_positions
    t = 0.1
    dt = dt = t / float(iterations + 1)

    delta = torch.zeros((positions.shape[0], positions.shape[0], positions.shape[1]))
    # the inscrutable (but fast) version

    for _ in pp.tqdm(range(iterations)):
        # matrix of difference between points
        delta = positions[torch.newaxis, :, :] - positions[:, torch.newaxis, :]
        # distance between points
        distance = torch.linalg.norm(delta, dim=-1)
        # enforce minimum distance of 0.01
        torch.clip(distance, 0.01, None, out=distance)
        # calculate displacement of all nodes
        displacement = torch.einsum('ijk,ij->ik', delta,
                                (A * distance / force - force**2 / distance**2))
        # calculate length of displacements
        length = torch.linalg.norm(displacement, dim=-1)
        # enforce minimum length of 0.01
        length = torch.where(length < 0.01, 0.1, length)
        # add temperature
        length_with_temp = torch.clamp(length, max=t)
        # calculate the change of the postionions
        delta_positions = torch.einsum('ij,i->ij', displacement, length_with_temp / length)
        # update positions
        positions += delta_positions
        # cool temperature
        t -= dt

    layout = {}
    for node in mo_model.layers[1].nodes:
        layout[node] = positions[mo_model.layers[1].mapping.to_idx(node)].tolist()

    return layout

def HotVisSlow(data: pp.TemporalGraph | pp.PathData, orders: int, iterations: int, delta: int, 
           alpha: torch.Tensor | None = None, initial_positions: torch.Tensor | None = None, force: int = 1) -> torch.Tensor:
    
    t = 0.1  
    dt = t / float(iterations + 1)

    if(isinstance(data, pp.TemporalGraph)):   
        mo_model = pp.MultiOrderModel.from_temporal_graph(data, delta=delta, max_order=orders)
    elif(isinstance(data, pp.PathData)):
        mo_model = pp.MultiOrderModel.from_PathData(data, max_order=orders)
    else:
        return
    

    if alpha is None:
        alpha = torch.ones(orders)
    if initial_positions is None:
        initial_positions = torch.rand((mo_model.layers[1].N, 2))*100
    A = torch.zeros((mo_model.layers[1].N, mo_model.layers[1].N))

    for i in range(orders):
        ho_graph = mo_model.layers[i+1]
        # iterate over edges of higher order graph
        for edge in ho_graph.edges:
            # for edge ((v_0, ..., v_{k-1}), (v_1,...,v_k)) get nodes v_0 and v_k
            # for i == 0, edge has form (v_0, v_1)
            if(i == 0):
                node_start = edge[0]
                node_end = edge[1]
            # for i > 0, edge has form ((v_0, ..., v_{i-1}), (v_1,...,v_i))
            else:
                node_start = edge[0][0]
                node_end = edge[1][-1]

            # get indices of the nodes
            index_node_start = mo_model.layers[1].mapping.to_idx(node_start)
            index_node_end = mo_model.layers[1].mapping.to_idx(node_end)

            # add to A
            A[index_node_start, index_node_end] += alpha[i] * ho_graph['edge_weight', edge[0], edge[1]]
        
    positions = initial_positions

    # every nodes "movement" or displacement gets describet by an tuple (x, y) 
    displacement = torch.zeros((mo_model.layers[1].N, 2))
    for _ in range(iterations):
        # reset displacement
        displacement *= 0
        # loop over rows/nodes
        for i in range(A.shape[0]):
            # difference between this row's node position and all others
            delta = positions - positions[i]
            # distance between the nodes
            distance = torch.sqrt((delta**2).sum(dim=1))
            # enforce minimum distance of 0.01
            distance = torch.where(distance < 0.01, 0.01, distance)
            # calculate displacement of node i
            displacement[i] += (delta/distance.view(-1, 1) * (A[i] * distance**2 / force -  force**2 / distance).view(-1,1)).sum(dim=0)
        # get length of displacement
        length = torch.sqrt((displacement**2).sum(dim=1))
        # enforce minimum length of 0.01
        length = torch.where(length < 0.01, 0.1, length)
        # add temperature
        length_with_temp = torch.clamp(length, max=t)
        # update positions
        positions += displacement / length.view(-1, 1) * length_with_temp.view(-1, 1)
        # cool temperature
        t -= dt

    layout = {}
    for node in mo_model.layers[1].nodes:
        layout[node] = positions[mo_model.layers[1].mapping.to_idx(node)].tolist()

    return layout


def barycentre(layout, nodes=None):
    if nodes is None:
        node_positions = torch.tensor(list(layout.values())).to(torch.float64)
    else:
        node_positions = torch.tensor([layout[node] for node in nodes]).to(torch.float64)
    return torch.mean(node_positions, dim=0)

# causal_path_set über temporal_shortest_paths bekommen?
def causal_path_dispersion_paper(data, layout, delta=1):
    if isinstance(data, pp.TemporalGraph):
        paths = get_shortest_paths_as_pathdata(data, delta)
    elif isinstance(data, pp.PathData):
        paths = data
    else:
        return 0
    
    numerator = 0
    for i in range(paths.num_paths):
        path = paths.get_walk(i)
        # get positions of nodes of path
        position_nodes = torch.tensor([layout[node] for node in path])
        # Add the summand of the corresponding path to the counter
        numerator += torch.sum(torch.norm(position_nodes - barycentre(layout, path), dim=1))
    numerator *= len(layout)
    # calculate denominator
    positions = torch.tensor(list(layout.values()))
    denominator = torch.sum(torch.norm(positions - barycentre(layout), dim=1)) * paths.num_paths
    return numerator/denominator

def causal_path_dispersion(data, layout, delta=1):
    if isinstance(data, pp.TemporalGraph):
        paths = get_shortest_paths_as_pathdata(data, delta)
    elif isinstance(data, pp.PathData):
        paths = data
    else:
        return 0
    
    numerator = 0
    multiplicator = 0
    for i in range(paths.num_paths):
        path = paths.get_walk(i)
        # get positions of nodes of path
        position_nodes = torch.tensor([layout[node] for node in path])
        # Add the summand of the corresponding path to the counter
        numerator += torch.sum(torch.norm(position_nodes - barycentre(layout, path), dim=1))
        multiplicator += len(path)
    numerator *= len(layout)
    # calculate denominator
    positions = torch.tensor(list(layout.values()))
    denominator = torch.sum(torch.norm( positions - barycentre(layout), dim=1)) * multiplicator
    return numerator/denominator

def get_shortest_paths_as_pathdata_slow(graph, delta):
    dist, pred = pp.algorithms.temporal_shortest_paths(graph, delta)
    paths = pp.PathData(graph.mapping)
    for node_i in range(graph.N):
           for node_j in range(graph.N):
                if dist[node_i, node_j] > 0 and pred[node_i, node_j] != -1:
                    # initialize path
                    causal_path = [graph.mapping.to_id(node_j)]
                    current_node = node_j

                    # append predecessor\n",
                    while pred[node_i, current_node] != node_i:
                            current_node = pred[node_i, current_node]
                            causal_path.insert(0, graph.mapping.to_id(current_node))

                    # insert starting node
                    causal_path.insert(0, graph.mapping.to_id(node_i))
                    # add path to set of paths
                    paths.append_walk(causal_path)

    return paths

def get_shortest_paths_as_pathdata(graph, delta):

    dist, pred = pp.algorithms.temporal_shortest_paths(graph, delta)
    dist = torch.tensor(dist)
    pred = torch.tensor(pred)

    paths = pp.PathData(graph.mapping)

    causal_paths = [[None]*graph.N] * graph.N

    idxs = torch.nonzero(dist == 1)
    for idx1, idx2 in idxs:
         causal_paths[idx1][idx2] = [graph.mapping.to_id(idx1)] + [graph.mapping.to_id(idx2)]
         paths.append_walk(causal_paths[idx1][idx2])
         
    for i in range(2,graph.N+1):
        idxs = torch.nonzero(dist == i)
        for idx1, idx2 in idxs:
            predecessor = pred[idx1, idx2]
            causal_paths[idx1][idx2] = causal_paths[idx1][predecessor] + [graph.mapping.to_id(idx2)]
            paths.append_walk(causal_paths[idx1][idx2])

    return paths
    

def closeness_centrality_paths_slow(paths):
    ret_dict = {v: 0 for v in paths.mapping.node_ids}
    for v in paths.mapping.node_ids:
        for w in paths.mapping.node_ids:
            numerator = 0
            denominator = 0
            if w != v:
                for i in range(paths.num_paths):
                    path = paths.get_walk(i)
                    if v in path and w in path:
                        numerator += 1
                        denominator += abs(path.index(v)-path.index(w))
            if(numerator > 0):                
                ret_dict[v] += numerator/denominator
            else:
                ret_dict[v] += 0
    return ret_dict

def closeness_centrality_paths(paths):
    # Idea: construct two nxn-matrices, where each entry [i,j] is the numerator/denomintor 
    # of the summand for node j for cc of node i
    num_nodes = paths.mapping.num_ids()
    num_paths = paths.num_paths

    # initialize numerator and denumerator
    numerator = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    denominator = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

    # Go through all paths
    for i in range(num_paths):
        path = paths.get_walk(i)
        path_indices = torch.tensor(paths.mapping.to_idxs(path))
        # get distances between all nodes (first row is distances between node path[0] and all others and so on)
        distances = torch.abs(torch.arange(len(path)).unsqueeze(0) - torch.arange(len(path)).unsqueeze(1))

        # Update numerator and denominator
        numerator[path_indices.unsqueeze(1), path_indices] += 1
        denominator[path_indices.unsqueeze(1), path_indices] += distances

    # calculate Closeness Centrality
    mask = denominator != 0
    closeness = torch.sum(torch.where(mask, numerator / denominator, torch.zeros_like(denominator)), dim=1)

    closeness_dict  = {id: closeness[paths.mapping.to_idx(id)].item() for id in paths.mapping.node_ids}

    return closeness_dict


def closeness_eccentricity(data, layout, delta, percentile):
    # get closeness centrality of all nodes
    if isinstance(data, pp.TemporalGraph):
        #closeness_centrality = pp.algorithms.centrality.temporal_closeness_centrality(data, delta)
        paths = get_shortest_paths_as_pathdata(data, delta)
        closeness_centrality = closeness_centrality_paths(paths)
    elif isinstance(data, pp.PathData):
        closeness_centrality = closeness_centrality_paths(data)
    else:
        return
    closeness_values = torch.tensor(list(closeness_centrality.values()), dtype=torch.float32)

    # determine treshold for upper percentile
    threshold = torch.quantile(closeness_values, 1 - percentile)
    
    # filter nodes based on treshold
    keys = list(closeness_centrality.keys())
    percentile_keys = [keys[i] for i in torch.where(closeness_values >= threshold)[0]]
    layout_percentile_nodes = torch.tensor([layout[key] for key in percentile_keys])
    
    # determine barycenter
    barycenter_layout = barycentre(layout)
    
    # determine numerator and denominator of formula for closeness_eccentricity
    numerator = torch.sum(torch.norm(layout_percentile_nodes - barycenter_layout, dim=1)) * len(layout)
    all_layout_values = torch.tensor(list(layout.values()))
    denominator = torch.sum(torch.norm(all_layout_values - barycenter_layout, dim=1)) * len(percentile_keys)
    
    return numerator / denominator

def edge_crossing_slow(data, layout):
    # initialize counter
    counter = 0
    if isinstance(data, pp.TemporalGraph):
        # get undirected (since direction doesn't matter) static graph
        static_graph = data.to_static_graph().to_undirected()
    elif isinstance(data, pp.PathData):
        static_graph = pp.MultiOrderModel.from_PathData(data, 1).layers[1]
    else:
        return
    # get edges
    edges = list(static_graph.edges)
    # every edge {'a','b'} is contained two times (as ('a','b') and as ('b','a'))
    # remove second entry, since direction isn't important for edge crossing
    edges = list(set(tuple(sorted(edge)) for edge in edges))
    # for every pair of edges
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            
            # check if the edges intersect (if no two nodes are the same) -> if so, increase counter
            if edges[i][0] not in edges[j] and edges[i][1] not in edges[j] and edge_intersection(layout[edges[i][0]], layout[edges[i][1]],
                            layout[edges[j][0]], layout[edges[j][1]]):
                counter += 1

    return counter

def edge_intersection(A1, A2, B1, B2):
    # formula for intersection is
    # x = det1 * (x3 - x4) - (x1 - x2) * det2 / (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    # y = det1 * (y3 - y4) - (y1 - y2) * det2 / (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    # where
    # det1 = x1 * y2 - y1 * x2
    # det2 = x3 * y4 - y3 * x4

    # get coordinates
    x1, y1 = A1
    x2, y2 = A2
    x3, y3 = B1
    x4, y4 = B2

    # determine denomitator
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # check if edges are parallel (denominator = 0)
    if torch.isclose(torch.tensor(float(denominator)), torch.tensor(0.0)):
        return False

    # determine determinants for nummerator
    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4

    # determine intersection of lines going through A1 and A2 resp. B1 and B2
    x = (det1 * (x3 - x4) - (x1 - x2) * det2) / denominator
    y = (det1 * (y3 - y4) - (y1 - y2) * det2) / denominator
    intersection = torch.tensor([x, y])

    # check if intersection is on edges 
    if is_on_segment(A1, intersection, A2) and is_on_segment(B1, intersection, B2):
        return True
    else:
        return False 

# checks if intersectiion q lies on line from p to r
def is_on_segment(p, q, r):
    return (torch.min(torch.tensor([p[0], r[0]])) <= q[0] <= torch.max(torch.tensor([p[0], r[0]]))) and \
           (torch.min(torch.tensor([p[1], r[1]])) <= q[1] <= torch.max(torch.tensor([p[1], r[1]])))

def edge_crossing(data, layout):

    # get static graph
    if isinstance(data, pp.TemporalGraph):
        # get undirected (since direction doesn't matter) static graph
        static_graph = data.to_static_graph().to_undirected()
    elif isinstance(data, pp.PathData):
        static_graph = pp.MultiOrderModel.from_PathData(data, 1).layers[1]
    else:
        return
    # get edges
    edges = list(static_graph.edges)
    # every edge {'a','b'} is contained two times (as ('a','b') and as ('b','a'))
    # remove second entry, since direction isn't important for edge crossing
    edges = list(set(tuple(sorted(edge)) for edge in edges))
    # create tensor containing [x_0, y_0, x_1, y_1] for each edge, where [x_0, y_0] is start and [x_1, y_1]is endpoint of the edge
    edge_coordinates = torch.tensor([layout[key1] + layout[key2] for key1, key2 in edges], dtype=torch.float)

    # for edges [x1,y1,x2,y2] and [x3,y3,x4,y4] formula for intersection is
    # x = det1 * (x3 - x4) - (x1 - x2) * det2 / (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    # y = det1 * (y3 - y4) - (y1 - y2) * det2 / (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    # where
    # det1 = x1 * y2 - y1 * x2
    # det2 = x3 * y4 - y3 * x4

    # initialize matrix containing the intersection coordinates for every two edges or the lines defined by the edges (if existant)
    intersection_coordinates = torch.zeros((edge_coordinates.shape[0], edge_coordinates.shape[0],2))
    # initialize matrix containing bool if two edges intersect 
    intersections = torch.zeros((edge_coordinates.shape[0],edge_coordinates.shape[0]), dtype=torch.bool)

    # determine denomitator
    dx = edge_coordinates[:, 0] - edge_coordinates[:, 2]  # x1 - x2
    dy = edge_coordinates[:, 1] - edge_coordinates[:, 3]  # y1 - y2

    # denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    denominator = (dx.view(-1, 1) @ dy.view(1, -1)) - (dy.view(-1, 1) @ dx.view(1, -1))

    # denominator == 0 if edges are parallel and therefor the lines definied by the edges don't intersect -> build mask to ignore them
    mask = ~torch.isclose(denominator, torch.tensor(0.0))

    # determine determinant for nummerator, det= x1 * y2 - y1 * x2
    det = edge_coordinates[:,0] * edge_coordinates[:,3] - edge_coordinates[:,1] * edge_coordinates[:,2]

    # initialize intersection coordinates 
    x = torch.zeros_like(denominator)
    y = torch.zeros_like(denominator)

    # determine intersection of lines defined by edges
    # det_dx = (x1 * y2 - y1 * x2) * (x3 - x4)
    det_dx = dx.view(-1, 1) @ det.view(1, -1)
    # det_dx = (x1 * y2 - y1 * x2) * (y3 - y4)
    det_dy = dy.view(-1, 1) @ det.view(1, -1)
    # calculate coordinates of intersections
    x[mask] = (det_dx.T - det_dx)[mask] / denominator[mask]
    y[mask] = (det_dy.T - det_dy)[mask] / denominator[mask]
    intersection_coordinates = torch.stack((x, y), dim=-1)

    # check if intersections are on the linesegments (means on the edge)
    min_edges_x = torch.minimum(edge_coordinates[:, 0], edge_coordinates[:, 2])
    min_edges_y = torch.minimum(edge_coordinates[:, 1], edge_coordinates[:, 3])
    max_edges_x = torch.maximum(edge_coordinates[:, 0], edge_coordinates[:, 2])
    max_edges_y = torch.maximum(edge_coordinates[:, 1], edge_coordinates[:, 3])
    intersections = (min_edges_x.view(-1,1) < intersection_coordinates[:, :, 0]) & (intersection_coordinates[:, :, 0] < max_edges_x.view(-1,1)) & (min_edges_y.view(-1,1) < intersection_coordinates[:, :, 1]) & (intersection_coordinates[:, :, 1] < max_edges_y.view(-1,1))
    intersections &= (min_edges_x.view(1,-1) < intersection_coordinates[:, :, 0]) & (intersection_coordinates[:, :, 0] < max_edges_x.view(1,-1)) & (min_edges_y.view(1,-1) < intersection_coordinates[:, :, 1]) & (intersection_coordinates[:, :, 1] < max_edges_y.view(1,-1))
    
    counter = torch.sum(intersections[mask])/2

    return counter


def cluster_distance_ratio(graph: pp.TemporalGraph, cluster, layout):
    distance_clusters = torch.zeros(len(cluster))
    for idx, c in enumerate(cluster):
        barycentre_cluster = barycentre(layout, c)
        mean_distance_cluster = torch.mean(torch.stack([torch.norm(torch.tensor(layout[node])-barycentre_cluster) for node in c])) 
        mean_distance_all = torch.mean(torch.stack([torch.norm(torch.tensor(layout[node])-barycentre_cluster) for node in graph.nodes])) 
        distance_clusters[idx] = mean_distance_cluster / mean_distance_all
    
    return distance_clusters