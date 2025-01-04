import torch
import torch_geometric
import pathpyG as pp
from typing import Iterable, Union, Any, Optional
import time


def HotVis(
    data: pp.TemporalGraph| pp.PathData, 
    orders: int, 
    iterations: int, 
    delta: int, 
    alpha: torch.Tensor = None, 
    initial_positions: torch.Tensor = None, 
    force: int = 1
) -> dict:
    """
    Generates a layout for visualizing a temporal graph or path data using a force-directed model. (GPU compatible)

    Args:
        data: TemporalGraph or PathData for which the layout is to be created.
        orders: Number of higher orders to consider.
        iterations: Number of iterations for the optimization.
        delta: Time window for paths in the temporal graph.
        alpha: Tensor of weights for each order (optional).
        initial_positions: Initial positions of nodes (optional).
        force: Controls the repulsive and attractive forces (default is 1).

    Returns:
        dict: Dictionary mapping nodes to their 2D positions in the layout.
    """
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(data, pp.TemporalGraph):
        mo_model = pp.MultiOrderModel.from_temporal_graph(data, delta=delta, max_order=orders)
    elif isinstance(data, pp.PathData):
        mo_model = pp.MultiOrderModel.from_PathData(data, max_order=orders)
    else:
        raise ValueError("Input data must be of type `pp.TemporalGraph` or `pp.PathData`.")

    # Initialize alpha and initial positions
    alpha = alpha.to(device) if alpha is not None else torch.ones(orders, device=device)
    initial_positions = (initial_positions.to(device) 
                         if initial_positions is not None 
                         else torch.rand((mo_model.layers[1].n, 2), device=device) * 100)
    
    # Adjacency matrix on device
    A = torch.zeros((mo_model.layers[1].n, mo_model.layers[1].n), device=device)


    # Iterate over higher orders
    for i in range(orders):
        ho_graph = mo_model.layers[i + 1]
        edge_index = torch.tensor(ho_graph.data.edge_index, device=device)
        node_sequence = torch.tensor(ho_graph.data.node_sequence, device=device)
        # Get start and end nodes of higher-order edges
        nodes_start = torch.tensor(node_sequence[:, 0][edge_index[0]], device=device)
        nodes_end =  torch.tensor(node_sequence[:, -1][edge_index[1]], device=device)
        indices = torch.stack((nodes_start, nodes_end), dim=0).to(device)
        # Edge weights
        edge_weights = ho_graph['edge_weight'].to(device)
        indices, edge_weights = torch_geometric.utils.coalesce(indices, edge_weights)
        A[indices[0], indices[1]] += alpha[i] * edge_weights

    # Position update
    positions = initial_positions
    t = 0.1
    dt = t / float(iterations + 1)

    for _ in pp.tqdm(range(iterations)):
        # Difference between points
        delta = positions.unsqueeze(1) - positions.unsqueeze(0)

        # Distance and its inverse
        distance = torch.linalg.norm(delta, dim=-1)
        torch.clip(distance, 0.01, None, out=distance)

        # Displacement
        displacement = torch.einsum('ijk,ij->ik', delta,
                                    (A * distance / force - force**2 / distance**2))

        # Normalize displacement length
        length = torch.linalg.norm(displacement, dim=-1)
        length = torch.where(length < 0.01, 0.1, length)
        length_with_temp = torch.clamp(length, max=t)

        # Update positions
        delta_positions = displacement * (length_with_temp / length).unsqueeze(-1)
        positions += delta_positions

        # Cool temperature
        t -= dt

    # Create layout dictionary
    layout = {node: positions[mo_model.layers[1].mapping.to_idx(node)].tolist() 
              for node in mo_model.layers[1].nodes}

    return layout


def barycentre(layout: dict, nodes=None):
    """
    Computes the barycentre (geometric center) of a set of nodes in a given layout. (GPU compatible)

    Args:
        layout (dict): A dictionary mapping nodes to their 2D positions. 
            Keys are node identifiers, and values are lists or tensors representing [x, y] coordinates.
        nodes (list, optional): A list of specific nodes for which the barycentre should be calculated. 
            If None, the barycentre is computed for all nodes in the layout.

    Returns:
        torch.Tensor: A tensor representing the [x, y] coordinates of the barycentre.
    """
    # Select device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if nodes is None:
        node_positions = torch.tensor(list(layout.values()), device=device, dtype=torch.float64)
    else:
        node_positions = torch.tensor([layout[node] for node in nodes], device=device, dtype=torch.float64)
    
    return torch.mean(node_positions, dim=0)


def causal_path_dispersion(data: pp.TemporalGraph|pp.PathData, layout: dict, delta: int = 1, steps: list = [], runs: list = []):
    """
    Computes the causal path dispersion, a measure of the spatial variability of paths in a graph layout.

    Args:
        data (pp.TemporalGraph | pp.PathData): The input data, either a temporal graph or path data.
        layout (dict): A dictionary mapping nodes to their 2D positions. 
            Keys are node identifiers, and values are lists or tensors representing [x, y] coordinates.
        delta (int): The time window parameter for paths in temporal graphs. Default is 1.
        steps (list, optional): A list of path lengths for random walks on the temporal graph. Not considered for PathData objects.
            Defaults to `[max(3, int(data.n/3))]` if not provided.
        runs (list, optional): A list of the number of random walk runs to perform. Not considered for PathData objects.
            Defaults to `[int(data.n/2)]` if not provided.

    Returns:
        float: The causal path dispersion value, a ratio of spatial variability of paths to overall variability in the layout.

    Example:
        >>> dispersion = causal_path_dispersion(data=temporal_graph, layout=layout, delta=2)
        >>> print(dispersion)
        0.85
    """

    # Device selection (use CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(data, pp.TemporalGraph):
        if len(steps) == 0:
            steps = [max(3, int(data.n/3))]
        if len(runs) == 0:
            runs = [int(data.n/2)]
        paths = random_walk_temporal_graph(data, delta=delta, steps=steps, runs=runs)
    elif isinstance(data, pp.PathData):
        paths = data

    # Initialize variables for computation
    numerator = 0
    multiplicator = 0

    # Compute the numerator and multiplicator
    for i in range(paths.num_paths):
        path = paths.get_walk(i)
        position_nodes = torch.tensor([layout[node] for node in path], device=device)  # Positions on the device
        numerator += torch.sum(torch.norm(position_nodes - barycentre(layout, path).to(device), dim=1))  # Barycentre on the device
        multiplicator += len(path)

    numerator *= len(layout)

    # Calculate the denominator
    positions = torch.tensor(list(layout.values()), device=device)  # Positions on the device
    denominator = torch.sum(torch.norm(positions - barycentre(layout).to(device), dim=1)) * multiplicator  # Barycentre on the device

    return numerator / denominator



def closeness_centrality_paths(paths: pp.PathData):
    """
    Computes the closeness centrality for nodes based on paths in the provided path data. (GPU compatible)

    Args:
        paths (PathData): The PathData object the closeness centralities should be based on.

    Returns:
        dict: A dictionary mapping nodes to their closeness centrality values.

    Example:
        >>> closeness = closeness_centrality_paths(paths)
        >>> print(closeness)
        {'node1': 0.75, 'node2': 0.65, ...}
    """

    # Device selection (use CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Idea: construct two nxn-matrices, where each entry [i,j] is the numerator/denominator 
    # of the summand for node j for cc of node i
    num_nodes = paths.mapping.num_ids()
    num_paths = paths.num_paths

    # Initialize numerator and denominator on the selected device (GPU or CPU)
    numerator = torch.zeros(num_nodes, num_nodes, dtype=torch.float32, device=device)
    denominator = torch.zeros(num_nodes, num_nodes, dtype=torch.float32, device=device)

    # Go through all paths
    for i in range(num_paths):
        path = paths.get_walk(i)
        path_indices = torch.tensor(paths.mapping.to_idxs(path), device=device)
        
        # Get distances between all nodes (first row is distances between node path[0] and all others and so on)
        distances = torch.abs(torch.arange(len(path), device=device).unsqueeze(0) - torch.arange(len(path), device=device).unsqueeze(1))

        # Update numerator and denominator
        numerator[path_indices.unsqueeze(1), path_indices] += 1
        denominator[path_indices.unsqueeze(1), path_indices] += distances

    # Calculate Closeness Centrality
    mask = denominator != 0
    closeness = torch.sum(torch.where(mask, numerator / denominator, torch.zeros_like(denominator)), dim=1)

    # Mapping closeness values to nodes
    closeness_dict = {id: closeness[paths.mapping.to_idx(id)].item() for id in paths.mapping.node_ids}

    return closeness_dict



def closeness_eccentricity(data: pp.TemporalGraph|pp.PathData, layout: dict, delta: int = 1, percentile: float = 0.1, steps: list = [], runs: list = []):
    """
    Computes the closeness eccentricity, a measure of how central nodes with high closeness centrality 
    are positioned relative to the overall layout.

    Args:
        data (pp.TemporalGraph | pp.PathData): The input data, either a temporal graph or path data.
        layout (dict): A dictionary mapping nodes to their 2D positions.
        delta (int): The time window parameter for paths in temporal graphs. Not considered if data is a PathData object. Default is 1.
        percentile (float): The upper percentile of nodes based on closeness centrality to include in the calculation. Default is 0.1.
        steps (list, optional): A list of step lengths for random walks on the temporal graph. Not considered if data is a PathData object. 
            Defaults to `[max(3, int(data.n/3))]` if not provided.
        runs (list, optional): A list of the number of random walk runs to perform. Not considered if data is a PathData object.
            Defaults to `[int(data.n/2)]` if not provided.

    Returns:
        float: The closeness eccentricity value, a ratio comparing the centrality of high-centrality nodes 
        to the centrality of all nodes in the layout.


    Example:
        >>> eccentricity = closeness_eccentricity(data=temporal_graph, layout=layout, percentile=0.1)
        >>> print(eccentricity)
        0.85
    """

    # Device selection (use CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get closeness centrality of all nodes
    if isinstance(data, pp.TemporalGraph):
        if len(steps) == 0:
            steps = [max(3, int(data.n/3))]
        if len(runs) == 0:
            runs = [int(data.n/2)]
        paths = random_walk_temporal_graph(data, delta=delta, steps=steps, runs=runs)
        closeness_centrality = closeness_centrality_paths(paths)
    elif isinstance(data, pp.PathData):
        closeness_centrality = closeness_centrality_paths(data)
    else:
        return

    # Convert closeness values to tensor on device
    closeness_values = torch.tensor(list(closeness_centrality.values()), dtype=torch.float32, device=device)

    # Determine threshold for upper percentile
    threshold = torch.quantile(closeness_values, 1 - percentile)

    # Filter nodes based on threshold
    keys = list(closeness_centrality.keys())
    percentile_keys = [keys[i] for i in torch.where(closeness_values >= threshold)[0]]
    layout_percentile_nodes = torch.tensor([layout[key] for key in percentile_keys], device=device)

    # Determine barycenter
    barycenter_layout = barycentre(layout).to(device)

    # Determine numerator and denominator of formula for closeness_eccentricity
    numerator = torch.sum(torch.norm(layout_percentile_nodes - barycenter_layout, dim=1)) * len(layout)
    all_layout_values = torch.tensor(list(layout.values()), device=device)
    denominator = torch.sum(torch.norm(all_layout_values - barycenter_layout, dim=1)) * len(percentile_keys)

    return numerator / denominator


def within_bounds(min_x, max_x, min_y, max_y, intersection_coordinates):
    """
    Checks if the intersection points are within the bounds of the edges.

    Args:
        min_x (torch.Tensor): Minimum x-coordinates of the intersecting edges.
        max_x (torch.Tensor): Maximum x-coordinates of the intersecting edges.
        min_y (torch.Tensor): Minimum y-coordinates of the intersecting edges.
        max_y (torch.Tensor): Maximum y-coordinates of the intersecting edges.
        intersection_coordinates (torch.Tensor): Tensor of intersection points as [x, y].

    Returns:
        torch.Tensor: Boolean tensor indicating whether each intersection point is within bounds.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return (
        (min_x <= intersection_coordinates[:, 0]) & (intersection_coordinates[:, 0] <= max_x) &
        (min_y <= intersection_coordinates[:, 1]) & (intersection_coordinates[:, 1] <= max_y)
    ).to(device)

# intersection point must not be endpoint of edge
def is_not_endpoint(coordinates, intersection_coordinates):
    """
    Verifies that intersection points are not endpoints of edges.

    Args:
        coordinates (torch.Tensor): Tensor of edge endpoints in the format [x1, y1, x2, y2].
        intersection_coordinates (torch.Tensor): Tensor of intersection points as [x, y].

    Returns:
        torch.Tensor: Boolean tensor indicating whether each intersection point is not an endpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return ~(
        ((intersection_coordinates[:, 0] == coordinates[0]) & (intersection_coordinates[:, 1] == coordinates[1])) |
        ((intersection_coordinates[:, 0] == coordinates[2]) & (intersection_coordinates[:, 1] == coordinates[3]))
    ).to(device)

def edge_crossing(data: pp.TemporalGraph | pp.PathData, layout: dict):
    """
    Counts the number of edge crossings in a graph layout.

    Args:
        data (pp.TemporalGraph | pp.PathData): Input graph, either a temporal graph or path data.
        layout (dict): A dictionary mapping nodes to their 2D positions.

    Returns:
        int: The total number of edge crossings in the layout.

    Example:
        >>> crossings = edge_crossing(temporal_graph, layout)
        >>> print(crossings)
        15
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get static graph
    if isinstance(data, pp.TemporalGraph):
        # get undirected (since direction doesn't matter) static graph
        static_graph = data.to_static_graph().to_undirected()
    elif isinstance(data, pp.PathData):
        static_graph = pp.MultiOrderModel.from_PathData(data, 1).layers[1]
    else:
        return
    
    # Get edges
    edges = list(static_graph.edges)
    # Remove second entry, since direction isn't important for edge crossing
    edges = list(set(tuple(sorted(edge)) for edge in edges))
    
    # Create tensor containing [x_0, y_0, x_1, y_1] for each edge
    edge_coordinates = torch.tensor([layout[key1] + layout[key2] for key1, key2 in edges], dtype=torch.float, device=device)

    counter = 0

    for edge in edges:
        intersection_coordinates = torch.zeros((edge_coordinates.shape[0], 2), device=device)
        intersections = torch.zeros((edge_coordinates.shape[0]), dtype=torch.bool, device=device)

        current_edge_coordinates = torch.tensor(layout[edge[0]] + layout[edge[1]], dtype=torch.float, device=device)
        current_edge_dx = current_edge_coordinates[0] - current_edge_coordinates[2]  # x1 - x2
        current_edge_dy = current_edge_coordinates[1] - current_edge_coordinates[3]  # y1 - y2

        # Determine denominator
        dx = edge_coordinates[:, 0] - edge_coordinates[:, 2]  # x1 - x2
        dy = edge_coordinates[:, 1] - edge_coordinates[:, 3]  # y1 - y2

        denominator = (current_edge_dx * dy) - (current_edge_dy * dx)

        mask = ~torch.isclose(denominator, torch.tensor(0.0, device=device))

        det1 = current_edge_coordinates[0] * current_edge_coordinates[3] - current_edge_coordinates[1] * current_edge_coordinates[2]
        det2 = edge_coordinates[:, 0] * edge_coordinates[:, 3] - edge_coordinates[:, 1] * edge_coordinates[:, 2]

        # Initialize intersection coordinates 
        x = torch.zeros_like(denominator, device=device)
        y = torch.zeros_like(denominator, device=device)

        nominator_x = det1 * dx - current_edge_dx * det2
        nominator_y = det1 * dy - current_edge_dy * det2
        x[mask] = nominator_x[mask] / denominator[mask]
        y[mask] = nominator_y[mask] / denominator[mask]
        intersection_coordinates = torch.stack((x, y), dim=-1)

        min_edges_x = torch.minimum(edge_coordinates[:, 0], edge_coordinates[:, 2])
        min_edges_y = torch.minimum(edge_coordinates[:, 1], edge_coordinates[:, 3])
        max_edges_x = torch.maximum(edge_coordinates[:, 0], edge_coordinates[:, 2])
        max_edges_y = torch.maximum(edge_coordinates[:, 1], edge_coordinates[:, 3])

        min_current_x = torch.minimum(current_edge_coordinates[0], current_edge_coordinates[2])
        min_current_y = torch.minimum(current_edge_coordinates[1], current_edge_coordinates[3])
        max_current_x = torch.maximum(current_edge_coordinates[0], current_edge_coordinates[2])
        max_current_y = torch.maximum(current_edge_coordinates[1], current_edge_coordinates[3])

        valid_intersections = within_bounds(min_edges_x, max_edges_x, min_edges_y, max_edges_y, intersection_coordinates) & \
                            within_bounds(min_current_x, max_current_x, min_current_y, max_current_y, intersection_coordinates)
        
        valid_intersections &= is_not_endpoint(edge_coordinates.T, intersection_coordinates) & \
                               is_not_endpoint(current_edge_coordinates, intersection_coordinates)

        counter += torch.sum(valid_intersections[mask])

    return counter / 2



def cluster_distance_ratio(graph: pp.TemporalGraph, cluster: list, layout: dict):
    """
    Computes the cluster distance ratio, a measure of how tightly nodes within a cluster 
    are positioned relative to their barycenter compared to all nodes in the graph.

    Args:
        graph (pp.TemporalGraph): The input temporal graph containing nodes and edges.
        cluster (list): A list of clusters, where each cluster is a list of node IDs.
        layout (dict): A dictionary mapping nodes to their 2D positions.

    Returns:
        torch.Tensor: A tensor containing the distance ratio for each cluster.

    Example:
        >>> cluster_distances = cluster_distance_ratio(graph, clusters, layout)
        >>> print(cluster_distances)
        tensor([0.85, 0.92, 1.10])
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    distance_clusters = torch.zeros(len(cluster), device=device)
    for idx, c in enumerate(cluster):
        barycentre_cluster = barycentre(layout, c)
        # Calculate distances for nodes in the cluster
        mean_distance_cluster = torch.mean(torch.stack([torch.norm(torch.tensor(layout[node], device=device)-barycentre_cluster) for node in c])) 
        # Calculate distances for all nodes in the graph
        mean_distance_all = torch.mean(torch.stack([torch.norm(torch.tensor(layout[node], device=device)-barycentre_cluster) for node in graph.nodes])) 
        distance_clusters[idx] = mean_distance_cluster / mean_distance_all
    
    return distance_clusters

def random_walk_temporal_graph(graph: pp.TemporalGraph, delta: int = 1, steps: list = [10], runs: list = [1]):
    """
    Performs random walks on a temporal graph using a higher-order model and returns the generated paths.

    Args:
        graph (pp.TemporalGraph): The input temporal graph on which the random walks are performed.
        delta (int, optional): The time window parameter used for the higher-order model. Default is 1.
        steps (list, optional): A list of step lengths for each random walk. Default is `[10]`.
        runs (list, optional): A list of the number of random walk runs to perform for each step length. Default is `[1]`.

    Returns:
        pp.PathData: A `PathData` object containing the paths generated by the random walks.

    Example:
        >>> paths = random_walk_temporal_graph(temporal_graph, delta=2, steps=[5, 10], runs=[2, 3])
    """

    # Create higher order model of order 2
    g_ho = pp.MultiOrderModel.from_temporal_graph(graph, delta=delta, max_order=2, cached=False).layers[2]
    # Get instance of RandomWalk
    rw = pp.processes.RandomWalk(g_ho, weight='edge_weight', restart_prob=0.0)
    # Get instance of PathData
    paths = pp.PathData(graph.mapping)
    
    for s, r in zip(steps, runs):
        # Create r paths on higher order model with s-1 steps
        current_steps_paths = rw.get_paths(rw.run_experiment(steps=s-1, runs=r))
        
        for idx in range(current_steps_paths.num_paths):
            # Get path
            current_path_ho = current_steps_paths.get_walk(idx)
            current_path = [current_path_ho[0][0], current_path_ho[0][1]]
            
            i = 1
            while i < len(current_path_ho) and (current_path_ho[i][0] == current_path[-1]):
                # Check if we stayed on the same node -> skip
                if not current_path_ho[i][0] == current_path_ho[i][1]:
                    # Append node
                    current_path.append(current_path_ho[i][1])
                i += 1
            # Append path to PathData object
            paths.append_walk(current_path)
            
    return paths





#################################################################################################################################

########################## Functions you don't need, but which I wrote in the process of the project ###########################

#################################################################################################################################

def HotVis_time(
    data: pp.TemporalGraph | pp.PathData,
    orders: int,
    iterations: int,
    delta: int,
    alpha: torch.Tensor = None,
    initial_positions: torch.Tensor = None,
    force: int = 1
) -> dict:
    """
    Generates a layout for visualizing a temporal graph or path data using a force-directed model. (GPU compatible)

    Args:
        data: TemporalGraph or PathData for which the layout is to be created.
        orders: Number of higher orders to consider.
        iterations: Number of iterations for the optimization.
        delta: Time window for paths in the temporal graph.
        alpha: Tensor of weights for each order (optional).
        initial_positions: Initial positions of nodes (optional).
        force: Controls the repulsive and attractive forces (default is 1).

    Returns:
        dict: Dictionary mapping nodes to their 2D positions in the layout.
    """
    def log_with_timing(message, start_time=None):
        """Logs a message with timing information."""
        current_time = time.time()
        if start_time is not None:
            elapsed_time = current_time - start_time
            print(f"{message} (Elapsed: {elapsed_time:.4f} seconds)")
        else:
            print(message)
        return current_time

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(data, pp.TemporalGraph):
        mo_model = pp.MultiOrderModel.from_temporal_graph(data, delta=delta, max_order=orders)
    elif isinstance(data, pp.PathData):
        mo_model = pp.MultiOrderModel.from_PathData(data, max_order=orders)
    else:
        raise ValueError("Input data must be of type `pp.TemporalGraph` or `pp.PathData`.")

    # Initialize alpha and initial positions
    alpha = alpha.to(device) if alpha is not None else torch.ones(orders, device=device)
    initial_positions = (initial_positions.to(device)
                         if initial_positions is not None
                         else torch.rand((mo_model.layers[1].n, 2), device=device) * 100)
    
    # Adjacency matrix on device
    A = torch.zeros((mo_model.layers[1].n, mo_model.layers[1].n), device=device)

    start_time = log_with_timing("Starting loop one")

    # Iterate over higher orders
    for i in range(orders):
        ho_graph = mo_model.layers[i + 1]
        start_time = log_with_timing("Get higher order model", start_time)
        edge_index = torch.tensor(ho_graph.data.edge_index, device=device)
        start_time = log_with_timing("Get edge indices", start_time)
        node_sequence = torch.tensor(ho_graph.data.node_sequence, device=device)
        start_time = log_with_timing("Get node sequence", start_time)
        nodes_start = torch.tensor(node_sequence[:, 0][edge_index[0]], device=device)
        start_time = log_with_timing("Get nodes start", start_time)
        nodes_end = torch.tensor(node_sequence[:, -1][edge_index[1]], device=device)
        start_time = log_with_timing("Get nodes end", start_time)
        indices = torch.stack((nodes_start, nodes_end), dim=0).to(device)
        start_time = log_with_timing("Get indices", start_time)
        edge_weights = ho_graph['edge_weight'].to(device)
        start_time = log_with_timing("Get edge weights", start_time)
        indices, edge_weights = torch_geometric.utils.coalesce(indices, edge_weights)
        start_time = log_with_timing("Coalesce", start_time)
        A[indices[0], indices[1]] += alpha[i] * edge_weights
        start_time = log_with_timing("Update A", start_time)

    positions = initial_positions
    t = 0.1
    dt = t / float(iterations + 1)

    start_time = log_with_timing("Starting second loop")

    for _ in pp.tqdm(range(iterations)):
        delta = positions.unsqueeze(1) - positions.unsqueeze(0)
        start_time = log_with_timing("Calculate delta", start_time)
        distance = torch.linalg.norm(delta, dim=-1)
        start_time = log_with_timing("Calculate distance", start_time)
        torch.clip(distance, 0.01, None, out=distance)
        start_time = log_with_timing("Clip", start_time)
        displacement = torch.einsum('ijk,ij->ik', delta,
                                    (A * distance / force - force**2 / distance**2))
        start_time = log_with_timing("Calculate displacement", start_time)
        length = torch.linalg.norm(displacement, dim=-1)
        start_time = log_with_timing("Get length", start_time)
        length = torch.where(length < 0.01, 0.1, length)
        start_time = log_with_timing("Cut length", start_time)
        length_with_temp = torch.clamp(length, max=t)
        start_time = log_with_timing("Include temp", start_time)
        delta_positions = displacement * (length_with_temp / length).unsqueeze(-1)
        start_time = log_with_timing("Get delta", start_time)
        positions += delta_positions
        start_time = log_with_timing("Update positions", start_time)
        t -= dt
        start_time = log_with_timing("Update t", start_time)

    layout = {node: positions[mo_model.layers[1].mapping.to_idx(node)].tolist()
              for node in mo_model.layers[1].nodes}
    start_time = log_with_timing("Create layout", start_time)

    return layout


# slow but easier to understand
def HotVisSlow(data: pp.TemporalGraph | pp.PathData, orders: int, iterations: int, delta: int, 
           alpha: torch.Tensor | None = None, initial_positions: torch.Tensor | None = None, force: int = 1) -> dict:
    
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
        initial_positions = torch.rand((mo_model.layers[1].n, 2))*100
    A = torch.zeros((mo_model.layers[1].n, mo_model.layers[1].n))

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
    displacement = torch.zeros((mo_model.layers[1].n, 2))
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
    
# version from paper. There is porobabily a mistake in it.
def causal_path_dispersion_paper_(data, layout, delta=1, steps: list = [], runs: list = []):
    if isinstance(data, pp.TemporalGraph):
        if len(steps) == 0:
            steps = [max(3, int(data.n/3))]
        if len(runs)==0:
            runs = [int(data.n/2)]
        paths = random_walk_temporal_graph(data, delta=delta, steps=steps, runs=runs)
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
    denominator = torch.sum(torch.norm( positions - barycentre(layout), dim=1)) * paths.num_paths
    return numerator/denominator

# doesn't work, because there isn't  necesseray a path from a predecessor of the predecessor of a node to the node.
def get_shortest_paths_as_pathdata_slow(graph, delta):
    dist, pred = pp.algorithms.temporal_shortest_paths(graph, delta)
    paths = pp.PathData(graph.mapping)
    for node_i in range(graph.n):
           for node_j in range(graph.n):
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

# doesn't work, because there isn't  necesseray a path from a predecessor of the predecessor of a node to the node.
def get_shortest_paths_as_pathdata(graph, delta):

    dist, pred = pp.algorithms.temporal_shortest_paths(graph, delta)
    dist = torch.tensor(dist)
    pred = torch.tensor(pred)

    paths = pp.PathData(graph.mapping)

    causal_paths = [[None]*graph.n] * graph.n

    idxs = torch.nonzero(dist == 1)
    for idx1, idx2 in idxs:
         causal_paths[idx1][idx2] = [graph.mapping.to_id(idx1)] + [graph.mapping.to_id(idx2)]
         paths.append_walk(causal_paths[idx1][idx2])
         
    for i in range(2,graph.n+1):
        idxs = torch.nonzero(dist == i)
        for idx1, idx2 in idxs:
            predecessor = pred[idx1, idx2]
            causal_paths[idx1][idx2] = causal_paths[idx1][predecessor] + [graph.mapping.to_id(idx2)]
            paths.append_walk(causal_paths[idx1][idx2])

    return paths

# slow version of closeness centrality
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

# slow version of edge crossing
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

# fast but memory expensive version of edge crossing. The results are not correct, since the intersection check at the end ins't correct.
# Didn't fixed it, since it is to expensive on memory anyways.
def edge_crossing_fast(data, layout):

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