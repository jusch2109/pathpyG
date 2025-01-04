from pathpyG.visualisations.Project_JS.HotVisFunctions import *

import torch
import pathpyG as pp

def shortest_paths_path_data(path_data: pp.PathData):
    """
    Computes the shortest path distances between all pairs of nodes in a PathData object using a second-order model.

    Args:
        path_data (pp.PathData): The `PathData` object containing the nodes for which shortest paths are calculated.

    Returns:
        torch.Tensor: A matrix of shortest path distances between all node pairs, where `dist[i, j]` represents the 
                      shortest path distance from node `i` to node `j`.

    Example:
        >>> dist_matrix = shortest_paths_path_data(path_data)
        >>> print(dist_matrix)
        tensor([[0., 1., 2.],
                [1., 0., 3.],
                [2., 3., 0.]])
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create second order model
    mo_graph = pp.MultiOrderModel.from_PathData(path_data, max_order=2, cached=True)
    
    # Create distance matrix for all nodes: default value is 'inf'
    dist = torch.full((mo_graph.layers[1].n, mo_graph.layers[1].n), float('inf'), device=device)
    
    # Get distances of second order model
    mo_dist, _ = pp.algorithms.shortest_paths.shortest_paths_dijkstra(mo_graph.layers[2])

    # Iterate through values adjacency matrix
    for i in range(mo_dist.shape[0]):
        for j in range(mo_dist.shape[0]):
            # Get node ids
            node_i = mo_graph.layers[2].mapping.node_ids[i]
            node_j = mo_graph.layers[2].mapping.node_ids[j]
            if dist[path_data.mapping.to_idx(node_i[0]), path_data.mapping.to_idx(node_j[1])] > mo_dist[i, j] + 1:
                # Write distance into 'dist'
                dist[path_data.mapping.to_idx(node_i[0]), path_data.mapping.to_idx(node_j[1])] = mo_dist[i, j] + 1
    
    # Insert all distances of length 1
    for node in mo_graph.layers[2].nodes:
        dist[path_data.mapping.to_idx(node[0]), path_data.mapping.to_idx(node[1])] = 1

    # Fill diagonals with 0
    torch.Tensor.fill_diagonal_(dist, 0)

    return dist

def stress_loss(layout: torch.nn.Embedding | torch.Tensor, shortest_path_dist: torch.Tensor) -> float:
    """
    Computes the stress loss between the pairwise distances in a layout and the corresponding shortest path distances.
    It can be used to minimize the difference between the distances of points in a layout and the original distances (e.g., shortest path distances in a graph).

    Args:
        layout (torch.nn.Embedding or torch.Tensor): The layout of nodes, either as an `Embedding` or a 2D tensor.
        dist (torch.Tensor): A tensor of shortest path distances between node pairs in the original graph.

    Returns:
        float: The computed stress loss value.

    Example:
        >>> layout = torch.rand((5, 2))  # Example layout with 5 nodes in 2D space
        >>> dist = torch.rand((5, 5))  # Example shortest path distance matrix
        >>> loss = stress_loss(layout, dist)
        >>> print(loss)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss = 0

    if isinstance(layout, torch.nn.Embedding):
        for i in range(layout.num_embeddings):
            for j in range(layout.num_embeddings):
                if j != i:
                    delta = layout(torch.tensor(i, device=device)) - layout(torch.tensor(j, device=device))
                    distance = torch.norm(delta)
                    loss += ((distance - shortest_path_dist[i, j]) / shortest_path_dist[i, j]) ** 2

    elif isinstance(layout, torch.Tensor):
        for i in range(layout.shape[0]):
            for j in range(layout.shape[0]):
                if j != i:
                    delta = layout[i] - layout[j]
                    distance = torch.norm(delta)
                    loss += ((distance - shortest_path_dist[i, j]) / shortest_path_dist[i, j]) ** 2  
    else:
        return None

    return loss / 2



import torch
import pathpyG as pp

def SGD_stress_torch(data: pp.TemporalGraph | pp.PathData, iterations: int, delta: int = 1, learning_rate: float = 0.01, initial_positions: torch.Tensor | None = None) -> tuple[dict, bool]:
    """
    Performs stress minimization using stochastic gradient descent (SGD) to optimize the layout of nodes in a graph or path data.

    This method aims to learn a 2D layout for the nodes, such that the pairwise distances between nodes in the layout 
    are as close as possible to the given shortest path distances. It uses a stress function to measure the difference 
    between the layout distances and the original shortest path distances.

    Args:
        data (pp.TemporalGraph or pp.PathData): The input graph or path data to be used for layout optimization.
        iterations (int): The number of iterations for the optimization process.
        delta (int, optional): The temporal window size for paths in temporal graphs. Not considered for Pathdata objects. Default is 1.
        learning_rate (float, optional): The learning rate for the SGD optimizer. Default is 0.01.
        initial_positions (torch.Tensor, optional): The initial 2D positions of the nodes for layout optimization. If None, random positions are used.

    Returns:
        dict: A dictionary with node identifiers as keys and their corresponding 2D layout coordinates as values.
        bool: True, if layout is not random

    Example:
        >>> data = pp.TemporalGraph(...)  # A temporal graph object
        >>> layout = SGD_stress_torch(data, iterations=1000)
        >>> print(layout)
        {'node1': [x1, y1], 'node2': [x2, y2], ...}
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dependent on input type, get graph for which we need the layout and distance matrix
    if isinstance(data, pp.TemporalGraph):
        graph = data
        dist, _ = pp.algorithms.temporal_shortest_paths(graph, delta)
    elif isinstance(data, pp.PathData):
        graph = pp.MultiOrderModel.from_PathData(data, max_order=1).layers[1]
        dist = shortest_paths_path_data(path_data=data)
    else:
        return {}, False
    
    dist = torch.tensor(dist, device=device)
    
    if(torch.isinf(dist).any()):
        print("Error: The graph or PathData isn't connected.")
        positions = torch.rand((graph.n, 2), device=device) * 100
        layout = {}
        for node in graph.nodes:
            layout[node] = positions[graph.mapping.to_idx(node)].tolist()
        return layout, False

    # Initialize embedding
    num_nodes = graph.n
    embedding_dim = 2
    embedding = torch.nn.Embedding(num_nodes, embedding_dim).to(device)

    # Initialize initial_positions
    if initial_positions is not None:
        with torch.no_grad():
            embedding.weight = torch.nn.Parameter(initial_positions.to(device))
    else:
        initial_positions = torch.rand((graph.n, 2), device=device) * 100
        embedding.weight = torch.nn.Parameter(initial_positions)

    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.SGD(embedding.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999999)

    # Training loop
    for i in range(iterations):
        loss = stress_loss(embedding, dist)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Show loss
        if (i + 1) % 10 == 0:
            print(f"Epoch [{i + 1}/{iterations}], Loss: {loss.item():.4f}")

    # Create layout out of embedding
    layout = {}
    for node in graph.nodes:
        layout[node] = embedding(torch.tensor(graph.mapping.to_idx(node), device=device)).tolist()

    return layout, True


def Adam_stress_torch(data: pp.TemporalGraph | pp.PathData, iterations: int, delta: int = 1, learning_rate: float = 0.01, initial_positions: torch.Tensor | None = None):
    """
    Performs stress minimization using the Adam optimizer to optimize the layout of nodes in a graph or path data.

    This method aims to learn a 2D layout for the nodes such that the pairwise distances between nodes in the layout
    are as close as possible to the given shortest path distances. It uses a stress function to measure the difference
    between the layout distances and the original shortest path distances. The Adam optimizer is used for gradient-based 
    optimization of the layout.

    Args:
        data (pp.TemporalGraph or pp.PathData): The input graph or path data to be used for layout optimization.
        iterations (int): The number of iterations for the optimization process.
        delta (int, optional): The temporal window size for paths in temporal graphs. Not considered for Pathdata objects. Default is 1.
        learning_rate (float, optional): The learning rate for the Adam optimizer. Default is 0.01.
        initial_positions (torch.Tensor, optional): The initial 2D positions of the nodes for layout optimization. If None, random positions are used.

    Returns:
        dict: A dictionary with node identifiers as keys and their corresponding 2D layout coordinates as values.
        bool: True, if layout is not random

    Example:
        >>> data = pp.TemporalGraph(...)  # A temporal graph object
        >>> layout = Adam_stress_torch(data, iterations=1000)
        >>> print(layout)
        {'node1': [x1, y1], 'node2': [x2, y2], ...}
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dependent on input type, get graph for which we need the layout and distance matrix
    if isinstance(data, pp.TemporalGraph):
        graph = data
        dist, _ = pp.algorithms.temporal_shortest_paths(graph, delta)
    elif isinstance(data, pp.PathData):
        graph = pp.MultiOrderModel.from_PathData(data, max_order=1).layers[1]
        dist = shortest_paths_path_data(path_data=data)
    else:
        return {}, False

    dist = torch.tensor(dist, device=device)
    
    if(torch.isinf(dist).any()):
        print("Error: The graph or PathData isn't connected.")
        positions = torch.rand((graph.n, 2), device=device) * 100
        layout = {}
        for node in graph.nodes:
            layout[node] = positions[graph.mapping.to_idx(node)].tolist()
        return layout, False

    # Initialize embedding
    num_nodes = graph.n
    embedding_dim = 2
    embedding = torch.nn.Embedding(num_nodes, embedding_dim).to(device)

    # Initialize initial_positions
    if initial_positions is not None:
        with torch.no_grad():
            embedding.weight = torch.nn.Parameter(initial_positions.to(device))
    else:
        initial_positions = torch.rand((graph.n, 2), device=device) * 100
        embedding.weight = torch.nn.Parameter(initial_positions)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(embedding.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999999)

    # Training loop
    for i in range(iterations):
        loss = stress_loss(embedding, dist)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Show loss
        if (i + 1) % 10 == 0:
            print(f"Epoch [{i + 1}/{iterations}], Loss: {loss.item():.4f}")

    # Create layout out of embedding
    layout = {}
    for node in graph.nodes:
        layout[node] = embedding(torch.tensor(graph.mapping.to_idx(node), device=device)).tolist()

    return layout, True

def SGD_stress_paper(data: pp.TemporalGraph | pp.PathData, iterations: int, delta: int = 1, initial_positions: torch.Tensor | None = None, learning_rate: float = 0.01, eta: float = 1, decay: float = 0.5) -> tuple[dict, bool]:
    """
    Performs stress minimization using Stochastic Gradient Descent (SGD) to optimize the layout of nodes in a graph or path data.

    This method aims to learn a 2D layout for the nodes such that the pairwise distances between nodes in the layout 
    are as close as possible to the given shortest path distances. It uses a stress function to measure the difference 
    between the layout distances and the original shortest path distances. The optimizer updates the node positions 
    iteratively based on the gradient of the stress function.

    The algorithm follows the paper: 
    Börsig, K., Brandes, U., Pasztor, B. (2020). Stochastic Gradient Descent Works Really Well for Stress Minimization. 
    In: Auber, D., Valtr, P. (eds) Graph Drawing and Network Visualization. GD 2020. Lecture Notes in Computer Science(), vol 12590. Springer, Cham. 
    https://doi.org/10.1007/978-3-030-68766-3_2

    Args:
        data (pp.TemporalGraph or pp.PathData): The input graph or path data to be used for layout optimization.
        iterations (int): The number of iterations for the optimization process.
        delta (int, optional): The temporal window size for temporal graphs. Not considered for Pathdata objects. Default is 1.
        initial_positions (torch.Tensor, optional): The initial 2D positions of the nodes for layout optimization. If None, random positions are used.
        learning_rate (float, optional): The learning rate for the SGD optimizer. Default is 0.01.
        eta (float, optional): A scaling factor that affects the step width, based on the shortest path distance. Default is 1.
        decay (float, optional): The decay rate for the learning rate over iterations. Default is 0.5.

    Returns:
        dict: A dictionary with node identifiers as keys and their corresponding 2D layout coordinates as values.
        bool: True, if layout is not random

    Example:
        >>> data = pp.TemporalGraph(...)  # A temporal graph object
        >>> layout = SGD_stress_paper(data, iterations=1000)
        >>> print(layout)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dependent on input type, get graph for which we need the layout and distance matrix
    if isinstance(data, pp.TemporalGraph):
        graph = data
        dist, _ = pp.algorithms.temporal_shortest_paths(graph, delta)
    elif isinstance(data, pp.PathData):
        graph = pp.MultiOrderModel.from_PathData(data, max_order=1).layers[1]
        dist = shortest_paths_path_data(path_data=data)
    else:
        return {}, False
    
    dist = torch.tensor(dist, device=device)
    
    if torch.isinf(dist).any():
        print("Error: The graph or PathData isn't connected.")
        positions = torch.rand((graph.n, 2), device=device) * 100
        layout = {}
        for node in graph.nodes:
            layout[node] = positions[graph.mapping.to_idx(node)].tolist()
        return layout, False
    
    # Initialize initial_positions if not given
    if initial_positions is None:
        initial_positions = torch.rand((graph.n, 2), device=device) * 100
    
    positions = torch.clone(initial_positions).to(device)

    # Get all possible node pairs
    node_pairs = torch.cartesian_prod(torch.arange(graph.n, device=device), torch.arange(graph.n, device=device))
    node_pairs = node_pairs[node_pairs[:, 0] != node_pairs[:, 1]]

    for i in range(iterations):
        # Shuffle order of node pairs
        shuffled_pairs = node_pairs[torch.randperm(node_pairs.size(0))]
        # Calculate step width (called eta in paper)
        step_width = eta * torch.exp(torch.tensor(-decay * i, device=device))

        # Iterate through node pairs
        for pair in shuffled_pairs:
            # Get distance between pairs
            shortest_path_dist = dist[pair[0], pair[1]]
            # Skip if nodes are the same
            if shortest_path_dist == 0:
                continue
            # Calculate learning rate
            learning_rate = min(1, ((1 / (shortest_path_dist ** 2)) * step_width)) / 2
            # Calculate distance of nodes in layout (not in graph)
            norm = torch.norm(positions[pair[0]] - positions[pair[1]])
            # Determine distance and direction the nodes are moved 
            delta = (norm - shortest_path_dist) / norm * (positions[pair[0]] - positions[pair[1]])
            positions[pair[0]] -= learning_rate * delta
            positions[pair[1]] += learning_rate * delta

        # Show loss
        if (i + 1) % 10 == 0:
            print(f"Epoch [{i + 1}/{iterations}], Stress: {stress_loss(positions, dist):.4f}")

    # Create layout 
    layout = {}
    for node in graph.nodes:
        layout[node] = positions[graph.mapping.to_idx(node)].tolist()

    return layout, True
