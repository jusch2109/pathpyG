from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    Tuple,
    List,
    Union,
    Any,
    Optional,
    Generator,
)

import torch

import torch_geometric
import torch_geometric.utils
from torch_geometric import EdgeIndex
from torch_geometric.data import Data
from torch_geometric.transforms.to_undirected import ToUndirected
from torch_geometric.utils import scatter

from pathpyG.utils.config import config
from pathpyG.core.index_map import IndexMap


class Graph:
    """
    A graph object storing nodes, edges, and attributes.

    An object than be be used to store directed or undirected graphs with node
    and edge attributes. Data on nodes and edges are stored in an underlying instance of
    [`torch_geometric.Data`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data).
    """

    def __init__(self, data: Data, mapping: Optional[IndexMap] = None):
        """Generate graph instance from a pyG `Data` object.

        Generate a Graph instance from a `torch_geometric.Data` object that contains an EdgeIndex as well as
        optional node-, edge- or graph-level attributes. An optional mapping can be used to transparently map
        node indices to string identifiers.

        Args:
            data: A pyG Data object containing an EdgeIndex and additional attributes
            mapping: `IndexMap` object that maps node indices to string identifiers

        Example:
            ```py
            import pathpyG as pp
            from torch_geometric.data import Data
            from torch_geometric import EdgeIndex

            data = Data(edge_index=EdgeIndex([[1,1,2],[0,2,1]], sparse_size=(3,3)))
            g = pp.Graph(data)

            g = pp.Graph(data, mapping=pp.IndexMap(['a', 'b', 'c']))
            ```
        """
        if mapping is None:
            self.mapping = IndexMap()
        else:
            self.mapping = mapping

        # set num_nodes property
        if "num_nodes" not in data:
            data.num_nodes = data.edge_index.max().item() + 1

        # turn edge index tensor into EdgeIndex object
        if not isinstance(data.edge_index, EdgeIndex):
            data.edge_index = EdgeIndex(data=data.edge_index, sparse_size=(data.num_nodes, data.num_nodes))

        if (
            data.edge_index.get_sparse_size(dim=0) != data.num_nodes
            or data.edge_index.get_sparse_size(dim=1) != data.num_nodes
        ):
            raise Exception("sparse size of EdgeIndex should match number of nodes!")

        # sort EdgeIndex and validate
        data.edge_index = data.edge_index.sort_by("row").values
        data.edge_index.validate()

        self.data = data

        # create mapping between edge tuples and edge indices
        self.edge_to_index = {
            (e[0].item(), e[1].item()): i for i, e in enumerate([e for e in self.data.edge_index.t()])
        }

        ((self.row_ptr, self.col), _) = self.data.edge_index.get_csr()
        ((self.col_ptr, self.row), _) = self.data.edge_index.get_csc()

    @staticmethod
    def from_edge_index(edge_index: torch.Tensor, mapping: Optional[IndexMap] = None, num_nodes: int = None) -> Graph:
        """Construct a graph from a torch Tensor containing an edge index. An optional mapping can
        be used to transparently map node indices to string identifiers.

        Args:
            edge_index:  torch.Tensor or torch_geometric.EdgeIndex object containing an edge_index
            mapping: `IndexMap` object that maps node indices to string identifiers
            num_nodes: optional number of nodes (default: None). If None, the number of nodes will be
                inferred based on the maximum node index in the edge index

        Examples:
            You can create a graph from an edge index tensor as follows:

            >>> import torch
            >>> import pathpyG as pp
            >>> g = pp.Graph.from_edge_index(torch.LongTensor([[1, 1, 2], [0, 2, 1]]))
            >>> print(g)
            Directed graph with 3 nodes and 3 edges ...

            You can also include a mapping of node IDs:

            >>> g = pp.Graph.from_edge_index(torch.LongTensor([[1, 1, 2], [0, 2, 1]]),
            >>>                              mapping=pp.IndexMap(['a', 'b', 'c']))
            >>> print(g.mapping)
            a -> 0
            b -> 1
            c -> 2
        """

        if not num_nodes:
            d = Data(edge_index=edge_index)
        else:
            d = Data(edge_index=edge_index, num_nodes=num_nodes)
        return Graph(d, mapping=mapping)

    @staticmethod
    def from_edge_list(
        edge_list: Iterable[Tuple[str, str]],
        is_undirected: bool = False,
        mapping: Optional[IndexMap] = None,
        num_nodes: Optional[int] = None,
    ) -> Graph:
        """Generate a Graph based on an edge list.

        Edges can be given as string or integer tuples. If strings are used and no mapping is given,
        a mapping of node IDs to indices will be automatically created based on a lexicographic ordering of
        node IDs.

        Args:
            edge_list: Iterable of edges represented as tuples
            is_undirected: Whether the edge list contains all bidorectional edges
            mapping: optional mapping of string IDs to node indices
            num_nodes: optional number of nodes (useful in case not all nodes have incident edges)

        Examples:
            >>> import pathpyG as pp
            >>> l = [('a', 'b'), ('a', 'c'), ('b', 'c')]
            >>> g = pp.Graph.from_edge_list(l)
            >>> print(list(g.edges))
            [('a', 'b'), ('a', 'c'), ('b', 'c')]
        """

        if mapping is None:
            node_ids = set()
            for v, w in edge_list:
                node_ids.add(v)
                node_ids.add(w)
            numeric_ids = True
            for x in node_ids:
                if not x.isnumeric():
                    numeric_ids = False
            node_list = list(node_ids)
            if numeric_ids:  # sort numerically
                node_list.sort(key=int)
            else:  # sort lexicograpbically
                node_list.sort()
            mapping = IndexMap(node_list)

        sources = []
        targets = []
        for v, w in edge_list:
            sources.append(mapping.to_idx(v))
            targets.append(mapping.to_idx(w))

        if num_nodes is None:
            num_nodes = mapping.num_ids()

        edge_index = EdgeIndex(
            [sources, targets],
            sparse_size=(num_nodes, num_nodes),
            is_undirected=is_undirected,
            device=config["torch"]["device"],
        )
        return Graph(Data(edge_index=edge_index, num_nodes=num_nodes), mapping=mapping)

    @staticmethod
    def from_csv(
        filename: str, sep: str = "", header: bool = True, is_undirected: bool = False, multiedges: bool = False
    ) -> Graph:
        """Read temporal graph from csv file, using pandas module

        Args:
            filename: path to csv file
            sep: separator used in csv file
            header: whether csv file contains a header
            is_undirected: whether graph is undirected
            multiedges: whether graph contains multiple edges between the same nodes
        """
        from pathpyG.io.pandas import read_csv_graph

        return read_csv_graph(filename, sep=sep, header=header, is_undirected=is_undirected, multiedges=multiedges)

    def to_undirected(self) -> Graph:
        """
        Returns an undirected version of a directed graph.

        This method transforms the current graph instance into an undirected graph by
        adding all directed edges in opposite direction. It applies [`ToUndirected`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.ToUndirected.html#torch_geometric.transforms.ToUndirected)
        transform to the underlying [`torch_geometric.Data`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data) object, which automatically
        duplicates edge attributes for newly created directed edges.

        Examples:
            >>> import pathpyG as pp
            >>> g = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('c', 'a')])
            >>> g_u = g.to_undirected()
            >>> print(g_u)
            Undirected graph with 3 nodes and 6 (directed) edges
        """
        tf = ToUndirected()
        d = tf(self.data)
        # unfortunately, the application of a transform creates a new edge_index of type tensor
        # so we have to recreate the EdgeIndex tensor and sort it again

        e = EdgeIndex(data=d.edge_index, sparse_size=(self.data.num_nodes, self.data.num_nodes), is_undirected=True)
        d.edge_index = e
        d.num_nodes = self.data.num_nodes
        return Graph(d, self.mapping)

    def to_weighted_graph(self) -> Graph:
        """Coalesces multi-edges to single-edges with an additional weight attribute

        If the graph contains multiple edges between the same nodes, this method will coalesce
        them into a single edge with an additional weight attribute called `edge_weight` that
        contains the number of coalesced edges. The method returns a new graph instance with
        the coalesced edges.

        Returns:
            Graph: Graph with coalesced edges
        """
        i, w = torch_geometric.utils.coalesce(
            self.data.edge_index.as_tensor(), torch.ones(self.M, device=self.data.edge_index.device)
        )
        return Graph(Data(edge_index=i, edge_weight=w, num_nodes=self.data.num_nodes), mapping=self.mapping)

    @staticmethod
    def attr_types(attr: Dict) -> Dict:
        """
        Return name, type, and size of all node, edge, and graph attributes.

        This method returns a dictionary that contains the name (key), as well as
        the type and size of all attributes.

        Args:
            attr: dictionary of attributes

        Returns:
            dict: dictionary containing type and size of all attributes in `attr`
        """
        a = {}
        for k in attr:
            t = type(attr[k])
            if t == torch.Tensor:
                a[k] = str(t) + " -> " + str(attr[k].size())
            else:
                a[k] = str(t)
        return a

    def node_attrs(self) -> List:
        """
        Return a list of node attributes.

        This method returns a list containing the names of all node-level attributes,
        ignoring the special `node_id` attribute.

        Returns:
            list: list of node attributes
        """
        attrs = []
        for k in self.data.keys():
            if k != "node_id" and k.startswith("node_"):
                attrs.append(k)
        return attrs

    def edge_attrs(self) -> List:
        """
        Return a list of edge attributes.

        This method returns a list containing the names of all edge-level attributes,
        ignoring the special `edge_index` attribute.

        Returns:
            list: list of edge attributes
        """
        attrs = []
        for k in self.data.keys():
            if k != "edge_index" and k.startswith("edge_"):
                attrs.append(k)
        return attrs

    @property
    def nodes(self) -> Generator[Union[int, str], None, None]:
        """
        Return indices or IDs of all nodes in the graph.

        This method returns a generator object that yields all nodes.
        If an IndexMap is used, nodes
        are returned as str IDs. If no IndexMap is used, nodes
        are returned as integer indices.

        Returns:
            generator: generator object yielding all nodes using IDs or indices (if no mapping is used)
        """
        for i in range(self.N):
            yield self.mapping.to_id(i)

    @property
    def edges(self) -> Generator[Union[Tuple[int, int], Tuple[str, str]], None, None]:
        """Return all edges in the graph.

        This method returns a generator object that yields all edges.
        If an IndexMap is used to map node indices to string IDs, edges
        are returned as tuples of str IDs. If no mapping is used, edges
        are returned as tuples of integer indices.

        Returns:
            generator: generator object yielding all edges using IDs or indices (if no mapping is used)
        """
        for e in self.data.edge_index.t():
            yield self.mapping.to_id(e[0].item()), self.mapping.to_id(e[1].item())

    def get_successors(self, row_idx: int) -> torch.Tensor:
        """Return a tensor containing the indices of all successor nodes for a given node identified by an index.

        Args:
            row_idx:   Index of node for which predecessors shall be returned.

        Returns:
            tensor: tensor containing indices of all successor nodes of the node indexed by `row_idx`
        """

        if row_idx + 1 < self.row_ptr.size(0):
            row_start = self.row_ptr[row_idx]
            row_end = self.row_ptr[row_idx + 1]
            return self.col[row_start:row_end]
        else:
            return torch.tensor([], device=self.data.edge_index.device)

    def get_predecessors(self, col_idx: int) -> torch.Tensor:
        """Return a tensor containing the indices of all predecessor nodes for a given node identified by an index.

        Args:
            col_idx:   Index of node for which predecessors shall be returned.

        Returns:
            tensor: tensor containing indices of all predecessor nodes of the node indexed by `col_idx`
        """
        if col_idx + 1 < self.col_ptr.size(0):
            col_start = self.col_ptr[col_idx]
            col_end = self.col_ptr[col_idx + 1]
            return self.row[col_start:col_end]
        else:
            return torch.tensor([], device=self.data.edge_index.device)

    def successors(self, node: Union[int, str] | tuple) -> Generator[Union[int, str] | tuple, None, None]:
        """Return all successors of a given node.

        This method returns a generator object that yields all successors of a
        given node. If an IndexMap is used, successors are returned
        as string IDs. If no mapping is used, successors are returned as indices.

        Args:
            node:   Index or string ID of node for which successors shall be returned.

        Returns:
            generator: generator object yielding all successors of the node identified
                by `node` using ID or index (if no mapping is used)
        """

        for j in self.get_successors(self.mapping.to_idx(node)):  # type: ignore
            yield self.mapping.to_id(j.item())

    def predecessors(self, node: Union[str, int] | tuple) -> Generator[Union[int, str] | tuple, None, None]:
        """Return the predecessors of a given node.

        This method returns a generator object that yields all predecessors of a
        given node. If a `node_id` mapping is used, predecessors will be returned
        as string IDs. If no mapping is used, predecessors are returned as indices.

        Args:
            node:   Index or string ID of node for which predecessors shall be returned.

        Returns:
            generator: generator object yielding all predecessors of the node identified
                by `node` using ID or index (if no mapping is used)
        """
        for i in self.get_predecessors(self.mapping.to_idx(node)):  # type: ignore
            yield self.mapping.to_id(i.item())

    def is_edge(self, v: Union[str, int], w: Union[str, int]) -> bool:
        """Return whether edge $(v,w)$ exists in the graph.

        If an index to ID mapping is used, nodes are assumed to be string IDs. If no
        mapping is used, nodes are assumed to be integer indices.

        Args:
            v: source node of edge as integer index or string ID
            w: target node of edge as integer index or string ID

        Returns:
            bool: True if edge exists, False otherwise
        """
        row = self.mapping.to_idx(v)
        row_start = self.row_ptr[row]
        row_end = self.row_ptr[row + 1]

        return self.mapping.to_idx(w) in self.col[row_start:row_end]

    def get_sparse_adj_matrix(self, edge_attr: Any = None) -> Any:
        """Return sparse adjacency matrix representation of (weighted) graph.

        Args:
            edge_attr: the edge attribute that shall be used as edge weight

        Returns:
            scipy.sparse.coo_matrix: sparse adjacency matrix representation of graph
        """
        if edge_attr is None:
            return torch_geometric.utils.to_scipy_sparse_matrix(self.data.edge_index.as_tensor())
        else:
            return torch_geometric.utils.to_scipy_sparse_matrix(
                self.data.edge_index.as_tensor(), edge_attr=self.data[edge_attr], num_nodes=self.N
            )

    @property
    def in_degrees(self) -> Dict[str, float]:
        """Return in-degrees of nodes in directed network.

        Returns:
            dict: dictionary containing in-degrees of nodes
        """
        return self.degrees(mode="in")

    @property
    def out_degrees(self) -> Dict[str, float]:
        """Return out-degrees of nodes in directed network.

        Returns:
            dict: dictionary containing out-degrees of nodes
        """
        return self.degrees(mode="out")

    def degrees(self, mode: str = "in") -> Dict[str, float]:
        """
        Return degrees of nodes.

        Args:
            mode: `in` or `out` to calculate the in- or out-degree for
                directed networks.

        Returns:
            dict: dictionary containing degrees of nodes
        """
        if mode == "in":
            d = torch_geometric.utils.degree(self.data.edge_index[1], num_nodes=self.N, dtype=torch.int)
        else:
            d = torch_geometric.utils.degree(self.data.edge_index[0], num_nodes=self.N, dtype=torch.int)
        return {self.mapping.to_id(i): d[i].item() for i in range(self.N)}

    def weighted_outdegrees(self) -> torch.Tensor:
        """
        Compute the weighted outdegrees of each node in the graph.

        Args:
            graph (Graph): pathpy graph object.

        Returns:
            tensor: Weighted outdegrees of nodes.
        """
        weighted_outdegree = scatter(
            self.data.edge_weight, self.data.edge_index[0], dim=0, dim_size=self.data.num_nodes, reduce="sum"
        )
        return weighted_outdegree

    def transition_probabilities(self) -> torch.Tensor:
        """
        Compute transition probabilities based on weighted outdegrees.

        Returns:
            tensor: Transition probabilities.
        """
        weighted_outdegree = self.weighted_outdegrees()
        source_ids = self.data.edge_index[0]
        return self.data.edge_weight / weighted_outdegree[source_ids]

    def get_laplacian(self, normalization: Any = None, edge_attr: Any = None) -> Any:
        """Return Laplacian matrix for a given graph.

        This wrapper method will use [`torch_geometric.utils.get_laplacian`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.get_laplacian)
        to return a Laplcian matrix representation of a given graph.

        Args:
            normalization: normalization parameter passed to pyG `get_laplacian`
                function
            edge_attr: optinal name of numerical edge attribute that shall
                be passed to pyG `get_laplacian` function as edge weight

        Returns:
            scipy.sparse.coo_matrix: Laplacian matrix representation of graph
        """
        if edge_attr is None:
            index, weight = torch_geometric.utils.get_laplacian(
                self.data.edge_index.as_tensor(), normalization=normalization
            )
            return torch_geometric.utils.to_scipy_sparse_matrix(index, weight)
        else:
            index, weight = torch_geometric.utils.get_laplacian(
                self.data.edge_index.as_tensor(),
                normalization=normalization,
                edge_weight=self.data[edge_attr],
            )
            return torch_geometric.utils.to_scipy_sparse_matrix(index, weight)

    def __getitem__(self, key: Union[tuple, str]) -> Any:
        """Return node, edge, or graph attribute.

        Args:
            key: name of attribute to be returned
        """
        if not isinstance(key, tuple):
            if key in self.data.keys():
                return self.data[key]
            else:
                raise KeyError(key + " is not a graph attribute")
        elif key[0] in self.node_attrs():
            return self.data[key[0]][self.mapping.to_idx(key[1])]
        elif key[0] in self.edge_attrs():
            return self.data[key[0]][self.edge_to_index[self.mapping.to_idx(key[1]), self.mapping.to_idx(key[2])]]
        else:
            raise KeyError(key[0] + " is not a node or edge attribute")

    def __setitem__(self, key: str, val: torch.Tensor) -> None:
        """Store node, edge, or graph attribute.

        Args:
            key: name of attribute to be stored
            val: value of attribute
        """
        if not isinstance(key, tuple):
            if key.startswith("node_"):
                if val.size(0) != self.N:
                    raise ValueError("Attribute must have same length as number of nodes")
                self.data[key] = val
            elif key.startswith("edge_"):
                if val.size(0) != self.M:
                    raise ValueError("Attribute must have same length as number of edges")
                self.data[key] = val
            else:
                self.data[key] = val
        elif key[0].startswith("node_"):  # type: ignore
            if key[0] not in self.data.keys():
                raise KeyError(
                    "Attribute does not yet exist. Setting the value of a specific node attribute"
                    + "requires that the attribute already exists."
                )
            self.data[key[0]][self.mapping.to_idx(key[1])] = val
        elif key[0].startswith("edge_"):  # type: ignore
            if key[0] not in self.data.keys():
                raise KeyError(
                    "Attribute does not yet exist. Setting the value of a specific node attribute"
                    + "requires that the attribute already exists."
                )
            self.data[key[0]][self.edge_to_index[self.mapping.to_idx(key[1]), self.mapping.to_idx(key[2])]] = val
        else:
            raise KeyError("node and edge specific attributes should be prefixed with 'node_' or 'edge_'")

    @property
    def N(self) -> int:
        """
        Return number of nodes.

        Returns:
            int: number of nodes in the graph
        """
        return self.data.num_nodes  # type: ignore

    @property
    def M(self) -> int:
        """
        Return number of edges.

        Returns the number of edges in the graph. For an undirected graph, the number of directed edges is returned.

        Returns:
            int: number of edges in the graph
        """
        return self.data.num_edges  # type: ignore

    def is_directed(self) -> bool:
        """Return whether graph is directed.

        Returns:
            bool: True if graph is directed, False otherwise
        """
        return not self.data.edge_index.is_undirected

    def is_undirected(self) -> bool:
        """Return whether graph is undirected.

        Returns:
            bool: True if graph is undirected, False otherwise
        """
        return self.data.edge_index.is_undirected

    def has_self_loops(self) -> bool:
        """Return whether graph contains self-loops.

        Returns:
            bool: True if graph contains self-loops, False otherwise
        """
        return self.data.has_self_loops()

    def __add__(self, other: Graph) -> Graph:
        """Combine Graph object with other Graph object.

        The semantics of this operation depends on the optional IndexMap
        of both graphs. If no IndexMap is included, the two underlying data objects
        are concatenated, thus merging edges from both graphs while leaving node indices
        unchanged. If both graphs include IndexMaps that assign node IDs to indices,
        indiced will be adjusted, creating a new mapping for the union of node Ids in both graphs.

        Node IDs of graphs to be combined can be disjoint, partly overlapping or non-overlapping.

        Examples:

            Adding two graphs without node IDs:

            >>> g1 = pp.Graph.from_edge_index(torch.Tensor([[0,1,1],[1,2,3]]))
            >>> g1 = pp.Graph.from_edge_index(torch.Tensor([[0,2,3],[3,2,1]]))
            >>> print(g1 + g2)
            Graph with 3 nodes and 6 edges

            Adding two graphs with identical node IDs:

            >>> g1 = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c')])
            >>> g2 = pp.Graph.from_edge_list([('a', 'c'), ('c', 'b')])
            >>> print(g1 + g2)
            Graph with 3 nodes and 4 edges

            Adding two graphs with non-overlapping node IDs:

            >>> g1 = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c')])
            >>> g2 = pp.Graph.from_edge_list([('c', 'd'), ('d', 'e')])
            >>> print(g1 + g2)
            Graph with 6 nodes and 4 edges

            Adding two graphs with partly overlapping node IDs:

            >>> g1 = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c')])
            >>> g2 = pp.Graph.from_edge_list([('b', 'd'), ('d', 'e')])
            >>> print(g1 + g2)
            Graph with 5 nodes and 4 edges
        """
        d1 = self.data.clone()
        m1 = self.mapping

        d2 = other.data.clone()
        m2 = other.mapping

        # compute overlap and additional nodes in g2 over g1
        overlap = set(m2.node_ids).intersection(m1.node_ids)
        additional_nodes = set(m2.node_ids).difference(m1.node_ids)

        d2_idx_translation = {}
        node_ids = [""] * (self.N + len(additional_nodes))
        # keep mappings of nodes in g1
        for v in m1.node_ids:
            node_ids[m1.to_idx(v)] = v
        for v in m2.node_ids:
            d2_idx_translation[m2.to_idx(v)] = m2.to_idx(v)
        # for overlapping node IDs we must correct node indices in m2
        for v in overlap:
            d2_idx_translation[m2.to_idx(v)] = m1.to_idx(v)
        # add mapping for nodes in g2 that are not in g1 and correct indices in g2
        for v in additional_nodes:
            new_idx = m2.to_idx(v) + self.N - len(overlap)
            node_ids[new_idx] = v
            d2_idx_translation[m2.to_idx(v)] = new_idx
        # apply index translation to d2
        # fast dictionary based mapping using torch
        palette, key = zip(*d2_idx_translation.items())
        key = torch.tensor(key)
        palette = torch.tensor(palette)

        index = torch.bucketize(d2.edge_index.ravel(), palette)
        d2.edge_index = key[index].reshape(d2.edge_index.shape)
        d = d1.concat(d2)
        mapping = IndexMap(node_ids)
        d.num_nodes = self.N + len(additional_nodes)
        d.edge_index = EdgeIndex(d.edge_index, sparse_size=(d.num_nodes, d.num_nodes))
        return Graph(d, mapping=mapping)

    def __str__(self) -> str:
        """Return a string representation of the graph."""

        attr_types = Graph.attr_types(self.data.to_dict())

        if self.is_undirected():
            s = "Undirected graph with {0} nodes and {1} (directed) edges\n".format(self.N, self.M)
        else:
            s = "Directed graph with {0} nodes and {1} edges\n".format(self.N, self.M)
        if len(self.data.node_attrs()) > 0:
            s += "\nNode attributes\n"
            for a in self.data.node_attrs():
                s += "\t{0}\t\t{1}\n".format(a, attr_types[a])
        if len(self.data.edge_attrs()) > 1:
            s += "\nEdge attributes\n"
            for a in self.data.edge_attrs():
                if a != "edge_index":
                    s += "\t{0}\t\t{1}\n".format(a, attr_types[a])
        if len(self.data.keys()) > len(self.data.edge_attrs()) + len(self.data.node_attrs()):
            s += "\nGraph attributes\n"
            for a in self.data.keys():
                if not self.data.is_node_attr(a) and not self.data.is_edge_attr(a):
                    s += "\t{0}\t\t{1}\n".format(a, attr_types[a])
        return s