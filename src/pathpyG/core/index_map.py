from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Union

import torch
import numpy as np


class IndexMap:
    """Maps node indices to string ids"""

    def __init__(self, node_ids: Union[List[str], None] = None) -> None:
        """Initialize mapping from indices to node IDs.

        Args:
            node_ids: List of node IDs to initialize mapping.
        """
        if node_ids is not None:
            assert len(node_ids) == len(set(node_ids)), "node_id entries must be unique"
            # first-order: nodes = integer indices
            self.node_ids: np.ndarray = np.array(node_ids)
            self.id_to_idx: Dict[str, int] = {v: i for i, v in enumerate(node_ids)}
            self.has_ids = True
        else:
            self.node_ids = np.array([])
            self.id_to_idx = {}
            self.has_ids = False

    def num_ids(self) -> int:
        """Return number of IDs.

        Returns:
            Number of IDs.
        """
        return len(self.node_ids)

    def add_id(self, node_id: str) -> None:
        """Assigns additional ID to next consecutive index.

        Args:
            node_id: ID to assign.
        """
        if node_id not in self.id_to_idx:
            idx = self.num_ids()
            self.node_ids = np.append(self.node_ids, node_id)
            self.id_to_idx[node_id] = idx
            self.has_ids = True

    def add_ids(self, node_ids: list | np.ndarray) -> None:
        """Assigns additional IDs to next consecutive indices.

        Args:
            node_ids: IDs to assign
        """
        cur_num_ids = self.num_ids()
        node_ids = np.array(node_ids)
        mask = np.isin(node_ids, self.node_ids)
        new_ids = np.unique(node_ids[~mask])
        self.node_ids = np.append(self.node_ids, new_ids)
        self.id_to_idx.update({v: i + cur_num_ids for i, v in enumerate(new_ids)})
        self.has_ids = True

    def to_id(self, idx: int) -> Union[int, str, tuple]:
        """Map index to ID if mapping is defined, return index otherwise.

        Args:
            idx: Index to map.

        Returns:
            ID if mapping is defined, index otherwise.
        """
        if self.has_ids:
            if self.node_ids.ndim == 1:
                return self.node_ids[idx]
            return tuple(self.node_ids[idx])
        return idx

    def to_ids(self, idxs: list | tuple) -> list:
        """Map list of indices to IDs if mapping is defined, return indices otherwise.

        Args:
            idxs: Indices to map.

        Returns:
            IDs if mapping is defined, indices otherwise.
        """
        if self.has_ids:
            return self.node_ids[idxs].tolist()
        return idxs

    def to_idx(self, node: Union[str, int]) -> int:
        """Map argument (ID or index) to index if mapping is defined, return argument otherwise.

        Args:
            node: ID or index to map.

        Returns:
            Index if mapping is defined, argument otherwise.
        """
        if self.has_ids:
            return self.id_to_idx[node]
        return node

    def to_idxs(self, nodes: list | tuple) -> torch.Tensor:
        """Map list of arguments (IDs or indices) to indices if mapping is defined, return argument otherwise.

        Args:
            nodes: IDs or indices to map.

        Returns:
            Indices if mapping is defined, arguments otherwise.
        """
        if self.has_ids:
            return torch.tensor([self.id_to_idx[node] for node in nodes])
        return torch.tensor(nodes)

    def __str__(self) -> str:
        s = ""
        for v in self.id_to_idx:
            s += str(v) + " -> " + str(self.to_idx(v)) + "\n"
        return s
