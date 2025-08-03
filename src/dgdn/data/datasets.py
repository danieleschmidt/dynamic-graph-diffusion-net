"""Temporal graph data structures and datasets for DGDN."""

import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass
import pandas as pd


@dataclass
class TemporalData:
    """Temporal graph data structure.
    
    Represents a dynamic graph with temporal edge information and node features.
    
    Attributes:
        edge_index: Edge connectivity [2, num_edges]
        edge_attr: Edge features [num_edges, edge_dim] (optional)
        timestamps: Edge timestamps [num_edges]
        node_features: Node features [num_nodes, node_dim] (optional)
        y: Labels for prediction tasks (optional)
        num_nodes: Number of nodes in the graph
    """
    edge_index: torch.Tensor
    timestamps: torch.Tensor
    edge_attr: Optional[torch.Tensor] = None
    node_features: Optional[torch.Tensor] = None
    y: Optional[torch.Tensor] = None
    num_nodes: Optional[int] = None
    
    def __post_init__(self):
        """Validate and set default values after initialization."""
        if self.num_nodes is None:
            self.num_nodes = int(self.edge_index.max().item()) + 1
        
        # Ensure tensors are on the same device
        device = self.edge_index.device
        self.timestamps = self.timestamps.to(device)
        
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device)
        if self.node_features is not None:
            self.node_features = self.node_features.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
    
    def to(self, device: torch.device) -> 'TemporalData':
        """Move all tensors to specified device."""
        return TemporalData(
            edge_index=self.edge_index.to(device),
            timestamps=self.timestamps.to(device),
            edge_attr=self.edge_attr.to(device) if self.edge_attr is not None else None,
            node_features=self.node_features.to(device) if self.node_features is not None else None,
            y=self.y.to(device) if self.y is not None else None,
            num_nodes=self.num_nodes
        )
    
    def subgraph(self, edge_mask: torch.Tensor) -> 'TemporalData':
        """Extract subgraph based on edge mask.
        
        Args:
            edge_mask: Boolean mask for edges to include
            
        Returns:
            New TemporalData with selected edges
        """
        return TemporalData(
            edge_index=self.edge_index[:, edge_mask],
            timestamps=self.timestamps[edge_mask],
            edge_attr=self.edge_attr[edge_mask] if self.edge_attr is not None else None,
            node_features=self.node_features,
            y=self.y,
            num_nodes=self.num_nodes
        )
    
    def time_window(self, start_time: float, end_time: float) -> 'TemporalData':
        """Extract subgraph within time window.
        
        Args:
            start_time: Start of time window
            end_time: End of time window
            
        Returns:
            TemporalData within the specified time window
        """
        mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
        return self.subgraph(mask)
    
    def get_snapshot(self, time: float) -> Data:
        """Get static graph snapshot up to a specific time.
        
        Args:
            time: Timestamp cutoff
            
        Returns:
            PyTorch Geometric Data object
        """
        mask = self.timestamps <= time
        edge_index = self.edge_index[:, mask]
        edge_attr = self.edge_attr[mask] if self.edge_attr is not None else None
        
        return Data(
            x=self.node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=self.y,
            num_nodes=self.num_nodes
        )
    
    def get_temporal_statistics(self) -> Dict[str, float]:
        """Get statistics about temporal patterns."""
        timestamps = self.timestamps.cpu().numpy()
        
        return {
            "min_time": float(timestamps.min()),
            "max_time": float(timestamps.max()),
            "time_span": float(timestamps.max() - timestamps.min()),
            "num_edges": len(timestamps),
            "num_nodes": self.num_nodes,
            "avg_edges_per_timestep": len(timestamps) / (timestamps.max() - timestamps.min() + 1),
            "temporal_density": len(np.unique(timestamps)) / len(timestamps)
        }


class TemporalDataset:
    """Dataset for temporal graph learning tasks."""
    
    SUPPORTED_DATASETS = {
        "wikipedia": "Wikipedia link prediction dataset",
        "reddit": "Reddit hyperlink network dataset", 
        "mooc": "MOOC user activity dataset",
        "lastfm": "Last.fm music listening dataset",
        "synthetic": "Synthetic temporal graph dataset"
    }
    
    def __init__(self, data: TemporalData, name: str = "custom"):
        """Initialize temporal dataset.
        
        Args:
            data: TemporalData object
            name: Dataset name
        """
        self.data = data
        self.name = name
        self._train_data = None
        self._val_data = None
        self._test_data = None
    
    @classmethod
    def load(cls, name: str, root: str = "data/") -> 'TemporalDataset':
        """Load a standard temporal graph dataset.
        
        Args:
            name: Dataset name
            root: Root directory for datasets
            
        Returns:
            TemporalDataset instance
        """
        if name not in cls.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {name} not supported. Available: {list(cls.SUPPORTED_DATASETS.keys())}")
        
        dataset_path = os.path.join(root, name)
        
        if name == "synthetic":
            return cls._create_synthetic_dataset()
        else:
            # For real datasets, we'd implement specific loaders
            # For now, create synthetic data as placeholder
            return cls._create_synthetic_dataset()
    
    @classmethod
    def _create_synthetic_dataset(cls, num_nodes: int = 1000, num_edges: int = 5000) -> 'TemporalDataset':
        """Create synthetic temporal graph dataset for testing."""
        # Generate random temporal graph
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        timestamps = torch.sort(torch.rand(num_edges) * 1000)[0]  # Sorted timestamps
        edge_attr = torch.randn(num_edges, 64)  # Random edge features
        node_features = torch.randn(num_nodes, 128)  # Random node features
        
        # Create binary edge prediction labels
        y = torch.randint(0, 2, (num_edges,)).float()
        
        data = TemporalData(
            edge_index=edge_index,
            timestamps=timestamps,
            edge_attr=edge_attr,
            node_features=node_features,
            y=y,
            num_nodes=num_nodes
        )
        
        return cls(data, name="synthetic")
    
    def split(
        self, 
        ratios: List[float] = [0.7, 0.15, 0.15],
        method: str = "temporal"
    ) -> Tuple['TemporalDataset', 'TemporalDataset', 'TemporalDataset']:
        """Split dataset into train/val/test sets.
        
        Args:
            ratios: Split ratios [train, val, test]
            method: Split method ("temporal" or "random")
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6
        
        num_edges = len(self.data.timestamps)
        
        if method == "temporal":
            # Split based on temporal order
            sorted_indices = torch.argsort(self.data.timestamps)
            
            train_end = int(num_edges * ratios[0])
            val_end = int(num_edges * (ratios[0] + ratios[1]))
            
            train_indices = sorted_indices[:train_end]
            val_indices = sorted_indices[train_end:val_end]
            test_indices = sorted_indices[val_end:]
            
        elif method == "random":
            # Random split
            indices = torch.randperm(num_edges)
            
            train_end = int(num_edges * ratios[0])
            val_end = int(num_edges * (ratios[0] + ratios[1]))
            
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]
        else:
            raise ValueError(f"Unknown split method: {method}")
        
        # Create split datasets
        train_data = self._create_split_data(train_indices)
        val_data = self._create_split_data(val_indices)
        test_data = self._create_split_data(test_indices)
        
        self._train_data = TemporalDataset(train_data, f"{self.name}_train")
        self._val_data = TemporalDataset(val_data, f"{self.name}_val")
        self._test_data = TemporalDataset(test_data, f"{self.name}_test")
        
        return self._train_data, self._val_data, self._test_data
    
    def _create_split_data(self, indices: torch.Tensor) -> TemporalData:
        """Create TemporalData for a subset of edges."""
        return TemporalData(
            edge_index=self.data.edge_index[:, indices],
            timestamps=self.data.timestamps[indices],
            edge_attr=self.data.edge_attr[indices] if self.data.edge_attr is not None else None,
            node_features=self.data.node_features,
            y=self.data.y[indices] if self.data.y is not None else None,
            num_nodes=self.data.num_nodes
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = self.data.get_temporal_statistics()
        stats.update({
            "dataset_name": self.name,
            "has_edge_features": self.data.edge_attr is not None,
            "has_node_features": self.data.node_features is not None,
            "has_labels": self.data.y is not None,
            "edge_feature_dim": self.data.edge_attr.shape[1] if self.data.edge_attr is not None else 0,
            "node_feature_dim": self.data.node_features.shape[1] if self.data.node_features is not None else 0
        })
        
        return stats
    
    def save(self, path: str):
        """Save dataset to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_from_file(cls, path: str) -> 'TemporalDataset':
        """Load dataset from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class TemporalGraphDataset(Dataset):
    """PyTorch Geometric compatible temporal graph dataset."""
    
    def __init__(self, temporal_data: TemporalData, window_size: int = 100):
        """Initialize dataset with temporal windows.
        
        Args:
            temporal_data: TemporalData object
            window_size: Size of temporal windows for batching
        """
        super().__init__()
        self.temporal_data = temporal_data
        self.window_size = window_size
        
        # Create temporal windows
        self._create_windows()
    
    def _create_windows(self):
        """Create temporal windows for efficient batching."""
        timestamps = self.temporal_data.timestamps.cpu().numpy()
        min_time, max_time = timestamps.min(), timestamps.max()
        time_range = max_time - min_time
        
        # Create overlapping windows
        self.windows = []
        step_size = self.window_size // 2  # 50% overlap
        
        for start in np.arange(min_time, max_time - self.window_size, step_size):
            end = start + self.window_size
            mask = (timestamps >= start) & (timestamps < end)
            
            if mask.sum() > 0:  # Only include non-empty windows
                self.windows.append({
                    'start_time': start,
                    'end_time': end,
                    'edge_mask': torch.from_numpy(mask)
                })
    
    def len(self) -> int:
        """Return number of temporal windows."""
        return len(self.windows)
    
    def get(self, idx: int) -> Data:
        """Get temporal window as PyTorch Geometric Data object."""
        window = self.windows[idx]
        edge_mask = window['edge_mask']
        
        # Extract subgraph for this window
        subgraph = self.temporal_data.subgraph(edge_mask)
        
        # Convert to PyTorch Geometric Data
        return Data(
            x=subgraph.node_features,
            edge_index=subgraph.edge_index,
            edge_attr=subgraph.edge_attr,
            edge_time=subgraph.timestamps,
            y=subgraph.y,
            num_nodes=subgraph.num_nodes,
            window_start=window['start_time'],
            window_end=window['end_time']
        )
    
    def process(self):
        """Process raw data (placeholder for custom datasets)."""
        pass


def create_temporal_batch(data_list: List[TemporalData]) -> TemporalData:
    """Create a batch from a list of TemporalData objects.
    
    Args:
        data_list: List of TemporalData objects
        
    Returns:
        Batched TemporalData object
    """
    if len(data_list) == 1:
        return data_list[0]
    
    edge_indices = []
    timestamps = []
    edge_attrs = []
    node_features = []
    ys = []
    
    node_offset = 0
    
    for data in data_list:
        # Offset edge indices for batching
        edge_index = data.edge_index + node_offset
        edge_indices.append(edge_index)
        timestamps.append(data.timestamps)
        
        if data.edge_attr is not None:
            edge_attrs.append(data.edge_attr)
        if data.node_features is not None:
            node_features.append(data.node_features)
        if data.y is not None:
            ys.append(data.y)
        
        node_offset += data.num_nodes
    
    # Concatenate all components
    batched_data = TemporalData(
        edge_index=torch.cat(edge_indices, dim=1),
        timestamps=torch.cat(timestamps),
        edge_attr=torch.cat(edge_attrs) if edge_attrs else None,
        node_features=torch.cat(node_features) if node_features else None,
        y=torch.cat(ys) if ys else None,
        num_nodes=node_offset
    )
    
    return batched_data