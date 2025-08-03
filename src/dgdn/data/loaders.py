"""Data loading utilities for temporal graphs."""

import torch
from torch.utils.data import DataLoader, Sampler
from typing import List, Iterator, Optional, Callable
import numpy as np
from .datasets import TemporalData, TemporalDataset, create_temporal_batch


class DynamicBatchSampler(Sampler):
    """Sampler that creates batches with adaptive sizes based on graph complexity.
    
    For temporal graphs, batch size should adapt to the temporal density and
    graph size to maintain consistent memory usage and training stability.
    """
    
    def __init__(
        self,
        dataset: TemporalDataset,
        max_batch_size: int = 32,
        min_batch_size: int = 1,
        max_edges_per_batch: int = 10000,
        shuffle: bool = True
    ):
        """Initialize dynamic batch sampler.
        
        Args:
            dataset: TemporalDataset to sample from
            max_batch_size: Maximum number of samples per batch
            min_batch_size: Minimum number of samples per batch
            max_edges_per_batch: Maximum edges allowed per batch 
            shuffle: Whether to shuffle samples
        """
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.max_edges_per_batch = max_edges_per_batch
        self.shuffle = shuffle
        
        # Pre-compute complexity metrics for each sample
        self._compute_complexity_metrics()
        
    def _compute_complexity_metrics(self):
        """Compute complexity metrics for dynamic batching."""
        # For now, use number of edges as complexity metric
        # In practice, this could include temporal density, node degree distribution, etc.
        timestamps = self.dataset.data.timestamps
        
        # Create windows and compute edges per window
        self.sample_complexities = []
        window_size = 100  # Should match dataset windowing
        min_time, max_time = timestamps.min(), timestamps.max()
        
        for start in torch.arange(min_time, max_time - window_size, window_size // 2):
            end = start + window_size
            mask = (timestamps >= start) & (timestamps < end)
            num_edges = mask.sum().item()
            self.sample_complexities.append(num_edges)
        
        self.num_samples = len(self.sample_complexities)
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with adaptive sizes."""
        indices = list(range(self.num_samples))
        
        if self.shuffle:
            torch.manual_seed(torch.initial_seed())
            indices = torch.randperm(self.num_samples).tolist()
        
        batch = []
        batch_edges = 0
        
        for idx in indices:
            sample_edges = self.sample_complexities[idx]
            
            # Check if adding this sample would exceed limits
            would_exceed_edges = batch_edges + sample_edges > self.max_edges_per_batch
            would_exceed_size = len(batch) >= self.max_batch_size
            
            if batch and (would_exceed_edges or would_exceed_size):
                # Yield current batch if it meets minimum size
                if len(batch) >= self.min_batch_size:
                    yield batch
                # Start new batch
                batch = [idx]
                batch_edges = sample_edges
            else:
                # Add to current batch
                batch.append(idx)
                batch_edges += sample_edges
        
        # Yield remaining batch if it meets minimum size
        if len(batch) >= self.min_batch_size:
            yield batch
    
    def __len__(self) -> int:
        """Estimate number of batches."""
        # This is an approximation since batch sizes are dynamic
        avg_edges_per_sample = sum(self.sample_complexities) / len(self.sample_complexities)
        avg_batch_size = min(
            self.max_batch_size,
            max(self.min_batch_size, self.max_edges_per_batch // max(1, int(avg_edges_per_sample)))
        )
        return max(1, self.num_samples // avg_batch_size)


class TemporalDataLoader:
    """DataLoader for temporal graph data with specialized batching."""
    
    def __init__(
        self,
        dataset: TemporalDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        dynamic_batching: bool = True,
        max_edges_per_batch: int = 10000,
        collate_fn: Optional[Callable] = None
    ):
        """Initialize temporal data loader.
        
        Args:
            dataset: TemporalDataset to load from
            batch_size: Batch size (max if dynamic_batching=True)
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to use pinned memory
            dynamic_batching: Whether to use dynamic batch sizing
            max_edges_per_batch: Maximum edges per batch
            collate_fn: Custom collation function
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dynamic_batching = dynamic_batching
        
        # Set up collation function
        if collate_fn is None:
            self.collate_fn = self._default_collate
        else:
            self.collate_fn = collate_fn
        
        # Set up sampler
        if dynamic_batching:
            self.sampler = DynamicBatchSampler(
                dataset=dataset,
                max_batch_size=batch_size,
                max_edges_per_batch=max_edges_per_batch,
                shuffle=shuffle
            )
            # Use batch_sampler instead of sampler for DataLoader
            self.dataloader = DataLoader(
                self._create_indexed_dataset(),
                batch_sampler=self.sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=self.collate_fn
            )
        else:
            # Standard batching
            self.dataloader = DataLoader(
                self._create_indexed_dataset(),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=self.collate_fn
            )
    
    def _create_indexed_dataset(self):
        """Create an indexed version of the dataset for DataLoader compatibility."""
        return IndexedTemporalDataset(self.dataset)
    
    def _default_collate(self, batch: List[TemporalData]) -> TemporalData:
        """Default collation function for temporal data."""
        return create_temporal_batch(batch)
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Number of batches."""
        return len(self.dataloader)


class IndexedTemporalDataset:
    """Wrapper to make TemporalDataset compatible with PyTorch DataLoader."""
    
    def __init__(self, temporal_dataset: TemporalDataset, window_size: int = 100):
        """Initialize indexed dataset.
        
        Args:
            temporal_dataset: TemporalDataset to wrap
            window_size: Size of temporal windows
        """
        self.temporal_dataset = temporal_dataset
        self.window_size = window_size
        
        # Create windows
        self._create_windows()
    
    def _create_windows(self):
        """Create temporal windows for indexing."""
        data = self.temporal_dataset.data
        timestamps = data.timestamps.cpu().numpy()
        min_time, max_time = timestamps.min(), timestamps.max()
        
        self.windows = []
        step_size = self.window_size // 2  # 50% overlap
        
        for start in np.arange(min_time, max_time - self.window_size, step_size):
            end = start + self.window_size
            mask = (timestamps >= start) & (timestamps < end)
            
            if mask.sum() > 0:  # Only include non-empty windows
                self.windows.append(torch.from_numpy(mask))
    
    def __len__(self) -> int:
        """Number of temporal windows."""
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> TemporalData:
        """Get temporal window as TemporalData."""
        edge_mask = self.windows[idx]
        return self.temporal_dataset.data.subgraph(edge_mask)


class TemporalBatchSampler(Sampler):
    """Batch sampler that respects temporal ordering."""
    
    def __init__(
        self,
        dataset: TemporalDataset,
        batch_size: int,
        temporal_stride: int = 1,
        shuffle_within_batch: bool = False
    ):
        """Initialize temporal batch sampler.
        
        Args:
            dataset: TemporalDataset to sample from
            batch_size: Number of samples per batch
            temporal_stride: Stride for temporal sampling
            shuffle_within_batch: Whether to shuffle samples within each batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.temporal_stride = temporal_stride
        self.shuffle_within_batch = shuffle_within_batch
        
        # Sort indices by timestamp for temporal consistency
        timestamps = dataset.data.timestamps
        self.temporal_indices = torch.argsort(timestamps).tolist()
        
    def __iter__(self) -> Iterator[List[int]]:
        """Generate temporally consistent batches."""
        indices = self.temporal_indices[::self.temporal_stride]
        
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            
            if self.shuffle_within_batch:
                np.random.shuffle(batch)
            
            yield batch
    
    def __len__(self) -> int:
        """Number of batches."""
        indices = self.temporal_indices[::self.temporal_stride]
        return (len(indices) + self.batch_size - 1) // self.batch_size


def create_data_loaders(
    dataset: TemporalDataset,
    batch_size: int = 32,
    val_batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    dynamic_batching: bool = True
) -> tuple:
    """Create train/val/test data loaders from a split dataset.
    
    Args:
        dataset: TemporalDataset (should be split first)
        batch_size: Training batch size
        val_batch_size: Validation batch size (defaults to batch_size)
        test_batch_size: Test batch size (defaults to val_batch_size)
        num_workers: Number of worker processes
        pin_memory: Whether to use pinned memory
        dynamic_batching: Whether to use dynamic batch sizing
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if val_batch_size is None:
        val_batch_size = batch_size
    if test_batch_size is None:
        test_batch_size = val_batch_size
    
    # Check if dataset has been split
    if not hasattr(dataset, '_train_data') or dataset._train_data is None:
        raise ValueError("Dataset must be split before creating data loaders")
    
    # Create loaders
    train_loader = TemporalDataLoader(
        dataset._train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        dynamic_batching=dynamic_batching
    )
    
    val_loader = TemporalDataLoader(
        dataset._val_data,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        dynamic_batching=False  # Use fixed batching for evaluation
    )
    
    test_loader = TemporalDataLoader(
        dataset._test_data,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        dynamic_batching=False  # Use fixed batching for evaluation
    )
    
    return train_loader, val_loader, test_loader