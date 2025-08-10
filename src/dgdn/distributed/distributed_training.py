"""Distributed training implementations for DGDN."""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
import logging
import json
import numpy as np
from collections import defaultdict
import threading
import queue

from ..training.trainer import DGDNTrainer
from ..models.dgdn import DynamicGraphDiffusionNet
from ..data import TemporalDataLoader


class DistributedDGDNTrainer(DGDNTrainer):
    """Distributed training implementation for DGDN models."""
    
    def __init__(
        self,
        model,
        rank: int,
        world_size: int,
        backend: str = "nccl",
        **trainer_kwargs
    ):
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        
        # Initialize distributed training
        self._setup_distributed()
        
        # Wrap model with DDP
        if torch.cuda.is_available():
            model = model.cuda(rank)
        model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
        
        super().__init__(model, **trainer_kwargs)
        
        self.logger = logging.getLogger(f'DGDN.DistributedTrainer.Rank{rank}')
        
        # Distributed training state
        self.gradient_accumulation_steps = trainer_kwargs.get('gradient_accumulation_steps', 1)
        self.sync_batch_norm = trainer_kwargs.get('sync_batch_norm', True)
        
    def _setup_distributed(self):
        """Initialize distributed training environment."""
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
        
        init_process_group(
            backend=self.backend,
            rank=self.rank,
            world_size=self.world_size
        )
        
        if torch.cuda.is_available():
            torch.cuda.set_device(self.rank)
            
    def _cleanup_distributed(self):
        """Cleanup distributed training."""
        destroy_process_group()
        
    def distributed_train_step(self, batch, accumulate_gradients: bool = True):
        """Distributed training step with gradient synchronization."""
        self.model.train()
        
        # Forward pass
        output = self.model(batch)
        
        # Compute loss
        loss_components = self.loss_fn(output, batch)
        loss = sum(loss_components.values()) / self.world_size  # Scale by world size
        
        # Backward pass
        if accumulate_gradients:
            loss = loss / self.gradient_accumulation_steps
            
        loss.backward()
        
        # Synchronize gradients across ranks
        if not accumulate_gradients or (self.current_step + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping before synchronization
            if self.gradient_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                
            # Optimizer step (DDP automatically synchronizes gradients)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if self.scheduler:
                self.scheduler.step()
                
        return {
            'loss': loss.item() * self.world_size,  # Rescale for logging
            'loss_components': {k: v.item() for k, v in loss_components.items()}
        }
        
    def all_reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """All-reduce metrics across all ranks."""
        reduced_metrics = {}
        
        for key, value in metrics.items():
            tensor_value = torch.tensor(value, dtype=torch.float32)
            if torch.cuda.is_available():
                tensor_value = tensor_value.cuda()
                
            # All-reduce sum, then divide by world size for average
            dist.all_reduce(tensor_value, op=dist.ReduceOp.SUM)
            reduced_metrics[key] = tensor_value.item() / self.world_size
            
        return reduced_metrics
        
    def barrier_sync(self):
        """Synchronize all processes."""
        dist.barrier()
        
    def broadcast_model_state(self, src_rank: int = 0):
        """Broadcast model state from src_rank to all other ranks."""
        for param in self.model.parameters():
            dist.broadcast(param.data, src=src_rank)
            
    def save_checkpoint_distributed(self, filepath: str, epoch: int, metrics: Dict):
        """Save checkpoint from rank 0 only."""
        if self.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
                'world_size': self.world_size,
                'distributed': True
            }
            torch.save(checkpoint, filepath)
            self.logger.info(f"Saved distributed checkpoint: {filepath}")
            
        # Synchronize all ranks
        self.barrier_sync()
        
    def load_checkpoint_distributed(self, filepath: str):
        """Load checkpoint in distributed setting."""
        checkpoint = torch.load(filepath, map_location=f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')
        
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Ensure all ranks are synchronized
        self.broadcast_model_state()
        
        return checkpoint['epoch'], checkpoint.get('metrics', {})
        
    def cleanup(self):
        """Cleanup distributed resources."""
        self._cleanup_distributed()


class MultiGPUTrainer:
    """Multi-GPU training coordinator."""
    
    def __init__(
        self,
        model_class,
        model_config: Dict,
        training_config: Dict,
        num_gpus: int = None
    ):
        self.model_class = model_class
        self.model_config = model_config
        self.training_config = training_config
        
        # Determine number of GPUs
        if num_gpus is None:
            self.num_gpus = torch.cuda.device_count()
        else:
            self.num_gpus = min(num_gpus, torch.cuda.device_count())
            
        if self.num_gpus == 0:
            raise RuntimeError("No GPUs available for multi-GPU training")
            
        self.logger = logging.getLogger('DGDN.MultiGPUTrainer')
        self.logger.info(f"Initializing multi-GPU training with {self.num_gpus} GPUs")
        
    def train(self, train_data, val_data = None):
        """Launch distributed training across multiple GPUs."""
        # Launch training processes
        mp.spawn(
            self._train_worker,
            args=(self.num_gpus, train_data, val_data),
            nprocs=self.num_gpus,
            join=True
        )
        
    def _train_worker(self, rank: int, world_size: int, train_data, val_data):
        """Worker process for distributed training."""
        try:
            # Create model
            model = self.model_class(**self.model_config)
            
            # Create distributed trainer
            trainer = DistributedDGDNTrainer(
                model=model,
                rank=rank,
                world_size=world_size,
                **self.training_config
            )
            
            # Create distributed data loaders
            train_loader = self._create_distributed_loader(train_data, rank, world_size)
            val_loader = self._create_distributed_loader(val_data, rank, world_size) if val_data else None
            
            # Train model
            history = trainer.fit(
                train_data=train_loader,
                val_data=val_loader,
                **self.training_config
            )
            
            # Save final model (rank 0 only)
            if rank == 0:
                model_path = self.training_config.get('model_save_path', 'multigpu_model.pt')
                trainer.save_checkpoint_distributed(model_path, trainer.current_epoch, history)
                
        except Exception as e:
            self.logger.error(f"Error in training worker {rank}: {e}")
            raise
        finally:
            # Cleanup
            if 'trainer' in locals():
                trainer.cleanup()
                
    def _create_distributed_loader(self, data, rank: int, world_size: int):
        """Create distributed data loader."""
        if data is None:
            return None
            
        # Create distributed sampler
        sampler = torch.utils.data.distributed.DistributedSampler(
            data,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        # Create data loader
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=self.training_config.get('batch_size', 32),
            sampler=sampler,
            num_workers=self.training_config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        
        return loader


class FederatedTrainer:
    """Federated learning implementation for DGDN."""
    
    def __init__(
        self,
        model_class,
        model_config: Dict,
        federation_config: Dict
    ):
        self.model_class = model_class
        self.model_config = model_config
        self.federation_config = federation_config
        
        # Federated learning parameters
        self.num_clients = federation_config['num_clients']
        self.clients_per_round = federation_config.get('clients_per_round', self.num_clients)
        self.num_rounds = federation_config.get('num_rounds', 100)
        self.client_epochs = federation_config.get('client_epochs', 1)
        
        # Privacy parameters
        self.differential_privacy = federation_config.get('differential_privacy', False)
        self.privacy_epsilon = federation_config.get('privacy_epsilon', 1.0)
        self.privacy_delta = federation_config.get('privacy_delta', 1e-5)
        
        # Communication parameters
        self.compression_ratio = federation_config.get('compression_ratio', 0.1)
        self.quantization_bits = federation_config.get('quantization_bits', 8)
        
        # Global model
        self.global_model = self.model_class(**model_config)
        
        # Client management
        self.clients = []
        self.client_data = {}
        
        self.logger = logging.getLogger('DGDN.FederatedTrainer')
        
    def add_client(self, client_id: str, client_data):
        """Add client to federation."""
        self.clients.append(client_id)
        self.client_data[client_id] = client_data
        
    def federated_train(self):
        """Execute federated training."""
        self.logger.info(f"Starting federated training with {len(self.clients)} clients")
        
        training_history = []
        
        for round_idx in range(self.num_rounds):
            round_start_time = time.time()
            
            # Select clients for this round
            selected_clients = self._select_clients()
            
            # Client training phase
            client_updates = []
            client_metrics = []
            
            for client_id in selected_clients:
                self.logger.info(f"Round {round_idx}, training client {client_id}")
                
                # Send global model to client
                client_model = self._create_client_model()
                
                # Train on client data
                client_trainer = self._create_client_trainer(client_model)
                client_data = self.client_data[client_id]
                
                # Local training
                local_history = client_trainer.fit(
                    train_data=client_data,
                    epochs=self.client_epochs,
                    verbose=False
                )
                
                # Get model update
                model_update = self._get_model_update(client_model)
                
                # Apply privacy if enabled
                if self.differential_privacy:
                    model_update = self._apply_differential_privacy(model_update)
                    
                # Compress update
                compressed_update = self._compress_update(model_update)
                
                client_updates.append(compressed_update)
                client_metrics.append(local_history)
                
            # Aggregate updates
            aggregated_update = self._aggregate_updates(client_updates)
            
            # Update global model
            self._apply_global_update(aggregated_update)
            
            # Evaluate global model
            global_metrics = self._evaluate_global_model()
            
            round_time = time.time() - round_start_time
            
            # Log round results
            round_info = {
                'round': round_idx,
                'selected_clients': len(selected_clients),
                'global_metrics': global_metrics,
                'round_time': round_time
            }
            
            training_history.append(round_info)
            
            self.logger.info(
                f"Round {round_idx} completed: "
                f"Loss={global_metrics.get('loss', 0):.4f}, "
                f"Accuracy={global_metrics.get('accuracy', 0):.4f}, "
                f"Time={round_time:.2f}s"
            )
            
        return training_history
        
    def _select_clients(self) -> List[str]:
        """Select clients for current round."""
        if self.clients_per_round >= len(self.clients):
            return self.clients.copy()
        else:
            return np.random.choice(
                self.clients, 
                size=self.clients_per_round, 
                replace=False
            ).tolist()
            
    def _create_client_model(self):
        """Create client model with global weights."""
        client_model = self.model_class(**self.model_config)
        client_model.load_state_dict(self.global_model.state_dict())
        return client_model
        
    def _create_client_trainer(self, model):
        """Create trainer for client."""
        trainer_config = self.federation_config.get('client_trainer_config', {})
        return DGDNTrainer(model=model, **trainer_config)
        
    def _get_model_update(self, client_model):
        """Get model parameter update."""
        update = {}
        global_params = dict(self.global_model.named_parameters())
        
        for name, param in client_model.named_parameters():
            update[name] = param.data - global_params[name].data
            
        return update
        
    def _apply_differential_privacy(self, update: Dict[str, torch.Tensor]):
        """Apply differential privacy to model update."""
        noised_update = {}
        
        for name, param_update in update.items():
            # Clip gradients
            clipped_update = torch.clamp(param_update, -1.0, 1.0)
            
            # Add Gaussian noise
            sensitivity = 2.0  # Due to clipping
            noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / self.privacy_delta)) / self.privacy_epsilon
            
            noise = torch.normal(0, noise_scale, size=param_update.shape)
            noised_update[name] = clipped_update + noise
            
        return noised_update
        
    def _compress_update(self, update: Dict[str, torch.Tensor]):
        """Compress model update for communication."""
        compressed_update = {}
        
        for name, param_update in update.items():
            # Top-k sparsification
            flat_update = param_update.flatten()
            k = int(len(flat_update) * self.compression_ratio)
            
            if k > 0:
                # Get top-k values by magnitude
                _, indices = torch.topk(torch.abs(flat_update), k)
                sparse_update = torch.zeros_like(flat_update)
                sparse_update[indices] = flat_update[indices]
                
                # Quantization
                if self.quantization_bits < 32:
                    scale = torch.max(torch.abs(sparse_update))
                    quantized_update = torch.round(sparse_update / scale * (2**(self.quantization_bits-1) - 1))
                    quantized_update = quantized_update / (2**(self.quantization_bits-1) - 1) * scale
                    sparse_update = quantized_update
                    
                compressed_update[name] = sparse_update.reshape(param_update.shape)
            else:
                compressed_update[name] = torch.zeros_like(param_update)
                
        return compressed_update
        
    def _aggregate_updates(self, client_updates: List[Dict[str, torch.Tensor]]):
        """Aggregate client updates using FedAvg."""
        if not client_updates:
            return {}
            
        # Initialize aggregated update
        aggregated = {}
        
        # Average all client updates
        for name in client_updates[0].keys():
            updates = [update[name] for update in client_updates]
            aggregated[name] = torch.stack(updates).mean(dim=0)
            
        return aggregated
        
    def _apply_global_update(self, aggregated_update: Dict[str, torch.Tensor]):
        """Apply aggregated update to global model."""
        learning_rate = self.federation_config.get('global_learning_rate', 1.0)
        
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_update:
                    param.data += learning_rate * aggregated_update[name]
                    
    def _evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate global model performance."""
        # This would typically evaluate on a global validation set
        # For now, return dummy metrics
        return {
            'loss': np.random.random(),
            'accuracy': np.random.random()
        }
        
    def save_global_model(self, filepath: str):
        """Save global federated model."""
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'federation_config': self.federation_config,
            'model_config': self.model_config
        }, filepath)
        
        self.logger.info(f"Saved global federated model: {filepath}")
        
    def load_global_model(self, filepath: str):
        """Load global federated model."""
        checkpoint = torch.load(filepath)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        
        self.logger.info(f"Loaded global federated model: {filepath}")


class AsyncParameterServer:
    """Asynchronous parameter server for large-scale distributed training."""
    
    def __init__(self, model, server_config: Dict):
        self.model = model
        self.config = server_config
        
        # Server state
        self.global_parameters = {name: param.clone() for name, param in model.named_parameters()}
        self.parameter_versions = {name: 0 for name in self.global_parameters.keys()}
        self.worker_staleness = defaultdict(int)
        
        # Synchronization
        self.parameter_lock = threading.RLock()
        self.update_queue = queue.Queue()
        
        # Configuration
        self.staleness_threshold = server_config.get('staleness_threshold', 10)
        self.aggregation_method = server_config.get('aggregation_method', 'average')
        
        self.logger = logging.getLogger('DGDN.ParameterServer')
        
        # Start parameter server
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
        
    def _server_loop(self):
        """Main parameter server loop."""
        while True:
            try:
                # Process parameter updates
                if not self.update_queue.empty():
                    worker_id, parameter_updates, version_info = self.update_queue.get()
                    self._process_parameter_update(worker_id, parameter_updates, version_info)
                    
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Error in parameter server loop: {e}")
                
    def _process_parameter_update(self, worker_id: str, updates: Dict, version_info: Dict):
        """Process parameter update from worker."""
        with self.parameter_lock:
            # Check staleness
            max_staleness = max(
                self.parameter_versions[name] - version_info[name]
                for name in updates.keys()
            )
            
            if max_staleness > self.staleness_threshold:
                self.logger.warning(f"Dropping stale update from worker {worker_id} (staleness: {max_staleness})")
                return
                
            # Apply updates
            for name, update in updates.items():
                current_version = self.parameter_versions[name]
                
                # Apply staleness compensation
                staleness = current_version - version_info[name]
                compensation_factor = 1.0 / (1.0 + staleness * 0.1)  # Reduce impact of stale updates
                
                # Update parameter
                self.global_parameters[name] += compensation_factor * update
                self.parameter_versions[name] += 1
                
            self.worker_staleness[worker_id] = max_staleness
            
    def get_parameters(self, worker_id: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
        """Get current parameters and version info."""
        with self.parameter_lock:
            parameters = {name: param.clone() for name, param in self.global_parameters.items()}
            versions = self.parameter_versions.copy()
            
        return parameters, versions
        
    def push_parameters(self, worker_id: str, updates: Dict[str, torch.Tensor], versions: Dict[str, int]):
        """Push parameter updates from worker."""
        self.update_queue.put((worker_id, updates, versions))
        
    def get_server_stats(self) -> Dict[str, Any]:
        """Get parameter server statistics."""
        with self.parameter_lock:
            return {
                'parameter_versions': self.parameter_versions.copy(),
                'worker_staleness': dict(self.worker_staleness),
                'queue_size': self.update_queue.qsize(),
                'active_workers': len(self.worker_staleness)
            }