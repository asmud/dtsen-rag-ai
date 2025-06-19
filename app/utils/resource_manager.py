import asyncio
import logging
import platform
import psutil
from typing import Dict, Any, Optional, List, Tuple
import torch
import numpy as np

from config import get_settings

logger = logging.getLogger(__name__)

class ResourceManager:
    """Manages CPU and GPU resources for optimal performance"""
    
    def __init__(self):
        self.settings = get_settings()
        self.gpu_available = False
        self.gpu_devices = []
        self.selected_device = "cpu"
        self.gpu_memory_info = {}
        self.cpu_info = {}
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize resource manager and detect available hardware"""
        try:
            await self._detect_hardware()
            await self._configure_devices()
            self._initialized = True
            logger.info(f"Resource manager initialized - Device: {self.selected_device}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize resource manager: {e}")
            return False
    
    async def _detect_hardware(self):
        """Detect available CPU and GPU hardware"""
        # CPU Detection
        self.cpu_info = {
            'cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'architecture': platform.machine(),
            'platform': platform.system(),
            'memory_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        # GPU Detection
        if self.settings.GPU_ENABLED and torch.cuda.is_available():
            self.gpu_available = True
            device_count = torch.cuda.device_count()
            
            for i in range(device_count):
                gpu_props = torch.cuda.get_device_properties(i)
                memory_info = {
                    'total': torch.cuda.get_device_properties(i).total_memory,
                    'reserved': torch.cuda.memory_reserved(i),
                    'allocated': torch.cuda.memory_allocated(i)
                }
                
                device_info = {
                    'id': i,
                    'name': gpu_props.name,
                    'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
                    'total_memory_gb': gpu_props.total_memory / (1024**3),
                    'memory_info': memory_info,
                    'multi_processor_count': gpu_props.multi_processor_count
                }
                
                self.gpu_devices.append(device_info)
                self.gpu_memory_info[f"cuda:{i}"] = memory_info
        
        logger.info(f"Hardware detected - CPU cores: {self.cpu_info['cores']}, "
                   f"GPU devices: {len(self.gpu_devices)}")
    
    async def _configure_devices(self):
        """Configure optimal device based on settings and availability"""
        if self.settings.EMBEDDING_DEVICE.lower() == "auto":
            if self.gpu_available and len(self.gpu_devices) > 0:
                # Select best GPU device
                if self.settings.GPU_DEVICE_ID is not None:
                    if self.settings.GPU_DEVICE_ID < len(self.gpu_devices):
                        self.selected_device = f"cuda:{self.settings.GPU_DEVICE_ID}"
                    else:
                        logger.warning(f"GPU device {self.settings.GPU_DEVICE_ID} not available, using cuda:0")
                        self.selected_device = "cuda:0"
                else:
                    # Auto-select GPU with most free memory
                    best_gpu = await self._select_best_gpu()
                    self.selected_device = f"cuda:{best_gpu}"
            else:
                self.selected_device = "cpu"
        else:
            # Use explicitly configured device
            configured_device = self.settings.EMBEDDING_DEVICE.lower()
            if configured_device.startswith("cuda"):
                if self.gpu_available:
                    self.selected_device = configured_device
                else:
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    self.selected_device = "cpu"
            else:
                self.selected_device = "cpu"
        
        # Configure GPU memory if using GPU
        if self.selected_device.startswith("cuda") and self.gpu_available:
            await self._configure_gpu_memory()
    
    async def _select_best_gpu(self) -> int:
        """Select GPU with most available memory"""
        if not self.gpu_devices:
            return 0
        
        best_gpu = 0
        max_free_memory = 0
        
        for device in self.gpu_devices:
            memory_info = device['memory_info']
            free_memory = memory_info['total'] - memory_info['allocated']
            
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = device['id']
        
        return best_gpu
    
    async def _configure_gpu_memory(self):
        """Configure GPU memory usage"""
        try:
            if self.selected_device.startswith("cuda"):
                device_id = int(self.selected_device.split(":")[1]) if ":" in self.selected_device else 0
                
                # Set memory fraction if configured
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(
                        self.settings.GPU_MEMORY_FRACTION, 
                        device_id
                    )
                
                # Enable mixed precision if supported and configured
                if self.settings.MIXED_PRECISION:
                    torch.backends.cudnn.benchmark = True
                    
        except Exception as e:
            logger.warning(f"Failed to configure GPU memory: {e}")
    
    def get_optimal_batch_size(self, task_type: str = "embedding") -> int:
        """Get optimal batch size based on device and task type"""
        if not self._initialized:
            return self.settings.EMBEDDING_BATCH_SIZE
        
        if self.selected_device.startswith("cuda"):
            # GPU batch sizes
            if task_type == "embedding":
                return self.settings.EMBEDDING_BATCH_SIZE_GPU
            else:
                return self.settings.EMBEDDING_BATCH_SIZE_GPU
        else:
            # CPU batch sizes
            cpu_cores = self.cpu_info.get('logical_cores', 4)
            if task_type == "embedding":
                return min(self.settings.EMBEDDING_BATCH_SIZE, cpu_cores * 8)
            else:
                return min(self.settings.EMBEDDING_BATCH_SIZE, cpu_cores * 4)
    
    def get_device(self) -> str:
        """Get the selected device"""
        return self.selected_device if self._initialized else "cpu"
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available and selected"""
        return self._initialized and self.selected_device.startswith("cuda")
    
    def get_worker_count(self, task_type: str = "cpu") -> int:
        """Get optimal worker count for parallel processing"""
        if task_type == "cpu":
            return min(self.settings.CPU_WORKERS, self.cpu_info.get('logical_cores', 4))
        elif task_type == "gpu" and self.is_gpu_available():
            # For GPU tasks, usually 1-2 workers per GPU is optimal
            return min(2, len(self.gpu_devices))
        else:
            return 1
    
    async def get_resource_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization"""
        utilization = {
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'cores_used': psutil.cpu_count(logical=True),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / (1024**3)
            }
        }
        
        if self.gpu_available:
            gpu_utilization = {}
            for i, device in enumerate(self.gpu_devices):
                try:
                    # Get current memory info
                    torch.cuda.synchronize(i)
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    memory_total = device['total_memory_gb'] * (1024**3)
                    
                    gpu_utilization[f'gpu_{i}'] = {
                        'name': device['name'],
                        'memory_allocated_gb': memory_allocated / (1024**3),
                        'memory_reserved_gb': memory_reserved / (1024**3),
                        'memory_total_gb': device['total_memory_gb'],
                        'memory_percent': (memory_allocated / memory_total) * 100,
                        'memory_free_gb': (memory_total - memory_allocated) / (1024**3)
                    }
                except Exception as e:
                    gpu_utilization[f'gpu_{i}'] = {
                        'name': device['name'],
                        'error': str(e)
                    }
            
            utilization['gpu'] = gpu_utilization
        
        return utilization
    
    async def optimize_for_task(self, task_type: str, estimated_load: str = "medium") -> Dict[str, Any]:
        """Get optimization recommendations for specific task"""
        recommendations = {
            'device': self.get_device(),
            'batch_size': self.get_optimal_batch_size(task_type),
            'workers': self.get_worker_count(task_type),
            'settings': {}
        }
        
        if task_type == "embedding":
            if self.is_gpu_available():
                recommendations['settings'] = {
                    'use_gpu': True,
                    'mixed_precision': self.settings.MIXED_PRECISION,
                    'pin_memory': True,
                    'non_blocking': True
                }
            else:
                recommendations['settings'] = {
                    'use_gpu': False,
                    'num_threads': self.get_worker_count("cpu"),
                    'pin_memory': False
                }
        
        elif task_type == "llm_inference":
            # LLM is handled by Ollama, but we can provide recommendations
            recommendations['settings'] = {
                'context_length': 2048 if estimated_load == "high" else 4096,
                'parallel_requests': 2 if self.is_gpu_available() else 1,
                'use_gpu': self.is_gpu_available()
            }
        
        return recommendations
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        if self.gpu_available:
            try:
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'synchronize'):
                    torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"Failed to cleanup GPU memory: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on resource management"""
        try:
            health = {
                'status': 'healthy',
                'initialized': self._initialized,
                'selected_device': self.selected_device,
                'gpu_available': self.gpu_available,
                'cpu_cores': self.cpu_info.get('cores', 0),
                'total_memory_gb': round(self.cpu_info.get('memory_gb', 0), 2)
            }
            
            if self.gpu_available:
                health['gpu_devices'] = len(self.gpu_devices)
                health['gpu_memory_total_gb'] = sum(
                    device['total_memory_gb'] for device in self.gpu_devices
                )
            
            # Test device access
            if self.is_gpu_available():
                try:
                    test_tensor = torch.tensor([1.0, 2.0, 3.0]).to(self.selected_device)
                    test_result = test_tensor.sum().item()
                    health['gpu_test'] = 'passed'
                except Exception as e:
                    health['gpu_test'] = f'failed: {e}'
                    health['status'] = 'degraded'
            
            return health
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            'resource_manager': {
                'initialized': self._initialized,
                'selected_device': self.selected_device,
                'auto_device_selection': self.settings.EMBEDDING_DEVICE.lower() == "auto"
            },
            'cpu': self.cpu_info,
            'gpu': {
                'available': self.gpu_available,
                'device_count': len(self.gpu_devices),
                'devices': self.gpu_devices
            },
            'configuration': {
                'gpu_enabled': self.settings.GPU_ENABLED,
                'gpu_memory_fraction': self.settings.GPU_MEMORY_FRACTION,
                'mixed_precision': self.settings.MIXED_PRECISION,
                'cpu_workers': self.settings.CPU_WORKERS,
                'embedding_batch_size_cpu': self.settings.EMBEDDING_BATCH_SIZE,
                'embedding_batch_size_gpu': self.settings.EMBEDDING_BATCH_SIZE_GPU
            }
        }
        
        return info

# Global resource manager instance
_resource_manager = None

async def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance"""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
        await _resource_manager.initialize()
    return _resource_manager

def cleanup_resource_manager():
    """Cleanup global resource manager"""
    global _resource_manager
    if _resource_manager:
        _resource_manager.cleanup_gpu_memory()
        _resource_manager = None