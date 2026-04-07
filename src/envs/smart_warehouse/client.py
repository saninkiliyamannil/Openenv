"""
SmartWarehouse HTTP Client
==========================

Client for connecting to SmartWarehouse environment servers.
Following the OpenEnv HTTPEnvClient pattern.
"""

from typing import Dict, Any, Optional, List
from dataclasses import fields

from core import HTTPEnvClient, StepResult
from .models import WarehouseAction, WarehouseObservation, WarehouseState


class SmartWarehouseEnv(HTTPEnvClient[WarehouseAction, WarehouseObservation]):
    """
    HTTP client for SmartWarehouse environment.
    
    This follows the OpenEnv pattern where the client communicates
    with an environment server via HTTP/JSON.
    
    Usage:
        # Connect to remote server
        env = SmartWarehouseEnv(base_url="http://localhost:8000")
        obs = env.reset()
        result = env.step(WarehouseAction(restock_level=1))
        
        # From Docker
        env = SmartWarehouseEnv.from_docker_image("registry.hf.space/smart-warehouse:latest")
    """
    
    SUPPORTS_CONCURRENT_SESSIONS = True
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        task: str = "easy",
    ):
        """
        Initialize the SmartWarehouse client.
        
        Args:
            base_url: Base URL of the environment server
            timeout: Request timeout in seconds
            task: Task difficulty ("easy", "medium", "hard")
        """
        super().__init__(base_url, timeout)
        self.task = task
        self._last_observation: Optional[WarehouseObservation] = None
    
    @classmethod
    def from_docker_image(cls, image_name: str, **kwargs) -> "SmartWarehouseEnv":
        """
        Create environment from Docker image.
        
        Args:
            image_name: Docker image name (e.g., "registry.hf.space/smart-warehouse:latest")
            **kwargs: Additional arguments for constructor
            
        Returns:
            SmartWarehouseEnv instance
        """
        import os
        import subprocess
        import time
        
        # Extract port from image or use default
        port = int(os.getenv("SMART_WAREHOUSE_PORT", "8001"))
        base_url = f"http://localhost:{port}"
        
        # Check if container is already running
        try:
            import requests
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                return cls(base_url=base_url, **kwargs)
        except:
            pass
        
        # Start container
        container_id = subprocess.check_output([
            "docker", "run", "-d", "-p", f"{port}:8000",
            "--platform", "linux/amd64",
            image_name
        ]).decode().strip()
        
        # Wait for container to be ready
        max_wait = 60
        for _ in range(max_wait):
            try:
                import requests
                response = requests.get(f"{base_url}/health", timeout=2)
                if response.status_code == 200:
                    break
            except:
                pass
            time.sleep(1)
        
        return cls(base_url=base_url, **kwargs)
    
    def _step_payload(self, action: WarehouseAction) -> Dict[str, Any]:
        """
        Convert WarehouseAction to JSON payload.
        
        Args:
            action: The typed action
            
        Returns:
            JSON-serializable dict
        """
        return {"action": action.to_dict()}
    
    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        """
        Parse JSON response to StepResult.
        
        Args:
            payload: JSON response from server
            
        Returns:
            Typed StepResult
        """
        obs_data = payload.get("observation", {})
        
        # Parse observation
        observation = WarehouseObservation(
            inventory_levels=obs_data.get("inventory_levels", []),
            pending_orders=obs_data.get("pending_orders", []),
            warehouse_zones=obs_data.get("warehouse_zones", []),
            worker_availability=obs_data.get("worker_availability", []),
            delivery_vehicles=obs_data.get("delivery_vehicles", []),
            time_features=obs_data.get("time_features", []),
            demand_forecast=obs_data.get("demand_forecast", []),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward", 0.0),
            messages=obs_data.get("messages", []),
        )
        
        self._last_observation = observation
        
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            info=payload.get("info", {}),
        )
    
    def _parse_state(self, payload: Dict[str, Any]) -> WarehouseState:
        """
        Parse JSON response to WarehouseState.
        
        Args:
            payload: JSON response from server
            
        Returns:
            Typed WarehouseState
        """
        return WarehouseState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task=payload.get("task", "easy"),
            total_reward=payload.get("total_reward", 0.0),
            metrics=payload.get("metrics", {}),
        )
    
    def reset(self, **kwargs) -> StepResult:
        """
        Reset the environment.
        
        Args:
            **kwargs: Optional parameters (e.g., seed, task)
        """
        # Merge task from init with any override
        reset_kwargs = {"task": kwargs.get("task", self.task)}
        reset_kwargs.update(kwargs)
        
        return super().reset(**reset_kwargs)
    
    def close(self):
        """Close the client connection."""
        pass
    
    def __str__(self) -> str:
        return f"SmartWarehouseEnv(base_url={self.base_url}, task={self.task})"
    
    def __repr__(self) -> str:
        return self.__str__()


# =============================================================================
# Tool-Based Interface (for GRPO/Agent Training)
# =============================================================================

class SmartWarehouseToolEnv:
    """
    Tool-based wrapper for SmartWarehouse.
    
    This follows the TRL environment_factory pattern where
    environment methods are exposed as tools for LLM agents.
    
    Usage:
        class MyEnvFactory:
            def __init__(self):
                self.client = SmartWarehouseEnv(base_url="http://localhost:8000")
                self.reward = 0.0
                self.done = False
            
            def reset(self, **kwargs):
                result = self.client.reset(**kwargs)
                self.done = False
                return result.observation.to_text()
            
            def restock(self, level: int) -> str:
                '''Restock inventory at the specified level.
                
                Args:
                    level: Restock level (0=none, 1=low, 2=medium, 3=high, 4=emergency)
                
                Returns:
                    Status message
                '''
                action = WarehouseAction(restock_level=level)
                result = self.client.step(action)
                self.reward = result.reward
                self.done = result.done
                return result.observation.to_text()
            
            # ... other tools
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", task: str = "easy"):
        self.client = SmartWarehouseEnv(base_url=base_url, task=task)
        self.reward = 0.0
        self.done = False
        self._last_observation: Optional[WarehouseObservation] = None
    
    def reset(self, **kwargs) -> str:
        """
        Reset the warehouse environment.
        
        Returns:
            Initial observation as text
        """
        result = self.client.reset(**kwargs)
        self.done = False
        self.reward = 0.0
        self._last_observation = result.observation
        return result.observation.to_text()
    
    def restock(self, level: int) -> str:
        """
        Order more inventory.
        
        Args:
            level: 0=none, 1=low(10%), 2=medium(25%), 3=high(50%), 4=emergency(100%)
        
        Returns:
            Status message
        """
        if self.done:
            return "Episode already complete. Call reset() to start a new episode."
        
        action = WarehouseAction(restock_level=level)
        result = self.client.step(action)
        self.reward = result.reward
        self.done = result.done
        self._last_observation = result.observation
        
        metrics = result.observation.current_metrics if hasattr(result.observation, 'current_metrics') else {}
        return result.observation.to_text()
    
    def set_allocation(self, strategy: int) -> str:
        """
        Set order allocation strategy.
        
        Args:
            strategy: 0=round_robin, 1=priority, 2=proximity, 3=balance, 4=dynamic
        
        Returns:
            Status message
        """
        if self.done:
            return "Episode already complete. Call reset() to start a new episode."
        
        action = WarehouseAction(allocation_strategy=strategy)
        result = self.client.step(action)
        self.reward = result.reward
        self.done = result.done
        self._last_observation = result.observation
        
        return result.observation.to_text()
    
    def assign_workers(self, mode: int) -> str:
        """
        Assign worker shift mode.
        
        Args:
            mode: 0=current, 1=overtime, 2=contractors, 3=automation
        
        Returns:
            Status message
        """
        if self.done:
            return "Episode already complete. Call reset() to start a new episode."
        
        action = WarehouseAction(worker_mode=mode)
        result = self.client.step(action)
        self.reward = result.reward
        self.done = result.done
        self._last_observation = result.observation
        
        return result.observation.to_text()
    
    def prioritize_delivery(self, priority: int) -> str:
        """
        Set delivery priority mode.
        
        Args:
            priority: 0=fifo, 1=urgency, 2=profit, 3=proximity
        
        Returns:
            Status message
        """
        if self.done:
            return "Episode already complete. Call reset() to start a new episode."
        
        action = WarehouseAction(delivery_priority=priority)
        result = self.client.step(action)
        self.reward = result.reward
        self.done = result.done
        self._last_observation = result.observation
        
        return result.observation.to_text()
    
    def optimize_storage(self, action: int) -> str:
        """
        Perform storage optimization.
        
        Args:
            action: 0=none, 1=rebalance, 2=consolidate
        
        Returns:
            Status message
        """
        if self.done:
            return "Episode already complete. Call reset() to start a new episode."
        
        action = WarehouseAction(storage_action=action)
        result = self.client.step(action)
        self.reward = result.reward
        self.done = result.done
        self._last_observation = result.observation
        
        return result.observation.to_text()
    
    def get_state(self) -> str:
        """
        Get current warehouse state.
        
        Returns:
            Current state as text
        """
        if self._last_observation is None:
            return "Environment not initialized. Call reset() first."
        return self._last_observation.to_text()
    
    def close(self):
        """Close the environment."""
        self.client.close()
