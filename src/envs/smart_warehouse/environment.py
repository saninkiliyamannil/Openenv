"""
SmartWarehouse Environment Implementation
=========================================

The core environment logic following OpenEnv's Environment base class.
"""

import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import numpy as np

from core import Environment
from .models import (
    WarehouseAction,
    WarehouseObservation,
    WarehouseState,
    TaskGrader,
    InventoryRestockingGrader,
    OrderFulfillmentGrader,
    WarehouseOptimizationGrader,
)


# =============================================================================
# Configuration
# =============================================================================

WAREHOUSE_CONFIG = {
    "num_products": 20,
    "num_zones": 6,
    "num_workers": 4,
    "num_vehicles": 4,
    "order_priorities": 3,
    "max_inventory": 500,
    "max_episode_steps": 168,
}


# =============================================================================
# Environment Implementation
# =============================================================================

class WarehouseEnvironment(Environment):
    """
    SmartWarehouse RL environment.
    
    Simulates real-world warehouse management operations including:
    - Inventory restocking decisions
    - Order allocation to zones
    - Worker shift assignments
    - Delivery scheduling
    - Storage optimization
    
    This is NOT a game - it's a realistic supply chain simulation
    that humans perform as actual jobs.
    
    Usage:
        env = WarehouseEnvironment(task="easy")
        obs = env.reset()
        obs = env.step(WarehouseAction(restock_level=1))
        state = env.state
    """
    
    SUPPORTS_CONCURRENT_SESSIONS = True
    
    def __init__(
        self,
        task: str = "easy",
        seed: Optional[int] = None,
        max_episode_steps: int = 168,
    ):
        """
        Initialize the warehouse environment.
        
        Args:
            task: Difficulty level - "easy", "medium", or "hard"
            seed: Random seed for reproducibility
            max_episode_steps: Maximum steps per episode
        """
        self.task = task.lower()
        self.max_episode_steps = max_episode_steps
        
        # Set up RNG
        self.seed = seed
        self._np_random = np.random.RandomState(seed)
        random.seed(seed)
        
        # Episode tracking
        self._step_count = 0
        self._total_reward = 0.0
        self._trajectory: List[Dict] = []
        self._messages: List[str] = []
        
        # Set up grader
        self._setup_grader()
        
        # Initialize warehouse state
        self._state: Optional[WarehouseState] = None
        self._init_warehouse_state()
    
    def _setup_grader(self):
        """Set up the grader based on task difficulty."""
        graders = {
            "easy": InventoryRestockingGrader(),
            "medium": OrderFulfillmentGrader(),
            "hard": WarehouseOptimizationGrader(),
        }
        self._grader: TaskGrader = graders.get(self.task, graders["easy"])
        
        # Adjust max steps based on task
        task_steps = {
            "easy": 72,
            "medium": 120,
            "hard": 168,
        }
        self.max_episode_steps = task_steps.get(self.task, self.max_episode_steps)
    
    def _init_warehouse_state(self):
        """Initialize warehouse state variables."""
        cfg = WAREHOUSE_CONFIG
        
        # Inventory levels for each product
        self._inventory = self._np_random.uniform(
            50, cfg["max_inventory"] * 0.5, cfg["num_products"]
        ).astype(np.float32)
        
        # Pending orders by priority
        self._pending_orders = np.zeros(cfg["order_priorities"], dtype=np.float32)
        
        # Zone utilization
        self._zone_utilization = self._np_random.uniform(
            0.2, 0.8, cfg["num_zones"]
        ).astype(np.float32)
        
        # Worker availability
        self._worker_availability = self._np_random.uniform(
            5, 15, cfg["num_workers"]
        ).astype(np.float32)
        
        # Delivery vehicles
        self._vehicles = np.array([8, 5, 3, 2], dtype=np.float32)
        
        # Demand forecast
        self._demand_forecast = self._np_random.uniform(
            10, 50, cfg["num_products"]
        ).astype(np.float32)
        
        # Time features
        self._hour_of_day = 8
        self._day_of_week = 0
        self._season = 0
        
        # Operational metrics
        self._fill_rate = 0.85
        self._avg_delivery_hours = 6.0
        self._operational_cost = 80.0
        self._customer_satisfaction = 0.8
        self._throughput = 100.0
        
        # Episode tracking
        self._total_orders_fulfilled = 0
        self._total_stockout_hours = 0.0
        self._ordering_costs = 0.0
        
        # Baseline for grading
        self._baseline_cost = 100.0
        self._baseline_throughput = 100.0
    
    # =========================================================================
    # Environment API (Core Methods)
    # =========================================================================
    
    def reset(self) -> WarehouseObservation:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        import uuid
        
        self._step_count = 0
        self._total_reward = 0.0
        self._trajectory = []
        self._messages = []
        
        self._init_warehouse_state()
        
        # Create state object
        self._state = WarehouseState(
            episode_id=str(uuid.uuid4())[:8],
            step_count=0,
            task=self.task,
            total_reward=0.0,
            metrics=self._get_metrics(),
        )
        
        self._messages.append(f"Episode started. Task: {self.task}. Maximize fill rate while minimizing costs.")
        
        return self._get_observation()
    
    def step(self, action: WarehouseAction) -> WarehouseObservation:
        """
        Execute one environment step.
        
        Args:
            action: The warehouse action to take
            
        Returns:
            The resulting observation
        """
        # Execute action effects
        self._execute_action(action)
        
        # Update environment dynamics
        self._update_dynamics()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        self._total_reward += reward
        
        # Update step count
        self._step_count += 1
        
        # Record trajectory for grading
        self._trajectory.append(self._get_metrics())
        
        # Update state
        self._state.step_count = self._step_count
        self._state.total_reward = self._total_reward
        self._state.metrics = self._get_metrics()
        
        # Check termination
        done = self._check_termination()
        
        obs = self._get_observation(done=done, reward=reward if done else 0.0)
        
        # Add messages for tool-calling agents
        if done:
            final_score = self._grader.grade(self._trajectory, self._get_final_state())
            self._messages.append(
                f"Episode complete! Score: {final_score:.3f}. "
                f"Fill rate: {self._fill_rate:.1%}, Cost: ${self._operational_cost:.2f}"
            )
        
        return obs
    
    @property
    def state(self) -> WarehouseState:
        """Get current episode state."""
        if self._state is None:
            self.reset()
        return self._state
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _execute_action(self, action: WarehouseAction):
        """Execute action effects on warehouse state."""
        cfg = WAREHOUSE_CONFIG
        
        # 1. Restock inventory
        multipliers = [0, 0.1, 0.25, 0.5, 1.0]
        multiplier = multipliers[action.restock_level]
        restock_amount = self._demand_forecast * multiplier
        
        for i in range(len(self._inventory)):
            if multiplier > 0:
                order_cost = restock_amount[i] * 2
                self._ordering_costs += order_cost
                self._inventory[i] = min(
                    self._inventory[i] + restock_amount[i],
                    cfg["max_inventory"]
                )
        
        # 2. Allocate orders based on strategy
        new_orders = self._np_random.poisson(5, cfg["order_priorities"])
        self._pending_orders += new_orders
        
        # Calculate allocation efficiency
        allocation_efficiency = [0.85, 0.95, 0.90, 0.92, 0.98][action.allocation_strategy]
        
        # 3. Process orders
        for i in range(len(self._pending_orders)):
            fulfilled = min(
                self._pending_orders[i],
                np.sum(self._inventory > 20) * allocation_efficiency
            )
            self._pending_orders[i] = max(0, self._pending_orders[i] - fulfilled)
            self._total_orders_fulfilled += fulfilled
        
        # 4. Worker assignment effects
        worker_effects = [1.0, 1.2, 1.5, 0.8]
        efficiency = worker_effects[action.worker_mode]
        self._throughput *= efficiency
        
        # 5. Delivery priority effects
        delivery_effects = [0.9, 1.1, 1.0, 1.05]
        self._avg_delivery_hours *= delivery_effects[action.delivery_priority]
        
        # 6. Storage optimization
        if action.storage_action == 1:
            self._zone_utilization = self._np_random.uniform(0.4, 0.7, cfg["num_zones"])
        elif action.storage_action == 2:
            avg_util = np.mean(self._zone_utilization)
            self._zone_utilization = np.full(cfg["num_zones"], avg_util)
        
        # Update derived metrics
        self._update_metrics()
    
    def _update_dynamics(self):
        """Update environment stochastic dynamics."""
        cfg = WAREHOUSE_CONFIG
        
        # Time progression
        self._hour_of_day = (self._hour_of_day + 1) % 24
        if self._hour_of_day == 0:
            self._day_of_week = (self._day_of_week + 1) % 7
        
        # Demand patterns
        hourly_factor = 1.0 + 0.3 * np.sin(2 * np.pi * self._hour_of_day / 24)
        weekday_factor = 1.2 if self._day_of_week < 5 else 0.7
        
        # Update demand forecast
        base_demand = self._np_random.uniform(10, 50, cfg["num_products"])
        self._demand_forecast = base_demand * hourly_factor * weekday_factor
        
        # Supply disruptions
        if self._np_random.random() < 0.02:
            for i in range(len(self._inventory)):
                self._inventory[i] *= 0.9
        
        # Natural consumption
        consumption = self._np_random.uniform(5, 15, cfg["num_products"])
        self._inventory = np.maximum(0, self._inventory - consumption)
        
        # Stockout tracking
        stockout_count = np.sum(self._inventory < 10)
        self._total_stockout_hours += stockout_count
    
    def _update_metrics(self):
        """Update derived operational metrics."""
        total_demand = np.sum(self._demand_forecast)
        if total_demand > 0:
            self._fill_rate = min(1.0, self._total_orders_fulfilled / max(1, total_demand))
        
        self._customer_satisfaction = (
            0.6 * self._fill_rate +
            0.4 * max(0, 1 - self._avg_delivery_hours / 48)
        )
        
        total_cost = self._operational_cost + self._ordering_costs
        self._operational_cost = total_cost / max(1, self._step_count + 1)
    
    def _get_observation(self, done: bool = False, reward: float = 0.0) -> WarehouseObservation:
        """Build current observation."""
        return WarehouseObservation(
            inventory_levels=self._inventory.tolist(),
            pending_orders=self._pending_orders.tolist(),
            warehouse_zones=self._zone_utilization.tolist(),
            worker_availability=self._worker_availability.tolist(),
            delivery_vehicles=self._vehicles.tolist(),
            time_features=[
                float(self._hour_of_day),
                float(self._day_of_week),
                1.0 if self._day_of_week >= 5 else 0.0,
                float(self._season),
            ],
            demand_forecast=self._demand_forecast.tolist(),
            done=done,
            reward=reward,
            messages=self._messages.copy(),
        )
    
    def _get_metrics(self) -> Dict[str, float]:
        """Get current step metrics."""
        return {
            "fill_rate": float(self._fill_rate),
            "fulfillment_rate": float(self._total_orders_fulfilled / max(1, self._step_count + 1)),
            "avg_delivery_hours": float(self._avg_delivery_hours),
            "operational_cost": float(self._operational_cost),
            "customer_satisfaction": float(self._customer_satisfaction),
            "throughput": float(self._throughput),
            "stockout_hours": float(self._total_stockout_hours),
        }
    
    def _get_final_state(self) -> Dict:
        """Get final state for grading."""
        return {
            "baseline_cost": self._baseline_cost,
            "baseline_throughput": self._baseline_throughput,
            "total_orders_fulfilled": self._total_orders_fulfilled,
        }
    
    def _calculate_reward(self, action: WarehouseAction) -> float:
        """
        Calculate reward with partial progress signals.
        
        Reward components:
        - stockout_avoidance: Higher is better
        - cost_efficiency: Lower cost is better
        - service_level: Higher fill rate is better
        """
        metrics = self._get_metrics()
        
        # Component 1: Stockout avoidance (0 to 1)
        stockout_reward = max(0, 1 - metrics["stockout_hours"] / 10)
        stockout_reward *= 0.3
        
        # Component 2: Cost efficiency (-1 to 1)
        cost_ratio = metrics["operational_cost"] / max(1, self._baseline_cost)
        cost_reward = (1 - cost_ratio) * 0.25
        cost_reward = max(-1, min(1, cost_reward))
        
        # Component 3: Service level (-1 to 1)
        fill_rate = metrics["fill_rate"]
        service_reward = (fill_rate - 0.5) * 0.6
        service_reward = max(-1, min(1, service_reward))
        
        total_reward = stockout_reward + cost_reward + service_reward
        return float(np.clip(total_reward, -1, 1))
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Time limit
        if self._step_count >= self.max_episode_steps:
            return True
        
        # Critical stockout
        if np.sum(self._inventory < 5) > 10:
            return True
        
        # Perfect performance
        if self._fill_rate >= 0.99 and self._step_count >= 24:
            return True
        
        return False
    
    def grade(self) -> float:
        """Grade the current episode."""
        return self._grader.grade(self._trajectory, self._get_final_state())
    
    def grade_task(self, task: str) -> float:
        """Grade the current episode with a specific task grader."""
        from .models import (
            InventoryRestockingGrader,
            OrderFulfillmentGrader,
            WarehouseOptimizationGrader,
        )
        
        graders = {
            "easy": InventoryRestockingGrader(),
            "medium": OrderFulfillmentGrader(),
            "hard": WarehouseOptimizationGrader(),
        }
        
        grader = graders.get(task, self._grader)
        return grader.grade(self._trajectory, self._get_final_state())


# Alias for consistency
create_smart_warehouse_environment = WarehouseEnvironment
