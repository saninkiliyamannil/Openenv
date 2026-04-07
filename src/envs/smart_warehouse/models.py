"""
SmartWarehouse Typed Models
===========================

Type-safe action, observation, and state models for SmartWarehouse.
Following the OpenEnv pattern for typed contracts.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

from core import Action, Observation, State


# =============================================================================
# Action Model
# =============================================================================

@dataclass
class WarehouseAction(Action):
    """
    Action for warehouse management decisions.
    
    Attributes:
        restock_level: 0=none, 1=low, 2=medium, 3=high, 4=emergency
        allocation_strategy: 0=round_robin, 1=priority_based, 2=proximity,
                           3=balance, 4=dynamic
        worker_mode: 0=current, 1=overtime, 2=contractors, 3=automation
        delivery_priority: 0=fifo, 1=urgency, 2=profit, 3=proximity
        storage_action: 0=none, 1=rebalance, 2=consolidate
    """
    restock_level: int = 0
    allocation_strategy: int = 0
    worker_mode: int = 0
    delivery_priority: int = 0
    storage_action: int = 0
    
    def __post_init__(self):
        self.restock_level = max(0, min(4, self.restock_level))
        self.allocation_strategy = max(0, min(4, self.allocation_strategy))
        self.worker_mode = max(0, min(3, self.worker_mode))
        self.delivery_priority = max(0, min(3, self.delivery_priority))
        self.storage_action = max(0, min(2, self.storage_action))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "restock_level": self.restock_level,
            "allocation_strategy": self.allocation_strategy,
            "worker_mode": self.worker_mode,
            "delivery_priority": self.delivery_priority,
            "storage_action": self.storage_action,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WarehouseAction":
        """Create from dictionary."""
        return cls(
            restock_level=int(data.get("restock_level", 0)),
            allocation_strategy=int(data.get("allocation_strategy", 0)),
            worker_mode=int(data.get("worker_mode", 0)),
            delivery_priority=int(data.get("delivery_priority", 0)),
            storage_action=int(data.get("storage_action", 0)),
        )
    
    def __str__(self) -> str:
        return (f"WarehouseAction(restock={self.restock_level}, "
                f"alloc={self.allocation_strategy}, workers={self.worker_mode}, "
                f"delivery={self.delivery_priority}, storage={self.storage_action})")


# =============================================================================
# Observation Model
# =============================================================================

@dataclass
class WarehouseObservation(Observation):
    """
    Observation from the warehouse environment.
    
    Attributes:
        inventory_levels: Current stock for 20 product categories [0-500]
        pending_orders: Pending orders by priority level [3 values]
        warehouse_zones: Zone utilization percentages [6 values, 0-1]
        worker_availability: Available workers per shift [4 values]
        delivery_vehicles: Available vehicles by capacity [4 values]
        time_features: Hour, day, weekend flag, season [4 values]
        demand_forecast: Predicted demand [10 values]
        done: Whether episode is complete
        reward: Current reward (if done)
        messages: Text messages for tool-calling agents
    """
    inventory_levels: List[float] = field(default_factory=list)
    pending_orders: List[float] = field(default_factory=list)
    warehouse_zones: List[float] = field(default_factory=list)
    worker_availability: List[float] = field(default_factory=list)
    delivery_vehicles: List[float] = field(default_factory=list)
    time_features: List[float] = field(default_factory=list)
    demand_forecast: List[float] = field(default_factory=list)
    done: bool = False
    reward: float = 0.0
    messages: List[str] = field(default_factory=list)
    
    def to_text(self) -> str:
        """Convert to human-readable text for LLM agents."""
        lines = ["=== WAREHOUSE STATE ==="]
        
        # Inventory summary
        inv_avg = sum(self.inventory_levels) / len(self.inventory_levels) if self.inventory_levels else 0
        lines.append(f"Inventory: avg {inv_avg:.1f}/500 across {len(self.inventory_levels)} categories")
        
        # Pending orders
        if self.pending_orders:
            lines.append(f"Pending Orders: urgent={self.pending_orders[0]:.0f}, "
                         f"normal={self.pending_orders[1]:.0f}, low={self.pending_orders[2]:.0f}")
        
        # Zone utilization
        if self.warehouse_zones:
            zones_str = ", ".join([f"{z:.0%}" for z in self.warehouse_zones[:3]])
            lines.append(f"Zone Utilization: [{zones_str}, ...]")
        
        # Time
        if self.time_features:
            hour = int(self.time_features[0])
            day = int(self.time_features[1])
            weekend = "Weekend" if self.time_features[2] else "Weekday"
            lines.append(f"Time: Hour {hour}, Day {day}, {weekend}")
        
        # Demand
        if self.demand_forecast:
            demand_avg = sum(self.demand_forecast) / len(self.demand_forecast)
            lines.append(f"Demand Forecast: avg {demand_avg:.1f}")
        
        # Messages for tool-calling
        if self.messages:
            lines.append("")
            lines.append("Messages:")
            for msg in self.messages[-3:]:
                lines.append(f"  - {msg}")
        
        return "\n".join(lines)
    
    def to_array(self) -> List[float]:
        """Convert to flat array for RL."""
        return (
            self.inventory_levels +
            self.pending_orders +
            self.warehouse_zones +
            self.worker_availability +
            self.delivery_vehicles +
            self.time_features +
            self.demand_forecast
        )


# =============================================================================
# State Model
# =============================================================================

@dataclass
class WarehouseState(State):
    """
    Episode state and metadata.
    
    Attributes:
        episode_id: Unique episode identifier
        step_count: Number of steps taken
        task: Current task difficulty
        total_reward: Cumulative reward
        metrics: Current operational metrics
    """
    episode_id: str = ""
    step_count: int = 0
    task: str = "easy"
    total_reward: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.episode_id:
            import uuid
            self.episode_id = str(uuid.uuid4())[:8]


# =============================================================================
# Task Graders
# =============================================================================

class TaskGrader:
    """Base grader for scoring agent performance."""
    
    def grade(self, trajectory: List[Dict], final_state: Dict) -> float:
        """Grade agent performance (0.0 to 1.0)."""
        raise NotImplementedError
    
    def get_success_criteria(self) -> Dict[str, Any]:
        """Return success criteria."""
        raise NotImplementedError


class InventoryRestockingGrader(TaskGrader):
    """Grader for easy inventory restocking task."""
    
    def __init__(self):
        self.target_fill_rate = 0.95
        self.max_stockout_hours = 4
    
    def grade(self, trajectory: List[Dict], final_state: Dict) -> float:
        if not trajectory:
            return 0.0
        
        fill_rates = [s.get("fill_rate", 0) for s in trajectory]
        avg_fill = sum(fill_rates) / len(fill_rates) if fill_rates else 0
        fill_score = min(1.0, avg_fill / self.target_fill_rate)
        
        stockouts = [s.get("stockout_hours", 0) for s in trajectory]
        max_stockout = max(stockouts) if stockouts else 0
        stockout_score = max(0, 1 - max_stockout / self.max_stockout_hours)
        
        score = 0.6 * fill_score + 0.4 * stockout_score
        return float(max(0.0, min(1.0, score)))
    
    def get_success_criteria(self) -> Dict[str, Any]:
        return {"target_fill_rate": self.target_fill_rate, "passing_score": 0.6}


class OrderFulfillmentGrader(TaskGrader):
    """Grader for medium order fulfillment task."""
    
    def __init__(self):
        self.target_fulfillment = 0.92
        self.target_delivery_hours = 4.0
    
    def grade(self, trajectory: List[Dict], final_state: Dict) -> float:
        if not trajectory:
            return 0.0
        
        fulfillments = [s.get("fulfillment_rate", 0) for s in trajectory]
        avg_fulfill = sum(fulfillments) / len(fulfillments) if fulfillments else 0
        fulfill_score = min(1.0, avg_fulfill / self.target_fulfillment)
        
        deliveries = [s.get("avg_delivery_hours", 0) for s in trajectory]
        avg_delivery = sum(deliveries) / len(deliveries) if deliveries else 0
        delivery_score = max(0, 1 - avg_delivery / self.target_delivery_hours)
        
        score = 0.5 * fulfill_score + 0.5 * delivery_score
        return float(max(0.0, min(1.0, score)))
    
    def get_success_criteria(self) -> Dict[str, Any]:
        return {"target_fulfillment": self.target_fulfillment, "passing_score": 0.65}


class WarehouseOptimizationGrader(TaskGrader):
    """Grader for hard warehouse optimization task."""
    
    def __init__(self):
        self.target_composite = 0.75
    
    def grade(self, trajectory: List[Dict], final_state: Dict) -> float:
        if not trajectory:
            return 0.0
        
        scores = {}
        
        satisfactions = [s.get("customer_satisfaction", 0) for s in trajectory]
        scores["satisfaction"] = sum(satisfactions) / len(satisfactions) if satisfactions else 0
        
        costs = [s.get("operational_cost", 100) for s in trajectory]
        baseline = final_state.get("baseline_cost", 100)
        scores["cost"] = max(0, min(1, 1 - (sum(costs) / len(costs)) / baseline))
        
        throughputs = [s.get("throughput", 0) for s in trajectory]
        scores["throughput"] = min(1.0, sum(throughputs) / len(throughputs) / 100) if throughputs else 0
        
        score = (0.35 * scores["satisfaction"] + 
                 0.35 * scores["cost"] + 
                 0.30 * scores["throughput"])
        
        return float(max(0.0, min(1.0, score)))
    
    def get_success_criteria(self) -> Dict[str, Any]:
        return {"target_composite": self.target_composite, "passing_score": 0.70}
