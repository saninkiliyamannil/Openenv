"""
SmartWarehouse Graders
=====================

Explicit grader definitions for all tasks.
These graders are used to score agent performance.
"""

from typing import Dict, Any, List
from .models import TaskGrader, InventoryRestockingGrader, OrderFulfillmentGrader, WarehouseOptimizationGrader


AVAILABLE_GRADERS: Dict[str, TaskGrader] = {
    "easy": InventoryRestockingGrader(),
    "medium": OrderFulfillmentGrader(),
    "hard": WarehouseOptimizationGrader(),
}

GRADER_METADATA: Dict[str, Dict[str, Any]] = {
    "easy": {
        "id": "inventory_restocking",
        "name": "Inventory Restocking",
        "description": "Optimize reorder decisions to maintain fill rate above 95%",
        "passing_score": 0.6,
        "max_score": 1.0,
        "criteria": {
            "target_fill_rate": 0.95,
            "max_stockout_hours": 4,
        },
    },
    "medium": {
        "id": "order_fulfillment",
        "name": "Order Fulfillment",
        "description": "Optimize order picking and delivery across warehouse zones",
        "passing_score": 0.65,
        "max_score": 1.0,
        "criteria": {
            "target_fulfillment_rate": 0.92,
            "target_delivery_hours": 4.0,
        },
    },
    "hard": {
        "id": "warehouse_optimization",
        "name": "Warehouse Optimization",
        "description": "Multi-objective optimization of customer satisfaction, costs, and throughput",
        "passing_score": 0.70,
        "max_score": 1.0,
        "criteria": {
            "target_composite_score": 0.75,
        },
    },
}


def get_grader(task: str) -> TaskGrader:
    """Get grader instance for the specified task."""
    return AVAILABLE_GRADERS.get(task, AVAILABLE_GRADERS["easy"])


def get_grader_metadata(task: str) -> Dict[str, Any]:
    """Get grader metadata for the specified task."""
    return GRADER_METADATA.get(task, GRADER_METADATA["easy"])


def list_all_tasks() -> List[str]:
    """List all available tasks."""
    return list(AVAILABLE_GRADERS.keys())


def get_all_grader_metadata() -> Dict[str, Dict[str, Any]]:
    """Get metadata for all graders."""
    return GRADER_METADATA.copy()
