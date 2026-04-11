"""
SmartWarehouse Environment
=========================

A real-world warehouse management environment following OpenEnv patterns.

This environment simulates actual supply chain operations that warehouse
managers perform daily:
- Inventory restocking decisions
- Order allocation to zones
- Worker shift assignments
- Delivery scheduling
- Storage optimization

Real task, not a game or toy.
"""

__version__ = "1.0.0"

from .models import (
    WarehouseAction,
    WarehouseObservation,
    WarehouseState,
    TaskGrader,
    InventoryRestockingGrader,
    OrderFulfillmentGrader,
    WarehouseOptimizationGrader,
    TaskDefinition,
)
from .environment import WarehouseEnvironment, create_smart_warehouse_environment
from .client import SmartWarehouseEnv, SmartWarehouseToolEnv
from .graders import (
    AVAILABLE_GRADERS,
    GRADER_METADATA,
    get_grader,
    get_grader_metadata,
    list_all_tasks,
    get_all_grader_metadata,
)
from .tasks import TASKS

__all__ = [
    "WarehouseAction",
    "WarehouseObservation", 
    "WarehouseState",
    "TaskGrader",
    "InventoryRestockingGrader",
    "OrderFulfillmentGrader",
    "WarehouseOptimizationGrader",
    "TaskDefinition",
    "WarehouseEnvironment",
    "create_smart_warehouse_environment",
    "SmartWarehouseEnv",
    "SmartWarehouseToolEnv",
    "AVAILABLE_GRADERS",
    "GRADER_METADATA",
    "get_grader",
    "get_grader_metadata",
    "list_all_tasks",
    "get_all_grader_metadata",
    "TASKS",
]
