"""
SmartWarehouse Tasks Definition
=============================

Tasks for the SmartWarehouse RL environment.
"""

from .models import TaskDefinition

TASKS = [
    TaskDefinition(
        task_id=1,
        name="easy",
        description="Inventory Restocking - Optimize reorder decisions to maintain fill rate above 95%",
        required_fields=["restock_level"],
        level="easy",
    ),
    TaskDefinition(
        task_id=2,
        name="medium",
        description="Order Fulfillment - Optimize order picking and delivery across warehouse zones",
        required_fields=["restock_level", "allocation_strategy", "worker_mode"],
        level="medium",
    ),
    TaskDefinition(
        task_id=3,
        name="hard",
        description="Warehouse Optimization - Multi-objective optimization of customer satisfaction, costs, and throughput",
        required_fields=["restock_level", "allocation_strategy", "worker_mode", "delivery_priority", "storage_action"],
        level="hard",
    ),
]
