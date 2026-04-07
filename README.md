---
title: Openenv
emoji: 📦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.11"
app_file: app.py
pinned: false
suggested_hardware: t4-small
suggested_storage: small
license: mit
library: gradio
---

# SmartWarehouse - OpenEnv
## Real-World Warehouse Management RL Environment

A production-ready OpenEnv environment that simulates actual supply chain operations.

### What This Is

**Not a game** - This is a realistic simulation of real-world job tasks:

- **Inventory Management**: Deciding when and how much to reorder products
- **Order Processing**: Allocating and fulfilling customer orders
- **Resource Scheduling**: Managing workers and delivery vehicles
- **Storage Optimization**: Organizing warehouse space efficiently

### Features

- **3 Difficulty Levels**: Easy (0.6), Medium (0.65), Hard (0.70)
- **OpenEnv Compliant**: `reset()` / `step()` / `state` API
- **Agent Graders**: Programmatic scoring (0.0-1.0)
- **Partial Rewards**: Continuous reward signal

## Quick Start

```python
import sys
sys.path.insert(0, "src")

from envs.smart_warehouse.environment import WarehouseEnvironment
from envs.smart_warehouse.models import WarehouseAction

env = WarehouseEnvironment(task="easy")
obs = env.reset()
action = WarehouseAction(restock_level=1, allocation_strategy=1)
obs = env.step(action)
```

## Tasks with Graders

| Task | Passing Score | Objective |
|------|--------------|-----------|
| Easy | 0.6 | Inventory restocking |
| Medium | 0.65 | Order fulfillment |
| Hard | 0.70 | Warehouse optimization |

## License

MIT
