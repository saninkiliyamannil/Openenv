title: SmartWarehouse
emoji: 📦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
suggested_hardware: t4-small
suggested_storage: 3gb
license: mit
library: gradio
---

# SmartWarehouse - OpenEnv

Real-world warehouse logistics optimization RL environment.

## Features

- **3 Tasks**: Easy (restocking), Medium (fulfillment), Hard (optimization)
- **Real Simulation**: Models actual supply chain operations
- **Agent Graders**: Scores in 0.0-1.0 range
- **Partial Rewards**: Continuous reward signal

## Usage

```python
from envs.smart_warehouse import SmartWarehouseEnv
from envs.smart_warehouse.models import WarehouseAction

env = SmartWarehouseEnv(base_url="http://localhost:8000")
obs = env.reset()
action = WarehouseAction(restock_level=2)
result = env.step(action)
```
