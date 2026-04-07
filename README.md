# SmartWarehouse - OpenEnv
## Real-World Warehouse Management RL Environment

A production-ready OpenEnv environment that simulates actual supply chain operations that humans perform as warehouse managers.

### What This Is

**Not a game** - This is a realistic simulation of real-world job tasks:

- **Inventory Management**: Deciding when and how much to reorder products
- **Order Processing**: Allocating and fulfilling customer orders
- **Resource Scheduling**: Managing workers and delivery vehicles
- **Storage Optimization**: Organizing warehouse space efficiently

### OpenEnv Compliant

Follows the official Meta PyTorch OpenEnv patterns:

- ✅ Typed `Action`, `Observation`, `State` models
- ✅ `reset()` / `step()` / `state` API
- ✅ HTTP client-server architecture
- ✅ Docker deployment ready
- ✅ Gradio web interface

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Python API

```python
import sys
sys.path.insert(0, "src")

from envs.smart_warehouse import SmartWarehouseEnv
from envs.smart_warehouse.models import WarehouseAction

# Connect to server
env = SmartWarehouseEnv(base_url="http://localhost:8000")

# Reset
obs = env.reset()
print(obs.to_text())

# Take action
action = WarehouseAction(restock_level=2, allocation_strategy=1)
result = env.step(action)
print(f"Reward: {result.reward}")
```

### Local Server

```bash
cd src/envs/smart_warehouse/server
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t smart-warehouse -f src/envs/smart_warehouse/server/Dockerfile .
docker run -p 8000:8000 smart-warehouse
```

## 3 Tasks with Graders

### Easy: Inventory Restocking
- **Objective**: Maintain 95%+ fill rate while minimizing costs
- **Passing Score**: 0.6
- **Grader**: Rewards high fill rate, penalizes stockouts

### Medium: Order Fulfillment
- **Objective**: Optimize order picking and delivery
- **Passing Score**: 0.65
- **Grader**: Rewards fulfillment rate and delivery speed

### Hard: Warehouse Optimization
- **Objective**: Multi-objective optimization
- **Passing Score**: 0.70
- **Grader**: Composite of satisfaction, cost, throughput

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | - | API key |
| `TASK_NAME` | `easy` | Task difficulty |
| `MAX_STEPS` | `50` | Max episode steps |

## Inference Script

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."
export TASK_NAME="easy"

python inference.py
```

Output format:
```
[START] task=easy env=SmartWarehouse model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=restock(1) alloc(1) reward=0.15 done=false error=null
[STEP] step=2 action=restock(2) alloc(1) reward=0.22 done=false error=null
[END] success=true steps=50 score=0.723 rewards=0.15,0.22,...
```

## Validation

```bash
python validate.py
```

## File Structure

```
OpenEnv/
├── src/
│   ├── core/                  # OpenEnv core abstractions
│   │   └── __init__.py
│   └── envs/smart_warehouse/
│       ├── models.py          # Typed models (Action, Observation, State)
│       ├── environment.py      # Core environment logic
│       ├── client.py          # HTTP client
│       └── server/
│           ├── app.py         # FastAPI server
│           └── Dockerfile     # Container deployment
├── inference.py              # LLM baseline evaluation
├── validate.py               # Validation suite
├── requirements.txt
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment |
| `/step` | POST | Execute action |
| `/state` | GET | Get current state |
| `/grade` | POST | Get episode score |

## License

MIT
