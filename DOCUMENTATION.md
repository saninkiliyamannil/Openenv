# OpenEnv SmartWarehouse - Complete Documentation

## Table of Contents

1. [What is OpenEnv?](#what-is-openenv)
2. [What is SmartWarehouse?](#what-is-smartwarehouse)
3. [Why This Matters](#why-this-matters)
4. [Architecture Overview](#architecture-overview)
5. [Installation](#installation)
6. [Usage Guide](#usage-guide)
7. [API Reference](#api-reference)
8. [Tasks and Graders](#tasks-and-graders)
9. [Reward System](#reward-system)
10. [Inference and Evaluation](#inference-and-evaluation)
11. [Deployment](#deployment)
12. [Troubleshooting](#troubleshooting)

---

## What is OpenEnv?

**OpenEnv** is Meta PyTorch's framework for building reinforcement learning (RL) environments. It's designed to make RL as easy to use as REST APIs.

### The Problem OpenEnv Solves

Traditional RL environments have several issues:

| Issue | Traditional Approach | OpenEnv Solution |
|-------|---------------------|------------------|
| **Type Safety** | `obs[0][3]` - what is this? | `obs.inventory_levels` - IDE knows! |
| **Isolation** | Same process (crashes affect training) | Docker containers (fully isolated) |
| **Deployment** | "Works on my machine" | Same container everywhere |
| **Scaling** | Hard to distribute | Deploy to cloud with one command |
| **Language** | Python only | Any language (HTTP API) |

### Key Concepts

OpenEnv environments consist of:

1. **Typed Models** - Actions, Observations, and States are proper classes
2. **Client-Server Architecture** - Environments run in isolated containers
3. **Standard API** - `reset()`, `step()`, `state` work the same everywhere
4. **Tool Interface** - Methods exposed as tools for LLM agents

---

## What is SmartWarehouse?

**SmartWarehouse** is a real-world RL environment that simulates actual warehouse operations - NOT a game or toy problem.

### Real-World Task Simulation

This environment models decisions that humans actually make as warehouse managers:

```
┌─────────────────────────────────────────────────────────────┐
│                    WAREHOUSE MANAGER                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐│
│  │  INVENTORY   │    │     ORDERS    │    │   WORKERS    ││
│  │  Deciding    │    │  Allocating   │    │  Scheduling  ││
│  │  when and    │    │  orders to    │    │  shifts and  ││
│  │  how much    │    │  zones based  │    │  assignments ││
│  │  to reorder  │    │  on priority  │    │              ││
│  └──────────────┘    └──────────────┘    └──────────────┘│
│                                                             │
│  ┌──────────────┐    ┌──────────────┐                     │
│  │   DELIVERY   │    │   STORAGE    │                     │
│  │  Scheduling  │    │  Optimizing  │                     │
│  │  routes and  │    │  warehouse   │                     │
│  │  priorities  │    │  space usage  │                     │
│  └──────────────┘    └──────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### What Makes It Real-World?

Unlike games where:
- The "goal" is arbitrary
- Success criteria are invented
- The task has no real-world parallel

SmartWarehouse simulates:
- **Actual business operations** - Warehouse management is a real $1.5 trillion industry
- **Real constraints** - Limited inventory, workers, vehicles, storage space
- **Real consequences** - Stockouts lose customers, excess inventory costs money
- **Real trade-offs** - Fast delivery costs more, cheap ordering risks stockouts

---

## Why This Matters

### For AI Research

1. **Beyond Games** - Most RL benchmarks are games. Real-world tasks are harder.
2. **Production Ready** - Docker deployment means environments work in production
3. **LLM Integration** - Tool-based interface enables training language models as agents

### For Industry

1. **Simulation Before Deployment** - Test warehouse strategies in simulation
2. **Training AI Agents** - Train models to make operational decisions
3. **Benchmarking** - Compare different AI approaches on the same task

### For Education

1. **Practical RL** - Learn RL concepts with realistic examples
2. **System Design** - Understand client-server architecture
3. **Deployment** - See how to ship ML systems to production

---

## Architecture Overview

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING CODE                                 │
│                                                                     │
│   from envs.smart_warehouse import SmartWarehouseEnv                │
│   from envs.smart_warehouse.models import WarehouseAction           │
│                                                                     │
│   env = SmartWarehouseEnv(base_url="http://localhost:8000")        │
│   obs = env.reset()                                               │
│   result = env.step(WarehouseAction(restock_level=1))             │
│                                                                     │
└─────────────────────────┬─────────────────────────────────────────┘
                          │
                          │ HTTP/JSON
                          │ POST /reset, POST /step, GET /state
                          │
┌─────────────────────────▼─────────────────────────────────────────┐
│                      DOCKER CONTAINER                               │
│                                                                     │
│   ┌───────────────────────────────────────────────────────────┐    │
│   │  FastAPI Server (uvicorn)                                │    │
│   │                                                           │    │
│   │   POST /reset    → WarehouseEnvironment.reset()          │    │
│   │   POST /step     → WarehouseEnvironment.step()           │    │
│   │   GET /state     → WarehouseEnvironment.state            │    │
│   │                                                           │    │
│   └───────────────────────────────────────────────────────────┘    │
│                                                                     │
│   Isolated • Reproducible • Secure                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### File Structure

```
OpenEnv/
│
├── src/                              # Source code
│   │
│   ├── core/                        # OpenEnv framework
│   │   └── __init__.py              # Base classes (Environment, Action, Observation)
│   │
│   └── envs/smart_warehouse/        # SmartWarehouse environment
│       │
│       ├── models.py                # Typed models
│       │   ├── WarehouseAction      # What agent can do
│       │   ├── WarehouseObservation # What agent sees
│       │   └── WarehouseState      # Episode metadata
│       │
│       ├── environment.py           # Core logic
│       │   └── WarehouseEnvironment # The RL environment
│       │
│       ├── client.py                # HTTP client
│       │   ├── SmartWarehouseEnv   # HTTPEnvClient implementation
│       │   └── SmartWarehouseToolEnv # For LLM agents
│       │
│       └── server/                  # Deployment
│           ├── app.py               # FastAPI application
│           ├── Dockerfile           # Container definition
│           └── requirements.txt     # Dependencies
│
├── inference.py                      # LLM evaluation script
├── validate.py                       # Validation suite
├── app.py                           # Gradio web UI
├── openenv.yaml                     # Environment specification
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Installation

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- pip or conda

### Option 1: Full Installation

```bash
# Clone the repository
git clone https://github.com/your-org/OpenEnv.git
cd OpenEnv

# Install dependencies
pip install -r requirements.txt

# Verify installation
python validate.py
```

### Option 2: Install as Package

```bash
# Install in development mode
pip install -e src/

# Or install specific environment
pip install -e src/envs/smart_warehouse/
```

### Option 3: Docker Only

```bash
# Build container
docker build -t smart-warehouse -f src/envs/smart_warehouse/server/Dockerfile .

# Run container
docker run -p 8000:8000 smart-warehouse
```

---

## Usage Guide

### 1. Basic RL Loop

```python
import sys
sys.path.insert(0, "src")

from envs.smart_warehouse import SmartWarehouseEnv
from envs.smart_warehouse.models import WarehouseAction

# Create environment
env = SmartWarehouseEnv(base_url="http://localhost:8000")

# Reset to get initial observation
obs = env.reset()
print(obs.to_text())

# Run episode
for step in range(100):
    # Choose action (here: random)
    action = WarehouseAction(
        restock_level=step % 5,
        allocation_strategy=step % 5,
        worker_mode=step % 4,
        delivery_priority=step % 4,
        storage_action=step % 3
    )
    
    # Execute action
    result = env.step(action)
    
    print(f"Step {step}: reward={result.reward:.3f}, done={result.done}")
    
    if result.done:
        print(f"Episode complete! Score: {env.grade():.3f}")
        break

env.close()
```

### 2. Direct Environment (No Server)

```python
import sys
sys.path.insert(0, "src")

from envs.smart_warehouse.environment import WarehouseEnvironment
from envs.smart_warehouse.models import WarehouseAction

# Create environment directly
env = WarehouseEnvironment(task="easy", seed=42, max_episode_steps=100)

# Reset
obs = env.reset()

# Run episode
for step in range(100):
    action = WarehouseAction(restock_level=1, allocation_strategy=1)
    obs = env.step(action)
    
    if obs.done:
        print(f"Episode done! Score: {env.grade():.3f}")
        break

# Get final state
state = env.state
print(f"Total reward: {state.total_reward}")
print(f"Steps taken: {state.step_count}")
```

### 3. Tool-Based Interface (For LLM Agents)

```python
import sys
sys.path.insert(0, "src")

from envs.smart_warehouse.client import SmartWarehouseToolEnv

# Create tool environment
env = SmartWarehouseToolEnv(base_url="http://localhost:8000", task="easy")

# Reset
initial_state = env.reset()
print(initial_state)

# Use tools (these become function-calling tools for LLMs)
result = env.restock(level=2)
print(result)

result = env.set_allocation(strategy=1)
print(result)

result = env.assign_workers(mode=1)
print(result)

# Check state
state = env.get_state()
print(state)

env.close()
```

### 4. Gradio Web Interface

```bash
# Run the web UI
python app.py

# Opens at http://localhost:7860
```

This provides a visual interface to interact with the environment.

---

## API Reference

### WarehouseAction

```python
@dataclass
class WarehouseAction:
    restock_level: int        # 0=none, 1=low, 2=medium, 3=high, 4=emergency
    allocation_strategy: int   # 0=round_robin, 1=priority, 2=proximity, 3=balance, 4=dynamic
    worker_mode: int          # 0=current, 1=overtime, 2=contractors, 3=automation
    delivery_priority: int    # 0=fifo, 1=urgency, 2=profit, 3=proximity
    storage_action: int       # 0=none, 1=rebalance, 2=consolidate
```

**Methods:**
- `to_dict()` → Convert to JSON-serializable dict
- `from_dict(data)` → Create from dict

### WarehouseObservation

```python
@dataclass
class WarehouseObservation:
    inventory_levels: List[float]    # [20] Stock for each category (0-500)
    pending_orders: List[float]     # [3] Orders by priority
    warehouse_zones: List[float]   # [6] Zone utilization (0-1)
    worker_availability: List[float] # [4] Workers per shift
    delivery_vehicles: List[float]  # [4] Vehicles by capacity
    time_features: List[float]      # [4] Hour, day, weekend, season
    demand_forecast: List[float]    # [10] Predicted demand
    done: bool                      # Episode complete
    reward: float                   # Current reward
    messages: List[str]             # Status messages
```

**Methods:**
- `to_text()` → Human-readable format for LLM agents
- `to_array()` → Flat numpy array for RL

### WarehouseEnvironment

```python
class WarehouseEnvironment:
    def __init__(self, task="easy", seed=None, max_episode_steps=168)
    
    def reset(self) -> WarehouseObservation:
        """Start new episode"""
        
    def step(self, action: WarehouseAction) -> WarehouseObservation:
        """Execute action"""
        
    @property
    def state(self) -> WarehouseState:
        """Get episode metadata"""
        
    def grade(self) -> float:
        """Score episode (0.0 to 1.0)"""
```

### HTTP API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Execute action |
| `/state` | GET | Get episode metadata |
| `/grade` | POST | Get final score |

---

## Tasks and Graders

### Easy: Inventory Restocking

**Objective:** Maintain high fill rate while minimizing costs

**Success Criteria:**
- Fill rate ≥ 95%
- Stockout hours ≤ 4

**Grading:**
```
score = 0.6 × fill_rate_score + 0.4 × stockout_score

fill_rate_score = min(1.0, avg_fill_rate / 0.95)
stockout_score = max(0, 1 - max_stockout_hours / 4)
```

**Passing Score:** 0.6

### Medium: Order Fulfillment

**Objective:** Optimize order picking and delivery across zones

**Success Criteria:**
- Fulfillment rate ≥ 92%
- Average delivery time ≤ 4 hours

**Grading:**
```
score = 0.5 × fulfillment_score + 0.5 × delivery_score

fulfillment_score = min(1.0, avg_fulfillment / 0.92)
delivery_score = max(0, 1 - avg_delivery_hours / 4)
```

**Passing Score:** 0.65

### Hard: Warehouse Optimization

**Objective:** Multi-objective optimization of all operations

**Success Criteria:**
- Customer satisfaction ≥ 75%
- Cost reduction ≥ 10%
- Throughput improvement ≥ 15%

**Grading:**
```
score = 0.35 × satisfaction + 0.35 × cost + 0.30 × throughput
```

**Passing Score:** 0.70

---

## Reward System

### Reward Components

The reward function provides continuous signal throughout the episode:

```
total_reward = stockout_reward + cost_reward + service_reward
```

| Component | Weight | Range | Description |
|-----------|--------|-------|-------------|
| stockout_avoidance | 0.3 | [0, 0.3] | Avoiding inventory shortages |
| cost_efficiency | 0.25 | [-0.25, 0.25] | Operational cost control |
| service_level | 0.45 | [-0.45, 0.45] | Fill rate performance |

### Partial Progress Signals

Unlike binary rewards (only at episode end), this provides signal at every step:

```python
# Step 1: Moderate action
reward = 0.15  # Good progress, but room for improvement

# Step 2: Better action
reward = 0.25  # Improved performance

# Step 3: Optimal action
reward = 0.35  # Excellent!

# Step 4: Stockout occurs
reward = -0.10  # Negative signal - bad action
```

### Penalty Examples

The reward also penalizes undesirable behavior:

```python
# Stockout penalty (inventory goes to zero)
if inventory[i] < 10:
    stockout_hours += 1
    reward -= 0.05

# Excessive ordering cost
if restock_level == 4:  # Emergency
    ordering_cost += unit_cost * quantity * 2  # 2x normal cost
```

---

## Inference and Evaluation

### Running Baseline Evaluation

```bash
# Set environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"
export TASK_NAME="easy"

# Run inference
python inference.py
```

### Output Format

```
[START] task=easy env=SmartWarehouse model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=restock(1) alloc(1) reward=0.15 done=false error=null
[STEP] step=2 action=restock(2) alloc(1) reward=0.22 done=false error=null
[STEP] step=3 action=restock(1) alloc(2) reward=0.18 done=false error=null
...
[END] success=true steps=50 score=0.723 rewards=0.15,0.22,0.18,...
```

### Custom LLM Evaluation

```python
import asyncio
from openai import OpenAI

async def evaluate_with_llm(llm_client, task="easy", max_steps=50):
    """Custom evaluation with your LLM."""
    
    env = WarehouseEnvironment(task=task)
    obs = env.reset()
    
    rewards = []
    
    for step in range(max_steps):
        if obs.done:
            break
        
        # Get LLM's action decision
        response = llm_client.chat.completions.create(
            model="your-model",
            messages=[
                {"role": "system", "content": "You are a warehouse manager..."},
                {"role": "user", "content": obs.to_text()}
            ]
        )
        
        # Parse action from response
        action_dict = parse_llm_response(response)
        action = WarehouseAction.from_dict(action_dict)
        
        # Execute
        obs = env.step(action)
        rewards.append(float(obs.reward))
    
    score = env.grade()
    return score, rewards

# Run
client = OpenAI(base_url="your-endpoint", api_key="your-key")
score, rewards = asyncio.run(evaluate_with_llm(client))
print(f"Score: {score:.3f}")
```

---

## Deployment

### Local Server

```bash
cd src/envs/smart_warehouse/server

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app:app --host 0.0.0.0 --port 8000

# Test
curl http://localhost:8000/health
```

### Docker Container

```bash
# Build
docker build -t smart-warehouse -f src/envs/smart_warehouse/server/Dockerfile .

# Run
docker run -d -p 8000:8000 \
  -e SMART_WAREHOUSE_TASK=easy \
  -e SMART_WAREHOUSE_SEED=42 \
  smart-warehouse

# Check logs
docker logs -f smart-warehouse
```

### HuggingFace Spaces

```bash
# Clone to HF Spaces
git clone https://huggingface.co/spaces/your-username/smart-warehouse
cd smart-warehouse

# Push changes
git add .
git commit -m "Deploy SmartWarehouse"
git push
```

The Space will automatically build and deploy.

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smart-warehouse
spec:
  replicas: 3
  selector:
    matchLabels:
      app: smart-warehouse
  template:
    spec:
      containers:
      - name: smart-warehouse
        image: your-registry/smart-warehouse:latest
        ports:
        - containerPort: 8000
        env:
        - name: SMART_WAREHOUSE_TASK
          value: "easy"
```

---

## Troubleshooting

### Common Issues

#### 1. Connection Refused

```bash
# Check if server is running
curl http://localhost:8000/health

# If not running, start it
cd src/envs/smart_warehouse/server
uvicorn app:app --host 0.0.0.0 --port 8000
```

#### 2. Import Errors

```bash
# Make sure src is in path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or in Python
import sys
sys.path.insert(0, "src")
```

#### 3. Docker Build Fails

```bash
# Check Docker is running
docker version

# Build with no cache
docker build --no-cache -t smart-warehouse .

# Check specific step
docker run -it smart-warehouse bash
```

#### 4. Invalid Actions

```python
# Actions are auto-clipped to valid ranges
action = WarehouseAction(
    restock_level=10,  # Will be clipped to 4
    allocation_strategy=-1,  # Will be clipped to 0
    worker_mode=5,  # Will be clipped to 3
    delivery_priority=10,  # Will be clipped to 3
    storage_action=100  # Will be clipped to 2
)
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all HTTP requests will be logged
env = SmartWarehouseEnv(base_url="http://localhost:8000")
```

### Getting Help

1. Check the [validation script](validate.py) runs successfully
2. Review the [example code](src/envs/smart_warehouse/)
3. Check the [openenv.yaml](openenv.yaml) specification

---

## Appendix: Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SMART_WAREHOUSE_TASK` | `easy` | Task difficulty |
| `SMART_WAREHOUSE_SEED` | `42` | Random seed |
| `PORT` | `8000` | Server port |
| `API_BASE_URL` | - | LLM API endpoint |
| `MODEL_NAME` | - | LLM model name |
| `HF_TOKEN` | - | API authentication |
| `TASK_NAME` | `easy` | Evaluation task |
| `MAX_STEPS` | `50` | Max episode steps |
| `TEMPERATURE` | `0.7` | LLM sampling temperature |
| `MAX_TOKENS` | `300` | LLM max output tokens |

---

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please:
1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Submit a pull request
