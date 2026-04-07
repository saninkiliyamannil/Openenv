openenv-smartwarehouse
=======================

Real-world warehouse management RL environment following OpenEnv patterns.

Installation
------------

From source:
    pip install -e .

As dependency (for Docker deployment):
    Add to requirements: git+https://github.com/your-org/OpenEnv.git#subdirectory=src/envs/smart_warehouse

Usage
-----

Python API:
    from envs.smart_warehouse import SmartWarehouseEnv
    from envs.smart_warehouse.models import WarehouseAction
    
    env = SmartWarehouseEnv(base_url="http://localhost:8000")
    result = env.reset()
    result = env.step(WarehouseAction(restock_level=1))

Docker:
    docker build -t smart-warehouse -f src/envs/smart_warehouse/server/Dockerfile .
    docker run -p 8000:8000 smart-warehouse

Environment Variables
---------------------

- SMART_WAREHOUSE_TASK: Task difficulty (easy, medium, hard)
- SMART_WAREHOUSE_SEED: Random seed
- PORT: Server port (default: 8000)

API Endpoints
-------------

GET  /health - Health check
POST /reset  - Reset environment
POST /step   - Execute action
GET  /state  - Get current state
POST /grade  - Get episode score
