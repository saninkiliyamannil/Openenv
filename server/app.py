"""
SmartWarehouse Server
====================

Main server entry point for SmartWarehouse environment.

Run:
    python server/app.py
    or
    python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
import uvicorn
import json

from envs.smart_warehouse.environment import WarehouseEnvironment
from envs.smart_warehouse.models import WarehouseAction


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(title="SmartWarehouse - OpenEnv")

env = None


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "smart-warehouse"}


@app.post("/reset")
async def reset(task: str = "easy", seed: int = 42, max_episode_steps: int = 50):
    """Reset the environment."""
    global env
    env = WarehouseEnvironment(task=task, seed=seed, max_episode_steps=max_episode_steps)
    obs = env.reset()
    
    return {
        "observation": {
            "inventory_levels": obs.inventory_levels,
            "pending_orders": obs.pending_orders,
            "warehouse_zones": obs.warehouse_zones,
            "worker_availability": obs.worker_availability,
            "delivery_vehicles": obs.delivery_vehicles,
            "time_features": obs.time_features,
            "demand_forecast": obs.demand_forecast,
            "done": obs.done,
            "reward": obs.reward,
            "messages": obs.messages,
        },
        "reward": 0.0,
        "done": False,
        "info": {}
    }


@app.post("/step")
async def step(action: dict):
    """Execute one environment step."""
    global env
    
    if env is None:
        env = WarehouseEnvironment()
        env.reset()
    
    warehouse_action = WarehouseAction(
        restock_level=action.get("restock_level", 0),
        allocation_strategy=action.get("allocation_strategy", 0),
        worker_mode=action.get("worker_mode", 0),
        delivery_priority=action.get("delivery_priority", 0),
        storage_action=action.get("storage_action", 0)
    )
    
    obs = env.step(warehouse_action)
    state = env.state
    
    return {
        "observation": {
            "inventory_levels": obs.inventory_levels,
            "pending_orders": obs.pending_orders,
            "warehouse_zones": obs.warehouse_zones,
            "worker_availability": obs.worker_availability,
            "delivery_vehicles": obs.delivery_vehicles,
            "time_features": obs.time_features,
            "demand_forecast": obs.demand_forecast,
            "done": obs.done,
            "reward": obs.reward,
            "messages": obs.messages,
        },
        "reward": float(obs.reward) if obs.reward else 0.0,
        "done": obs.done,
        "info": {
            "step": state.step_count,
            "metrics": state.metrics,
        }
    }


@app.get("/state")
async def get_state():
    """Get current environment state."""
    global env
    
    if env is None:
        return {"error": "Environment not initialized"}
    
    state = env.state
    
    return {
        "episode_id": state.episode_id,
        "step_count": state.step_count,
        "task": state.task,
        "total_reward": state.total_reward,
        "metrics": state.metrics,
    }


@app.post("/grade")
async def grade():
    """Get episode score."""
    global env
    
    if env is None:
        return {"score": 0.0}
    
    score = env.grade()
    return {"score": score}


# =============================================================================
# Web Interface
# =============================================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>SmartWarehouse - OpenEnv</title>
    <style>
        body { font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #00d9ff; text-align: center; }
        .card { background: #16213e; padding: 20px; margin: 10px 0; border-radius: 10px; }
        .metric { display: inline-block; margin: 10px; padding: 15px; background: #0f3460; border-radius: 8px; text-align: center; }
        .metric .value { font-size: 24px; color: #00d9ff; }
        .metric .label { font-size: 12px; color: #888; }
        button { padding: 10px 20px; background: #e94560; color: white; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #ff6b6b; }
        input, select { padding: 10px; background: #0f3460; color: white; border: 1px solid #333; border-radius: 5px; margin: 5px; }
        pre { background: #0f3460; padding: 15px; border-radius: 5px; overflow-x: auto; }
        .log { max-height: 200px; overflow-y: auto; background: #0f3460; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>SmartWarehouse - OpenEnv</h1>
        
        <div class="card">
            <h3>Configuration</h3>
            <select id="task">
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
            </select>
            <input type="number" id="seed" value="42" placeholder="Seed">
            <button onclick="reset()">Reset</button>
        </div>
        
        <div class="card">
            <h3>Metrics</h3>
            <div class="metric"><div class="value" id="fillRate">-</div><div class="label">Fill Rate</div></div>
            <div class="metric"><div class="value" id="cost">-</div><div class="label">Cost</div></div>
            <div class="metric"><div class="value" id="steps">0</div><div class="label">Steps</div></div>
            <div class="metric"><div class="value" id="reward">-</div><div class="label">Last Reward</div></div>
        </div>
        
        <div class="card">
            <h3>Actions</h3>
            <input type="number" id="restock" value="1" min="0" max="4" placeholder="Restock 0-4">
            <input type="number" id="alloc" value="1" min="0" max="4" placeholder="Alloc 0-4">
            <input type="number" id="worker" value="0" min="0" max="3" placeholder="Worker 0-3">
            <input type="number" id="delivery" value="0" min="0" max="3" placeholder="Delivery 0-3">
            <input type="number" id="storage" value="0" min="0" max="2" placeholder="Storage 0-2">
            <button onclick="step()">Execute Step</button>
            <button onclick="autoRun()">Auto Run (10)</button>
        </div>
        
        <div class="card">
            <h3>Log</h3>
            <div class="log" id="log"></div>
        </div>
    </div>
    
    <script>
        async function reset() {
            const task = document.getElementById('task').value;
            const seed = document.getElementById('seed').value;
            const res = await fetch('/reset?task=' + task + '&seed=' + seed, { method: 'POST' });
            const data = await res.json();
            updateMetrics(data.info.metrics || {});
            addLog('Reset - Task: ' + task);
        }
        
        async function step() {
            const action = {
                restock_level: parseInt(document.getElementById('restock').value),
                allocation_strategy: parseInt(document.getElementById('alloc').value),
                worker_mode: parseInt(document.getElementById('worker').value),
                delivery_priority: parseInt(document.getElementById('delivery').value),
                storage_action: parseInt(document.getElementById('storage').value)
            };
            const res = await fetch('/step', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(action)
            });
            const data = await res.json();
            updateMetrics(data.info.metrics || {});
            document.getElementById('reward').textContent = data.reward.toFixed(2);
            addLog('Step: reward=' + data.reward.toFixed(3) + (data.done ? ' [DONE]' : ''));
            if (data.done) { const g = await fetch('/grade', {method:'POST'}); const s = await g.json(); addLog('FINAL SCORE: ' + s.score.toFixed(3)); }
        }
        
        async function autoRun() { for (let i=0; i<10; i++) { await step(); await new Promise(r => setTimeout(r, 100)); } }
        
        function updateMetrics(m) {
            document.getElementById('fillRate').textContent = ((m.fill_rate || 0) * 100).toFixed(0) + '%';
            document.getElementById('cost').textContent = '$' + (m.operational_cost || 0).toFixed(0);
            document.getElementById('steps').textContent = m.step_count || 0;
        }
        
        function addLog(msg) {
            const log = document.getElementById('log');
            log.innerHTML = '> ' + msg + '<br>' + log.innerHTML;
        }
        
        reset();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve web interface."""
    return HTML


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the server."""
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
