"""
SmartWarehouse - OpenEnv Web Interface
====================================

A simple HTML-based interface for SmartWarehouse environment.
Run with: python app.py
Or: python -m uvicorn app:app --host 0.0.0.0 --port 7860
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from envs.smart_warehouse.environment import WarehouseEnvironment
from envs.smart_warehouse.models import WarehouseAction

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import json

# Create FastAPI app
app = FastAPI(title="SmartWarehouse - OpenEnv")

# Global state
env = None
current_task = "easy"


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SmartWarehouse - OpenEnv</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #60a5fa; text-align: center; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #94a3b8; margin-bottom: 30px; }
        
        .card { background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        .card h2 { color: #f472b6; margin-bottom: 15px; font-size: 1.2rem; }
        
        .config { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .config select, .config input { width: 100%; padding: 10px; background: #334155; border: 1px solid #475569; border-radius: 8px; color: white; font-size: 14px; }
        .config label { display: block; color: #94a3b8; font-size: 12px; margin-bottom: 5px; }
        
        .btn { padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 600; transition: all 0.2s; }
        .btn-primary { background: #3b82f6; color: white; }
        .btn-primary:hover { background: #2563eb; }
        .btn-secondary { background: #475569; color: white; }
        .btn-secondary:hover { background: #64748b; }
        
        .state-display { background: #0f172a; border-radius: 8px; padding: 15px; font-family: monospace; font-size: 13px; line-height: 1.6; white-space: pre-wrap; max-height: 400px; overflow-y: auto; }
        
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; }
        .metric { background: #0f172a; padding: 12px; border-radius: 8px; text-align: center; }
        .metric .value { font-size: 1.5rem; font-weight: bold; color: #60a5fa; }
        .metric .label { font-size: 11px; color: #94a3b8; text-transform: uppercase; }
        
        .action-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .action-group label { display: block; color: #94a3b8; font-size: 12px; margin-bottom: 5px; }
        .action-group input { width: 100%; padding: 10px; background: #0f172a; border: 1px solid #475569; border-radius: 8px; color: white; }
        
        .logs { background: #0f172a; border-radius: 8px; padding: 15px; font-family: monospace; font-size: 12px; max-height: 300px; overflow-y: auto; }
        .log-line { padding: 4px 0; border-bottom: 1px solid #1e293b; }
        .log-step { color: #60a5fa; }
        .log-reward { color: #4ade80; }
        .log-score { color: #fbbf24; font-weight: bold; }
        
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <h1>SmartWarehouse - OpenEnv</h1>
        <p class="subtitle">Real-world warehouse management RL environment</p>
        
        <div class="card">
            <h2>Configuration</h2>
            <div class="config">
                <div>
                    <label>Task</label>
                    <select id="task">
                        <option value="easy">Easy - Inventory</option>
                        <option value="medium">Medium - Fulfillment</option>
                        <option value="hard">Hard - Optimization</option>
                    </select>
                </div>
                <div>
                    <label>Seed</label>
                    <input type="number" id="seed" value="42" min="0" max="1000">
                </div>
                <div>
                    <label>Max Steps</label>
                    <input type="number" id="maxSteps" value="50" min="10" max="200">
                </div>
            </div>
            <button class="btn btn-primary" onclick="resetEnv()">Reset Environment</button>
        </div>
        
        <div class="grid-2">
            <div class="card">
                <h2>Metrics</h2>
                <div class="metrics">
                    <div class="metric"><div class="value" id="fillRate">-</div><div class="label">Fill Rate</div></div>
                    <div class="metric"><div class="value" id="cost">-</div><div class="label">Cost</div></div>
                    <div class="metric"><div class="value" id="satisfaction">-</div><div class="label">Satisfaction</div></div>
                    <div class="metric"><div class="value" id="stepCount">0</div><div class="label">Step</div></div>
                </div>
            </div>
            
            <div class="card">
                <h2>State</h2>
                <div class="state-display" id="state">Click Reset to start</div>
            </div>
        </div>
        
        <div class="card">
            <h2>Take Action</h2>
            <div class="action-grid">
                <div class="action-group">
                    <label>Restock Level (0-4)</label>
                    <input type="number" id="restockLevel" value="1" min="0" max="4">
                </div>
                <div class="action-group">
                    <label>Allocation Strategy (0-4)</label>
                    <input type="number" id="allocationStrategy" value="1" min="0" max="4">
                </div>
                <div class="action-group">
                    <label>Worker Mode (0-3)</label>
                    <input type="number" id="workerMode" value="0" min="0" max="3">
                </div>
                <div class="action-group">
                    <label>Delivery Priority (0-3)</label>
                    <input type="number" id="deliveryPriority" value="0" min="0" max="3">
                </div>
                <div class="action-group">
                    <label>Storage Action (0-2)</label>
                    <input type="number" id="storageAction" value="0" min="0" max="2">
                </div>
            </div>
            <div style="margin-top: 15px; display: flex; gap: 10px;">
                <button class="btn btn-primary" onclick="takeStep()">Execute Step</button>
                <button class="btn btn-secondary" onclick="runEpisode()">Auto Run (10 steps)</button>
            </div>
        </div>
        
        <div class="card">
            <h2>Episode Log</h2>
            <div class="logs" id="logs"></div>
        </div>
    </div>
    
    <script>
        async function resetEnv() {
            const task = document.getElementById('task').value;
            const seed = document.getElementById('seed').value;
            const maxSteps = document.getElementById('maxSteps').value;
            
            try {
                const response = await fetch('/reset?task=' + task + '&seed=' + seed + '&max_steps=' + maxSteps, {
                    method: 'POST'
                });
                const data = await response.json();
                updateDisplay(data);
                document.getElementById('logs').innerHTML = '';
                addLog('Environment reset - Task: ' + task, 'info');
            } catch (e) {
                addLog('Error: ' + e.message, 'error');
            }
        }
        
        async function takeStep() {
            const action = {
                restock_level: parseInt(document.getElementById('restockLevel').value),
                allocation_strategy: parseInt(document.getElementById('allocationStrategy').value),
                worker_mode: parseInt(document.getElementById('workerMode').value),
                delivery_priority: parseInt(document.getElementById('deliveryPriority').value),
                storage_action: parseInt(document.getElementById('storageAction').value)
            };
            
            try {
                const response = await fetch('/step', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action})
                });
                const data = await response.json();
                
                updateDisplay(data);
                addLog('Step ' + data.step + ': reward=' + data.reward.toFixed(3), 'step');
                
                if (data.done) {
                    addLog('EPISODE COMPLETE! Score: ' + data.score.toFixed(3), 'score');
                }
            } catch (e) {
                addLog('Error: ' + e.message, 'error');
            }
        }
        
        async function runEpisode() {
            for (let i = 0; i < 10; i++) {
                await takeStep();
                await new Promise(r => setTimeout(r, 200));
            }
        }
        
        function updateDisplay(data) {
            document.getElementById('fillRate').textContent = (data.metrics.fill_rate * 100).toFixed(0) + '%';
            document.getElementById('cost').textContent = '$' + data.metrics.operational_cost.toFixed(0);
            document.getElementById('satisfaction').textContent = (data.metrics.customer_satisfaction * 100).toFixed(0) + '%';
            document.getElementById('stepCount').textContent = data.step;
            
            let stateText = '=== WAREHOUSE STATE ===\n';
            stateText += 'Inventory avg: ' + data.inventory_avg.toFixed(1) + '/500\n';
            stateText += 'Pending: ' + data.pending_orders.map(p => p.toFixed(0)).join(', ') + '\n';
            stateText += 'Zones: ' + data.zones.map(z => (z * 100).toFixed(0) + '%').join(', ') + '\n';
            stateText += 'Time: Hour ' + data.hour + ', Day ' + data.day;
            document.getElementById('state').textContent = stateText;
        }
        
        function addLog(text, type) {
            const logs = document.getElementById('logs');
            const line = document.createElement('div');
            line.className = 'log-line log-' + type;
            line.textContent = '> ' + text;
            logs.insertBefore(line, logs.firstChild);
        }
        
        // Initialize
        resetEnv();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main page."""
    global env, current_task
    
    if env is None:
        env = WarehouseEnvironment(task="easy", seed=42, max_episode_steps=50)
        env.reset()
    
    return HTML_TEMPLATE


@app.post("/reset")
async def reset(task: str = "easy", seed: int = 42, max_steps: int = 50):
    """Reset the environment."""
    global env, current_task
    
    current_task = task
    env = WarehouseEnvironment(task=task, seed=seed, max_episode_steps=max_steps)
    obs = env.reset()
    
    return format_response(obs, env.state.step_count, 0.0, False, 0.0)


@app.post("/step")
async def step(action: dict):
    """Execute a step."""
    global env
    
    if env is None:
        env = WarehouseEnvironment(task="easy", seed=42)
        env.reset()
    
    # Parse action
    warehouse_action = WarehouseAction(
        restock_level=action.get("restock_level", 0),
        allocation_strategy=action.get("allocation_strategy", 0),
        worker_mode=action.get("worker_mode", 0),
        delivery_priority=action.get("delivery_priority", 0),
        storage_action=action.get("storage_action", 0)
    )
    
    obs = env.step(warehouse_action)
    score = env.grade() if obs.done else 0.0
    
    return format_response(obs, env.state.step_count, obs.reward, obs.done, score)


def format_response(obs, step, reward, done, score):
    """Format observation for JSON response."""
    metrics = env.state.metrics if env else {}
    
    return {
        "inventory_avg": sum(obs.inventory_levels) / len(obs.inventory_levels) if obs.inventory_levels else 0,
        "pending_orders": obs.pending_orders if obs.pending_orders else [0, 0, 0],
        "zones": obs.warehouse_zones if obs.warehouse_zones else [0] * 6,
        "hour": int(obs.time_features[0]) if obs.time_features else 0,
        "day": int(obs.time_features[1]) if obs.time_features else 0,
        "metrics": metrics,
        "step": step,
        "reward": reward,
        "done": done,
        "score": score
    }


def main():
    """Entry point for HuggingFace Spaces."""
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
