"""
OpenEnv SmartWarehouse Inference Script
======================================

Evaluates LLM agents against SmartWarehouse tasks.
Follows exact STDOUT format for evaluation.

Usage (Windows PowerShell):
    $env:HF_TOKEN="hf_..."
    $env:TASK_NAME="easy"
    python inference.py

Usage (Windows CMD):
    set HF_TOKEN=hf_...
    set TASK_NAME=easy
    python inference.py

Usage (Linux/Mac):
    export HF_TOKEN="hf_..."
    export TASK_NAME="easy"
    python inference.py
"""

import asyncio
import os
import json
import sys
import textwrap
import random
from typing import List, Optional, Dict, Any

# Check for openai module
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: openai module not installed.")
    print("To install: pip install openai")
    print("Will use random baseline actions.\n")

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from envs.smart_warehouse.environment import WarehouseEnvironment
from envs.smart_warehouse.models import WarehouseAction, WarehouseObservation


# =============================================================================
# Configuration (from environment variables)
# =============================================================================

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or ""
TASK_NAME = os.environ.get("TASK_NAME", "easy")
BENCHMARK = os.environ.get("BENCHMARK", "SmartWarehouse")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "50"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "300"))
SUCCESS_SCORE_THRESHOLD = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.6"))


# =============================================================================
# System Prompt for Warehouse Manager Agent
# =============================================================================

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert warehouse manager AI optimizing real-world supply chain operations.

Current task: Manage a distribution warehouse efficiently by making optimal decisions.

You must respond with a JSON object containing your action choices.

Available actions (all values are integers):
- restock_level: 0=none, 1=low(10%), 2=medium(25%), 3=high(50%), 4=emergency(100%)
- allocation_strategy: 0=round_robin, 1=priority_based, 2=proximity, 3=balance, 4=dynamic
- worker_mode: 0=current, 1=overtime, 2=contractors, 3=automation
- delivery_priority: 0=fifo, 1=urgency, 2=profit, 3=proximity
- storage_action: 0=none, 1=rebalance, 2=consolidate

Respond ONLY with valid JSON in this format:
{"restock_level": 0-4, "allocation_strategy": 0-4, "worker_mode": 0-3, "delivery_priority": 0-3, "storage_action": 0-2}
""").strip()


# =============================================================================
# Logging Functions (EXACT STDOUT FORMAT)
# =============================================================================

def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] log line."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    """Emit [STEP] log line."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] log line."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )


# =============================================================================
# LLM Client
# =============================================================================

class LLMClient:
    """Client for interacting with LLM API."""
    
    def __init__(self, base_url: str, model: str, api_key: str):
        self.base_url = base_url
        self.model = model
        self.client = None
        
        if OPENAI_AVAILABLE and api_key:
            try:
                self.client = OpenAI(base_url=base_url, api_key=api_key)
                print(f"Connected to LLM at {base_url}")
            except Exception as e:
                print(f"Failed to connect to LLM: {e}")
        else:
            print("Using RANDOM baseline (no API key or openai not installed)")
    
    def get_action(
        self,
        state_description: str,
        history: List[str],
    ) -> Dict[str, int]:
        """Get action from LLM based on current state."""
        if not self.client:
            return self._random_action()
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_prompt(state_description, history)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            
            content = completion.choices[0].message.content or ""
            return self._parse_action(content)
            
        except Exception as e:
            print(f"[DEBUG] LLM request failed: {e}", flush=True)
            return self._random_action()
    
    def _build_prompt(self, state_desc: str, history: List[str]) -> str:
        """Build user prompt from state and history."""
        history_block = "\n".join(history[-5:]) if history else "None"
        return textwrap.dedent(f"""
        Current Warehouse State:
        {state_desc}
        
        Recent History:
        {history_block}
        
        Choose your next action (JSON only):
        """).strip()
    
    def _parse_action(self, content: str) -> Dict[str, int]:
        """Parse action from LLM response."""
        import re
        try:
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                action_dict = json.loads(json_match.group())
            else:
                action_dict = json.loads(content)
            
            return {
                "restock_level": max(0, min(4, int(action_dict.get("restock_level", 0)))),
                "allocation_strategy": max(0, min(4, int(action_dict.get("allocation_strategy", 0)))),
                "worker_mode": max(0, min(3, int(action_dict.get("worker_mode", 0)))),
                "delivery_priority": max(0, min(3, int(action_dict.get("delivery_priority", 0)))),
                "storage_action": max(0, min(2, int(action_dict.get("storage_action", 0)))),
            }
        except json.JSONDecodeError:
            return self._random_action()
    
    def _random_action(self) -> Dict[str, int]:
        """Return random action for baseline."""
        return {
            "restock_level": random.randint(0, 4),
            "allocation_strategy": random.randint(0, 4),
            "worker_mode": random.randint(0, 3),
            "delivery_priority": random.randint(0, 3),
            "storage_action": random.randint(0, 2),
        }


# =============================================================================
# State Formatter
# =============================================================================

def format_state(obs: WarehouseObservation) -> str:
    """Format observation as human-readable state description."""
    lines = []
    
    # Inventory
    if obs.inventory_levels:
        avg_inv = sum(obs.inventory_levels) / len(obs.inventory_levels)
        lines.append(f"- Inventory: avg {avg_inv:.1f}/500 across {len(obs.inventory_levels)} categories")
    
    # Pending orders
    if obs.pending_orders:
        lines.append(f"- Pending Orders: urgent={obs.pending_orders[0]:.0f}, normal={obs.pending_orders[1]:.0f}, low={obs.pending_orders[2]:.0f}")
    
    # Zone utilization
    if obs.warehouse_zones:
        zones_str = ", ".join([f"{z:.0%}" for z in obs.warehouse_zones[:3]])
        lines.append(f"- Zone Utilization: [{zones_str}, ...]")
    
    # Time
    if obs.time_features:
        hour = int(obs.time_features[0])
        day = int(obs.time_features[1])
        weekend = "Weekend" if obs.time_features[2] else "Weekday"
        lines.append(f"- Time: Hour {hour}, Day {day}, {weekend}")
    
    # Demand forecast
    if obs.demand_forecast:
        avg_demand = sum(obs.demand_forecast) / len(obs.demand_forecast)
        lines.append(f"- Demand Forecast: avg {avg_demand:.1f}")
    
    return "\n".join(lines)


def format_action(action: WarehouseAction) -> str:
    """Format action as action string."""
    return f"restock({action.restock_level}) alloc({action.allocation_strategy}) worker({action.worker_mode}) delivery({action.delivery_priority}) storage({action.storage_action})"


# =============================================================================
# Main Inference Loop
# =============================================================================

async def run_inference() -> None:
    """Run main inference loop."""
    
    print(f"\n{'='*60}")
    print("OpenEnv SmartWarehouse Inference")
    print(f"{'='*60}")
    print(f"Task: {TASK_NAME}")
    print(f"Max Steps: {MAX_STEPS}")
    print(f"API: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"{'='*60}\n")
    
    # Initialize
    llm_client = LLMClient(API_BASE_URL, MODEL_NAME, HF_TOKEN)
    env = WarehouseEnvironment(task=TASK_NAME, max_episode_steps=MAX_STEPS)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # Reset environment
        obs = env.reset()
        steps_taken = 0
        
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break
            
            # Get state description for LLM
            state_desc = format_state(obs)
            
            # Get action from LLM
            action_dict = llm_client.get_action(state_desc, history)
            action = WarehouseAction.from_dict(action_dict)
            
            # Execute step
            obs = env.step(action)
            
            # Extract reward
            reward_value = float(obs.reward) if obs.done else 0.0
            done = obs.done
            
            # Record
            rewards.append(reward_value)
            steps_taken = step
            
            # Format action string for logging
            action_str = format_action(action)
            
            # Log step
            log_step(
                step=step,
                action=action_str,
                reward=reward_value,
                done=done,
                error=last_error
            )
            
            # Update history
            history.append(
                f"Step {step}: {action_str} -> reward {reward_value:.2f}, done={done}"
            )
            
            if done:
                break
        
        # Calculate final score
        if rewards:
            score = env.grade()
            score = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD
        
    except Exception as e:
        last_error = str(e)
        print(f"[DEBUG] Inference error: {e}", flush=True)
    
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        
        print(f"\n{'='*60}")
        print(f"Episode Complete!")
        print(f"Steps: {steps_taken}")
        print(f"Score: {score:.3f}")
        print(f"Success: {success}")
        print(f"{'='*60}\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Entry point for inference script."""
    asyncio.run(run_inference())


if __name__ == "__main__":
    main()
