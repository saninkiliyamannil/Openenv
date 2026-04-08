"""
SmartWarehouse FastAPI Server
===========================

FastAPI server for SmartWarehouse environment following OpenEnv patterns.

Run locally:
    cd src/envs/smart_warehouse/server
    uvicorn app:app --host 0.0.0.0 --port 8000

Run with Docker:
    docker build -t smart-warehouse .
    docker run -p 8000:8000 smart-warehouse
"""

import os
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import environment
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from envs.smart_warehouse.environment import WarehouseEnvironment, create_smart_warehouse_environment
from envs.smart_warehouse.models import WarehouseAction, WarehouseObservation, WarehouseState


# =============================================================================
# Request/Response Models
# =============================================================================

class ResetRequest(BaseModel):
    """Request for /reset endpoint."""
    task: Optional[str] = "easy"
    seed: Optional[int] = None
    max_episode_steps: Optional[int] = None


class StepRequest(BaseModel):
    """Request for /step endpoint."""
    action: Dict[str, Any] = Field(
        default_factory=lambda: {
            "restock_level": 0,
            "allocation_strategy": 0,
            "worker_mode": 0,
            "delivery_priority": 0,
            "storage_action": 0
        }
    )


class ObservationResponse(BaseModel):
    """Response containing observation data."""
    inventory_levels: list[float]
    pending_orders: list[float]
    warehouse_zones: list[float]
    worker_availability: list[float]
    delivery_vehicles: list[float]
    time_features: list[float]
    demand_forecast: list[float]
    done: bool
    reward: float
    messages: list[str]


class StateResponse(BaseModel):
    """Response containing state data."""
    episode_id: str
    step_count: int
    task: str
    total_reward: float
    metrics: Dict[str, float]


# =============================================================================
# Create Application
# =============================================================================

# Global environment instance
_env: Optional[WarehouseEnvironment] = None


def get_environment() -> WarehouseEnvironment:
    """Get or create the environment instance."""
    global _env
    if _env is None:
        _env = WarehouseEnvironment(
            task=os.getenv("SMART_WAREHOUSE_TASK", "easy"),
            seed=int(os.getenv("SMART_WAREHOUSE_SEED", "42")),
        )
    return _env


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="SmartWarehouse Environment",
        description="Real-world warehouse management RL environment",
        version="1.0.0",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "service": "smart-warehouse"}
    
    @app.post("/reset", response_model=ObservationResponse)
    async def reset(request: ResetRequest = None):
        """
        Reset the environment to initial state.
        
        Args:
            request: Optional reset parameters
            
        Returns:
            Initial observation
        """
        global _env
        
        if request:
            task = request.task or os.getenv("SMART_WAREHOUSE_TASK", "easy")
            seed = request.seed
            max_steps = request.max_episode_steps
        else:
            task = os.getenv("SMART_WAREHOUSE_TASK", "easy")
            seed = None
            max_steps = None
        
        # Create new environment if task changed
        if _env is not None and _env.task != task:
            _env = None
        
        if _env is None:
            _env = WarehouseEnvironment(task=task, seed=seed, max_episode_steps=max_steps or 168)
        else:
            _env.reset()
        
        obs = _env.reset()
        
        return ObservationResponse(
            inventory_levels=obs.inventory_levels,
            pending_orders=obs.pending_orders,
            warehouse_zones=obs.warehouse_zones,
            worker_availability=obs.worker_availability,
            delivery_vehicles=obs.delivery_vehicles,
            time_features=obs.time_features,
            demand_forecast=obs.demand_forecast,
            done=obs.done,
            reward=obs.reward,
            messages=obs.messages,
        )
    
    @app.post("/step", response_model=ObservationResponse)
    async def step(request: StepRequest):
        """
        Execute one environment step.
        
        Args:
            request: Step request with action
            
        Returns:
            New observation
        """
        env = get_environment()
        
        # Parse action
        try:
            action = WarehouseAction.from_dict(request.action)
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid action: {e}")
        
        # Execute step
        try:
            obs = env.step(action)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Step failed: {e}")
        
        return ObservationResponse(
            inventory_levels=obs.inventory_levels,
            pending_orders=obs.pending_orders,
            warehouse_zones=obs.warehouse_zones,
            worker_availability=obs.worker_availability,
            delivery_vehicles=obs.delivery_vehicles,
            time_features=obs.time_features,
            demand_forecast=obs.demand_forecast,
            done=obs.done,
            reward=obs.reward,
            messages=obs.messages,
        )
    
    @app.get("/state", response_model=StateResponse)
    async def get_state():
        """
        Get current environment state.
        
        Returns:
            Current state
        """
        env = get_environment()
        state = env.state
        
        return StateResponse(
            episode_id=state.episode_id,
            step_count=state.step_count,
            task=state.task,
            total_reward=state.total_reward,
            metrics=state.metrics,
        )
    
    @app.post("/grade")
    async def grade():
        """
        Grade the current episode.
        
        Returns:
            Final score
        """
        env = get_environment()
        score = env.grade()
        return {"score": score}
    
    return app


# Create app instance
app = create_app()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Entry point for the server."""
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
