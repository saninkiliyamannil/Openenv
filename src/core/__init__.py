"""
Core Abstractions for OpenEnv
=============================

Provides the base classes for building OpenEnv-compatible environments.
Following the official OpenEnv patterns from Meta PyTorch.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import uuid

import numpy as np


# =============================================================================
# Base Types (Following OpenEnv Core)
# =============================================================================

class Action(ABC):
    """Base class for all environment actions."""
    pass


class Observation(ABC):
    """Base class for all environment observations."""
    pass


class State(ABC):
    """Base class for environment state/metadata."""
    pass


@dataclass
class StepResult:
    """
    Result of an environment step.
    
    Attributes:
        observation: The new observation after the step
        reward: The reward for taking this action
        done: Whether the episode has ended
        info: Additional metadata
    """
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Environment Base Class
# =============================================================================

class Environment(ABC):
    """
    Abstract base class for all OpenEnv environments.
    
    This follows the OpenEnv specification where environments expose
    reset(), step(), and state properties.
    
    Example:
        class MyEnv(Environment):
            def reset(self) -> Observation:
                ...
            
            def step(self, action: Action) -> Observation:
                ...
    """
    
    @abstractmethod
    def reset(self) -> Observation:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        pass
    
    @abstractmethod
    def step(self, action: Action) -> Observation:
        """
        Execute one environment step.
        
        Args:
            action: The action to take
            
        Returns:
            The resulting observation
        """
        pass
    
    @property
    def state(self) -> State:
        """
        Get the current episode state/metadata.
        
        Returns:
            State object with episode information
        """
        raise NotImplementedError


# =============================================================================
# HTTP Client Base (Following OpenEnv HTTP Pattern)
# =============================================================================

from typing import TypeVar, Generic
import requests

ActionT = TypeVar("ActionT", bound=Action)
ObservationT = TypeVar("ObservationT", bound=Observation)


class HTTPEnvClient(ABC, Generic[ActionT, ObservationT]):
    """
    Base HTTP client for OpenEnv environments.
    
    This follows the OpenEnv pattern where clients communicate
    with environment servers via HTTP/JSON.
    
    Attributes:
        base_url: The base URL of the environment server
    """
    
    SUPPORTS_CONCURRENT_SESSIONS: bool = False
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session_id: Optional[str] = None
    
    def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request to the server."""
        url = f"{self.base_url}{endpoint}"
        response = requests.post(url, json=data, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def _get(self, endpoint: str) -> Dict[str, Any]:
        """Make a GET request to the server."""
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def reset(self, **kwargs) -> StepResult:
        """
        Reset the environment.
        
        Args:
            **kwargs: Optional parameters passed to the server
            
        Returns:
            StepResult with initial observation
        """
        payload = self._post("/reset", kwargs)
        return self._parse_result(payload)
    
    def step(self, action: ActionT) -> StepResult:
        """
        Execute an environment step.
        
        Args:
            action: The action to take
            
        Returns:
            StepResult with new observation and reward
        """
        payload = self._step_payload(action)
        result_payload = self._post("/step", payload)
        return self._parse_result(result_payload)
    
    def state(self) -> State:
        """
        Get the current environment state.
        
        Returns:
            Current state object
        """
        payload = self._get("/state")
        return self._parse_state(payload)
    
    @abstractmethod
    def _step_payload(self, action: ActionT) -> Dict[str, Any]:
        """
        Convert typed action to JSON payload.
        
        Args:
            action: The typed action
            
        Returns:
            JSON-serializable dict
        """
        pass
    
    @abstractmethod
    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        """
        Parse JSON response to StepResult.
        
        Args:
            payload: JSON response from server
            
        Returns:
            Typed StepResult
        """
        pass
    
    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse JSON response to State.
        
        Args:
            payload: JSON response from server
            
        Returns:
            Typed State object
        """
        raise NotImplementedError
    
    def close(self):
        """Close the client connection."""
        pass


# =============================================================================
# FastAPI App Creation Helper
# =============================================================================

def create_fastapi_app(
    env: Environment,
    max_concurrent_envs: int = 1
):
    """
    Create a FastAPI app for an OpenEnv environment.
    
    Args:
        env: The environment instance
        max_concurrent_envs: Maximum concurrent sessions
        
    Returns:
        FastAPI application
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    
    app = FastAPI(title="OpenEnv Server")
    
    class ResetRequest(BaseModel):
        class Config:
            extra = "allow"
    
    class StepRequest(BaseModel):
        action: Dict[str, Any]
    
    @app.get("/health")
    async def health():
        return {"status": "ok"}
    
    @app.post("/reset")
    async def reset(request: ResetRequest = None):
        try:
            obs = env.reset()
            return {
                "observation": _serialize_observation(obs),
                "reward": 0.0,
                "done": False,
                "info": {}
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/step")
    async def step(request: StepRequest):
        try:
            action = _deserialize_action(request.action)
            obs = env.step(action)
            return {
                "observation": _serialize_observation(obs),
                "reward": 0.0,
                "done": False,
                "info": {}
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/state")
    async def get_state():
        try:
            state = env.state
            return _serialize_state(state)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def _serialize_observation(obs: Observation) -> Dict[str, Any]:
    """Serialize an observation to JSON."""
    if hasattr(obs, 'model_dump'):
        return obs.model_dump()
    elif hasattr(obs, '__dict__'):
        return obs.__dict__
    return {}


def _serialize_state(state: State) -> Dict[str, Any]:
    """Serialize a state to JSON."""
    if hasattr(state, 'model_dump'):
        return state.model_dump()
    elif hasattr(state, '__dict__'):
        return state.__dict__
    return {}


def _deserialize_action(data: Dict[str, Any]) -> Action:
    """Deserialize an action from JSON."""
    return data  # Override in subclass
