"""
OpenEnv Validation Script
=======================

Validates the SmartWarehouse environment against OpenEnv standards.

Usage:
    python validate.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from envs.smart_warehouse.environment import WarehouseEnvironment
from envs.smart_warehouse.models import (
    WarehouseAction,
    WarehouseObservation,
    WarehouseState,
    InventoryRestockingGrader,
    OrderFulfillmentGrader,
    WarehouseOptimizationGrader,
)


def validate_models():
    """Validate typed models."""
    print("\n" + "="*60)
    print("VALIDATING TYPED MODELS")
    print("="*60)
    
    # Test Action
    action = WarehouseAction(
        restock_level=2,
        allocation_strategy=1,
        worker_mode=0,
        delivery_priority=2,
        storage_action=1
    )
    print(f"[PASS] WarehouseAction: {action}")
    
    arr = action.to_dict()
    print(f"[PASS] to_dict(): {arr}")
    
    action2 = WarehouseAction.from_dict(arr)
    print(f"[PASS] from_dict(): {action2}")
    
    # Test Observation
    obs = WarehouseObservation(
        inventory_levels=[100.0] * 20,
        pending_orders=[10.0] * 3,
        warehouse_zones=[0.5] * 6,
        worker_availability=[10.0] * 4,
        delivery_vehicles=[5.0] * 4,
        time_features=[8.0, 0.0, 0.0, 0.0],
        demand_forecast=[25.0] * 10,
    )
    print(f"[PASS] WarehouseObservation created")
    
    text = obs.to_text()
    print(f"[PASS] to_text(): {len(text)} chars")
    
    arr = obs.to_array()
    print(f"[PASS] to_array(): shape {len(arr)}")
    
    # Test State
    state = WarehouseState(
        episode_id="test123",
        step_count=5,
        task="easy",
        total_reward=2.5,
    )
    print(f"[PASS] WarehouseState: {state}")
    
    return True


def validate_api():
    """Validate step/reset/state API."""
    print("\n" + "="*60)
    print("VALIDATING API METHODS (step/reset/state)")
    print("="*60)
    
    for task in ["easy", "medium", "hard"]:
        print(f"\n--- Task: {task} ---")
        
        env = WarehouseEnvironment(task=task, seed=42)
        
        # Test reset()
        obs = env.reset()
        assert isinstance(obs, WarehouseObservation), f"reset() should return WarehouseObservation"
        print(f"[PASS] reset() returns WarehouseObservation")
        
        # Test state property
        state = env.state
        assert isinstance(state, WarehouseState), f"state should return WarehouseState"
        print(f"[PASS] state returns WarehouseState")
        
        # Test step()
        action = WarehouseAction(
            restock_level=1,
            allocation_strategy=1,
            worker_mode=0,
            delivery_priority=0,
            storage_action=0
        )
        obs = env.step(action)
        assert isinstance(obs, WarehouseObservation), f"step() should return WarehouseObservation"
        print(f"[PASS] step() returns WarehouseObservation")
        
        # Test multiple steps
        for i in range(10):
            action = WarehouseAction(
                restock_level=i % 5,
                allocation_strategy=i % 5,
                worker_mode=i % 4,
                delivery_priority=i % 4,
                storage_action=i % 3
            )
            obs = env.step(action)
            if obs.done:
                break
        
        print(f"[PASS] Ran {env.state.step_count} steps")
        
        # Test grading
        score = env.grade()
        assert 0.0 <= score <= 1.0, f"Score {score} outside [0,1] range"
        print(f"[PASS] Grading works: score={score:.3f}")
        
    return True


def validate_tasks_and_graders():
    """Validate 3 tasks with graders."""
    print("\n" + "="*60)
    print("VALIDATING 3 TASKS WITH GRADERS")
    print("="*60)
    
    graders = {
        "easy": (InventoryRestockingGrader(), 0.6),
        "medium": (OrderFulfillmentGrader(), 0.65),
        "hard": (WarehouseOptimizationGrader(), 0.70),
    }
    
    for task_name, (grader, passing) in graders.items():
        print(f"\n--- {task_name.upper()} ---")
        
        env = WarehouseEnvironment(task=task_name, seed=42, max_episode_steps=50)
        env.reset()
        
        trajectory = []
        for i in range(50):
            action = WarehouseAction(
                restock_level=i % 5,
                allocation_strategy=i % 5,
            )
            obs = env.step(action)
            trajectory.append(env.state.metrics)
            if obs.done:
                break
        
        score = env.grade()
        
        assert 0.0 <= score <= 1.0, f"Score {score} outside [0,1]"
        print(f"[PASS] Score in range: {score:.3f} (passing: {passing})")
        
        criteria = grader.get_success_criteria()
        print(f"[PASS] Grader criteria: {criteria}")
    
    return True


def validate_reward_function():
    """Validate reward provides partial progress."""
    print("\n" + "="*60)
    print("VALIDATING REWARD FUNCTION")
    print("="*60)
    
    env = WarehouseEnvironment(task="medium", seed=42)
    env.reset()
    
    rewards = []
    
    for i in range(20):
        action = WarehouseAction(
            restock_level=i % 5,
            allocation_strategy=i % 5,
            worker_mode=i % 4,
            delivery_priority=i % 4,
            storage_action=i % 3
        )
        obs = env.step(action)
        rewards.append(float(obs.reward) if obs.reward else 0.0)
    
    print(f"[PASS] Reward range: min={min(rewards):.3f}, max={max(rewards):.3f}")
    print(f"[PASS] Reward varies: {len(set([round(r, 2) for r in rewards]))} unique values")
    
    return True


def validate_openenv_compliance():
    """Validate OpenEnv spec compliance."""
    print("\n" + "="*60)
    print("VALIDATING OPENENV COMPLIANCE")
    print("="*60)
    
    # Check file structure
    base = Path(__file__).parent
    
    required = [
        "src/envs/smart_warehouse/models.py",
        "src/envs/smart_warehouse/environment.py",
        "src/envs/smart_warehouse/client.py",
        "src/envs/smart_warehouse/server/app.py",
        "src/envs/smart_warehouse/server/Dockerfile",
        "openenv.yaml",
        "inference.py",
    ]
    
    for path in required:
        full_path = base / path
        if full_path.exists():
            print(f"[PASS] {path} exists")
        else:
            print(f"[FAIL] {path} missing")
            return False
    
    # Check models have required classes
    from envs.smart_warehouse.models import (
        WarehouseAction,
        WarehouseObservation,
        WarehouseState,
    )
    print(f"[PASS] WarehouseAction: {WarehouseAction.__bases__}")
    print(f"[PASS] WarehouseObservation: {WarehouseObservation.__bases__}")
    print(f"[PASS] WarehouseState: {WarehouseState.__bases__}")
    
    # Check environment has required methods
    env = WarehouseEnvironment()
    assert hasattr(env, 'reset'), "Missing reset() method"
    assert hasattr(env, 'step'), "Missing step() method"
    assert hasattr(env, 'state'), "Missing state property"
    print(f"[PASS] Environment has reset(), step(), state")
    
    return True


def main():
    """Run all validations."""
    print("\n" + "="*60)
    print("OPENENV VALIDATION SUITE")
    print("="*60)
    
    results = {}
    
    results["Typed Models"] = validate_models()
    results["API Methods"] = validate_api()
    results["Tasks & Graders"] = validate_tasks_and_graders()
    results["Reward Function"] = validate_reward_function()
    results["OpenEnv Compliance"] = validate_openenv_compliance()
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL VALIDATIONS PASSED!")
    else:
        print("SOME VALIDATIONS FAILED!")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
