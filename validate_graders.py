"""
Grader Validation Test
=====================

This script validates that the SmartWarehouse environment has all required
graders for the Meta PyTorch OpenEnv Hackathon.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from envs.smart_warehouse.graders import AVAILABLE_GRADERS, GRADER_METADATA, list_all_tasks
from envs.smart_warehouse.models import (
    TaskGrader,
    InventoryRestockingGrader,
    OrderFulfillmentGrader,
    WarehouseOptimizationGrader,
)


def test_graders():
    """Test that all required graders exist and have correct passing scores."""
    
    print("=" * 60)
    print("GRADER VALIDATION TEST")
    print("=" * 60)
    
    # Test 1: Check AVAILABLE_GRADERS has 3 entries
    print(f"\n[Test 1] AVAILABLE_GRADERS count: {len(AVAILABLE_GRADERS)}")
    assert len(AVAILABLE_GRADERS) >= 3, "Must have at least 3 graders"
    print("✓ PASS: Has 3+ graders")
    
    # Test 2: Check all required tasks exist
    required_tasks = ["easy", "medium", "hard"]
    for task in required_tasks:
        print(f"\n[Test 2.{task}] Checking task '{task}'")
        assert task in AVAILABLE_GRADERS, f"Missing grader for task: {task}"
        assert task in GRADER_METADATA, f"Missing metadata for task: {task}"
        
        grader = AVAILABLE_GRADERS[task]
        assert hasattr(grader, 'passing_score'), f"Grader {task} missing passing_score property"
        assert hasattr(grader, 'grade'), f"Grader {task} missing grade method"
        assert hasattr(grader, 'get_success_criteria'), f"Grader {task} missing get_success_criteria method"
        
        print(f"  - passing_score: {grader.passing_score}")
        print(f"  - criteria: {grader.get_success_criteria()}")
        print(f"✓ PASS: Task '{task}' has valid grader")
    
    # Test 3: Check passing scores are correct
    print(f"\n[Test 3] Checking passing scores")
    expected_scores = {"easy": 0.6, "medium": 0.65, "hard": 0.70}
    for task, expected in expected_scores.items():
        actual = AVAILABLE_GRADERS[task].passing_score
        assert actual == expected, f"Task {task}: expected {expected}, got {actual}"
        print(f"  - {task}: {actual} ✓")
    
    # Test 4: Test grading functionality
    print(f"\n[Test 4] Testing grading functionality")
    sample_trajectory = [
        {"fill_rate": 0.95, "stockout_hours": 2, "fulfillment_rate": 0.92, "delivery_hours": 3.5, "customer_satisfaction": 0.85},
        {"fill_rate": 0.90, "stockout_hours": 3, "fulfillment_rate": 0.88, "delivery_hours": 4.0, "customer_satisfaction": 0.80},
    ]
    sample_state = {"baseline_cost": 100}
    
    for task in required_tasks:
        grader = AVAILABLE_GRADERS[task]
        score = grader.grade(sample_trajectory, sample_state)
        assert 0.0 <= score <= 1.0, f"Task {task}: score {score} out of range"
        print(f"  - {task}: score = {score:.3f} ✓")
    
    # Test 5: Check list_all_tasks
    print(f"\n[Test 5] list_all_tasks() = {list_all_tasks()}")
    assert len(list_all_tasks()) >= 3
    print("✓ PASS: list_all_tasks works")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        test_graders()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
