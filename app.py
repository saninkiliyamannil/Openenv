"""
Gradio Web Interface for SmartWarehouse
======================================

Run locally:
    python app.py

Deploy to HuggingFace Spaces.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr
import json
from typing import Optional

from envs.smart_warehouse.environment import WarehouseEnvironment
from envs.smart_warehouse.models import WarehouseAction, WarehouseObservation


# Global state
env: Optional[WarehouseEnvironment] = None


def reset_environment(task: str, seed: int, max_steps: int):
    """Reset the environment."""
    global env
    
    if env:
        pass  # Keep existing env
    
    env = WarehouseEnvironment(
        task=task,
        seed=seed,
        max_episode_steps=max_steps,
    )
    
    obs = env.reset()
    
    return (
        format_observation(obs),
        format_metrics(env.state.metrics),
        f"Reset! Task: {task}, Max steps: {max_steps}",
        0.0,
        False,
    )


def execute_step(action_str: str):
    """Execute one environment step."""
    global env
    
    if env is None:
        return "", "", "Reset environment first!", 0.0, True
    
    try:
        action_dict = json.loads(action_str)
        action = WarehouseAction.from_dict(action_dict)
    except (json.JSONDecodeError, ValueError) as e:
        return "", "", f"Invalid action: {e}", 0.0, False
    
    obs = env.step(action)
    
    done_msg = ""
    if obs.done:
        score = env.grade()
        done_msg = f"\n\nEpisode Complete! Score: {score:.3f}"
    
    return (
        format_observation(obs),
        format_metrics(env.state.metrics),
        done_msg or "Step executed.",
        float(obs.reward) if obs.reward else 0.0,
        obs.done,
    )


def format_observation(obs: WarehouseObservation) -> str:
    """Format observation as readable text."""
    lines = ["=== WAREHOUSE STATE ==="]
    
    if obs.inventory_levels:
        avg = sum(obs.inventory_levels) / len(obs.inventory_levels)
        lines.append(f"Inventory: avg {avg:.1f}/500")
    
    if obs.pending_orders:
        lines.append(f"Pending Orders: urgent={obs.pending_orders[0]:.0f}, "
                    f"normal={obs.pending_orders[1]:.0f}, low={obs.pending_orders[2]:.0f}")
    
    if obs.warehouse_zones:
        zones = ", ".join([f"{z:.0%}" for z in obs.warehouse_zones[:3]])
        lines.append(f"Zone Utilization: [{zones}, ...]")
    
    if obs.time_features:
        hour = int(obs.time_features[0])
        day = int(obs.time_features[1])
        lines.append(f"Time: Hour {hour}, Day {day}")
    
    if obs.messages:
        lines.append("")
        lines.append("Messages:")
        for msg in obs.messages[-2:]:
            lines.append(f"  {msg}")
    
    return "\n".join(lines)


def format_metrics(metrics: dict) -> str:
    """Format metrics as HTML table."""
    if not metrics:
        return "<p>No metrics yet</p>"
    
    rows = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if value < 1:
                rows.append(f"<tr><td>{key}</td><td>{value:.1%}</td></tr>")
            else:
                rows.append(f"<tr><td>{key}</td><td>{value:.2f}</td></tr>")
        else:
            rows.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
    
    return f"<table>{''.join(rows)}</table>"


def get_default_action() -> str:
    """Get default action JSON."""
    return json.dumps({
        "restock_level": 1,
        "allocation_strategy": 1,
        "worker_mode": 0,
        "delivery_priority": 0,
        "storage_action": 0,
    }, indent=2)


# Build Gradio UI
with gr.Blocks(title="SmartWarehouse - OpenEnv") as demo:
    gr.Markdown("# SmartWarehouse - OpenEnv")
    gr.Markdown("Real-world warehouse management RL environment")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Configuration")
            
            task_dropdown = gr.Dropdown(
                ["easy", "medium", "hard"],
                value="easy",
                label="Task"
            )
            seed_slider = gr.Slider(0, 1000, value=42, label="Seed")
            max_steps = gr.Slider(10, 200, value=50, label="Max Steps")
            reset_btn = gr.Button("Reset", variant="primary")
        
        with gr.Column(scale=2):
            gr.Markdown("### State")
            obs_display = gr.Textbox(label="Observation", lines=10, interactive=False)
            metrics_display = gr.HTML(label="Metrics")
    
    gr.Markdown("---")
    
    with gr.Row():
        gr.Markdown("""
        ### Action (JSON)
        ```json
        {
            "restock_level": 0-4,
            "allocation_strategy": 0-4,
            "worker_mode": 0-3,
            "delivery_priority": 0-3,
            "storage_action": 0-2
        }
        ```
        """)
    
    with gr.Row():
        action_input = gr.Textbox(
            value=get_default_action(),
            label="Action"
        )
        with gr.Column():
            step_btn = gr.Button("Step", variant="primary")
            done_check = gr.Checkbox(label="Done", interactive=False)
            reward_display = gr.Number(label="Reward", interactive=False)
            status_display = gr.Textbox(label="Status", interactive=False)
    
    # Initialize
    demo.load(
        reset_environment,
        inputs=[task_dropdown, seed_slider, max_steps],
        outputs=[obs_display, metrics_display, status_display, reward_display, done_check]
    )
    
    # Events
    reset_btn.click(
        reset_environment,
        inputs=[task_dropdown, seed_slider, max_steps],
        outputs=[obs_display, metrics_display, status_display, reward_display, done_check]
    )
    
    step_btn.click(
        execute_step,
        inputs=[action_input],
        outputs=[obs_display, metrics_display, status_display, reward_display, done_check]
    )


if __name__ == "__main__":
    # For HuggingFace Spaces, use the PORT environment variable
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
