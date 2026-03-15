"""
Kitchen ReAct Orchestrator — drives an LLM agent through RoboCasa kitchen tasks.

Follows agent_server_v3.py structure: bash() + sim_exec() tools, ReAct loop,
ATIF trajectory recording, skill saving.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from trajectory import TrajectoryRecorder, ToolCall, ObservationResult, Metrics
from rlvr.agents.prompts import KITCHEN_TASK_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("kitchen_orchestrator")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
MAX_STEPS = 30
MAX_ATTEMPTS = 15
PROJECT_ROOT = Path(__file__).parent.parent.parent  # SkillClaw/

# ============================================================================
# Tool Definitions
# ============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a shell command locally. Use for: reading files (cat), "
                "searching code (grep -r), finding files (find), listing directories (ls), "
                "writing files (cat > file), running Python scripts, creating directories (mkdir). "
                "Working directory is the SkillClaw project root."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sim_exec",
            "description": (
                "Execute Python code in the RoboCasa kitchen simulation sandbox. "
                "The code should define a `def solve(env, planner, pw):` function. "
                "The sandbox provides: env, planner, pw (SapienPlanningWorld), np, sapien, "
                "MPPose, execute_trajectory, actuate_gripper, make_action, wait_until_stable, "
                "select_grasps, get_robot_qpos, sync_planner. "
                "Returns: success (bool), task_success (bool), reward, error info, state_before, state_after."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code containing a def solve(env, planner, pw): function",
                    }
                },
                "required": ["code"],
            },
        },
    },
]


# ============================================================================
# Tool Execution
# ============================================================================

def execute_bash(command: str) -> str:
    """Execute a shell command locally and return output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"
        return output[:10000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out after 30 seconds"
    except Exception as e:
        return f"ERROR: {e}"


async def execute_sim(
    client: httpx.AsyncClient,
    resources_url: str,
    session_id: str,
    code: str,
    record_video: bool = False,
) -> str:
    """Execute code in remote kitchen sandbox. Returns formatted result string."""
    resp = await client.post(
        f"{resources_url}/execute_code",
        json={"session_id": session_id, "code": code, "record_video": record_video},
    )
    resp.raise_for_status()
    data = resp.json()

    lines = []
    if data.get("task_success"):
        lines.append("TASK SUCCESS! reward=1.0")
    elif data.get("success"):
        lines.append("Code executed without errors, but task NOT completed.")
    else:
        lines.append(f"EXECUTION ERROR: {data.get('error_type', 'Unknown')}")
        lines.append(f"Message: {data.get('error', '')}")
        if data.get("error_traceback"):
            lines.append(f"Traceback:\n{data['error_traceback']}")

    if data.get("state_before"):
        lines.append(f"\nState BEFORE execution:\n{json.dumps(data['state_before'], indent=2)}")
    if data.get("state_after"):
        lines.append(f"\nState AFTER execution:\n{json.dumps(data['state_after'], indent=2)}")

    if not data.get("task_success"):
        lines.append("\nBEFORE your next attempt: update brains/private_brain/scratch.md with what you tried, what happened, and what pattern you see across all attempts.")

    result_text = "\n".join(lines)

    video_path = data.get("video_path")
    if video_path and record_video:
        filename = os.path.basename(video_path)
        try:
            video_resp = await client.get(f"{resources_url}/video/{filename}")
            if video_resp.status_code == 200:
                local_path = PROJECT_ROOT / "demos" / filename
                local_path.parent.mkdir(exist_ok=True)
                local_path.write_bytes(video_resp.content)
                logger.info(f"  Video downloaded: {local_path.name}")
                result_text += f"\n\nVideo saved: demos/{filename}"
        except Exception as e:
            logger.warning(f"  Video download failed: {e}")

    return result_text


# ============================================================================
# ReAct Agent Loop
# ============================================================================

async def run_agent(
    resources_url: str,
    api_key: str,
    env_id: str,
    seed: int,
    model_id: str = DEFAULT_MODEL,
    max_steps: int = MAX_STEPS,
    record_video: bool = False,
    layout: int = None,
    style: int = None,
) -> Dict[str, Any]:
    """Run kitchen agent with bash + sim_exec tools in a ReAct loop."""

    start_time = time.time()

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Create kitchen session
        seed_body = {"env_id": env_id, "seed": seed}
        if layout is not None:
            seed_body["layout"] = layout
        if style is not None:
            seed_body["style"] = style

        resp = await client.post(
            f"{resources_url}/seed_session",
            json=seed_body,
        )
        resp.raise_for_status()
        session_data = resp.json()
        session_id = session_data["session_id"]
        initial_state = session_data["initial_state"]

        logger.info(f"Kitchen session {session_id} for {env_id} (seed={seed})")

        # Build system prompt
        task_desc = initial_state.get("task_instruction", env_id)
        system_prompt = KITCHEN_TASK_PROMPT.replace(
            "{env_id}", env_id
        ).replace(
            "{task_description}", task_desc or env_id
        ).replace(
            "{seed}", str(seed)
        ).replace(
            "{session_id}", session_id
        ).replace(
            "{initial_state}", json.dumps(initial_state, indent=2)
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Start by studying the tidyverse reference code, then solve the task."},
        ]

        # ATIF trajectory recording
        recorder = TrajectoryRecorder(
            model_name=model_id,
            tool_definitions=TOOLS,
        )
        recorder.trajectory.extra = {"env_id": env_id, "seed": seed, "mode": "kitchen"}
        recorder.add_system(system_prompt)
        recorder.add_user("Start by studying the tidyverse reference code, then solve the task.")

        task_success = False
        tool_log = []
        sim_attempts = 0

        try:
            for step in range(max_steps):
                llm_resp = await client.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_id,
                        "max_tokens": 4096,
                        "messages": messages,
                        "tools": TOOLS,
                    },
                )
                if llm_resp.status_code != 200:
                    logger.error(f"LLM error: {llm_resp.status_code}")
                    break

                choice = llm_resp.json()["choices"][0]
                assistant_msg = choice["message"]
                text = assistant_msg.get("content", "") or ""
                tool_calls = assistant_msg.get("tool_calls")

                usage = llm_resp.json().get("usage", {})
                llm_metrics = Metrics(
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                )

                if not tool_calls:
                    logger.info(f"  Step {step+1}: Agent finished. {text[:100]}")
                    recorder.add_agent(text, metrics=llm_metrics)
                    break

                atif_tool_calls = [
                    ToolCall(
                        tool_call_id=tc["id"],
                        function_name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"],
                    )
                    for tc in tool_calls
                ]
                recorder.add_agent(text, tool_calls=atif_tool_calls, metrics=llm_metrics)

                messages.append(assistant_msg)
                observation_results = []

                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    try:
                        func_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        func_args = {}

                    if func_name == "bash":
                        cmd = func_args.get("command", "")
                        logger.info(f"  Step {step+1}: bash({cmd[:80]})")
                        result = execute_bash(cmd)

                    elif func_name == "sim_exec":
                        code = func_args.get("code", "")
                        sim_attempts += 1
                        logger.info(f"  Step {step+1}: sim_exec ({len(code)} chars, attempt {sim_attempts})")

                        # Reset session for clean state on retry
                        if sim_attempts > 1:
                            await client.post(
                                f"{resources_url}/cleanup_session",
                                json={"session_id": session_id},
                            )
                            resp = await client.post(
                                f"{resources_url}/seed_session",
                                json=seed_body,
                            )
                            resp.raise_for_status()
                            session_id = resp.json()["session_id"]

                        result = await execute_sim(client, resources_url, session_id, code, record_video=record_video)

                        if "TASK SUCCESS" in result:
                            task_success = True
                            logger.info(f"  TASK SUCCESS on sim attempt {sim_attempts}!")

                    else:
                        result = f"Unknown tool: {func_name}"

                    observation_results.append(ObservationResult(
                        source_call_id=tc["id"],
                        content=result[:5000],
                    ))

                    tool_log.append({
                        "step": step + 1,
                        "tool": func_name,
                        "input": func_args,
                        "output_preview": result[:500],
                        "task_success": task_success,
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

                recorder.add_observation(observation_results)

                if task_success:
                    messages.append({
                        "role": "user",
                        "content": "Task succeeded! Save the skill to brains/private_brain/ and promote with sc-publish if quality is good.",
                    })
                    for save_step in range(15):
                        save_resp = await client.post(
                            f"{OPENROUTER_BASE_URL}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": model_id,
                                "max_tokens": 4096,
                                "messages": messages,
                                "tools": TOOLS,
                            },
                        )
                        if save_resp.status_code != 200:
                            break
                        save_choice = save_resp.json()["choices"][0]
                        save_msg = save_choice["message"]
                        save_tool_calls = save_msg.get("tool_calls", [])

                        if not save_tool_calls:
                            logger.info(f"  Save complete: {save_msg.get('content', '')[:100]}")
                            break

                        messages.append(save_msg)
                        for tc in save_tool_calls:
                            if tc["function"]["name"] == "bash":
                                cmd = json.loads(tc["function"]["arguments"]).get("command", "")
                                logger.info(f"  Saving skill: bash({cmd[:80]})")
                                result = execute_bash(cmd)
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "content": result,
                                })
                    break

                if sim_attempts >= MAX_ATTEMPTS:
                    logger.info(f"  Max sim attempts ({MAX_ATTEMPTS}) reached")
                    break

        finally:
            await client.post(
                f"{resources_url}/cleanup_session",
                json={"session_id": session_id},
            )

    elapsed = time.time() - start_time
    status = "SUCCESS" if task_success else "FAIL"
    logger.info(f"  {status} — {env_id} seed={seed} sim_attempts={sim_attempts} steps={len(tool_log)} time={elapsed:.1f}s")

    # Save ATIF trajectory
    recorder.finalize(extra={
        "env_id": env_id,
        "seed": seed,
        "mode": "kitchen",
        "task_success": task_success,
        "sim_attempts": sim_attempts,
        "elapsed_seconds": round(elapsed, 1),
    })
    traj_dir = PROJECT_ROOT / "trajectories"
    traj_path = recorder.save(traj_dir / f"kitchen_{env_id}_seed{seed}_{recorder.trajectory.session_id}.json")
    logger.info(f"  Trajectory saved: {traj_path.name} ({len(recorder.trajectory.steps)} steps)")

    return {
        "env_id": env_id,
        "seed": seed,
        "mode": "kitchen",
        "task_success": task_success,
        "sim_attempts": sim_attempts,
        "total_steps": len(tool_log),
        "elapsed_seconds": round(elapsed, 1),
        "tool_log": tool_log,
        "trajectory_path": str(traj_path),
        "session_id": recorder.trajectory.session_id,
    }
