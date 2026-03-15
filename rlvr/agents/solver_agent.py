"""
SkillClaw Solver Agent — extracted from agent_server_v3.py.

Runs the ReAct loop to solve a single ManiSkill task.
On success, pushes a SkillSave to the pr_queue instead of doing
skill save + PR creation itself.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from trajectory import TrajectoryRecorder, ToolCall, ObservationResult, Metrics
from agent_server import (
    OPENROUTER_BASE_URL,
    DEFAULT_MODEL,
    MAX_STEPS,
    MAX_ATTEMPTS,
    PROJECT_ROOT,
    TOOLS,
    TASK_DESCRIPTIONS,
    TASK_PROMPT,
    execute_bash,
    execute_sim,
)

logger = logging.getLogger("solver")


@dataclass
class SkillSave:
    """Message pushed to pr_queue when a solver succeeds."""
    skill_name: str
    env_id: str
    seed: int
    description: str
    trajectory_path: str
    video_file: str = ""
    agent_id: str = "solver_0"
    review_notes: str = ""


# Solver-specific system prompt suffix: save skill but do NOT publish/PR
SOLVER_SAVE_PROMPT = (
    "Task succeeded! Save the skill to your private workspace:\n\n"
    "1. `mkdir -p brains/private_brain/dev-sc-<name>/scripts`\n"
    "2. Write `SKILL.md` (description, strategy, key observations)\n"
    "3. Write `scripts/main.py` (the working solve() function)\n\n"
    "Do NOT run the publish script or create a PR — another agent handles that.\n"
    "When you're done saving, respond with a short summary."
)


async def run_solver(
    resources_url: str,
    api_key: str,
    env_id: str,
    seed: int,
    agent_id: str = "solver_0",
    pr_queue: Optional[asyncio.Queue] = None,
    model_id: str = DEFAULT_MODEL,
    max_steps: int = MAX_STEPS,
    record_video: bool = False,
    control_mode: str = "pd_joint_pos",
    robot_uid: str = "panda",
) -> Dict[str, Any]:
    """Run solver agent: solve task, save skill, push SkillSave to queue."""

    # Auto-detect robot for kitchen tasks
    if env_id == "RoboCasaKitchen-v1":
        control_mode = "whole_body"
        robot_uid = "tidyverse"

    start_time = time.time()
    traj_dir = PROJECT_ROOT / "trajectories" / agent_id
    traj_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Create sim session
        resp = await client.post(
            f"{resources_url}/seed_session",
            json={
                "env_id": env_id,
                "seed": seed,
                "control_mode": control_mode,
                "robot_uid": robot_uid,
            },
        )
        resp.raise_for_status()
        session_data = resp.json()
        session_id = session_data["session_id"]
        initial_state = session_data["initial_state"]

        logger.info(f"[{agent_id}] Session {session_id} for {env_id} (seed={seed})")

        # Build system prompt
        task_desc = TASK_DESCRIPTIONS.get(env_id, env_id)
        system_prompt = (
            TASK_PROMPT
            .replace("{env_id}", env_id)
            .replace("{task_description}", task_desc)
            .replace("{seed}", str(seed))
            .replace("{session_id}", session_id)
            .replace("{initial_state}", json.dumps(initial_state, indent=2))
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Start by studying the relevant reference code, then solve the task."},
        ]

        # ATIF trajectory recording
        recorder = TrajectoryRecorder(
            model_name=model_id,
            tool_definitions=TOOLS,
        )
        recorder.trajectory.extra = {"env_id": env_id, "seed": seed, "agent_id": agent_id}
        recorder.add_system(system_prompt)
        recorder.add_user("Start by studying the relevant reference code, then solve the task.")

        task_success = False
        tool_log = []
        sim_attempts = 0
        last_video_file = ""

        try:
            for step in range(max_steps):
                # Call LLM
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
                    logger.error(f"[{agent_id}] LLM error: {llm_resp.status_code}")
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
                    logger.info(f"[{agent_id}] Step {step+1}: Agent finished. {text[:100]}")
                    recorder.add_agent(text, metrics=llm_metrics)
                    break

                atif_tool_calls = [
                    ToolCall(
                        tool_call_id=tc["id"],
                        function_name=tc["function"]["name"],
                        arguments=(
                            json.loads(tc["function"]["arguments"])
                            if isinstance(tc["function"]["arguments"], str)
                            else tc["function"]["arguments"]
                        ),
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
                        logger.info(f"[{agent_id}] Step {step+1}: bash({cmd[:80]})")
                        result = execute_bash(cmd)

                    elif func_name == "sim_exec":
                        code = func_args.get("code", "")
                        sim_attempts += 1
                        logger.info(f"[{agent_id}] Step {step+1}: sim_exec ({len(code)} chars, attempt {sim_attempts})")

                        if sim_attempts > 1:
                            await client.post(
                                f"{resources_url}/cleanup_session",
                                json={"session_id": session_id},
                            )
                            resp = await client.post(
                                f"{resources_url}/seed_session",
                                json={
                                    "env_id": env_id,
                                    "seed": seed,
                                    "control_mode": control_mode,
                                    "robot_uid": robot_uid,
                                },
                            )
                            resp.raise_for_status()
                            session_id = resp.json()["session_id"]

                        result = await execute_sim(
                            client, resources_url, session_id, code, record_video=record_video,
                        )

                        # Track video file
                        if "Video saved: demos/" in result:
                            last_video_file = result.split("Video saved: demos/")[-1].split("\n")[0].strip()

                        if "TASK SUCCESS" in result:
                            task_success = True
                            logger.info(f"[{agent_id}] ✅ TASK SUCCESS on sim attempt {sim_attempts}!")

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
                    # Give agent turns to save skill (but NOT publish/PR)
                    messages.append({"role": "user", "content": SOLVER_SAVE_PROMPT})
                    for save_step in range(10):
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
                            logger.info(f"[{agent_id}] Save complete: {save_msg.get('content', '')[:100]}")
                            break

                        messages.append(save_msg)
                        for tc in save_tool_calls:
                            if tc["function"]["name"] == "bash":
                                cmd = json.loads(tc["function"]["arguments"]).get("command", "")
                                logger.info(f"[{agent_id}] Saving skill: bash({cmd[:80]})")
                                result = execute_bash(cmd)
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "content": result,
                                })
                    break

                if sim_attempts >= MAX_ATTEMPTS:
                    logger.info(f"[{agent_id}] Max sim attempts ({MAX_ATTEMPTS}) reached")
                    break

        finally:
            await client.post(
                f"{resources_url}/cleanup_session",
                json={"session_id": session_id},
            )

    elapsed = time.time() - start_time
    status = "✅ SUCCESS" if task_success else "❌ FAIL"
    logger.info(f"[{agent_id}] {status} — {env_id} seed={seed} sim_attempts={sim_attempts} steps={len(tool_log)} time={elapsed:.1f}s")

    # Save ATIF trajectory
    recorder.finalize(extra={
        "env_id": env_id,
        "seed": seed,
        "task_success": task_success,
        "sim_attempts": sim_attempts,
        "elapsed_seconds": round(elapsed, 1),
        "agent_id": agent_id,
    })
    traj_path = recorder.save(traj_dir / f"{env_id}_seed{seed}_{recorder.trajectory.session_id}.json")
    logger.info(f"[{agent_id}] Trajectory saved: {traj_path.name}")

    # Derive skill name from env_id (e.g. PullCube-v1 → pull)
    skill_name = env_id.replace("-v1", "").replace("-v0", "")
    # CamelCase → kebab: PullCube → pull-cube → pull
    import re
    skill_name = re.sub(r"(?<=[a-z])(?=[A-Z])", "-", skill_name).lower()
    # Use first word as short name (pull-cube → pull, lift-peg-upright → lift-peg-upright)
    skill_name = skill_name.lower()

    # Push to pr_queue if success
    if task_success and pr_queue is not None:
        skill_save = SkillSave(
            skill_name=skill_name,
            env_id=env_id,
            seed=seed,
            description=TASK_DESCRIPTIONS.get(env_id, env_id),
            trajectory_path=str(traj_path),
            video_file=last_video_file,
            agent_id=agent_id,
        )
        await pr_queue.put(skill_save)
        logger.info(f"[{agent_id}] Pushed SkillSave to pr_queue: {skill_name}")

    return {
        "env_id": env_id,
        "seed": seed,
        "task_success": task_success,
        "sim_attempts": sim_attempts,
        "total_steps": len(tool_log),
        "elapsed_seconds": round(elapsed, 1),
        "trajectory_path": str(traj_path),
        "agent_id": agent_id,
        "skill_name": skill_name,
    }
