"""
SkillClaw Agent Server v3 — Bash Shell + Remote Sim Exec.

The agent has two tools:
  1. bash(command) — execute any shell command locally (grep, cat, find, python...)
  2. sim_exec(code) — execute Python code in ManiSkill sandbox on remote GPU VM

No predefined load_reference_code(), load_skill_library(), save_skill().
Agent uses bash to do all of that itself — like Claude Code.

Usage:
    # Single task:
    python rlvr/agent_server_v3.py --env-id PickSingleYCB-v1 --seed 42

    # Flywheel:
    python rlvr/agent_server_v3.py --flywheel

    # Study (agent explores the codebase):
    python rlvr/agent_server_v3.py --study
"""

import argparse
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("agent_v3")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
MAX_STEPS = 30  # max tool calls per episode
MAX_ATTEMPTS = 15  # max sim_exec retries for one task
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# Tool Definitions (OpenAI function calling format)
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
                "Execute Python code in the ManiSkill simulation sandbox on the remote GPU. "
                "The code should define a `def solve(env, planner):` function. "
                "The sandbox provides: env, planner (None in delta control mode), torch, np, numpy, sapien, get_actor_obb, compute_grasp_info_by_obb. "
                "Returns: success (bool), task_success (bool), reward, error info, state_before, state_after."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code containing a def solve(env, planner): function",
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
    """Execute code in remote ManiSkill sandbox. Returns formatted result string."""
    resp = await client.post(
        f"{resources_url}/execute_code",
        json={"session_id": session_id, "code": code, "record_video": record_video},
    )
    resp.raise_for_status()
    data = resp.json()

    # Format as readable text (not JSON dump — agent reads this)
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

    # Add scratch pad reminder on failure
    if not data.get("task_success"):
        lines.append("\n⚠️ BEFORE your next attempt: update brains/private_brain/scratch.md with what you tried, what happened, and what pattern you see across all attempts. Then read it back before writing new code.")

    result_text = "\n".join(lines)

    # Download video if available
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
# System Prompts
# ============================================================================

TASK_PROMPT = """You are a robot skill agent. You solve ManiSkill robot manipulation tasks by writing Python code.

You have two tools:
- `bash(command)` — run any shell command locally (grep, cat, find, ls, mkdir, python, etc.)
- `sim_exec(code)` — execute Python code in the ManiSkill simulator on a remote GPU

## Your workflow

### Step 1: Understand the task (MOST IMPORTANT — do this BEFORE anything else)

Before writing any code, you MUST understand what the task requires using ONLY:
- The **task description** (provided below)
- The **initial state** (object names and positions, provided below)

From these two sources, reason about:

a) **What objects are in the scene?** Look at the initial state — every object is listed with its position. Think about what each object is and how they relate to each other physically.

b) **What needs to happen?** Read the task description carefully. What is the goal? Which objects need to move where?

c) **What strategy do I need?** Based on (a) and (b), reason about:
   - Which objects do I need to interact with, and in what order?
   - Do I need to use one object as a tool to manipulate another?
   - Is this a grasp task, a push task, a tool-use task, an insertion task, or something else?
   - What are the physical constraints? (object sizes, distances, orientations)

Write your understanding as a brief plan before proceeding.

**IMPORTANT: Do NOT read environment source code (envs/tasks/*.py) to understand the task.** In the real world, there is no source code for the environment. You must understand the task from the description and sensor data (state) alone — just like a real robot would.

### Step 2: Study how to use the framework

Now that you understand WHAT to do, learn HOW to do it:
   - `cat brains/shared_brain/README.md` — Task analysis + API reference + skill index
   - `cat brains/shared_brain/debug_patterns.md` — lessons from past failures
   - `ls brains/shared_brain/skills/sc-*/SKILL.md` — existing skills you've learned
   - Search official solutions for similar PATTERNS (not for this specific task):
     e.g. if you need to grasp an object, look at how pick_cube.py does grasping
     e.g. if you need to insert something, look at how peg_insertion_side.py does alignment
   - You CAN read framework code (planner, utils, API) — that's learning the tool
   - You CANNOT read environment code (envs/tasks/) — that's peeking at the world

### Step 3: Write & Execute

Write a `def solve(env, planner):` function and run it with sim_exec.

### Step 4: Debug + Scratch Pad

**MANDATORY after EVERY sim_exec attempt (success or failure):**

1. **IMMEDIATELY write to scratch pad** — do this BEFORE any other bash call or sim_exec:
   ```
   bash("cat >> brains/private_brain/scratch.md << 'EOF'
   ## Attempt N
   - What I tried: ...
   - What happened: ...
   - Key observation: ...
   - What to try next: ...
   EOF")
   ```
   This is NOT optional. Every attempt must be logged. If you skip this step, you will repeat the same mistakes.

2. **Read your scratch pad** before writing new code: `bash("cat brains/private_brain/scratch.md")`

3. Look for PATTERNS across attempts — don't just fix the immediate bug. Ask: what do all my observations tell me about the physics of this task?

The scratch pad is your thinking notebook. A human is watching it in real time to understand your reasoning. If you don't write to it, they can't help you.

### Step 5: Save to Private Brain

If task succeeds, save the skill to your private workspace:
   - `mkdir -p brains/private_brain/dev-sc-skill-name/scripts`
   - Write SKILL.md and scripts/main.py using bash

### Step 6: Promote to Shared Brain (if quality is good)

If the skill works well (you're confident it's correct and reusable), publish it in ONE command:
   ```
   bash brains/shared_brain/skills/sc-publish/scripts/main.sh <name> <video_file> "<description>"
   ```
   Example: `bash brains/shared_brain/skills/sc-publish/scripts/main.sh pull PullCube-v1_sess_abc.mp4 "Pull object to goal by grasping and dragging"`

   This script handles everything: copy to shared brain, clean private brain, GIF, branch, commit, push, PR.
   Once the PR is merged, ALL other agents can use this skill.

## Current Task
Task: {env_id}
Description: {task_description}
Seed: {seed}
Session ID: {session_id}

## Initial State
{initial_state}

## Important
- All ManiSkill data is batched torch tensors. Use [0] indexing and .cpu().numpy() to convert.
- Use move_to_pose_with_screw for precision (<10cm). Use move_to_pose_with_RRTConnect for large moves.
- move_to_pose_with_screw and move_to_pose_with_RRTConnect return -1 on failure.
- Always `env = env.unwrapped` to access task attributes.

## Direct Delta Control (for force-based tasks)
For tasks that need continuous contact (turning faucets, rolling balls), the motion planner
won't work because it opens the gripper for collision avoidance. In delta control mode,
`planner` is None — use `env.step()` directly instead.

Action format: `[dx, dy, dz, drx, dry, drz, gripper]` — shape is `(7,)` (1D, NOT batched)
- Position deltas: clipped to [-1, 1], mapped to ~0.1m per step
- Rotation deltas: clipped to [-1, 1], mapped to ~0.1rad per step
- Gripper: -1 = close, +1 = open

**CRITICAL tensor shapes:** action is 1D `(7,)`, but state tensors like `env.agent.tcp.pose.p`
are batched `(1, 3)`. Use `[0]` to index state tensors, but NOT action tensors:
- `tcp_pos = env.agent.tcp.pose.p[0].cpu().numpy()` — correct (removes batch dim)
- `action[:3] = ...` — correct (action is 1D)
- `env.target_link_pos[0].cpu().numpy()` — correct for getting handle position
- Do NOT call `env.get_obs()` — it returns a raw state tensor, not a dict.
  Instead access attributes directly: `env.target_link_pos`, `env.target_joint_axis`, etc.

```python
def solve(env, planner):
    # planner is None in delta mode — use env.step() directly
    import torch

    def move_to(env, target_pos, gripper=-1, gain=8.0, max_steps=100):
        \"\"\"Move TCP toward target_pos with proportional control.\"\"\"
        for step in range(max_steps):
            tcp_pos = env.agent.tcp.pose.p[0].cpu().numpy()
            error = target_pos - tcp_pos
            if np.linalg.norm(error) < 0.015:
                return True
            delta = np.clip(error * gain, -1, 1)
            action = torch.zeros(env.action_space.shape, device=env.device)
            action[:3] = torch.tensor(delta, device=env.device)
            action[6] = gripper
            env.step(action)
        return False

    def rotate(env, axis_deltas, gripper=-1, steps=50):
        \"\"\"Apply rotation deltas [drx, dry, drz] for N steps.\"\"\"
        for _ in range(steps):
            action = torch.zeros(env.action_space.shape, device=env.device)
            action[3:6] = torch.tensor(axis_deltas, device=env.device)
            action[6] = gripper
            env.step(action)

    # Example: move to object, close gripper, rotate
    obj_pos = ...  # get from initial state
    move_to(env, obj_pos, gripper=1)   # approach with open gripper
    move_to(env, obj_pos, gripper=-1)  # close gripper
    rotate(env, [0, 0, 0.5], gripper=-1, steps=100)  # rotate around z-axis
```

### TurnFaucet-specific: Use the joint axis from initial state!
The initial state includes `target_joint_axis` (the faucet's rotation axis) and `target_link_pos`
(the handle's center-of-mass position). Do NOT guess the rotation axis — use the one provided.

To turn the faucet, push the handle in the **tangential direction** (perpendicular to both
the joint axis and the radial vector from joint to handle). Think of it like pushing a door handle:
```python
joint_axis = np.array(initial_state["target_joint_axis"])  # e.g. [1, 0, 0]
handle_pos = np.array(initial_state["target_link_pos"])
faucet_pos = ...  # faucet base position from initial state
radial = handle_pos - faucet_pos
tangent = np.cross(joint_axis, radial)
tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
# Push in tangent direction while tracking handle position each step
```

**IMPORTANT:** To check progress, use `env.evaluate()['angle_dist']` — this is the REMAINING
angle to turn (decreases toward 0). Do NOT use `env.target_angle_diff` — that's the TOTAL angle
to turn (constant, never changes). Success = `angle_dist < 0`.

**Do NOT raise RuntimeError** at the end of solve() — just return. The state_after in the response
shows angle_dist so you can see progress between attempts. Raising RuntimeError hides your progress.

Use this when: the planner keeps opening the gripper, or you need continuous contact while moving.
Note: `torch` is available in the sandbox — no need to import it yourself.

## Critical: Handle motion planning failures LOUDLY
When a move fails, DO NOT silently return. Raise an exception so you get useful feedback:
```python
res = planner.move_to_pose_with_screw(target_pose)
if res == -1:
    raise RuntimeError(f"move_to_pose_with_screw failed for target {target_pose.p}. Try different approach angle, use RRTConnect, or break into smaller steps.")
```
This way you'll see WHICH step failed in the error feedback and can fix it specifically.
NEVER write `if res == -1: return res` — that silently swallows the error.

Start by understanding the task from the description and initial state, then study reference code, then solve.

## Kitchen / Mobile Robot (RoboCasaKitchen-v1)
If the task is RoboCasaKitchen-v1, you control a TidyVerse mobile robot (3-DOF base + 7-DOF Panda arm + Robotiq gripper) in a kitchen.

**Control mode: `whole_body`** — action format: `[arm_j1-j7, gripper, base_x, base_y, base_yaw]` = 11 dims
- `gripper`: 0.0 = open, 0.81 = closed

**Sandbox objects (available in solve()):**
- `planner` — SapienPlanner for motion planning
- `planning_world` — SapienPlanningWorld (for ACM, IK)
- `robot` — the robot articulation
- `scene` — SAPIEN sub-scene
- `fixtures` — dict of kitchen fixtures (counters, drawers, etc.)
- `MPPose` — mplib Pose class

**Available helpers:**
- `make_action(arm_qpos, gripper, base_cmd)` → batched action tensor
- `sync_planner(planner)` — update planner from simulation state
- `get_robot_qpos(robot)` → numpy array of current joint positions
- `execute_trajectory(traj, step_fn, gripper, ...)` — follow a planned path
- `actuate_gripper(step_fn, env, robot, gripper_val, label)` — open/close gripper
- `wait_until_stable(step_fn, hold_action, robot)` — wait for robot to settle
- `build_grasp_poses(cube_pos, arm_base)` → list of (name, pos, quat) grasp candidates
- `navigate_to(env, planner, pw, target_pos)` — move base near target
- `pick_up(env, planner, pw, obj_pos)` — full grasp pipeline
- `place_object(env, planner, pw, target_pos)` — place held object

**Constants:** `ARM_HOME`, `GRIPPER_OPEN` (0.0), `GRIPPER_CLOSED` (0.81), `MASK_ARM_ONLY`, `MASK_WHOLE_BODY`, `CUBE_HALF`

**Pattern:**
```python
def solve(env, planner):
    # Define step function
    def step_fn(action):
        env.step(action)

    # Get planning world from sandbox
    pw = planning_world

    # Read initial state to find target object position
    target_pos = ...  # from initial_state["spawned_objects"]["target_cube"]["position"]

    # Option A: Use high-level primitives
    result = pick_up(env, planner, pw, target_pos, step_fn)

    # Option B: Use lower-level helpers directly
    sync_planner(planner)
    cq = get_robot_qpos(robot)
    pose = MPPose(p=np.array([x, y, z]), q=[0, 1, 0, 0])
    result = planner.plan_pose(pose, cq, mask=MASK_WHOLE_BODY, planning_time=5.0)
    if result['status'] == 'Success':
        execute_trajectory(result['position'], step_fn, GRIPPER_OPEN, robot=robot)
```
"""

STUDY_PROMPT = """You are studying a robotics simulation framework (ManiSkill). Your goal is to read the official solution code and write a framework.md knowledge base.

You have one tool:
- `bash(command)` — run any shell command locally

## Your task

1. List all solution files: `ls ManiSkill/mani_skill/examples/motionplanning/panda/solutions/`
2. Read each solution file with `cat`
3. Write a comprehensive `framework.md` to `skill_library/framework.md` that covers:
   - Environment API (attribute names per task)
   - Tensor handling patterns
   - Motion planning (screw vs RRTConnect)
   - Pose algebra patterns
   - Grasp computation patterns
   - Common pitfalls

Use `bash` to write the file: `cat > skill_library/framework.md << 'EOF'\n...\nEOF`

Start by listing and reading the solution files.
"""

TASK_DESCRIPTIONS = {
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it.",
    "PullCube-v1": "Pull a cube onto a target.",
    "StackCube-v1": "Pick up a red cube and stack it on top of a green cube and let go without it falling.",
    "PlaceSphere-v1": "Place the sphere into the shallow bin.",
    "PickSingleYCB-v1": "Pick up a random object from the YCB dataset and move it to a random goal position.",
    "LiftPegUpright-v1": "Move a peg laying on the table to any upright position on the table.",
    "PokeCube-v1": "Poke a red cube with a peg and push it to a target goal position.",
    "RollBall-v1": "Push and roll a ball to a goal region at the other end of the table.",
    "TurnFaucet-v1": "Turn the faucet handle.",
    "PegInsertionSide-v1": "Pick up an orange-white peg and insert the orange end into the box with a hole in it.",
    "PlugCharger-v1": "Pick up a charger and insert it into the wall receptacle.",
    "StackPyramid-v1": "Pick up a red cube, place it next to the green cube, and stack the blue cube on top of both without it falling.",
    "AssemblingKits-v1": "Pick up a misplaced shape on the board and insert it into the correct empty slot.",
    "RoboCasaKitchen-v1": "Navigate a mobile robot in a kitchen to pick up objects from counters and surfaces.",
}


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
    system_prompt: str = None,
    record_video: bool = False,
    control_mode: str = "pd_joint_pos",
    robot_uid: str = "panda",
) -> Dict[str, Any]:
    """Run agent with bash + sim_exec tools in a ReAct loop."""

    # Auto-detect robot for kitchen tasks
    if env_id == "RoboCasaKitchen-v1":
        control_mode = "whole_body"
        robot_uid = "tidyverse"

    start_time = time.time()

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

        logger.info(f"Session {session_id} for {env_id} (seed={seed})")

        # Build system prompt
        if system_prompt is None:
            task_desc = TASK_DESCRIPTIONS.get(env_id, env_id)
            system_prompt = TASK_PROMPT.replace(
                "{env_id}", env_id
            ).replace(
                "{task_description}", task_desc
            ).replace(
                "{seed}", str(seed)
            ).replace(
                "{session_id}", session_id
            ).replace(
                "{initial_state}", json.dumps(initial_state, indent=2)
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
        recorder.trajectory.extra = {"env_id": env_id, "seed": seed}
        recorder.add_system(system_prompt)
        recorder.add_user("Start by studying the relevant reference code, then solve the task.")

        task_success = False
        tool_log = []
        sim_attempts = 0

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
                    logger.error(f"LLM error: {llm_resp.status_code}")
                    break

                choice = llm_resp.json()["choices"][0]
                assistant_msg = choice["message"]
                text = assistant_msg.get("content", "") or ""
                tool_calls = assistant_msg.get("tool_calls")

                # Extract usage metrics from LLM response
                usage = llm_resp.json().get("usage", {})
                llm_metrics = Metrics(
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                )

                # No tool calls — agent is done
                if not tool_calls:
                    logger.info(f"  Step {step+1}: Agent finished. {text[:100]}")
                    recorder.add_agent(text, metrics=llm_metrics)
                    break

                # Record agent step with tool calls
                atif_tool_calls = [
                    ToolCall(
                        tool_call_id=tc["id"],
                        function_name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"],
                    )
                    for tc in tool_calls
                ]
                recorder.add_agent(text, tool_calls=atif_tool_calls, metrics=llm_metrics)

                # Process each tool call
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
                                json={
                                    "env_id": env_id,
                                    "seed": seed,
                                    "control_mode": control_mode,
                                    "robot_uid": robot_uid,
                                },
                            )
                            resp.raise_for_status()
                            session_id = resp.json()["session_id"]

                        result = await execute_sim(client, resources_url, session_id, code, record_video=record_video)

                        if "TASK SUCCESS" in result:
                            task_success = True
                            logger.info(f"  ✅ TASK SUCCESS on sim attempt {sim_attempts}!")

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

                # Record observation (all tool results for this step)
                recorder.add_observation(observation_results)

                if task_success:
                    # Give agent multiple turns to save + promote skill
                    messages.append({
                        "role": "user",
                        "content": "Task succeeded! Do exactly these 2 steps:\n\nStep 1: Save to private brain:\n  mkdir -p brains/private_brain/dev-sc-<name>/scripts\n  Write SKILL.md and scripts/main.py\n\nStep 2: Publish using the sc-publish skill (ONE command):\n  bash brains/shared_brain/skills/sc-publish/scripts/main.sh <name> <video_filename> \"<description>\"\n\nThe publish script handles everything: copy to shared brain, clean private, GIF, branch, commit, push, PR.\nDo NOT do git/PR steps manually — use the script.",
                    })
                    # Allow up to 5 turns for save + promote + PR
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
    status = "✅ SUCCESS" if task_success else "❌ FAIL"
    logger.info(f"  {status} — {env_id} seed={seed} sim_attempts={sim_attempts} steps={len(tool_log)} time={elapsed:.1f}s")

    # Save ATIF trajectory
    recorder.finalize(extra={
        "env_id": env_id,
        "seed": seed,
        "task_success": task_success,
        "sim_attempts": sim_attempts,
        "elapsed_seconds": round(elapsed, 1),
    })
    traj_dir = PROJECT_ROOT / "trajectories"
    traj_path = recorder.save(traj_dir / f"{env_id}_seed{seed}_{recorder.trajectory.session_id}.json")
    logger.info(f"  Trajectory saved: {traj_path.name} ({len(recorder.trajectory.steps)} steps)")

    return {
        "env_id": env_id,
        "seed": seed,
        "task_success": task_success,
        "sim_attempts": sim_attempts,
        "total_steps": len(tool_log),
        "elapsed_seconds": round(elapsed, 1),
        "tool_log": tool_log,
        "trajectory_path": str(traj_path),
        "session_id": recorder.trajectory.session_id,
    }


# ============================================================================
# Study Mode
# ============================================================================

async def run_study(api_key: str, model_id: str = DEFAULT_MODEL):
    """Agent explores ManiSkill repo and writes framework.md using bash."""
    logger.info("=== STUDY MODE: Agent exploring ManiSkill codebase ===")

    messages = [
        {"role": "system", "content": STUDY_PROMPT},
        {"role": "user", "content": "Start by listing and reading the official solution files."},
    ]

    async with httpx.AsyncClient(timeout=120.0) as client:
        for step in range(30):
            resp = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "max_tokens": 8192,
                    "messages": messages,
                    "tools": [TOOLS[0]],  # bash only
                },
            )
            if resp.status_code != 200:
                logger.error(f"LLM error: {resp.status_code}")
                break

            choice = resp.json()["choices"][0]
            assistant_msg = choice["message"]
            tool_calls = assistant_msg.get("tool_calls")

            if not tool_calls:
                logger.info(f"  Step {step+1}: Agent finished studying.")
                break

            messages.append(assistant_msg)

            for tc in tool_calls:
                cmd = json.loads(tc["function"]["arguments"]).get("command", "")
                logger.info(f"  Step {step+1}: bash({cmd[:80]})")
                result = execute_bash(cmd)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result[:10000],
                })

    # Check if framework.md was created
    fw_path = PROJECT_ROOT / "brains" / "shared_brain" / "framework.md"
    if fw_path.exists():
        logger.info(f"  framework.md written ({len(fw_path.read_text())} chars)")
    else:
        logger.warning("  framework.md was NOT created — agent may need more steps")


# ============================================================================
# Flywheel
# ============================================================================

async def run_flywheel(
    resources_url: str,
    api_key: str,
    tasks: List[str],
    seed: int = 42,
    model_id: str = DEFAULT_MODEL,
    record_video: bool = False,
) -> List[Dict]:
    """Run tasks sequentially. Agent accumulates skills via bash."""
    logger.info(f"=== FLYWHEEL: {len(tasks)} tasks, seed={seed} ===")

    results = []
    for i, env_id in enumerate(tasks):
        logger.info(f"\n{'='*60}")
        logger.info(f"Task {i+1}/{len(tasks)}: {env_id}")
        logger.info(f"{'='*60}")

        result = await run_agent(
            resources_url=resources_url,
            api_key=api_key,
            env_id=env_id,
            seed=seed,
            model_id=model_id,
            record_video=record_video,
        )
        results.append(result)

        successes = sum(1 for r in results if r["task_success"])
        logger.info(f"Progress: {successes}/{len(results)}")

    logger.info(f"\n{'='*60}")
    logger.info("FLYWHEEL RESULTS")
    logger.info(f"{'='*60}")
    for r in results:
        status = "✅" if r["task_success"] else "❌"
        logger.info(f"  {status} {r['env_id']} — {r['sim_attempts']} sim attempts, {r['total_steps']} steps, {r['elapsed_seconds']}s")
    logger.info(f"{'='*60}")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SkillClaw Agent v3 — Bash + Sim Exec")
    parser.add_argument("--resources-url", default="http://localhost:8100")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--env-id", default="PickSingleYCB-v1")
    parser.add_argument("--study", action="store_true")
    parser.add_argument("--flywheel", action="store_true")
    parser.add_argument("--control-mode", default="pd_joint_pos",
                        choices=["pd_joint_pos", "pd_ee_delta_pose"],
                        help="Control mode: pd_joint_pos (with planner) or pd_ee_delta_pose (direct delta)")
    parser.add_argument("--record-video", action="store_true",
                        help="Record simulation video for each sim_exec call")
    parser.add_argument("--robot-uid", default="panda",
                        help="Robot UID: 'panda' (fixed base) or 'tidyverse' (mobile base)")
    parser.add_argument("--tasks", nargs="+",
                        default=["PickSingleYCB-v1", "StackCube-v1", "PegInsertionSide-v1", "PlugCharger-v1"])

    args = parser.parse_args()
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY or pass --api-key")
        exit(1)

    if args.study:
        asyncio.run(run_study(api_key=api_key, model_id=args.model))
    elif args.flywheel:
        asyncio.run(run_flywheel(
            resources_url=args.resources_url,
            api_key=api_key,
            tasks=args.tasks,
            seed=args.seed,
            model_id=args.model,
            record_video=args.record_video,
        ))
    else:
        result = asyncio.run(run_agent(
            resources_url=args.resources_url,
            api_key=api_key,
            env_id=args.env_id,
            seed=args.seed,
            model_id=args.model,
            record_video=args.record_video,
            control_mode=args.control_mode,
            robot_uid=args.robot_uid,
        ))
        status = "SUCCESS" if result["task_success"] else "FAIL"
        print(f"\nResult: {status}")
        print(f"Sim attempts: {result['sim_attempts']}")
        print(f"Total steps: {result['total_steps']}")


if __name__ == "__main__":
    main()
