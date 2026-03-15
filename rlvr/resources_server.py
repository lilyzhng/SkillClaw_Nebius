"""
SkillClaw Resources Server — FastAPI server for robot manipulation.

v1 endpoints (fixed tool calling):
- POST /seed_session — create ManiSkill env + planner for a rollout
- POST /{tool_name} — execute an atomic primitive, return result + state
- POST /verify — compute binary reward (task_success)
- POST /cleanup_session — teardown env + planner

v2 endpoint (code generation):
- POST /execute_code — exec() LLM-generated Python code in a sandboxed env

Usage:
    python rlvr/resources_server.py --port 8100

Reference:
    - Design doc: research/rlvr_v2_design.md
"""

import argparse
import json
import logging
import signal
import traceback
from typing import Any, Dict, List, Optional
from uuid import uuid4

import gymnasium as gym
import numpy as np
import sapien

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ManiSkill imports (deferred — only needed when actually creating envs)
import mani_skill.envs
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)

from primitives import RobotPrimitives, execute_tool, TASK_REGISTRY

# Kitchen/mobile robot support (optional — only if mplib + tidyverse are available)
try:
    from kitchen_helpers import (
        setup_kitchen_planner, get_kitchen_state,
        ARM_HOME, GRIPPER_OPEN, GRIPPER_CLOSED, MASK_ARM_ONLY, MASK_WHOLE_BODY,
        make_action, sync_planner, get_robot_qpos, wait_until_stable,
        execute_trajectory, actuate_gripper, build_grasp_poses,
        collect_placements, spawn_cube, build_kitchen_acm,
        navigate_to, pick_up, place_object,
        MPPose, SapienPlanner, SapienPlanningWorld,
        CUBE_HALF, PLANNING_TIMEOUT, IK_TIMEOUT,
        PRE_GRASP_HEIGHT, LIFT_HEIGHT,
        select_strategies, attempt_grasp,
    )
    KITCHEN_AVAILABLE = True
except ImportError as _kitchen_err:
    KITCHEN_AVAILABLE = False
    logger.info(f"Kitchen helpers not available: {_kitchen_err}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resources_server")

# ============================================================================
# Request/Response Models
# ============================================================================

class SeedSessionRequest(BaseModel):
    env_id: str = "PickCube-v1"
    seed: int = 42
    robot_uid: str = "panda"  # "panda" (fixed base) or "fetch" (mobile base)
    control_mode: str = "pd_joint_pos"  # "pd_joint_pos" (with planner) or "pd_ee_delta_pose" (direct delta control)


class SeedSessionResponse(BaseModel):
    session_id: str
    initial_state: Dict[str, Any]
    env_id: str


class ToolRequest(BaseModel):
    session_id: str
    tool_name: str
    arguments: Dict[str, Any] = {}


class ToolResponse(BaseModel):
    tool_result: Dict[str, Any]
    updated_state: Dict[str, Any]


class VerifyRequest(BaseModel):
    session_id: str


class VerifyResponse(BaseModel):
    reward: float
    done: bool
    info: Dict[str, Any]


class CleanupRequest(BaseModel):
    session_id: str


class ExecuteCodeRequest(BaseModel):
    session_id: str
    code: str
    record_video: bool = False


class ExecuteCodeResponse(BaseModel):
    success: bool
    task_success: bool = False
    reward: float = 0.0
    error_type: Optional[str] = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    state_before: Optional[Dict[str, Any]] = None
    state_after: Optional[Dict[str, Any]] = None
    steps_executed: Optional[int] = None
    video_path: Optional[str] = None


class ListSessionsResponse(BaseModel):
    sessions: List[Dict[str, Any]]


# ============================================================================
# Session Store
# ============================================================================

class SessionStore:
    """Manages ManiSkill environment sessions."""

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create(self, env_id: str, seed: int, robot_uid: str = "panda", control_mode: str = "pd_joint_pos") -> tuple:
        """Create a new session with ManiSkill env + planner + primitives."""
        session_id = f"sess_{uuid4().hex[:8]}"

        # Kitchen/tidyverse: force whole_body control mode
        if robot_uid == "tidyverse":
            control_mode = "whole_body"

        env = gym.make(
            env_id,
            obs_mode="state",
            control_mode=control_mode,
            render_mode="rgb_array",
            num_envs=1,
            robot_uids=robot_uid,
        )
        env.reset(seed=seed)

        planner = None
        primitives = None
        planning_world = None
        kitchen_fixtures = None

        if robot_uid == "tidyverse" and KITCHEN_AVAILABLE:
            # Kitchen path: use SapienPlanner from Simon's code
            env_unwrapped = env.unwrapped
            planner, planning_world = setup_kitchen_planner(env_unwrapped)

            # Spawn a target cube on the nearest counter surface
            scene = env_unwrapped.scene.sub_scenes[0]
            fixtures = env_unwrapped.scene_builder.scene_data[0]['fixtures']
            kitchen_fixtures = fixtures
            all_placements = collect_placements(fixtures)

            robot = env_unwrapped.agent.robot
            if all_placements:
                # Sort by distance from arm base (nearest first)
                arm_base = next(l for l in robot.get_links()
                                if l.get_name() == 'panda_link0').pose.p[0].cpu().numpy()
                # Filter: counter surfaces only, skip interiors (drawers/cabinets)
                open_surfaces = [
                    p for p in all_placements
                    if 'interior' not in p[0] and p[2] in ('Counter', 'Stove', 'Stovetop')
                ]
                if not open_surfaces:
                    open_surfaces = all_placements
                open_surfaces.sort(
                    key=lambda x: np.linalg.norm(arm_base - x[1]))
                # Use nearest counter — attempt_grasp handles shelf obstructions
                # via Front/Angled45 grasp strategies
                label, pos, ftype = open_surfaces[0]
                cube_pos = pos.copy()
                cube_pos[2] += CUBE_HALF + 0.002
                dist = np.linalg.norm(arm_base - cube_pos)
                spawn_cube(scene, "target_cube", cube_pos,
                           [1.0, 0.0, 0.0, 1])  # red
                logger.info(f"Spawned target_cube at {cube_pos} on {label} ({ftype}), dist={dist:.2f}m from arm base")

                # Build ACM to ignore kitchen fixtures in planning
                build_kitchen_acm(planning_world, planner, {"target_cube"})

            # Stabilize
            robot_pos = robot.pose.p[0].cpu().numpy()
            hold = make_action(ARM_HOME, GRIPPER_OPEN, robot_pos[:3])

            def _step_fn(action):
                env.step(action)

            wait_until_stable(_step_fn, hold, robot, max_steps=100)

            initial_state = get_kitchen_state(env_unwrapped)
        elif control_mode == "pd_joint_pos" and robot_uid == "panda":
            # Existing Panda path (unchanged)
            planner = PandaArmMotionPlanningSolver(
                env,
                debug=False,
                vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                print_env_info=False,
            )
            primitives = RobotPrimitives(env, planner)
            initial_state = primitives.get_state()
        else:
            # Delta control or other modes
            initial_state = self._get_basic_state(env)

        self._sessions[session_id] = {
            "env": env,
            "planner": planner,
            "primitives": primitives,
            "env_id": env_id,
            "seed": seed,
            "steps": 0,
            "control_mode": control_mode,
            "robot_uid": robot_uid,
            "planning_world": planning_world,
            "kitchen_fixtures": kitchen_fixtures,
        }

        logger.info(f"Created session {session_id} for {env_id} (seed={seed}, robot={robot_uid}, control={control_mode})")
        return session_id, initial_state

    def _get_basic_state(self, env, session=None) -> dict:
        """Get state without primitives (for delta control mode or kitchen).

        For force-based tasks like TurnFaucet, we include the obs_extra fields
        (target_joint_axis, target_link_pos, target_angle_diff) so the agent
        knows the rotation axis, handle position, and how far to turn.
        """
        # Kitchen/tidyverse: use dedicated state function
        if session and session.get("robot_uid") == "tidyverse" and KITCHEN_AVAILABLE:
            return get_kitchen_state(env.unwrapped if hasattr(env, 'unwrapped') else env)

        env_unwrapped = env.unwrapped
        state = {
            "gripper_position": env_unwrapped.agent.tcp.pose.p.cpu().numpy().flatten().tolist(),
        }
        # Try to get object and goal info from evaluate
        try:
            info = env_unwrapped.evaluate()
            state.update({k: v.item() if hasattr(v, 'item') else v for k, v in info.items()})
        except Exception:
            pass
        # Include obs_extra fields if available (e.g. target_joint_axis for TurnFaucet)
        try:
            obs_extra = env_unwrapped._get_obs_extra({})
            for key in ["target_joint_axis", "target_link_pos", "target_angle_diff"]:
                if key in obs_extra:
                    val = obs_extra[key]
                    if hasattr(val, 'cpu'):
                        val = val.cpu().numpy()
                    state[key] = val.flatten().tolist()
        except Exception:
            pass
        return state

    def get(self, session_id: str) -> Dict[str, Any]:
        """Get session by ID."""
        if session_id not in self._sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return self._sessions[session_id]

    def delete(self, session_id: str):
        """Cleanup and remove a session."""
        if session_id in self._sessions:
            session = self._sessions.pop(session_id)
            try:
                if session["planner"]:
                    session["planner"].close()
            except Exception:
                pass
            try:
                session["env"].close()
            except Exception:
                pass
            logger.info(f"Deleted session {session_id}")

    def list_sessions(self) -> List[Dict[str, str]]:
        """List all active sessions."""
        return [
            {"session_id": sid, "env_id": s["env_id"], "steps": s["steps"]}
            for sid, s in self._sessions.items()
        ]


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="SkillClaw Resources Server",
    description="RLVR-compatible Resources Server for robot manipulation (NeMo Gym pattern)",
    version="0.1.0",
)

store = SessionStore()


@app.post("/seed_session", response_model=SeedSessionResponse)
async def seed_session(body: SeedSessionRequest):
    """Create a new ManiSkill environment session."""
    if body.env_id not in TASK_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown env_id: {body.env_id}. Available: {list(TASK_REGISTRY.keys())}",
        )

    session_id, initial_state = store.create(body.env_id, body.seed, body.robot_uid, body.control_mode)
    return SeedSessionResponse(
        session_id=session_id,
        initial_state=initial_state,
        env_id=body.env_id,
    )


@app.post("/call_tool", response_model=ToolResponse)
async def call_tool(body: ToolRequest):
    """Execute an atomic primitive and return result + updated state."""
    session = store.get(body.session_id)
    primitives = session["primitives"]

    result = execute_tool(primitives, body.tool_name, body.arguments)
    state = primitives.get_state()
    session["steps"] += 1

    return ToolResponse(tool_result=result, updated_state=state)


# Convenience endpoints — one per primitive (NeMo Gym pattern)
PRIMITIVE_NAMES = [
    "close_gripper", "open_gripper", "move_to_position",
    "rotate_gripper", "move_base", "go_home",
    "detect_object", "get_camera_image", "tilt_camera",
    "grasp_object", "align_object_to_goal", "insert_object",
]


def _make_primitive_handler(prim_name: str):
    """Create a POST handler for a single primitive."""
    async def handler(session_id: str, body: Dict[str, Any] = None):
        if body is None:
            body = {}
        session = store.get(session_id)
        result = execute_tool(session["primitives"], prim_name, body)
        state = session["primitives"].get_state()
        session["steps"] += 1
        return {"tool_result": result, "updated_state": state}
    handler.__name__ = prim_name
    return handler


for _prim in PRIMITIVE_NAMES:
    app.post(f"/{_prim}")(_make_primitive_handler(_prim))


EXECUTE_CODE_TIMEOUT = 120  # seconds (increased for force-based tasks with many env.step calls)
VIDEO_DIR = "/tmp/skillclaw_videos"


def _save_video(frames, env_id: str, session_id: str) -> str:
    """Save captured frames as mp4 video. Returns file path."""
    import os
    os.makedirs(VIDEO_DIR, exist_ok=True)
    path = f"{VIDEO_DIR}/{env_id}_{session_id}.mp4"
    try:
        import imageio
        writer = imageio.get_writer(path, fps=20)
        for frame in frames:
            if frame is not None and len(frame.shape) == 3:
                writer.append_data(frame)
        writer.close()
        logger.info(f"Video saved: {path} ({len(frames)} frames)")
        return path
    except ImportError:
        # Fallback: save as numpy frames
        np.save(path.replace(".mp4", ".npy"), np.array(frames))
        logger.info(f"Video frames saved as npy: {path} ({len(frames)} frames)")
        return path.replace(".mp4", ".npy")
    except Exception as e:
        logger.warning(f"Failed to save video: {e}")
        return None


class _ExecutionTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _ExecutionTimeout("Code execution exceeded 30 second time limit")


@app.post("/execute_code", response_model=ExecuteCodeResponse)
async def execute_code(body: ExecuteCodeRequest):
    """Execute LLM-generated Python code in a sandboxed ManiSkill session.

    The code should define a `def solve(env, planner):` function.
    The sandbox provides: env, planner, np, sapien, get_actor_obb,
    compute_grasp_info_by_obb.
    """
    session = store.get(body.session_id)
    env = session["env"].unwrapped
    planner = session["planner"]
    primitives = session["primitives"]

    def _state():
        if primitives:
            return primitives.get_state()
        return store._get_basic_state(env, session=session)

    state_before = _state()

    # Video recording setup
    frames = []
    video_path = None
    if body.record_video:
        import os
        os.makedirs(VIDEO_DIR, exist_ok=True)
        # Capture initial frame
        try:
            frame = session["env"].render()
            if frame is not None:
                frames.append(frame.cpu().numpy()[0] if hasattr(frame, 'cpu') else frame)
        except Exception:
            pass

        # Monkey-patch env.step to capture frames after each step
        original_step = session["env"].step
        def recording_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            try:
                frame = session["env"].render()
                if frame is not None:
                    frames.append(frame.cpu().numpy()[0] if hasattr(frame, 'cpu') else frame)
            except Exception:
                pass
            return result
        session["env"].step = recording_step

    import torch
    sandbox = {
        "env": env,
        "planner": planner,  # None for delta control mode; SapienPlanner for tidyverse
        "torch": torch,
        "np": np,
        "numpy": np,
        "sapien": sapien,
        "get_actor_obb": get_actor_obb,
        "compute_grasp_info_by_obb": compute_grasp_info_by_obb,
        "__builtins__": __builtins__,
    }

    # Expand sandbox for kitchen/tidyverse sessions
    if session.get("robot_uid") == "tidyverse" and KITCHEN_AVAILABLE:
        robot = env.agent.robot
        scene = env.scene.sub_scenes[0]
        sandbox.update({
            # Core objects — wrapped env for step_fn (controllers need the wrapper)
            "wrapped_env": session["env"],
            "robot": robot,
            "scene": scene,
            "planning_world": session.get("planning_world"),
            "fixtures": session.get("kitchen_fixtures"),
            # MPLib types
            "MPPose": MPPose,
            "SapienPlanner": SapienPlanner,
            "SapienPlanningWorld": SapienPlanningWorld,
            # Motion helpers
            "make_action": make_action,
            "sync_planner": sync_planner,
            "get_robot_qpos": get_robot_qpos,
            "wait_until_stable": wait_until_stable,
            "execute_trajectory": execute_trajectory,
            "actuate_gripper": actuate_gripper,
            # Grasp helpers
            "build_grasp_poses": build_grasp_poses,
            "collect_placements": collect_placements,
            "spawn_cube": spawn_cube,
            "build_kitchen_acm": build_kitchen_acm,
            # High-level primitives
            "navigate_to": navigate_to,
            "pick_up": pick_up,
            "place_object": place_object,
            # Constants
            "ARM_HOME": ARM_HOME,
            "GRIPPER_OPEN": GRIPPER_OPEN,
            "GRIPPER_CLOSED": GRIPPER_CLOSED,
            "MASK_ARM_ONLY": MASK_ARM_ONLY,
            "MASK_WHOLE_BODY": MASK_WHOLE_BODY,
            "CUBE_HALF": CUBE_HALF,
            "PLANNING_TIMEOUT": PLANNING_TIMEOUT,
            "IK_TIMEOUT": IK_TIMEOUT,
            "PRE_GRASP_HEIGHT": PRE_GRASP_HEIGHT,
            "LIFT_HEIGHT": LIFT_HEIGHT,
            # Simon's grasp functions (for manual use)
            "select_strategies": select_strategies,
            "attempt_grasp": attempt_grasp,
        })

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(EXECUTE_CODE_TIMEOUT)
    try:
        exec(body.code, sandbox)
        if "solve" not in sandbox:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            return ExecuteCodeResponse(
                success=False,
                error_type="NoSolveFunction",
                error="Code must define a `def solve(env, planner):` function. No `solve` found in executed code.",
                state_before=state_before,
                state_after=_state(),
            )

        solve_result = sandbox["solve"](env, planner)

        state_after = _state()
        task_success = state_after.get("task_success", False)

        # Detect silent failures: solve() returned -1 or error indicator
        if solve_result == -1 or solve_result is False:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            if body.record_video and frames:
                video_path = _save_video(frames, session["env_id"], body.session_id)
            return ExecuteCodeResponse(
                success=False,
                error_type="SolveReturnedFailure",
                error=f"solve() returned {solve_result}. This usually means a motion planning call failed (returned -1) and your code exited early. Check which move_to_pose call is failing and try a different approach (e.g. different grasp angle, use RRTConnect instead of screw, or break the move into smaller steps).",
                state_before=state_before,
                state_after=state_after,
                video_path=video_path,
            )

        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        # Save video if recording
        if body.record_video and frames:
            video_path = _save_video(frames, session["env_id"], body.session_id)

        # Restore original step
        if body.record_video and 'original_step' in dir():
            session["env"].step = original_step

        return ExecuteCodeResponse(
            success=True,
            task_success=task_success,
            reward=1.0 if task_success else 0.0,
            state_before=state_before if not task_success else None,
            state_after=state_after,
            video_path=video_path,
        )
    except _ExecutionTimeout as e:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        state_after = _state()
        if body.record_video and frames:
            video_path = _save_video(frames, session["env_id"], body.session_id)
        return ExecuteCodeResponse(
            success=False,
            error_type="TimeoutError",
            error=str(e),
            state_before=state_before,
            state_after=state_after,
            video_path=video_path,
        )
    except Exception as e:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        state_after = _state()
        if body.record_video and frames:
            video_path = _save_video(frames, session["env_id"], body.session_id)
        return ExecuteCodeResponse(
            success=False,
            error_type=type(e).__name__,
            error=str(e),
            error_traceback=traceback.format_exc(),
            state_before=state_before,
            state_after=state_after,
            video_path=video_path,
        )


@app.post("/verify", response_model=VerifyResponse)
async def verify(body: VerifyRequest):
    """Compute binary reward for the current session state."""
    session = store.get(body.session_id)
    primitives = session["primitives"]
    if primitives:
        state = primitives.get_state()
    else:
        state = store._get_basic_state(session["env"].unwrapped, session=session)
    success = state.get("task_success", False)

    return VerifyResponse(
        reward=1.0 if success else 0.0,
        done=success,
        info={
            "task_success": success,
            "steps": session["steps"],
            "env_id": session["env_id"],
            **{k: v for k, v in state.items() if k != "task_success"},
        },
    )


@app.post("/cleanup_session")
async def cleanup_session(body: CleanupRequest):
    """Teardown a session and free resources."""
    store.delete(body.session_id)
    return {"ok": True}


@app.get("/sessions", response_model=ListSessionsResponse)
async def list_sessions():
    """List all active sessions."""
    return ListSessionsResponse(sessions=store.list_sessions())


@app.get("/tasks")
async def list_tasks():
    """List all available tasks."""
    return {
        "tasks": [
            {"env_id": k, **{kk: vv for kk, vv in v.items() if kk != "objects"}}
            for k, v in TASK_REGISTRY.items()
        ]
    }


@app.get("/video/{filename}")
async def get_video(filename: str):
    """Serve a recorded video file."""
    from fastapi.responses import FileResponse
    import os
    path = os.path.join(VIDEO_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Video {filename} not found")
    return FileResponse(path, media_type="video/mp4")


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(store._sessions)}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="SkillClaw Resources Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
