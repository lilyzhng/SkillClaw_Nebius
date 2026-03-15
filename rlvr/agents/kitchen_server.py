"""
SkillClaw Kitchen Server — FastAPI server for RoboCasa kitchen environments.

Mirrors resources_server.py but uses the TidyVerse robot (Panda arm + Robotiq
gripper + 3-DOF mobile base) in RoboCasa kitchen layouts.

Usage:
    python rlvr/agents/kitchen_server.py --port 8200
"""

import argparse
import json
import logging
import os
import signal
import traceback
from typing import Any, Dict, List, Optional
from uuid import uuid4

import gymnasium as gym
import mplib
import numpy as np
import sapien

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ManiSkill imports
import mani_skill.envs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kitchen_server")

# ============================================================================
# Request/Response Models (same shape as resources_server.py)
# ============================================================================

class SeedSessionRequest(BaseModel):
    env_id: str = "RoboCasaKitchen-v1"
    seed: int = 0
    layout: Optional[int] = None
    style: Optional[int] = None


class SeedSessionResponse(BaseModel):
    session_id: str
    initial_state: Dict[str, Any]
    env_id: str


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


class CleanupRequest(BaseModel):
    session_id: str


class ListSessionsResponse(BaseModel):
    sessions: List[Dict[str, Any]]


# ============================================================================
# Session Store
# ============================================================================

EXECUTE_CODE_TIMEOUT = 60  # kitchen planning is slower
VIDEO_DIR = "/tmp/skillclaw_kitchen_videos"


def _save_video(frames, env_id: str, session_id: str) -> str:
    """Save captured frames as mp4 video. Returns file path."""
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
        np.save(path.replace(".mp4", ".npy"), np.array(frames))
        logger.info(f"Video frames saved as npy: {path} ({len(frames)} frames)")
        return path.replace(".mp4", ".npy")
    except Exception as e:
        logger.warning(f"Failed to save video: {e}")
        return None


class _ExecutionTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _ExecutionTimeout(f"Code execution exceeded {EXECUTE_CODE_TIMEOUT} second time limit")


class KitchenSessionStore:
    """Manages RoboCasa kitchen environment sessions."""

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create(self, env_id: str, seed: int, layout: int = None, style: int = None) -> tuple:
        """Create a new kitchen session with TidyVerse robot."""
        session_id = f"ksess_{uuid4().hex[:8]}"

        # Import tidyverse agent to register the robot
        from rlvr.agents.tidyverse import tidyverse_agent  # noqa: F401

        # Import kitchen utilities
        from rlvr.agents.tidyverse.planning_utils import (
            SapienPlanningWorld, build_kitchen_acm, sync_planner,
        )
        from rlvr.agents.tidyverse.motion_utils import (
            wait_until_stable, make_action, get_robot_qpos,
        )
        from rlvr.agents.tidyverse.placement_utils import collect_placements

        # Build env kwargs
        env_kwargs = dict(
            robot_uids="tidyverse",
            control_mode="whole_body",
            render_mode="rgb_array",
            num_envs=1,
        )
        if layout is not None:
            env_kwargs["layout"] = layout
        if style is not None:
            env_kwargs["style"] = style

        env = gym.make(env_id, **env_kwargs)
        env.reset(seed=seed)

        env_unwrapped = env.unwrapped

        # Wait for physics to stabilize
        robot = env_unwrapped.agent.robot
        qpos = get_robot_qpos(robot)
        hold_action = make_action(qpos[3:10], 0.0, qpos[0:3])
        wait_until_stable(env.step, hold_action, robot, max_steps=200)

        # Set up mplib planning world + planner
        from mplib.sapien_utils import SapienPlanner as _SapienPlanner

        pw = SapienPlanningWorld(env_unwrapped.scene.sub_scenes[0], [robot._objs[0]])

        # Find the eef link name from the planning world articulation
        art = pw.get_planned_articulations()[0]
        eef_name = next(
            n for n in art.get_pinocchio_model().get_link_names()
            if 'eef' in n
        )

        planner = _SapienPlanner(pw, move_group=eef_name)

        # Sync planner to current robot state
        planner.update_from_simulation()

        # Enumerate fixtures and placements
        fixtures = {}
        placements = []
        try:
            sb = env_unwrapped.scene_builder
            if hasattr(sb, 'scene_data') and sb.scene_data:
                fixtures = sb.scene_data[0].get('fixtures', {})
                placements = collect_placements(fixtures)
        except Exception as e:
            logger.warning(f"Could not enumerate fixtures: {e}")

        # Build ACM (relaxed mode for initial setup)
        cube_names = []  # No cubes spawned yet by default
        try:
            build_kitchen_acm(pw, planner, cube_names, mode='relaxed')
        except Exception as e:
            logger.warning(f"ACM build failed: {e}")

        # Get task instruction
        task_instruction = ""
        try:
            if hasattr(env_unwrapped, 'task_description'):
                task_instruction = env_unwrapped.task_description
            elif hasattr(env_unwrapped, '_task_description'):
                task_instruction = env_unwrapped._task_description
        except Exception:
            pass

        # Build initial state
        initial_state = self._get_state(env_unwrapped, planner, pw, fixtures, placements, task_instruction)

        self._sessions[session_id] = {
            "env": env,
            "env_unwrapped": env_unwrapped,
            "planner": planner,
            "pw": pw,
            "fixtures": fixtures,
            "placements": placements,
            "env_id": env_id,
            "seed": seed,
            "steps": 0,
            "task_instruction": task_instruction,
        }

        logger.info(f"Created kitchen session {session_id} for {env_id} (seed={seed})")
        return session_id, initial_state

    def _get_state(self, env_unwrapped, planner, pw, fixtures, placements, task_instruction) -> dict:
        """Build state dict for kitchen environment."""
        robot = env_unwrapped.agent.robot
        agent = env_unwrapped.agent
        qpos = robot.get_qpos().cpu().numpy()[0]

        state = {
            "robot_base_pose": qpos[:3].tolist(),  # [x, y, yaw]
            "tcp_position": agent.tcp_pos[0].cpu().numpy().tolist(),
            "tcp_orientation": agent.tcp_pose.q[0].cpu().numpy().tolist(),
            "fixtures": [],
            "objects": [],
            "task_instruction": task_instruction,
        }

        # Fixtures info
        for fname, fix in fixtures.items():
            ftype = type(fix).__name__
            try:
                pos = list(fix.pos) if hasattr(fix, 'pos') else []
            except Exception:
                pos = []
            state["fixtures"].append({
                "name": fname,
                "type": ftype,
                "position": pos,
            })

        # Placements info
        state["placements"] = [
            {"label": label, "position": pos.tolist(), "fixture_type": ftype}
            for label, pos, ftype, _ in placements
        ]

        # Objects (actors that aren't robot or fixture links)
        try:
            for actor in env_unwrapped.scene.sub_scenes[0].entities:
                name = actor.name
                if name and not name.startswith("tidyverse") and "ground" not in name.lower():
                    try:
                        p = actor.pose.p
                        if hasattr(p, 'cpu'):
                            p = p.cpu().numpy()
                        if p.ndim > 1:
                            p = p[0]
                        # Check if grasped
                        is_grasped = False
                        try:
                            is_grasped = bool(agent.is_grasping(actor).item())
                        except Exception:
                            pass
                        state["objects"].append({
                            "name": name,
                            "position": p.tolist(),
                            "is_grasped": is_grasped,
                        })
                    except Exception:
                        pass
        except Exception:
            pass

        # Success flags
        try:
            info = env_unwrapped.evaluate()
            state["eval_info"] = {
                k: v.item() if hasattr(v, 'item') else v
                for k, v in info.items()
            }
        except Exception:
            pass

        return state

    def get(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self._sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return self._sessions[session_id]

    def delete(self, session_id: str):
        if session_id in self._sessions:
            session = self._sessions.pop(session_id)
            try:
                session["env"].close()
            except Exception:
                pass
            logger.info(f"Deleted session {session_id}")

    def list_sessions(self) -> List[Dict[str, str]]:
        return [
            {"session_id": sid, "env_id": s["env_id"], "steps": s["steps"]}
            for sid, s in self._sessions.items()
        ]


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="SkillClaw Kitchen Server",
    description="FastAPI server for RoboCasa kitchen environments with TidyVerse robot",
    version="0.1.0",
)

store = KitchenSessionStore()


@app.post("/seed_session", response_model=SeedSessionResponse)
async def seed_session(body: SeedSessionRequest):
    """Create a new RoboCasa kitchen session."""
    session_id, initial_state = store.create(
        body.env_id, body.seed, body.layout, body.style
    )
    return SeedSessionResponse(
        session_id=session_id,
        initial_state=initial_state,
        env_id=body.env_id,
    )


@app.post("/execute_code", response_model=ExecuteCodeResponse)
async def execute_code(body: ExecuteCodeRequest):
    """Execute LLM-generated Python code in kitchen sandbox.

    The code should define a `def solve(env, planner, pw):` function.
    The sandbox provides: env, planner, pw, np, sapien, MPPose,
    execute_trajectory, actuate_gripper, make_action, wait_until_stable,
    select_grasps, get_robot_qpos.
    """
    session = store.get(body.session_id)
    env = session["env"]
    env_unwrapped = session["env_unwrapped"]
    planner = session["planner"]
    pw = session["pw"]

    state_before = store._get_state(
        env_unwrapped, planner, pw,
        session["fixtures"], session["placements"],
        session["task_instruction"],
    )

    # Video recording setup
    frames = []
    video_path = None
    original_step = None
    if body.record_video:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        try:
            frame = env.render()
            if frame is not None:
                frames.append(frame.cpu().numpy()[0] if hasattr(frame, 'cpu') else frame)
        except Exception:
            pass

        original_step = env.step
        def recording_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(frame.cpu().numpy()[0] if hasattr(frame, 'cpu') else frame)
            except Exception:
                pass
            return result
        env.step = recording_step

    # Import tidyverse utilities for sandbox
    from rlvr.agents.tidyverse import motion_utils, grasp_strategies
    from rlvr.agents.tidyverse.planning_utils import sync_planner

    sandbox = {
        "env": env_unwrapped,
        "planner": planner,
        "pw": pw,
        "np": np,
        "numpy": np,
        "sapien": sapien,
        "MPPose": mplib.Pose,
        "execute_trajectory": motion_utils.execute_trajectory,
        "actuate_gripper": motion_utils.actuate_gripper,
        "make_action": motion_utils.make_action,
        "wait_until_stable": motion_utils.wait_until_stable,
        "select_grasps": grasp_strategies.select_grasps,
        "get_robot_qpos": motion_utils.get_robot_qpos,
        "sync_planner": sync_planner,
        "__builtins__": __builtins__,
    }

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
                error="Code must define a `def solve(env, planner, pw):` function.",
                state_before=state_before,
                state_after=store._get_state(
                    env_unwrapped, planner, pw,
                    session["fixtures"], session["placements"],
                    session["task_instruction"],
                ),
            )

        solve_result = sandbox["solve"](env_unwrapped, planner, pw)

        state_after = store._get_state(
            env_unwrapped, planner, pw,
            session["fixtures"], session["placements"],
            session["task_instruction"],
        )
        task_success = state_after.get("eval_info", {}).get("success", False)

        if solve_result == -1 or solve_result is False:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            if body.record_video and frames:
                video_path = _save_video(frames, session["env_id"], body.session_id)
            return ExecuteCodeResponse(
                success=False,
                error_type="SolveReturnedFailure",
                error=f"solve() returned {solve_result}. A motion planning call likely failed.",
                state_before=state_before,
                state_after=state_after,
                video_path=video_path,
            )

        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        if body.record_video and frames:
            video_path = _save_video(frames, session["env_id"], body.session_id)
        if body.record_video and original_step is not None:
            env.step = original_step

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
        state_after = store._get_state(
            env_unwrapped, planner, pw,
            session["fixtures"], session["placements"],
            session["task_instruction"],
        )
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
        state_after = store._get_state(
            env_unwrapped, planner, pw,
            session["fixtures"], session["placements"],
            session["task_instruction"],
        )
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


@app.post("/cleanup_session")
async def cleanup_session(body: CleanupRequest):
    """Teardown a session and free resources."""
    store.delete(body.session_id)
    return {"ok": True}


@app.get("/sessions", response_model=ListSessionsResponse)
async def list_sessions():
    """List all active sessions."""
    return ListSessionsResponse(sessions=store.list_sessions())


@app.get("/video/{filename}")
async def get_video(filename: str):
    """Serve a recorded video file."""
    from fastapi.responses import FileResponse
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

    parser = argparse.ArgumentParser(description="SkillClaw Kitchen Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
