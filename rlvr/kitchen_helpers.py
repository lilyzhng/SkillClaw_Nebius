"""Kitchen helpers — thin wrapper around Simon's test_robocasa_grasp.py.

Imports from maniskill-tidyverse/test_robocasa_grasp.py and re-exports what
the sandbox and resources_server need for kitchen/mobile robot tasks.

Usage:
    from kitchen_helpers import setup_kitchen_planner, get_kitchen_state
"""
import sys
import os
import numpy as np

# Add Simon's directory to path so we can import from test_robocasa_grasp
_TIDYVERSE_DIR = os.path.join(os.path.dirname(__file__), '..', 'maniskill-tidyverse')
if _TIDYVERSE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_TIDYVERSE_DIR))

# Register the tidyverse robot agent
import tidyverse_agent  # noqa: F401

# Import constants and helpers from Simon's code
from test_robocasa_grasp import (
    # Constants
    ARM_HOME,
    GRIPPER_OPEN,
    GRIPPER_CLOSED,
    PRE_GRASP_HEIGHT,
    LIFT_HEIGHT,
    CUBE_HALF,
    MASK_ARM_ONLY,
    MASK_WHOLE_BODY,
    PLANNING_TIMEOUT,
    IK_TIMEOUT,
    # Motion helpers
    make_action,
    sync_planner,
    get_robot_qpos,
    wait_until_stable,
    execute_trajectory,
    actuate_gripper,
    # Placement / spawning
    collect_placements,
    spawn_cube,
    local_to_world,
    # ACM
    build_kitchen_acm,
    # Grasp strategy
    build_grasp_poses,
    select_strategies,
    attempt_grasp,
)

from mplib import Pose as MPPose
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld


def setup_kitchen_planner(env):
    """Create SapienPlanner + PlanningWorld for a kitchen environment.

    Args:
        env: The ManiSkill env (unwrapped).

    Returns:
        (planner, planning_world) tuple.
    """
    import signal

    robot = env.agent.robot
    scene = env.scene.sub_scenes[0]

    def _timeout_handler(signum, frame):
        raise TimeoutError("planner setup timeout")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(30)
    try:
        pw = SapienPlanningWorld(scene, [robot._objs[0]])
        eef = next(n for n in pw.get_planned_articulations()[0]
                   .get_pinocchio_model().get_link_names() if 'eef' in n)
        planner = SapienPlanner(pw, move_group=eef)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return planner, pw


def get_kitchen_state(env):
    """Get state dict for kitchen environments.

    Returns robot position, TCP, fixture info, and spawned object positions.
    """
    robot = env.agent.robot
    tcp_pos = env.agent.tcp.pose.p.cpu().numpy().flatten().tolist()
    robot_pos = robot.pose.p[0].cpu().numpy().tolist()

    # Get arm base position
    arm_base = None
    for link in robot.get_links():
        if link.get_name() == 'panda_link0':
            arm_base = link.pose.p[0].cpu().numpy().tolist()
            break

    state = {
        "robot_position": [round(x, 4) for x in robot_pos],
        "gripper_position": [round(x, 4) for x in tcp_pos],
        "arm_base_position": [round(x, 4) for x in arm_base] if arm_base else None,
    }

    # Fixtures summary
    try:
        fixtures = env.scene_builder.scene_data[0]['fixtures']
        fixture_info = {}
        for fname, fix in fixtures.items():
            ftype = type(fix).__name__
            if ftype in ('Floor', 'Wall'):
                continue
            fixture_info[fname] = {
                "type": ftype,
                "position": [round(x, 4) for x in fix.pos],
            }
            if hasattr(fix, 'size') and fix.size is not None:
                fixture_info[fname]["size"] = [round(x, 4) for x in fix.size]
        state["fixtures"] = fixture_info
    except Exception:
        state["fixtures"] = {}

    # Find spawned cubes/objects in the scene
    try:
        scene = env.scene.sub_scenes[0]
        spawned_objects = {}
        for entity in scene.entities:
            name = entity.name
            if name and (name.startswith("obj_") or name.startswith("target_")):
                pos = entity.pose.p.tolist()
                spawned_objects[name] = {
                    "position": [round(x, 4) for x in pos],
                }
        if spawned_objects:
            state["spawned_objects"] = spawned_objects
    except Exception:
        pass

    # evaluate() for task success
    try:
        info = env.evaluate()
        state.update({k: v.item() if hasattr(v, 'item') else v
                      for k, v in info.items()})
    except Exception:
        pass

    return state


# ============================================================================
# High-level primitives for the sandbox
# ============================================================================

def navigate_to(env, planner, pw, target_pos, step_fn=None):
    """Move the robot base near target_pos while keeping arm at home.

    Uses direct position stepping (whole_body mode = base position control).
    No planner needed for base movement — just step toward target.

    Args:
        env: unwrapped ManiSkill env
        planner: SapienPlanner
        pw: SapienPlanningWorld
        target_pos: [x, y, z] target position to navigate near
        step_fn: function to call with action tensors (default: env.step)

    Returns:
        True on success, False on failure.
    """
    if step_fn is None:
        def step_fn(action):
            env.step(action)

    robot = env.agent.robot
    target_pos = np.array(target_pos)
    robot_world = robot.pose.p[0].cpu().numpy()

    # Get arm base (panda_link0) — this is what needs to be near the target
    arm_base_world = None
    for link in robot.get_links():
        if link.get_name() == 'panda_link0':
            arm_base_world = link.pose.p[0].cpu().numpy()
            break
    if arm_base_world is None:
        arm_base_world = robot_world

    # How far does the arm base need to move?
    direction = target_pos[:2] - arm_base_world[:2]
    dist = np.linalg.norm(direction)
    stop_dist = 0.25  # arm needs to be close to reach counter height (Z~0.94m)
    if dist <= stop_dist:
        return True  # Already close enough

    # Delta in world frame = how far the arm base needs to move
    desired_arm_base = target_pos[:2] - stop_dist * direction / dist
    delta_world = desired_arm_base - arm_base_world[:2]

    yaw_world = np.arctan2(direction[1], direction[0])

    # Apply same delta to qpos — base X/Y joints are prismatic,
    # so world displacement = qpos displacement
    cq = get_robot_qpos(robot)
    nav_qpos_xy = cq[:2] + delta_world
    yaw_qpos = yaw_world

    base_target = np.array([nav_qpos_xy[0], nav_qpos_xy[1], yaw_qpos])
    base_start = cq[:3].copy()

    # Step directly to target — whole_body mode is position-controlled
    # Interpolate to avoid sudden jumps
    n_steps = max(50, int(dist / 0.02))  # ~2cm per step

    for i in range(n_steps):
        t = (i + 1) / n_steps
        base_cmd = base_start + t * (base_target - base_start)
        step_fn(make_action(ARM_HOME, GRIPPER_OPEN, base_cmd))

    # Settle
    wait_until_stable(step_fn,
                      make_action(ARM_HOME, GRIPPER_OPEN, base_target),
                      robot, max_steps=100)

    # Verify we arrived (compare arm base in world frame)
    final_arm_base = None
    for link in robot.get_links():
        if link.get_name() == 'panda_link0':
            final_arm_base = link.pose.p[0].cpu().numpy()
            break
    if final_arm_base is not None:
        final_dist = np.linalg.norm(final_arm_base[:2] - target_pos[:2])
        if final_dist > 0.8:
            print(f"  navigate_to: WARNING — arm base {final_dist:.2f}m from target (want <0.5m)")
            # Update planner with new robot state anyway
            sync_planner(planner)
            return False
        else:
            print(f"  navigate_to: arm base {final_dist:.2f}m from target — OK")

    # Update planner with new robot state
    sync_planner(planner)
    return True


def pick_up(env, planner, pw, obj_pos, step_fn=None, label="target", ftype="Counter"):
    """Pick up an object at obj_pos using Simon's attempt_grasp directly.

    This is a thin wrapper around attempt_grasp from test_robocasa_grasp.py,
    which handles IK (arm-only → whole-body fallback), planning, and execution.

    Args:
        env: unwrapped ManiSkill env (or wrapped — attempt_grasp uses step_fn)
        planner: SapienPlanner
        pw: SapienPlanningWorld
        obj_pos: [x, y, z] position of object to pick up
        step_fn: function to call with action tensors
        label: placement label (affects grasp strategy selection)
        ftype: fixture type string (affects grasp strategy selection)

    Returns:
        'success', 'partial', or 'unreachable'
    """
    if step_fn is None:
        def step_fn(action):
            env.step(action)

    robot = env.agent.robot
    obj_pos = np.array(obj_pos)

    return attempt_grasp(
        cube_idx=0,
        cube_name="target_cube",
        cube_pos=obj_pos,
        label=label,
        ftype=ftype,
        robot=robot,
        planner=planner,
        pw=pw,
        step_fn=step_fn,
        env=env,
        total=1,
    )


def place_object(env, planner, pw, target_pos, step_fn=None):
    """Move to target_pos and release the held object.

    Args:
        env: unwrapped ManiSkill env
        planner: SapienPlanner
        pw: SapienPlanningWorld
        target_pos: [x, y, z] where to place the object
        step_fn: function to call with action tensors

    Returns:
        True on success, False on failure.
    """
    import signal

    if step_fn is None:
        def step_fn(action):
            env.step(action)

    robot = env.agent.robot
    target_pos = np.array(target_pos)

    # Plan to above the target
    place_pose = MPPose(p=target_pos + [0, 0, 0.05], q=[0, 1, 0, 0])
    sync_planner(planner)
    cq = get_robot_qpos(robot)

    signal.alarm(PLANNING_TIMEOUT)
    try:
        result = planner.plan_pose(place_pose, cq, mask=MASK_WHOLE_BODY,
                                    planning_time=5.0)
    except TimeoutError:
        return False
    finally:
        signal.alarm(0)

    if result['status'] != 'Success':
        return False

    execute_trajectory(result['position'], step_fn, GRIPPER_CLOSED,
                       robot=robot)

    # Open gripper to release
    actuate_gripper(step_fn, env, robot, GRIPPER_OPEN, "Release")

    # Retract upward
    retract_pose = MPPose(p=target_pos + [0, 0, LIFT_HEIGHT], q=[0, 1, 0, 0])
    sync_planner(planner)
    cq = get_robot_qpos(robot)
    signal.alarm(PLANNING_TIMEOUT)
    try:
        r_ret = planner.plan_pose(retract_pose, cq, mask=MASK_WHOLE_BODY,
                                   planning_time=5.0)
    except TimeoutError:
        return True  # Object was placed, retract failed — still success
    finally:
        signal.alarm(0)

    if r_ret['status'] == 'Success':
        execute_trajectory(r_ret['position'], step_fn, GRIPPER_OPEN,
                           robot=robot)
    return True
