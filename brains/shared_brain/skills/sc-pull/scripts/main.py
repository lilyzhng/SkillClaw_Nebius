import numpy as np
import sapien

from mani_skill.envs.tasks import PullCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver

def solve(env: PullCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    env = env.unwrapped
    
    # Get object position
    obj_pos = env.obj.pose.sp.p
    
    # Goal position (PullCubeEnv doesn't have goal_site attribute)
    # Use goal position from environment state or initial observation
    # For generality, we can try to access goal attributes
    if hasattr(env, 'goal_site'):
        goal_pos = env.goal_site.pose.sp.p
    elif hasattr(env, 'goal_region'):
        goal_pos = env.goal_region.pose.sp.p
    else:
        # Fallback: reconstruct from initial state or use a heuristic
        # For PullCube, goal is typically in negative x direction
        goal_pos = np.array([-0.1235, obj_pos[1], 0.001])
    
    # Use current TCP orientation (gripper pointing down)
    tcp_quat = env.agent.tcp.pose.sp.q
    
    # -------------------------------------------------------------------------- #
    # Open gripper and approach from positive x side
    # -------------------------------------------------------------------------- #
    planner.open_gripper()
    
    # Approach from the side closer to object (positive x for negative pull)
    approach_pos = obj_pos + np.array([0.04, 0, 0])
    approach_pose = sapien.Pose(p=approach_pos, q=tcp_quat)
    res = planner.move_to_pose_with_RRTConnect(approach_pose)
    if res == -1:
        raise RuntimeError(f"move_to_pose_with_RRTConnect failed for approach position {approach_pos}")
    
    # -------------------------------------------------------------------------- #
    # Move to grasp position and close gripper
    # -------------------------------------------------------------------------- #
    grasp_pos = obj_pos + np.array([0.015, 0, 0])
    grasp_pose = sapien.Pose(p=grasp_pos, q=tcp_quat)
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        raise RuntimeError(f"move_to_pose_with_screw failed for grasp position {grasp_pos}")
    
    planner.close_gripper()
    
    # -------------------------------------------------------------------------- #
    # Pull to goal position
    # -------------------------------------------------------------------------- #
    # Maintain same y and z, only change x to goal
    pull_target = np.array([goal_pos[0], obj_pos[1], obj_pos[2]])
    pull_pose = sapien.Pose(p=pull_target, q=tcp_quat)
    res = planner.move_to_pose_with_RRTConnect(pull_pose)
    if res == -1:
        raise RuntimeError(f"move_to_pose_with_RRTConnect failed for pull to {pull_target}")
    
    planner.close()
    return res
