import numpy as np
import sapien

def solve(env, planner):
    """
    Pull a cube from its current position to a goal position by grasping and dragging.
    
    Strategy:
    1. Open gripper
    2. Approach from positive X side (object is at x=0.0765, goal at x=-0.1235)
    3. Grasp the cube
    4. Pull to goal position
    """
    env = env.unwrapped
    
    # Get object and goal positions
    obj_pos = env.obj.pose.sp.p
    
    # Get goal position - try different attribute names
    if hasattr(env, 'goal_site'):
        goal_pos = env.goal_site.pose.sp.p
    elif hasattr(env, 'goal_region'):
        goal_pos = env.goal_region.pose.sp.p
    else:
        # Use from initial state knowledge
        goal_pos = np.array([-0.1235, 0.083, 0.001])
    
    # Use current TCP orientation (gripper pointing down)
    tcp_quat = env.agent.tcp.pose.sp.q
    
    # Step 1: Open gripper
    planner.open_gripper()
    
    # Step 2: Approach from positive X side (the side closer to current position)
    approach_pos = obj_pos + np.array([0.04, 0, 0])
    approach_pose = sapien.Pose(p=approach_pos, q=tcp_quat)
    res = planner.move_to_pose_with_RRTConnect(approach_pose)
    if res == -1:
        raise RuntimeError(f"move_to_pose_with_RRTConnect failed for approach position {approach_pos}")
    
    # Step 3: Move closer for grasp
    grasp_pos = obj_pos + np.array([0.015, 0, 0])
    grasp_pose = sapien.Pose(p=grasp_pos, q=tcp_quat)
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        raise RuntimeError(f"move_to_pose_with_screw failed for grasp position {grasp_pos}")
    
    # Step 4: Close gripper to grasp
    planner.close_gripper()
    
    # Step 5: Pull to goal position (maintain Y and Z, change X)
    pull_target = np.array([goal_pos[0], obj_pos[1], obj_pos[2]])
    pull_pose = sapien.Pose(p=pull_target, q=tcp_quat)
    res = planner.move_to_pose_with_RRTConnect(pull_pose)
    if res == -1:
        raise RuntimeError(f"move_to_pose_with_RRTConnect failed for pull to {pull_target}")
    
    planner.close()
    return res
