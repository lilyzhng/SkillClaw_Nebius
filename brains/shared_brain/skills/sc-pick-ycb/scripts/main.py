import numpy as np
import sapien

def solve(env, planner):
    """
    PickSingleYCB-v1: Pick up a YCB object and move it to a goal position.
    
    Strategy:
    1. Compute OBB-aligned grasp pose from above
    2. Approach to pre-grasp position (5cm above grasp)
    3. Move to grasp position
    4. Close gripper
    5. Lift object 
    6. Move to goal position
    """
    env = env.unwrapped
    FINGER_LENGTH = 0.025
    
    # Get object - try common attribute names
    if hasattr(env, 'obj'):
        obj = env.obj
    elif hasattr(env, 'cube'):
        obj = env.cube
    else:
        raise RuntimeError(f"Cannot find object. Available attributes: {[a for a in dir(env) if not a.startswith('_')]}")
    
    # Get goal position
    if hasattr(env, 'goal_site'):
        goal_pos = env.goal_site.pose.sp.p
    elif hasattr(env, 'goal_region'):
        goal_pos = env.goal_region.pose.sp.p
    else:
        raise RuntimeError("Cannot find goal position")
    
    # Compute OBB-aligned grasp
    obb = get_actor_obb(obj)
    approaching = np.array([0, 0, -1])  # Approach from above
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    
    # Build grasp pose
    grasp_pose = env.agent.build_grasp_pose(
        approaching, 
        grasp_info["closing"], 
        obj.pose.sp.p
    )
    
    # Step 1: Move to pre-grasp position (5cm above grasp)
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1:
        raise RuntimeError(f"Failed to reach pre-grasp pose at {reach_pose.p}. Try RRTConnect or different approach angle.")
    
    # Step 2: Move to grasp position
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        raise RuntimeError(f"Failed to reach grasp pose at {grasp_pose.p}. Object may be too close to surface.")
    
    # Step 3: Close gripper to grasp
    planner.close_gripper()
    
    # Step 4: Lift object up (10cm lift using RRTConnect for robustness)
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose  # World frame offset
    res = planner.move_to_pose_with_RRTConnect(lift_pose)
    if res == -1:
        raise RuntimeError(f"Failed to lift object to {lift_pose.p}. Check collision or joint limits.")
    
    # Step 5: Move to goal position (keep the grasp orientation)
    goal_pose = sapien.Pose(goal_pos, grasp_pose.q)
    res = planner.move_to_pose_with_RRTConnect(goal_pose)
    if res == -1:
        raise RuntimeError(f"Failed to move to goal pose at {goal_pose.p}. Path may be obstructed.")
    
    return res
