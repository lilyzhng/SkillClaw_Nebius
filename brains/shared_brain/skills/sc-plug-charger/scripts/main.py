def solve(env, planner):
    import numpy as np
    import sapien
    from mani_skill.examples.motionplanning.base_motionplanner.utils import (
        compute_grasp_info_by_obb, get_actor_obb
    )
    from transforms3d.euler import euler2quat
    
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    
    # -------------------------------------------------------------------------- #
    # Phase 1: Grasp the charger
    # -------------------------------------------------------------------------- #
    
    # Get the charger base for grasping
    charger_base_pose = env.charger_base_pose
    charger_base_size = np.array(env._base_size) * 2
    
    # Create OBB for the charger base
    import trimesh
    obb = trimesh.primitives.Box(
        extents=charger_base_size,
        transform=charger_base_pose.sp.to_transformation_matrix(),
    )
    
    # Approach from above
    approaching = np.array([0, 0, -1])
    
    # Get current TCP closing direction
    target_closing = env.agent.tcp.pose.sp.to_transformation_matrix()[:3, 1]
    
    # Compute grasp info
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    
    closing = grasp_info["closing"]
    center = grasp_info["center"]
    
    # Build grasp pose
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    
    # Add a slight angle to grasp (following reference pattern)
    grasp_angle = np.deg2rad(15)
    grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_angle, 0))
    
    # Reach (approach from 5cm above)
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1:
        return res
    
    # Grasp the charger
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        return res
    planner.close_gripper()
    
    # -------------------------------------------------------------------------- #
    # Phase 2: Align charger with receptacle
    # -------------------------------------------------------------------------- #
    
    # Pre-insertion pose: goal pose with offset, transformed to current TCP
    pre_insert_pose = (
        env.goal_pose.sp
        * sapien.Pose([-0.05, 0.0, 0.0])
        * env.charger.pose.sp.inv()
        * env.agent.tcp.pose.sp
    )
    
    # Move to pre-insertion pose (initial alignment)
    res = planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=0)
    if res == -1:
        return res
    
    # Refine alignment with multiple iterations
    res = planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=5)
    if res == -1:
        return res
    
    # -------------------------------------------------------------------------- #
    # Phase 3: Insert charger into receptacle
    # -------------------------------------------------------------------------- #
    
    # Final insertion pose: goal pose transformed to current TCP
    insert_pose = env.goal_pose.sp * env.charger.pose.sp.inv() * env.agent.tcp.pose.sp
    
    # Execute insertion
    res = planner.move_to_pose_with_screw(insert_pose)
    if res == -1:
        return res
    
    # -------------------------------------------------------------------------- #
    # Check success and cleanup
    # -------------------------------------------------------------------------- #
    result = env.evaluate()
    planner.close()
    
    return result["success"]