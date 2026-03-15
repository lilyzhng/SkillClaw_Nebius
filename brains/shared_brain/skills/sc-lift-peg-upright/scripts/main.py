import numpy as np
import sapien

def solve(env, planner):
    """
    Lift a peg from lying flat to upright position.
    
    Strategy:
    1. Compute OBB-aligned grasp with side offset
    2. Approach and grasp with partial grip
    3. Lift vertically
    4. Rotate to upright orientation
    5. Lower back to table
    6. Release
    """
    env = env.unwrapped
    FINGER_LENGTH = 0.025
    
    # Get the peg object
    peg = env.peg
    
    # Compute OBB-aligned grasp
    obb = get_actor_obb(peg)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    
    # Build grasp pose with side offset (for horizontal peg)
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose([0.10, 0, 0])
    grasp_pose = grasp_pose * offset
    
    # Reach - approach pose (5cm back from grasp)
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1:
        raise RuntimeError(f"Failed to reach approach pose")
    
    # Grasp - move to grasp pose
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        raise RuntimeError(f"Failed to reach grasp pose")
    
    # Close gripper with partial grip (good for thin objects)
    planner.close_gripper(gripper_state=-0.6)
    
    # Lift - raise peg 30cm vertically
    lift_pose = sapien.Pose([0, 0, 0.30]) * grasp_pose
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1:
        raise RuntimeError(f"Failed to lift peg")
    
    # Rotate - apply rotation to make peg upright
    theta = np.pi / 10
    rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])
    
    final_pose = lift_pose * sapien.Pose(p=[0, 0, 0], q=rotation_quat)
    res = planner.move_to_pose_with_screw(final_pose)
    if res == -1:
        raise RuntimeError(f"Failed to rotate to upright")
    
    # Lower - move peg back toward table
    lower_pose = sapien.Pose([0, 0, -0.10]) * final_pose
    res = planner.move_to_pose_with_screw(lower_pose)
    if res == -1:
        raise RuntimeError(f"Failed to lower peg")
    
    # Release - open gripper
    planner.open_gripper()
    
    return res
