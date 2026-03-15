import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb

def solve(env, planner):
    """
    Grasp a horizontal peg and reorient it to stand upright on the table.
    
    Strategy:
    1. Grasp the peg from above using OBB-aligned grasp
    2. Lift straight up to safe height
    3. Rotate gripper 90 degrees to make peg vertical
    4. Lower peg to table at proper height (peg_half_length)
    5. Release and retract
    """
    env = env.unwrapped
    FINGER_LENGTH = 0.025
    
    # Initialize planner
    if planner is None:
        planner = PandaArmMotionPlanningSolver(
            env,
            debug=False,
            vis=True,
            base_pose=env.agent.robot.pose,
            visualize_target_grasp_pose=True,
            print_env_info=False,
            joint_vel_limits=0.75,
            joint_acc_limits=0.75,
        )
    
    # Get the peg object
    peg = env.peg
    peg_pos = peg.pose.sp.p
    
    # Get OBB for grasp computation
    obb = get_actor_obb(peg)
    
    # Approach from above
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    # Compute grasp info
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    
    # Build grasp pose using peg's current position
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, peg_pos)
    
    # --------------------------------------------------------------------------
    # Open gripper first
    # --------------------------------------------------------------------------
    planner.open_gripper()
    
    # --------------------------------------------------------------------------
    # Reach and Grasp
    # --------------------------------------------------------------------------
    # Approach pose (5cm above grasp)
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1:
        raise RuntimeError(f"Failed to reach approach pose at {reach_pose.p}")
    
    # Move to grasp pose
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        raise RuntimeError(f"Failed to reach grasp pose at {grasp_pose.p}")
    
    # Close gripper to grasp
    planner.close_gripper()
    
    # --------------------------------------------------------------------------
    # Lift straight up
    # --------------------------------------------------------------------------
    lift_pose = sapien.Pose([peg_pos[0], peg_pos[1], 0.15], grasp_pose.q)
    res = planner.move_to_pose_with_RRTConnect(lift_pose)
    if res == -1:
        raise RuntimeError(f"Failed to lift to {lift_pose.p}")
    
    # --------------------------------------------------------------------------
    # Rotate to upright while in air
    # --------------------------------------------------------------------------
    # Rotate 90 degrees around y-axis to make peg vertical
    upright_quat = euler2quat(0, np.pi/2, 0, 'sxyz')
    upright_pose = sapien.Pose([peg_pos[0], peg_pos[1], 0.15], upright_quat)
    
    res = planner.move_to_pose_with_RRTConnect(upright_pose)
    if res == -1:
        raise RuntimeError(f"Failed to rotate to upright")
    
    # --------------------------------------------------------------------------
    # Lower to table
    # --------------------------------------------------------------------------
    # Peg should stand on its end at height = peg_half_length
    peg_half_length = max(obb.extents) / 2
    final_z = peg_half_length + 0.01  # Small margin above table
    
    final_pose = sapien.Pose([peg_pos[0], peg_pos[1], final_z], upright_quat)
    res = planner.move_to_pose_with_RRTConnect(final_pose)
    if res == -1:
        raise RuntimeError(f"Failed to lower to table")
    
    # Open gripper to release
    planner.open_gripper()
    
    # Retract upward in world frame (gripper's local z is horizontal after rotation)
    retract_pose = sapien.Pose([peg_pos[0], peg_pos[1], final_z + 0.1], upright_quat)
    res = planner.move_to_pose_with_RRTConnect(retract_pose)
    if res == -1:
        print("Warning: Failed to retract, but task may be complete")
    
    planner.close()
    return 0
