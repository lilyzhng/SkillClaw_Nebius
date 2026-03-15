import numpy as np
import sapien
from transforms3d.euler import euler2quat
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env, planner):
    """
    Stack cubeA on top of cubeB.
    
    Steps:
    1. Compute grasp pose for cubeA using OBB
    2. Reach approach pose
    3. Grasp cubeA
    4. Lift cubeA
    5. Move to position above cubeB
    6. Release cubeA
    """
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    
    # -------------------------------------------------------------------------- #
    # Compute grasp pose for cubeA
    # -------------------------------------------------------------------------- #
    obb = get_actor_obb(env.cubeA)
    
    approaching = np.array([0, 0, -1])  # Approach from above
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # Search for a valid grasp pose by trying different angles
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")

    # -------------------------------------------------------------------------- #
    # Reach approach pose (5cm away from grasp)
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp cubeA
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift cubeA (10cm up in world frame)
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    planner.move_to_pose_with_RRTConnect(lift_pose)  # Use RRTConnect for large moves

    # -------------------------------------------------------------------------- #
    # Stack: Move to position above cubeB
    # -------------------------------------------------------------------------- #
    # Goal position is on top of cubeB (height = 2 * cube_half_size)
    goal_pose = env.cubeB.pose * sapien.Pose([0, 0, (env.cube_half_size[2] * 2).item()])
    
    # Calculate offset from current cubeA position to goal position
    offset = (goal_pose.p - env.cubeA.pose.p).cpu().numpy()[0]
    
    # Apply offset to lift_pose to get aligned position
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_RRTConnect(align_pose)  # Use RRTConnect for large move

    # -------------------------------------------------------------------------- #
    # Release cubeA
    # -------------------------------------------------------------------------- #
    res = planner.open_gripper()
    planner.close()
    return res
