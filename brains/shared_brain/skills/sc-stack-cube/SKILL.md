# Stack Cube

**Task:** StackCcube-v1

**Description:** Pick up cubeA and stack it on top of cubeB.

## Solution Strategy

1. **Compute grasp pose** — Use OBB-based grasp computation for cubeA with downward approach
2. **Angle search** — Try multiple rotation angles to find a collision-free grasp pose
3. **Reach** — Move to approach pose (5cm away from grasp)
4. **Grasp** — Move to grasp pose and close gripper
5. **Lift** — Lift cubeA 10cm up using RRTConnect (large move)
6. **Stack** — Calculate goal position on top of cubeB (height = 2 * cube_half_size), align and move using RRTConnect
7. **Release** — Open gripper to drop cubeA

## Key Techniques

- **OBB-based grasping:** Uses `compute_grasp_info_by_obb` with downward approach vector
- **Angle search:** Tests multiple rotation angles to find valid grasp pose
- **Motion planner selection:**
  - `move_to_pose_with_screw` for precision moves (reach, grasp)
  - `move_to_pose_with_RRTConnect` for large moves (lift, stack alignment)
- **Stacking height calculation:** Goal position = cubeB.pose + [0, 0, 2*cube_half_size]
- **Offset-based alignment:** Calculate offset from current cubeA position to goal, apply to lift pose

## Parameters

- `FINGER_LENGTH = 0.025` — Depth for grasp computation
- Approach offset: 5cm in -Z (local frame)
- Lift height: 10cm in +Z (world frame)
- Stack height: 2 × cube_half_size above cubeB

## Test Results

- **Seed 42:** ✓ SUCCESS (reward=1.0)
  - Initial: cubeA at [0.053, 0.267, 0.02], cubeB at [0.055, 0.123, 0.02]
  - Final: cubeA at [0.054, 0.124, 0.06], cubeB at [0.055, 0.123, 0.02]
  - CubeA successfully stacked on top of cubeB

## Notes

- Using RRTConnect for lift and stack moves is critical (screw planner fails for large distances)
- The offset-based alignment approach keeps the gripper orientation consistent throughout the motion
- Angle search ensures robust grasping across different cube orientations and initial configurations
