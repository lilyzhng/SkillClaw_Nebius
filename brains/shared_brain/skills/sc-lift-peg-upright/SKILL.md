---
date: 2026-03-15
time: 14:37

name: sc-lift-peg-upright
description: Lift a peg from lying flat on a table to an upright (vertical) position
tasks: LiftPegUpright-v1
---

# Lift Peg Upright

Grasp a horizontally lying peg, lift it, rotate to vertical orientation, and place back on table.

## Strategy

1. **Grasp from side**: Compute OBB-aligned grasp with side offset (peg is lying flat)
2. **Approach**: Move to approach pose (5cm back from grasp)
3. **Grasp**: Move to grasp pose and close gripper with partial grip
4. **Lift**: Lift peg 30cm vertically
5. **Rotate**: Apply rotation to make peg upright
6. **Lower**: Lower peg back to table height
7. **Release**: Open gripper

## Key Observations

### Use screw planner throughout
Even for the 30cm vertical lift, `move_to_pose_with_screw` works better than RRTConnect for this precision manipulation task. The screw planner maintains better control.

### Partial gripper close
Use `gripper_state=-0.6` instead of full close. This provides enough grip for a thin peg without crushing it.

### Side offset for horizontal objects
When grasping a horizontally lying cylindrical object, apply a side offset (e.g., `[0.10, 0, 0]`) to the grasp pose to ensure proper contact with the gripper fingers.

### Rotation for reorientation
Apply quaternion rotation to the lifted pose to change orientation. For small rotations (π/10 around y-axis), use `[cos(θ), 0, sin(θ), 0]`.

## Pattern

```python
# Grasp with side offset
grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
offset = sapien.Pose([0.10, 0, 0])
grasp_pose = grasp_pose * offset

# Lift in world frame
lift_pose = sapien.Pose([0, 0, 0.30]) * grasp_pose

# Rotate in local frame
rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])
final_pose = lift_pose * sapien.Pose(p=[0, 0, 0], q=rotation_quat)
```

## Related Skills

- `sc-pick`: Basic OBB-aligned grasp pattern
- `sc-insert`: Pose algebra for alignment and rotation
