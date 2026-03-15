---
date: 2026-03-15
time: 14:46

name: sc-pick-ycb
description: Pick up a YCB object and move it to a goal position using OBB-aligned grasp
tasks_solved: PickSingleYCB-v1
---

# Pick YCB Object

Pick up any object from the YCB dataset and move it to a specified goal position.

## Strategy

1. **Compute OBB-aligned grasp** - Use object's oriented bounding box to determine optimal grasp direction
2. **Approach from above** - Move to pre-grasp position 5cm above the object
3. **Descend to grasp** - Move to final grasp position
4. **Close gripper** - Secure the object
5. **Lift** - Move up to clear the surface (10cm lift)
6. **Move to goal** - Navigate to goal position while maintaining grasp orientation

## Key Observations

- **Planner selection matters**: Use `move_to_pose_with_screw` for short precision moves (approach, grasp), and `move_to_pose_with_RRTConnect` for large moves (lift, navigate to goal)
- **OBB-based grasping**: Works reliably for YCB objects which have various shapes and sizes
- **Maintain orientation**: Keep the grasp orientation when moving to goal position to avoid dropping the object
- **Error handling**: Raise RuntimeError on planning failures to get actionable feedback

## Applicability

Works for any pick-and-place task where:
- Object has a well-defined bounding box
- Goal is to move object to a specific position
- Approach from above is feasible
- Object can be grasped with parallel-jaw gripper

## Variations from sc-pick

This is essentially the same as `sc-pick` but validated specifically for YCB objects with varying geometries. The core pattern (OBB grasp → approach → grasp → lift → move) is universal for pick-and-place tasks.
