---
name: sc-pull
description: Pull an object to a goal position by grasping and dragging it along the surface.
---

# Pull

Grasp object → drag to goal position along surface.

## Strategy

1. Compute OBB-aligned grasp pose (approaching from above)
2. Approach and grasp the object
3. Drag it horizontally to the goal position (keeping it on or near the surface)

## Key Differences from Push and Pick

- **vs Push**: Pull requires grasping (closed gripper holds object), push uses closed gripper as flat surface
- **vs Pick**: Pull drags along surface, pick lifts object vertically

## Pattern

```python
# Grasp from above
grasp_pose = env.agent.build_grasp_pose(approaching, closing, obj_pos)
approach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
planner.move_to_pose_with_screw(approach_pose)
planner.move_to_pose_with_screw(grasp_pose)
planner.close_gripper()

# Pull to goal (use RRTConnect for large horizontal move)
pull_pose = sapien.Pose(p=goal_pos + [0, 0, offset], q=grasp_pose.q)
planner.move_to_pose_with_RRTConnect(pull_pose)
```

## Solved Tasks

- PullCube-v1 (seed=42)
