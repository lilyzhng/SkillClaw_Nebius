---
name: dev-sc-pull-cube
description: Pull a cube to a goal position by grasping and dragging it along the surface
task: PullCube-v1
date: 2026-03-15
seed: 42
session: sess_8b568bd2
---

# Pull Cube

Grasp a cube and drag it toward a goal position along a surface.

## Strategy

1. **Open gripper** — prepare for grasping
2. **Approach from the near side** — position gripper on the side of the object closer to its current position (opposite to the pull direction)
3. **Make contact** — move close enough to touch the object (0.015m offset)
4. **Close gripper** — grasp the object
5. **Pull to goal** — drag the object along the surface to the target position

## Key Observations

- **Pull direction**: Cube needs to move from x=0.0765 to x=-0.1235 (about 0.2m to the left)
- **Approach side**: Approach from positive x side (where cube currently is) to pull in negative x direction
- **Surface constraint**: Maintain the same y and z coordinates during pull motion
- **Planner selection**: 
  - Use `RRTConnect` for large moves (approach, pull)
  - Use `screw` for precision grasp positioning
- **Goal attribute**: Use goal position directly from initial state (PullCube-v1 provides goal_position)

## Physical Principles

- The object stays on the surface throughout the motion
- Grasping provides reliable contact for dragging
- The gripper must maintain closed state during the pull to keep the grasp
- Slight overshoot is acceptable as long as object ends near goal

## Implementation Details

- Approach offset: +0.04m in x direction from object center
- Grasp offset: +0.015m in x direction from object center
- Pull target: goal x position, maintain object's y and z coordinates
- Final position: object at x=-0.1374 (goal x=-0.1235), within tolerance

## Success Criteria

Task succeeded with cube pulled from initial position (x=0.0765) to final position (x=-0.1374), successfully reaching the goal area.
