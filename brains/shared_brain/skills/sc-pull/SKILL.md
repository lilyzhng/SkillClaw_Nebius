---
name: sc-pull
description: Pull an object to a goal position by grasping and dragging it along the surface.
task: PullCube-v1
date: 2026-03-15
---

# Pull

Grasp an object and drag it toward a goal position along a surface.

## Strategy

1. **Open gripper** — prepare for grasping
2. **Approach from the near side** — position gripper on the side of the object closer to its current position (opposite to the pull direction)
3. **Make contact** — move close enough to touch the object
4. **Close gripper** — grasp the object
5. **Pull to goal** — drag the object along the surface to the target position

## Key Observations

- **Pull vs Push distinction**: Pull requires grasping the object first, then dragging it. Push uses a closed gripper as a flat surface without grasping.
- **Approach side**: For pulling from position A to position B, approach from side A and drag toward B
- **Surface constraint**: Maintain the same z-height (surface level) during the pull motion
- **Planner selection**: 
  - Use `RRTConnect` for large moves (approach, pull)
  - Use `screw` for precision grasp positioning
- **Goal attribute**: PullCube-v1 doesn't have `goal_site`, use the goal position directly from initial state

## Physical Principles

- The object stays on the surface throughout the motion
- Grasping provides reliable contact for dragging
- The gripper must maintain closed state during the pull to keep the grasp

## Implementation Notes

- Offset gripper slightly beyond cube edge for approach (~0.04m)
- Move closer for contact (~0.015m from center)
- Keep y and z coordinates constant during pull, only change x
- The final position may slightly overshoot but task still succeeds if within tolerance
