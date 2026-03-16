---
name: sc-pull
description: Pull an object to a goal position by grasping and dragging it
task: PullCube-v1
date: 2026-03-15
---

# Pull

Grasp object → drag to goal position while holding.

## Strategy

1. **Compute OBB-aligned grasp** from above (reuse pattern from sc-pick)
2. **Approach** from 5cm above grasp pose
3. **Move to grasp pose** and close gripper
4. **Pull to goal** using RRTConnect (handles large distances ~20cm)

## Key Observations

- **Pull = grasp + transport**: Unlike push (no grasp), pull requires holding the object throughout motion
- **Use RRTConnect for pull motion**: The pull distance can be large (>10cm), so screw planner would fail
- **Goal position from initial state**: PullCube doesn't expose `goal_site` attribute, but initial state provides `goal_position`

## Differences from Related Skills

- **vs sc-pick**: Pull moves to a specific ground-level goal, not lifting upward
- **vs sc-push**: Pull requires grasping first, push uses closed gripper as flat surface
- **vs sc-place**: Pull keeps object at surface level, place lifts then lowers

## Physical Intuition

Pulling is like dragging a heavy box across the floor:
- You must grip it first (can't just push)
- You move your whole body while maintaining grip
- The object slides along the surface as you move
