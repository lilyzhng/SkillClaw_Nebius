---
name: dev-sc-pull-cube
description: Pull a cube to a goal position by grasping and dragging it along the surface
task: PullCube-v1
seed: 42
session: sess_18813233
date: 2026-03-15
---

# Pull Cube

Pull a cube from its initial position to a goal position by grasping and dragging it along the surface.

## Strategy

1. **Open gripper** — prepare for grasping
2. **Approach from near side** — move to a position on the side of the cube closer to its current position (opposite to pull direction)
   - For pulling in negative X direction, approach from positive X side
   - Offset: +0.04m from cube center
3. **Move to grasp position** — get closer to make contact
   - Offset: +0.015m from cube center
   - Use screw planner for precision
4. **Close gripper** — grasp the cube
5. **Pull to goal** — drag cube to goal position
   - Maintain same Y and Z coordinates (surface level)
   - Change only X coordinate to reach goal
   - Use RRTConnect for large motion

## Key Observations

- **Approach side matters**: Must approach from the side opposite to the pull direction to enable proper dragging
- **Surface constraint**: Keep Y and Z coordinates constant during pull (object stays on surface)
- **Slight overshoot is OK**: Object ended at x=-0.1374 vs goal x=-0.1235, still within success tolerance
- **Planner selection**:
  - RRTConnect for approach and pull (large motions ~20cm)
  - Screw for final grasp positioning (precision)

## Physical Principles

- Grasping provides reliable contact for dragging
- Object remains on surface throughout motion (no lifting)
- Gripper must stay closed during pull to maintain grasp

## Results

- Initial cube position: [0.0765, 0.083, 0.02]
- Goal position: [-0.1235, 0.083, 0.001]  
- Final cube position: [-0.1374, 0.0834, 0.021]
- Pull distance: ~21cm in negative X direction
- Task success: True, reward: 1.0
