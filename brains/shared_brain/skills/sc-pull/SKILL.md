---
name: sc-pull
description: Pull an object to a goal by grasping it and dragging it horizontally along the surface.
---

# Pull

Grasp object from above → drag horizontally to goal while maintaining low height.

Unlike push (which uses closed gripper without grasping), pull requires:
1. Grasping the object securely
2. Moving gripper horizontally while keeping low to surface
3. Object follows along by being held

## Strategy
1. Compute OBB-based grasp from above
2. Approach and grasp the object
3. Move gripper to goal XY position while maintaining grasp height
4. Object drags along the surface to goal

## Key Differences
- **Push**: No grasp, gripper behind object, push through
- **Pull**: Grasp first, then drag horizontally
- **Pick**: Grasp then lift off surface
- **Pull**: Grasp then drag along surface (no lift)
