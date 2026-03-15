---
name: sc-pull
description: Pull an object to a goal position by grasping and dragging. Combines grasp from sc-pick with horizontal surface motion.
tasks: PullCube-v1
---

# Pull

Grasp object → drag to goal position (horizontal motion on surface).

## Strategy

1. **Grasp from above** - Use OBB-aligned grasp like sc-pick
2. **Close gripper** - Secure the object
3. **Drag horizontally** - Move to goal position while maintaining grasp height
4. **Use RRTConnect** - Pull distance is typically large (>10cm)

## Key Differences from Other Skills

- **vs sc-pick**: Pick lifts vertically, pull moves horizontally on surface
- **vs sc-push**: Push doesn't grasp (uses closed gripper as tool), pull grasps first
- **vs sc-place**: Place releases object at goal, pull keeps gripper closed

## Physical Insight

Pull tasks require:
- Secure grasp (object must not slip during drag)
- Horizontal motion at constant height
- Enough force to overcome friction (gripper must stay closed)

## Implementation Notes

- Goal position from initial state may have z ≈ 0 (floor level), but gripper must stay at grasp height
- Use `RRTConnect` for the pull motion (typically 10-30cm distance)
- Maintain grasp pose orientation during pull to avoid dropping object
