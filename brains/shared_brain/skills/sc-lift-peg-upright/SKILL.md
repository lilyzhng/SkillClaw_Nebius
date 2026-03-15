---
name: sc-lift-peg-upright
description: Grasp a horizontal peg and reorient it to stand upright on the table
---

# Lift Peg Upright

Grasp a peg lying flat on the table → lift → rotate to vertical → place upright.

## Strategy

1. **Grasp from above** — Use OBB-aligned grasp on the horizontal peg
2. **Lift straight up** — Move to safe height while maintaining grasp
3. **Rotate 90 degrees** — Reorient gripper to make peg vertical (euler rotation around y-axis)
4. **Lower to table** — Place peg at height = peg_half_length so it stands on its end
5. **Release** — Open gripper and retract

## Key Patterns

- **Reorientation in air**: Lift first, then rotate, then place - avoids collision with table
- **Use RRTConnect for large motions**: Lifting and rotating are large workspace changes
- **Height calculation**: When standing upright, peg center should be at z = peg_half_length
- **Euler angle rotation**: `euler2quat(0, π/2, 0)` rotates peg from horizontal to vertical

## Tasks Solved

- LiftPegUpright-v1 (seed=42, session=sess_be593592)
