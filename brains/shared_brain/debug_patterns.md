# Debug Patterns

Lessons learned from failed attempts. Read this BEFORE writing code to avoid known pitfalls.

---

## Motion Planner: screw vs RRTConnect

**Symptom:** Code executes without error, but robot doesn't move. Server log shows `screw plan failed.`

**Root cause:** `move_to_pose_with_screw` fails silently (returns -1, no exception) when the target pose is far from the current pose. The screw planner is designed for short, precise moves — it can't plan paths that require large joint angle changes.

**Fix:** Use the right planner for the right move:
- `move_to_pose_with_screw(pose)` — precision moves only (final approach to grasp, insertion alignment, small adjustments). Range: <10cm.
- `move_to_pose_with_RRTConnect(pose)` — large moves (lifting, navigating to a different part of the workspace, moving to goal). Range: any distance.

**Pattern:**
```python
# Large move — use RRTConnect
planner.move_to_pose_with_RRTConnect(lift_pose)

# Precision move — use screw
planner.move_to_pose_with_screw(grasp_pose)
```

**Source:** Experiment #6 smoke test (PickSingleYCB-v1, seed=42). Attempts 1-2 failed because lift used screw. Attempt 3 switched to RRTConnect → success.

---

## Silent Failure: state_before == state_after

**Symptom:** Code runs without exception, task not completed, and state_before is identical to state_after — robot didn't move at all.

**Root cause:** Every `move_to_pose` call returned -1 (path planning failed), and the code silently returned instead of raising an error. You don't know WHICH step failed.

**Fix:** NEVER silently return on failure. Always raise an exception with context:
```python
res = planner.move_to_pose_with_screw(target_pose)
if res == -1:
    raise RuntimeError(f"move_to_pose_with_screw FAILED for target position {target_pose.p}. "
                        f"Current TCP at {env.agent.tcp.pose.p}. Try: different approach angle, "
                        f"use RRTConnect instead, or break into smaller intermediate moves.")
```

**Why this matters:** Without the exception, the retry prompt just says "task not completed" — you can't diagnose whether step 1 or step 5 failed. With the exception, you see exactly which pose was unreachable and can fix that specific step.

**Source:** PokeCube-v1 (seed=42). Agent wrote correct strategy (grasp peg, push cube) but first move_to_pose_with_screw returned -1. Code did `return res` → robot never moved → 4 attempts all identical failure.

---

## Grasping objects near table surface

**Symptom:** move_to_pose fails when trying to reach an object lying flat on the table (z ≈ 0.02-0.03).

**Root cause:** The grasp pose computed by OBB may be too close to the table surface, causing collision with the table in the planner's collision checking.

**Fix:** Try multiple approaches:
1. Use angle search (try different rotation angles around z-axis for the grasp)
2. Approach from the side instead of from above (change `approaching` vector)
3. Use a small offset to lift the grasp pose slightly above the computed position
4. Try RRTConnect instead of screw for the approach (more robust path finding)


## Common Pitfalls
### Silent Failures
- **Motion planning returns -1 on failure** — Always `raise RuntimeError`, NEVER `return res`
- **Pose multiplication order matters** — `A * B ≠ B * A` (right = local frame, left = world frame)
- **Screw planner fails silently for large moves** — Use RRTConnect for >10cm moves

### Attribute Access
- **Check initial state keys first** — `object_position` → `env.obj`, `cubeA_position` → `env.cubeA`
- **Most single-object tasks use `env.obj`** — try this first before task-specific names
- **Missing `.unwrapped`** — Always `env = env.unwrapped` before accessing task attributes
- **Missing `[0]` indexing** — Batched arrays need `[0]` after `.cpu().numpy()`

### Tensor/Numpy Mixing
- **Not all data is torch** — Some attributes are already numpy. Use `to_numpy()` helper (see above)
- **`.cpu()` on numpy = AttributeError** — Check type first, or use safe `to_numpy()` pattern
- **Shape mismatches** — Batch dimension `[1, ...]` often needs `[0]` indexing

### Motion Planning
- **Collision with held objects** — Gripper must be closed before grasping
- **Unreachable poses** — Use angle search or `dry_run=True` validation
- **Missing cleanup** — Always call `planner.close()` to release resources
- **Speed limits** — Default velocity/acceleration may be too fast for precision tasks

### Coordinate Frame Confusion
- **Local vs world offsets** — Left-multiply for world, right-multiply for local
- **Inverse order** — `target * current.inv()` gives relative transform from current to target
- **Goal-relative poses** — Pattern: `goal_pose * object_pose.inv() * grasp_pose`