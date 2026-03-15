---
date: 2026-03-14
time: 00:27
---
# ManiSkill Framework Reference

## 1. Task Analysis (Do This First)

Before writing any code, analyze what the task physically requires.

### Step 1: Read the task name for physics cues

| Keyword in task name | Physical meaning | Key question |
|---|---|---|
| **pick**, **lift** | Object leaves the surface | Where does it go after lifting? |
| **place**, **stack** | Object must land at a target | What height/alignment is needed? |
| **push**, **pull**, **slide** | Object moves along a surface | What direction? From where to where? |
| **insert**, **plug** | Object fits into a receptacle | What axis? How precise? |
| **open**, **close** | Articulated joint motion | Revolute or prismatic? |

### Step 2: Identify the motion strategy

Ask these questions in order:

1. **Does the object need to leave the surface?**
   - Yes → Pick strategy (grasp → lift → move → place)
   - No → Surface strategy (contact → slide/push/pull)

2. **Does the object need to be held during motion?**
   - Yes → Grasp the object first, then move
   - No → Use gripper as a tool (push without grasping)

3. **How does the object behave physically?**
   - Flat-bottomed (cube, box) → stays where you push it, stops when contact ends
   - Round (ball, cylinder) → **rolls and keeps momentum** — a short push can travel a long distance
   - Heavy vs light → affects how much force is needed
   - Think: do I need to move the gripper the full distance, or can I give the object momentum and let physics do the rest?

4. **What direction does the object need to move?**
   - Compute the vector: `goal_position - object_position`
   - This vector determines where to position the gripper relative to the object

5. **How precise does the final position need to be?**
   - Coarse (near a region) → RRTConnect is fine
   - Precise (insertion, stacking) → screw planner with refine_steps

### Step 3: Plan gripper positioning

For any strategy, think about **where the gripper contacts the object relative to the motion direction:**

- **Pick/place:** Gripper approaches from above, grasps center
- **Push toward goal:** Gripper contacts the side of the object that faces AWAY from the goal
- **Pull toward goal:** Gripper grasps the object, then moves toward the goal
- **Insert:** Gripper holds object, aligns with receptacle axis

The general principle: **the gripper must be positioned so that the force it applies moves the object in the desired direction.**

### Step 4: Find a matching skill

After deciding your strategy, study the skill that matches your task type. Each skill has a SKILL.md (strategy description) and scripts/main.py (working code).

| Task type | Skill | Key pattern | Tasks solved |
|---|---|---|---|
| Pick / lift | `sc-pick` | OBB grasp → approach → close → lift | PickSingleYCB-v1, PickCube-v1 |
| Push / slide | `sc-push` | Close gripper → position behind → push through | PushCube-v1 |
| Stack | `sc-stack-cube` | Pick + offset alignment + place | StackCube-v1 |
| Insert (peg) | `sc-insert` | Grasp → pose algebra alignment → iterative refinement | PegInsertionSide-v1 |
| Plug (charger) | `sc-plug-charger` | Like insert, different geometry | PlugCharger-v1 |
| Place (sphere) | `sc-place-sphere` | Grasp + move to goal region | PlaceSphere-v1 |

Skills live in: `brains/shared_brain/skills/<skill-name>/`

**If no skill matches exactly**, find the closest one and adapt it. Combine skills for compound tasks (e.g., "pull" = grasp from `sc-pick` + surface motion direction reasoning). Think about what's physically different and modify accordingly.

---

## 2. Environment Setup

### Unwrap first
```python
env = env.unwrapped
```

### Discover objects from initial state keys

Don't memorize attribute names — discover them from the initial state:

| Initial state key pattern | Try env attribute |
|---|---|
| `object_position` or `cube_position` | `env.obj` (most common) |
| `cubeA_position`, `cubeB_position` | `env.cubeA`, `env.cubeB` |
| `peg_position` | `env.peg` |
| `goal_position` | `env.goal_site`, then `env.goal_region`, then `env.goal_pose` |
| Any `X_position` | `env.X` |

If AttributeError, try alternatives:
```python
try:
    obj = env.obj
except AttributeError:
    for attr in ['cube', 'peg', 'ball', 'charger', 'sphere']:
        if hasattr(env, attr):
            obj = getattr(env, attr)
            break
    else:
        raise RuntimeError(f"Cannot find object. Available: {[a for a in dir(env) if not a.startswith('_')]}")
```

### Other key attributes
- `env.agent` — robot agent with `.tcp` (tool center point) and `.robot`
- `env.agent.tcp.pose` — current gripper pose

---

## 3. Data Handling

### Safe conversion (handles both torch and numpy)
```python
def to_numpy(x):
    if hasattr(x, 'cpu'):
        return x.cpu().numpy()
    return np.array(x)

pos = to_numpy(env.obj.pose.p)[0]          # [0] removes batch dim
tcp_closing = to_numpy(env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1])
```

### Pose access
```python
env.obj.pose.p      # position [1, 3] torch tensor
env.obj.pose.q      # quaternion [1, 4] torch tensor
env.obj.pose.sp     # sapien.Pose object (numpy)
env.obj.pose.sp.p   # position, numpy array
```

---

## 4. Motion Planning API

**There are TWO different robots/planners. Check which one you have:**

### 4a. Panda (fixed base) — tabletop tasks

Used for: PickCube, StackCube, PegInsertionSide, PlugCharger, etc.

```python
# Planner is PandaArmMotionPlanningSolver (provided as `planner` in sandbox)
# Action: 7-9 dims (arm joints + gripper)

# Precision (<10cm): grasp approach, insertion, fine alignment
res = planner.move_to_pose_with_screw(target_pose)

# Large moves (any distance): lifting, navigating, repositioning
res = planner.move_to_pose_with_RRTConnect(target_pose)

# Refinement control
planner.move_to_pose_with_screw(pose, refine_steps=0)  # Fast
planner.move_to_pose_with_screw(pose, refine_steps=5)  # Precise
```

**CRITICAL: On failure (res == -1), always `raise RuntimeError(...)`. NEVER `return res`.**

```python
planner.open_gripper()
planner.close_gripper()
```

### 4b. TidyVerse (mobile base) — kitchen tasks

Used for: RoboCasaKitchen-v1

The TidyVerse is a mobile robot: 3-DOF base (x, y, yaw) + 7-DOF Panda arm + Robotiq gripper.

**Action format (`whole_body` mode):** `[arm_j1-j7, gripper, base_x, base_y, base_yaw]` = 11 dims
- Gripper: `0.0` = open, `0.81` = closed

**Planner is mplib `SapienPlanner` (NOT PandaArmMotionPlanningSolver).**
Do NOT use `move_to_pose_with_screw` or `move_to_pose_with_RRTConnect` — those are Panda-only.

**IMPORTANT: use `wrapped_env.step()` not `env.step()`.** The sandbox `env` is unwrapped, but the controller pipeline needs the wrapper to apply actions correctly.

```python
def step_fn(action):
    wrapped_env.step(action)  # NOT env.step()
```

**High-level primitives (recommended):**
```python
# Navigate base near target (direct position stepping, no planner needed)
navigate_to(env, planner, pw, target_pos, step_fn)

# Full grasp pipeline (IK-seeded, tries multiple strategies)
result = pick_up(env, planner, pw, obj_pos, step_fn)  # returns 'success'/'unreachable'

# Place held object at target
place_object(env, planner, pw, target_pos, step_fn)
```

**Mid-level helpers (for custom sequences):**
```python
# Build action tensor: [arm(7), gripper(1), base(3)]
action = make_action(arm_qpos, gripper_val, base_cmd)

# Update planner from simulation state (call before planning)
sync_planner(planner)

# Get current joint positions
qpos = get_robot_qpos(robot)  # numpy array, indices: [0:3]=base, [3:10]=arm

# Plan and execute arm motion
sync_planner(planner)
cq = get_robot_qpos(robot)
pose = MPPose(p=np.array([x, y, z]), q=[0, 1, 0, 0])  # top-down grasp orientation
result = planner.plan_pose(pose, cq, mask=MASK_ARM_ONLY, planning_time=5.0)
if result['status'] == 'Success':
    execute_trajectory(result['position'], step_fn, GRIPPER_OPEN, robot=robot)

# Open/close gripper
actuate_gripper(step_fn, env, robot, GRIPPER_CLOSED, "Closing")  # 0.81
actuate_gripper(step_fn, env, robot, GRIPPER_OPEN, "Opening")    # 0.0

# Wait for robot to settle
wait_until_stable(step_fn, make_action(ARM_HOME, GRIPPER_OPEN, base_cmd), robot)
```

**Planning masks:**
- `MASK_ARM_ONLY` — lock base, plan arm only (faster, for nearby targets)
- `MASK_WHOLE_BODY` — plan arm + base together (for distant targets)

**Constants:** `ARM_HOME`, `GRIPPER_OPEN` (0.0), `GRIPPER_CLOSED` (0.81), `CUBE_HALF` (0.02)

**Kitchen state:** Initial state includes `fixtures` (counters, drawers, stove, etc.) and `spawned_objects` (target cube positions).

**Full kitchen solve pattern:**
```python
def solve(env, planner):
    pw = planning_world

    def step_fn(action):
        wrapped_env.step(action)

    # 1. Find target from initial state
    target_pos = None
    for e in scene.entities:
        if e.name == "target_cube":
            target_pos = list(e.pose.p)  # already numpy, NOT torch
            break

    # 2. Pick up directly — the whole-body planner handles base movement
    # Do NOT call navigate_to first — let the planner figure out base+arm jointly
    result = pick_up(env, planner, pw, target_pos, step_fn)
    print(f"Result: {result}")
```

**IMPORTANT:** Do NOT call `navigate_to()` before `pick_up()`. The whole-body planner
moves the base as part of the grasp plan. Navigating first puts the base in a sub-optimal
position. Just call `pick_up()` directly and let the planner handle everything.

---

## 5. Pose Algebra

### Composition
```python
# Right-multiply = offset in LOCAL frame (gripper/object frame)
approach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])  # 5cm back in grasp frame

# Left-multiply = offset in WORLD frame
lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose  # 10cm up in world

# Inverse = relative transform
relative = target_pose * current_pose.inv()
```

### Building poses
```python
pose = sapien.Pose(p=[x, y, z], q=[w, x, y, z])
pose = sapien.Pose(q=euler2quat(roll, pitch, yaw))
new_pose = sapien.Pose(pose.p + offset, pose.q)  # Shift position, keep orientation
```

### Goal-relative alignment
```python
# "Where should the gripper be so that the held object aligns with the goal?"
aligned_pose = goal_pose * object_pose.inv() * grasp_pose
```

---

## 6. Debug Patterns & Common Pitfalls

See `brains/shared_brain/debug_patterns.md` for known failure modes, symptoms, and fixes.

---

## 7. Sharing Skills (PR Template)

After successfully learning a skill, promote it to the shared brain via PR. Follow the template at `brains/shared_brain/SKILL_PR_TEMPLATE.md`.

Steps:
1. Save to private brain: `mkdir -p brains/private_brain/dev-sc-<name>/scripts` + write files
2. Convert video to GIF: `ffmpeg -i demos/<task>_sess_<id>.mp4 -vf "fps=10,scale=320:-1" demos/<task>_sess_<id>.gif`
3. Copy to shared brain: `cp -r brains/private_brain/dev-sc-<name> brains/shared_brain/skills/sc-<name>`
4. Create branch + PR:
```bash
git checkout -b skill/sc-<name>
git add brains/shared_brain/skills/sc-<name> trajectories/<file>.json demos/<file>.gif
git commit -m "skill: sc-<name> — description"
git push -u origin skill/sc-<name>
gh pr create --title "skill: sc-<name>" --body "$(cat .github/SKILL_PR_TEMPLATE.md)"
```

Fill in the template fields (session_id, seed, result, etc.) before creating the PR.
