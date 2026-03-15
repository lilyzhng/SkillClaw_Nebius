"""Low-level motion helpers: action construction, trajectory execution, settling."""
import numpy as np
import torch


ARM_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.913, 0.785])
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.81

MASK_ARM_ONLY = np.array([True] * 3 + [False] * 7 + [True] * 6)
MASK_WHOLE_BODY = np.array([False] * 3 + [False] * 7 + [True] * 6)


def get_robot_qpos(robot):
    return robot.get_qpos().cpu().numpy()[0]


def make_action(arm_qpos, gripper, base_cmd):
    act = np.concatenate([arm_qpos, [gripper], base_cmd])
    return torch.tensor(act, dtype=torch.float32).unsqueeze(0)


def wait_until_stable(step_fn, hold, robot, max_steps=300,
                      vel_thresh=1e-3, window=10):
    stable_count = 0
    for si in range(max_steps):
        step_fn(hold)
        qvel = robot.get_qvel().cpu().numpy()[0]
        if np.max(np.abs(qvel)) < vel_thresh:
            stable_count += 1
            if stable_count >= window:
                return si + 1
        else:
            stable_count = 0
    return max_steps


def execute_trajectory(traj, step_fn, gripper, lock_base=False,
                       robot=None, settle_thresh=0.01, settle_steps=100):
    base_cmd = traj[0, 0:3] if lock_base else None
    for i in range(traj.shape[0]):
        b = base_cmd if lock_base else traj[i, 0:3]
        step_fn(make_action(traj[i, 3:10], gripper, b))

    final_arm = traj[-1, 3:10]
    final_base = base_cmd if lock_base else traj[-1, 0:3]
    final_act = make_action(final_arm, gripper, final_base)

    if robot is not None:
        for _ in range(settle_steps):
            step_fn(final_act)
            qpos = get_robot_qpos(robot)
            arm_err = np.max(np.abs(qpos[3:10] - final_arm))
            base_err = np.max(np.abs(qpos[0:3] - final_base))
            if arm_err < settle_thresh and base_err < settle_thresh:
                break


def actuate_gripper(step_fn, robot, gripper_val, n_steps=30):
    qpos = get_robot_qpos(robot)
    action = make_action(qpos[3:10], gripper_val, qpos[0:3])
    for _ in range(n_steps):
        step_fn(action)


def check_joint_limits(qpos, joint_limits, joint_names, label=""):
    qi = 0
    for limits, name in zip(joint_limits, joint_names):
        if limits.ndim == 2:
            for d in range(limits.shape[0]):
                if qi >= len(qpos):
                    return
                lo, hi = limits[d, 0], limits[d, 1]
                margin = (hi - lo) * 0.02
                if qpos[qi] <= lo + margin or qpos[qi] >= hi - margin:
                    print(f"    JOINT LIMIT {label}: {name}[{d}] = {qpos[qi]:.4f}")
                qi += 1
        elif limits.ndim == 1 and limits.shape[0] >= 2:
            qi += 1
        else:
            qi += 1
