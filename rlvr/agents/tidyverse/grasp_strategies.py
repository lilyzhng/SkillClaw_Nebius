"""Grasp pose generation for objects and handles."""
import numpy as np
from transforms3d.euler import euler2quat
from scipy.spatial.transform import Rotation as R


DROP_HEIGHT = 0.05


def build_object_grasps(obj_pos, arm_base, obj_yaw=0.0):
    """Grasp poses for free objects (blocks, cups, etc.): Angled45 + Top-Down.

    Generates 4 yaw rotations (0, 90, 180, 270 deg) relative to the
    object's own yaw so gripper fingers align with object faces.
    """
    grasps = []
    for i, yaw_offset in enumerate([0, np.pi / 2, np.pi, -np.pi / 2]):
        yaw = obj_yaw + yaw_offset
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        deg = int(np.degrees(yaw_offset))
        grasps.append((
            f'Angled45-{deg}',
            obj_pos + np.array([-0.02 * cos_y, -0.02 * sin_y, 0.02]),
            np.array(euler2quat(0, 3 * np.pi / 4, yaw)),
        ))
    for i, yaw_offset in enumerate([0, np.pi / 2, np.pi, -np.pi / 2]):
        yaw = obj_yaw + yaw_offset
        deg = int(np.degrees(yaw_offset))
        grasps.append((
            f'Top-Down-{deg}',
            obj_pos.copy(),
            np.array(euler2quat(0, np.pi, yaw)),
        ))
    return grasps


def build_handle_grasps(handle_pos, arm_base):
    """Grasp poses for handles: Front + Front-Vertical (rotated 90 deg)."""
    yaw = np.arctan2(handle_pos[1] - arm_base[1],
                     handle_pos[0] - arm_base[0])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    front_rot = R.from_euler('yz', [np.pi / 2, yaw])
    front_vert_rot = front_rot * R.from_euler('z', np.pi / 2)
    return [
        ('Front',
         handle_pos + np.array([-0.06 * cos_y, -0.06 * sin_y, 0.08]),
         np.array(front_rot.as_quat()[[3, 0, 1, 2]])),
        ('Front-Vertical',
         handle_pos + np.array([-0.06 * cos_y, -0.06 * sin_y, 0.08]),
         np.array(front_vert_rot.as_quat()[[3, 0, 1, 2]])),
    ]


def select_grasps(obj_pos, arm_base, ftype, label, obj_yaw=0.0):
    """Select ordered grasp list based on object/fixture type."""
    is_handle = 'handle' in label.lower()
    if is_handle:
        return build_handle_grasps(obj_pos, arm_base)
    # All non-handle objects: Angled45 preferred, then Top-Down
    grasps = build_object_grasps(obj_pos, arm_base, obj_yaw=obj_yaw)
    is_enclosed = 'interior' in label
    if is_enclosed:
        # Inside a fixture — reverse order to try Top-Down first
        return list(reversed(grasps))
    elif ftype in ('Stove', 'Stovetop'):
        return [g for g in grasps if g[0].startswith('Top-Down')] + \
               [g for g in grasps if not g[0].startswith('Top-Down')]
    return grasps


def build_place_poses(dest_pos, arm_base):
    """Place poses: Angled45 (preferred) then Top-Down, 4 world-frame yaws each."""
    poses = []
    for yaw in [0, np.pi / 2, np.pi, -np.pi / 2]:
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        deg = int(np.degrees(yaw))
        p = dest_pos.copy()
        p[2] += DROP_HEIGHT
        p += np.array([-0.02 * cos_y, -0.02 * sin_y, 0.0])
        q = np.array(euler2quat(0, 3 * np.pi / 4, yaw))
        poses.append((f'Place-Angled45-{deg}', p, q))
    for yaw in [0, np.pi / 2, np.pi, -np.pi / 2]:
        deg = int(np.degrees(yaw))
        p = dest_pos.copy()
        p[2] += DROP_HEIGHT
        q = np.array(euler2quat(0, np.pi, yaw))
        poses.append((f'Place-TopDown-{deg}', p, q))
    return poses
