"""Success-condition utilities ported from RoboCasa to ManiSkill/SAPIEN.

RoboCasa originals: robocasa/utils/object_utils.py (obj_inside_of, gripper_obj_far,
check_obj_upright, check_obj_fixture_contact).

SAPIEN API mappings:
    env.sim.data.body_xpos[id]        → actor.pose.p[0].cpu().numpy()
    env.sim.data.xquat[id]            → actor.pose.q[0].cpu().numpy()  (wxyz)
    env.sim.data.site_xpos[eef_id]    → agent.tcp_pos[0].cpu().numpy()
    fixture.get_int_sites(relative=F) → same API exists in ManiSkill fixture.py
    agent.is_grasping(obj)             → tidyverse_agent.py:341
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def _to_np(val):
    """Convert pose component (numpy or torch, 1D or batched) to 1D numpy."""
    if isinstance(val, torch.Tensor):
        val = val.cpu().numpy()
    if val.ndim >= 2:
        val = val[0]
    return np.asarray(val, dtype=np.float64)


def obj_inside_fixture(actor, fixture, th=0.05):
    """Check whether *actor* centre lies inside *fixture* interior volume.

    Port of RoboCasa ``obj_inside_of()`` (object_utils.py:14-64).
    Uses centre-only check — sufficient for small cubes (CUBE_HALF=0.02 m).

    Args:
        actor: SAPIEN Actor with ``.pose``.
        fixture: ManiSkill Fixture with ``get_int_sites(relative=False)``.
        th: tolerance (metres) added to each half-space boundary.

    Returns:
        bool
    """
    try:
        p0, px, py, pz = fixture.get_int_sites(relative=False)
    except Exception:
        return False

    u = px - p0
    v = py - p0
    w = pz - p0

    # Reject degenerate bounding boxes (e.g. Counter int_sites are all identical)
    vol = abs(np.dot(u, np.cross(v, w)))
    if vol < 1e-6:
        return False

    obj = _to_np(actor.pose.p)

    check_u = np.dot(u, p0) - th <= np.dot(u, obj) <= np.dot(u, px) + th
    check_v = np.dot(v, p0) - th <= np.dot(v, obj) <= np.dot(v, py) + th
    check_w = np.dot(w, p0) - th <= np.dot(w, obj) <= np.dot(w, pz) + th

    return bool(check_u and check_v and check_w)


def gripper_obj_far(agent, actor, th=0.15):
    """Check if gripper is far from object (Euclidean distance > *th*).

    Port of RoboCasa ``gripper_obj_far()`` (object_utils.py:645-652).
    Threshold lowered from RoboCasa 0.25 → 0.15 (smaller gripper).

    Args:
        agent: TidyVerse agent with ``.tcp_pos``.
        actor: SAPIEN Actor with ``.pose``.
        th: distance threshold in metres.

    Returns:
        bool
    """
    eef = _to_np(agent.tcp_pos)
    obj = _to_np(actor.pose.p)
    return bool(np.linalg.norm(eef - obj) > th)


def check_obj_upright(actor, th=15.0):
    """Check if object is upright (roll & pitch within *th* degrees).

    Port of RoboCasa ``check_obj_upright()`` (object_utils.py:604-609).

    Args:
        actor: SAPIEN Actor with ``.pose`` (quaternion in wxyz format).
        th: angle tolerance in degrees.

    Returns:
        bool
    """
    q_wxyz = _to_np(actor.pose.q)
    # SAPIEN quaternion is [w, x, y, z]; scipy expects [x, y, z, w]
    rot = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    euler = rot.as_euler('xyz', degrees=True)
    return bool(abs(euler[0]) < th and abs(euler[1]) < th)


def check_obj_lifted(actor, original_z, min_lift=0.05):
    """Check if object Z is above its original resting height.

    New utility (no direct RoboCasa equivalent).

    Args:
        actor: SAPIEN Actor.
        original_z: Z position when the object was first placed.
        min_lift: minimum height gain in metres.

    Returns:
        bool
    """
    obj_z = _to_np(actor.pose.p)[2]
    return bool(obj_z > original_z + min_lift)


def obj_on_surface(actor, fixture, xy_margin=0.05, z_tolerance=0.06):
    """Simplified surface-proximity check using fixture exterior bounds.

    Checks that the actor XY is within the fixture footprint (with margin)
    and that its Z is near the fixture top surface.

    Args:
        actor: SAPIEN Actor.
        fixture: ManiSkill Fixture with ``get_ext_sites(relative=False)``.
        xy_margin: tolerance for XY containment (metres).
        z_tolerance: maximum Z distance from surface top (metres).

    Returns:
        bool
    """
    try:
        p0, px, py, pz = fixture.get_ext_sites(relative=False)
    except Exception:
        return False

    obj = _to_np(actor.pose.p)

    # XY containment via dot-product half-spaces (same logic as obj_inside_fixture)
    u = px - p0
    v = py - p0
    check_u = np.dot(u, p0) - xy_margin <= np.dot(u, obj) <= np.dot(u, px) + xy_margin
    check_v = np.dot(v, p0) - xy_margin <= np.dot(v, obj) <= np.dot(v, py) + xy_margin

    # Z: object should be near the top surface
    top_z = max(p0[2], px[2], py[2], pz[2])
    check_z = abs(obj[2] - top_z) < z_tolerance

    return bool(check_u and check_v and check_z)


def compute_step_flags(scene, agent, actor, fixture, original_pos, phase):
    """Compute composite success flags for a single pipeline step.

    Args:
        scene: SAPIEN sub-scene (for contacts).
        agent: TidyVerse agent.
        actor: cube Actor being grasped.
        fixture: source Fixture where cube was placed (or None).
        original_pos: np.array — cube position at spawn time.
        phase: str label for the pipeline step (e.g. 'pre_grasp').

    Returns:
        dict with keys: phase, obj_at_source, obj_lifted, obj_upright,
                        gripper_far, is_grasped.
    """
    at_source = False
    if fixture is not None:
        try:
            at_source = obj_inside_fixture(actor, fixture, th=0.05)
        except Exception:
            pass
        if not at_source:
            try:
                at_source = obj_on_surface(actor, fixture)
            except Exception:
                pass

    lifted = check_obj_lifted(actor, original_pos[2], min_lift=0.05)
    upright = check_obj_upright(actor, th=15.0)
    far = gripper_obj_far(agent, actor, th=0.15)

    try:
        grasped = bool(agent.is_grasping(actor).item())
    except Exception:
        grasped = False

    return {
        'phase': phase,
        'obj_at_source': at_source,
        'obj_lifted': lifted,
        'obj_upright': upright,
        'gripper_far': far,
        'is_grasped': grasped,
    }


def format_flags(flags):
    """One-line summary of a flag dict for terminal output."""
    def _b(v):
        return 'T' if v else 'F'
    return (f"src={_b(flags['obj_at_source'])} "
            f"lift={_b(flags['obj_lifted'])} "
            f"up={_b(flags['obj_upright'])} "
            f"grasp={_b(flags['is_grasped'])} "
            f"far={_b(flags['gripper_far'])}")
