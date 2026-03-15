"""Planner setup: mplib monkey-patch, ACM builder, sync helper."""
import numpy as np

from mplib import Pose as MPPose
from mplib.sapien_utils import SapienPlanningWorld
from mplib.collision_detection.fcl import (
    Box, Capsule, Convex, Sphere, BVHModel, Halfspace, Cylinder,
    CollisionObject, FCLObject,
)
from sapien.physx import (
    PhysxCollisionShapeBox, PhysxCollisionShapeCapsule,
    PhysxCollisionShapeConvexMesh, PhysxCollisionShapeSphere,
    PhysxCollisionShapeTriangleMesh, PhysxCollisionShapePlane,
    PhysxCollisionShapeCylinder, PhysxArticulationLinkComponent,
)
from transforms3d.euler import euler2quat
import mplib.sapien_utils.conversion as _conv


# ─── Monkey-patch: apply scale to Robotiq convex collision meshes ─────────────

@staticmethod
def _convert_physx_component(comp):
    shapes, shape_poses = [], []
    for shape in comp.collision_shapes:
        shape_poses.append(MPPose(shape.local_pose))
        if isinstance(shape, PhysxCollisionShapeBox):
            geom = Box(side=shape.half_size * 2)
        elif isinstance(shape, PhysxCollisionShapeCapsule):
            geom = Capsule(radius=shape.radius, lz=shape.half_length * 2)
            shape_poses[-1] *= MPPose(q=euler2quat(0, np.pi / 2, 0))
        elif isinstance(shape, PhysxCollisionShapeConvexMesh):
            verts = shape.vertices
            if not np.allclose(shape.scale, 1.0):
                verts = verts * np.array(shape.scale)
            geom = Convex(vertices=verts, faces=shape.triangles)
        elif isinstance(shape, PhysxCollisionShapeSphere):
            geom = Sphere(radius=shape.radius)
        elif isinstance(shape, PhysxCollisionShapeTriangleMesh):
            geom = BVHModel()
            geom.begin_model()
            geom.add_sub_model(vertices=shape.vertices, faces=shape.triangles)
            geom.end_model()
        elif isinstance(shape, PhysxCollisionShapePlane):
            n = shape_poses[-1].to_transformation_matrix()[:3, 0]
            d = n.dot(shape_poses[-1].p)
            geom = Halfspace(n=n, d=d)
            shape_poses[-1] = MPPose()
        elif isinstance(shape, PhysxCollisionShapeCylinder):
            geom = Cylinder(radius=shape.radius, lz=shape.half_length * 2)
            shape_poses[-1] *= MPPose(q=euler2quat(0, np.pi / 2, 0))
        else:
            continue
        shapes.append(CollisionObject(geom))
    if not shapes:
        return None
    name = (comp.name if isinstance(comp, PhysxArticulationLinkComponent)
            else _conv.convert_object_name(comp.entity))
    return FCLObject(name, comp.entity.pose, shapes, shape_poses)

SapienPlanningWorld.convert_physx_component = _convert_physx_component


# ─── Helpers ──────────────────────────────────────────────────────────────────

def sync_planner(planner):
    try:
        planner.update_from_simulation()
    except Exception:
        pass


def _get_object_position(pw, name):
    try:
        return np.array(pw.get_object(name).pose.p)
    except Exception:
        return None


# ─── ACM builder ─────────────────────────────────────────────────────────────

def build_kitchen_acm(pw, planner, cube_names, mode='relaxed',
                      robot_pos=None, near_radius=3.0):
    acm = pw.get_allowed_collision_matrix()
    art_names = pw.get_articulation_names()
    robot_link_names = planner.pinocchio_model.get_link_names()
    robot_art = next(n for n in art_names if 'tidyverse' in n.lower())

    print(f"\n  Planning world contents (ACM mode={mode}):")
    print(f"    Robot links ({len(robot_link_names)}): {robot_link_names[:5]}...")
    print(f"    Articulations ({len(art_names)}):")

    # Classify articulations by distance
    relaxed_arts, checked_arts = [], []
    for an in art_names:
        if an == robot_art:
            print(f"      [ROBOT] {an}")
            continue
        fl = pw.get_articulation(an).get_pinocchio_model().get_link_names()
        if mode == 'relaxed':
            relaxed_arts.append(an)
            print(f"      [RELAXED] {an} ({len(fl)} links)")
        else:
            art = pw.get_articulation(an)
            fcl_model = art.get_fcl_model()
            col_objs = fcl_model.get_collision_objects()
            art_pos = None
            if col_objs:
                art_pos = np.array(col_objs[0].pose.p)
            if art_pos is not None and robot_pos is not None:
                dist = np.linalg.norm(art_pos[:2] - robot_pos[:2])
                if dist > near_radius:
                    relaxed_arts.append(an)
                    print(f"      [RELAXED] {an} ({len(fl)} links)  dist={dist:.2f}m")
                else:
                    checked_arts.append((an, dist))
                    print(f"      [CHECKED] {an} ({len(fl)} links)  dist={dist:.2f}m")
            else:
                checked_arts.append((an, -1))
                print(f"      [CHECKED] {an} ({len(fl)} links)  dist=unknown")

    obj_names = pw.get_object_names()
    checked_objs = [n for n in obj_names if n in cube_names]

    # Classify static objects by distance
    relaxed_static, checked_static = [], []
    for on in obj_names:
        if on in cube_names:
            continue
        if mode == 'relaxed':
            relaxed_static.append(on)
        else:
            pos = _get_object_position(pw, on)
            if pos is not None and robot_pos is not None:
                dist = np.linalg.norm(pos[:2] - robot_pos[:2])
                if dist > near_radius:
                    relaxed_static.append(on)
                else:
                    checked_static.append((on, dist))
            else:
                checked_static.append((on, -1))

    print(f"    Static objects ({len(obj_names)} total):")
    print(f"      Cubes (always checked): {len(checked_objs)}")
    if mode == 'strict':
        print(f"      Collision-checked arts (near): {len(checked_arts)}")
        print(f"      Collision-checked static (near, <{near_radius}m): {len(checked_static)}")
        for on, d in sorted(checked_static, key=lambda x: x[1]):
            print(f"        - {on}  dist={d:.2f}m")
    print(f"      ACM-relaxed arts: {len(relaxed_arts)}")
    print(f"      ACM-relaxed static: {len(relaxed_static)}")

    # Apply ACM — only relax far objects
    for an in relaxed_arts:
        fl = pw.get_articulation(an).get_pinocchio_model().get_link_names()
        for rl in robot_link_names:
            for f in fl:
                acm.set_entry(rl, f, True)
    for on in relaxed_static:
        for rl in robot_link_names:
            acm.set_entry(rl, on, True)

    # Resolve initial collisions
    sync_planner(planner)
    collisions = pw.check_collision()
    init_relaxed = []
    if collisions:
        print(f"\n  Initial planner collisions ({len(collisions)}):")
        for c in collisions:
            print(f"    {c.link_name1}({c.object_name1}) <-> "
                  f"{c.link_name2}({c.object_name2})")
            for obj_name in [c.object_name1, c.object_name2]:
                if obj_name and obj_name not in cube_names:
                    if not any('tidyverse' in obj_name.lower() for _ in [1]):
                        for rl in robot_link_names:
                            acm.set_entry(rl, obj_name, True)
                        init_relaxed.append(obj_name)
        if init_relaxed:
            print(f"    Auto-relaxed {len(set(init_relaxed))} objects")
            sync_planner(planner)
    else:
        print(f"\n  No initial planner collisions")

    n_checked_s = len(checked_static) - len(set(init_relaxed))
    print(f"\n  ACM summary: {len(relaxed_arts)} arts relaxed, "
          f"{len(checked_arts)} arts checked, "
          f"{len(relaxed_static)} static relaxed, "
          f"{max(0, n_checked_s)} static checked, "
          f"{len(checked_objs)} cubes checked")
