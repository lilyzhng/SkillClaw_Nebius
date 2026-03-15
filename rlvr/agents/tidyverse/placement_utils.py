"""RoboCasa fixture enumeration and cube spawning."""
import numpy as np
import sapien
from transforms3d.euler import euler2mat

from mani_skill.utils.scene_builder.robocasa.fixtures.counter import Counter
from mani_skill.utils.scene_builder.robocasa.fixtures.stove import Stove, Stovetop
from mani_skill.utils.scene_builder.robocasa.fixtures.cabinet import (
    SingleCabinet, HingeCabinet, OpenCabinet, Drawer,
)
from mani_skill.utils.scene_builder.robocasa.fixtures.microwave import Microwave
from mani_skill.utils.scene_builder.robocasa.fixtures.accessories import CoffeeMachine
from mani_skill.utils.scene_builder.robocasa.fixtures.sink import Sink
from mani_skill.utils.scene_builder.robocasa.fixtures.others import Floor, Wall


CUBE_HALF = 0.02

COLORS = [
    [1.0, 0.0, 0.0, 1], [0.0, 0.8, 0.0, 1], [0.0, 0.3, 1.0, 1],
    [1.0, 0.7, 0.0, 1], [0.8, 0.0, 0.8, 1], [0.0, 0.8, 0.8, 1],
    [1.0, 1.0, 0.0, 1], [1.0, 0.4, 0.7, 1], [0.6, 0.3, 0.0, 1],
    [0.5, 0.5, 0.5, 1],
]


def local_to_world(fixture, offset):
    rot_mat = euler2mat(0, 0, fixture.rot)
    return np.array(fixture.pos) + rot_mat @ np.array(offset)


def spawn_cube(scene, name, pos, color):
    builder = scene.create_actor_builder()
    hs = np.array([CUBE_HALF] * 3)
    builder.add_box_collision(half_size=hs)
    builder.add_box_visual(
        half_size=hs,
        material=sapien.render.RenderMaterial(base_color=color),
    )
    actor = builder.build(name=name)
    actor.set_pose(sapien.Pose(p=pos))
    return actor


def _region_placements(fix, regions):
    results = []
    for rname, region in regions.items():
        offset = np.array(region["offset"], dtype=float)
        wp = local_to_world(fix, offset)
        wp[2] = fix.pos[2] + offset[2]
        results.append((rname, wp))
    return results


def _int_sites_placement(fix, suffix="interior"):
    try:
        p0, px, py, pz = fix.get_int_sites()
        cx = np.mean([p0[0], px[0]])
        cy = np.mean([p0[1], py[1]])
        cz = p0[2]
        wp = local_to_world(fix, [cx, cy, cz])
        wp[2] = fix.pos[2] + cz
        return [(suffix, wp)]
    except Exception:
        return []


def collect_placements(fixtures):
    """Enumerate all placement surfaces.
    Returns [(label, world_pos, fixture_type_str, fixture_obj)].
    """
    all_placements = []
    for fname, fix in fixtures.items():
        if isinstance(fix, (Floor, Wall)):
            continue
        positions = []
        ftype = type(fix).__name__

        if isinstance(fix, Counter):
            try:
                regions = fix.get_reset_regions(
                    None, fixtures, ref=None, loc="any", top_size=(0.01, 0.01))
                positions = _region_placements(fix, regions)
            except Exception:
                sz = fix.pos[2] + fix.size[2] / 2
                positions = [("top", np.array([fix.pos[0], fix.pos[1], sz]))]
        elif isinstance(fix, (Stove, Stovetop)):
            try:
                positions = _region_placements(fix, fix.get_reset_regions(None))
            except Exception:
                pass
        elif isinstance(fix, Drawer):
            if 0.4 <= fix.pos[2] + fix.size[2] / 2 <= 1.2:
                positions = _int_sites_placement(fix)
        elif isinstance(fix, (SingleCabinet, HingeCabinet, OpenCabinet)):
            positions = _int_sites_placement(fix)
            top_z = fix.pos[2] + fix.size[2] / 2
            if 1.0 <= top_z <= 1.6:
                positions.append(("top", np.array([fix.pos[0], fix.pos[1], top_z])))
        elif isinstance(fix, Microwave):
            positions = _int_sites_placement(fix)
        elif isinstance(fix, CoffeeMachine):
            try:
                positions = _region_placements(fix, fix.get_reset_regions())
            except Exception:
                pass
        elif isinstance(fix, Sink):
            positions = _int_sites_placement(fix, suffix="basin")

        for rname, pos in positions:
            all_placements.append((f"{fname}_{rname}", pos, ftype, fix))

    return all_placements
