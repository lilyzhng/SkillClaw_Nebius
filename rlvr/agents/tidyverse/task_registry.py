"""RoboCasa task language instructions ported to ManiSkill.

Each entry maps a (source_fixture_type, dest_fixture_type) pair to a natural
language instruction template and the success condition names it implies.

Templates use {obj} for the object name and {src}/{dst} for fixture labels.

The success conditions reference functions in success_utils.py:
    obj_inside_fixture  — object centre inside fixture interior volume
    obj_on_surface      — object XY within fixture footprint, Z near top
    gripper_obj_far     — gripper disengaged from object
    check_obj_upright   — object roll/pitch within tolerance
    check_obj_lifted    — object Z above original resting height
"""

# ─── Pick-place instruction templates ─────────────────────────────────────────
# Keyed by (source_type, dest_type) where types are fixture class names.
# "any" matches any fixture type.

PICK_PLACE_TEMPLATES = {
    # Counter → enclosed fixtures
    ("Counter", "HingeCabinet"):
        "Pick the {obj} from the counter and place it in the cabinet.",
    ("Counter", "SingleCabinet"):
        "Pick the {obj} from the counter and place it in the cabinet.",
    ("Counter", "OpenCabinet"):
        "Pick the {obj} from the counter and place it in the cabinet.",
    ("Counter", "Drawer"):
        "Pick the {obj} from the counter and place it in the drawer.",
    ("Counter", "Microwave"):
        "Pick the {obj} from the counter and place it in the microwave.",
    ("Counter", "Sink"):
        "Pick the {obj} from the counter and place it in the sink.",
    ("Counter", "Stove"):
        "Pick the {obj} from the counter and place it on the stove.",
    ("Counter", "Stovetop"):
        "Pick the {obj} from the counter and place it on the stove.",
    ("Counter", "CoffeeMachine"):
        "Pick the {obj} from the counter and place it under the coffee machine.",

    # Enclosed fixtures → counter
    ("HingeCabinet", "Counter"):
        "Pick the {obj} from the cabinet and place it on the counter.",
    ("SingleCabinet", "Counter"):
        "Pick the {obj} from the cabinet and place it on the counter.",
    ("OpenCabinet", "Counter"):
        "Pick the {obj} from the cabinet and place it on the counter.",
    ("Drawer", "Counter"):
        "Pick the {obj} from the drawer and place it on the counter.",
    ("Microwave", "Counter"):
        "Pick the {obj} from the microwave and place it on the counter.",
    ("Sink", "Counter"):
        "Pick the {obj} from the sink and place it on the counter.",
    ("Stove", "Counter"):
        "Pick the {obj} from the stove and place it on the counter.",
    ("Stovetop", "Counter"):
        "Pick the {obj} from the stove and place it on the counter.",

    # Same-type moves
    ("Counter", "Counter"):
        "Move the {obj} to a different spot on the counter.",
}

# Fallback template when no specific match exists
PICK_PLACE_FALLBACK = "Pick the {obj} from the {src} and place it on the {dst}."


# ─── Grasp-only instruction templates ─────────────────────────────────────────
# Used when test only performs grasp (no destination), keyed by source type.

GRASP_TEMPLATES = {
    "Counter":       "Pick up the {obj} from the counter.",
    "HingeCabinet":  "Pick up the {obj} from inside the cabinet.",
    "SingleCabinet": "Pick up the {obj} from inside the cabinet.",
    "OpenCabinet":   "Pick up the {obj} from inside the cabinet.",
    "Drawer":        "Pick up the {obj} from the drawer.",
    "Microwave":     "Pick up the {obj} from the microwave.",
    "Sink":          "Pick up the {obj} from the sink.",
    "Stove":         "Pick up the {obj} from the stove.",
    "Stovetop":      "Pick up the {obj} from the stove.",
    "CoffeeMachine": "Pick up the {obj} from the coffee machine.",
}

GRASP_FALLBACK = "Pick up the {obj} from the {src}."


# ─── Expected success conditions per phase ────────────────────────────────────
# Maps phase name to the flags that should be True for a successful grasp cycle.
# Used for automated validation of flag outputs.

EXPECTED_FLAGS = {
    'pre_grasp': {
        'obj_at_source': True,
        'obj_lifted': False,
        'is_grasped': False,
    },
    'approach': {
        'obj_at_source': True,
        'obj_lifted': False,
        'is_grasped': False,
    },
    'grasp': {
        'obj_at_source': True,
        'obj_lifted': False,
        'is_grasped': True,
    },
    'lift': {
        'obj_at_source': False,
        'obj_lifted': True,
        'is_grasped': True,
    },
    'release': {
        'is_grasped': False,
        'gripper_far': True,
    },
    'home': {
        'is_grasped': False,
        'gripper_far': True,
    },
}


def get_grasp_instruction(fixture_type, obj_name="cube"):
    """Get natural language instruction for grasping an object from a fixture.

    Args:
        fixture_type: class name of the source fixture (e.g. 'Counter').
        obj_name: name of the object to grasp.

    Returns:
        str: natural language instruction.
    """
    template = GRASP_TEMPLATES.get(fixture_type, GRASP_FALLBACK)
    return template.format(obj=obj_name, src=fixture_type.lower())


def get_pick_place_instruction(src_type, dst_type, obj_name="cube"):
    """Get natural language instruction for a pick-place task.

    Args:
        src_type: class name of the source fixture.
        dst_type: class name of the destination fixture.
        obj_name: name of the object.

    Returns:
        str: natural language instruction.
    """
    template = PICK_PLACE_TEMPLATES.get(
        (src_type, dst_type), PICK_PLACE_FALLBACK)
    return template.format(obj=obj_name, src=src_type.lower(), dst=dst_type.lower())


def check_phase_expectations(flags, phase):
    """Check if flags match expected values for a phase.

    Args:
        flags: dict from compute_step_flags().
        phase: str phase name.

    Returns:
        (bool, list[str]): (all_match, list of violation descriptions).
    """
    expected = EXPECTED_FLAGS.get(phase, {})
    violations = []
    for key, expected_val in expected.items():
        actual_val = flags.get(key)
        if actual_val is not None and actual_val != expected_val:
            violations.append(
                f"{key}: expected={expected_val} got={actual_val}")
    return (len(violations) == 0, violations)
