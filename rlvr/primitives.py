"""
SkillClaw Primitives — 12 atomic robot actions + tool definitions.

Extracted from scripts/composition_agent.py and generalized to support
all 14 ManiSkill benchmark tasks (not just cube tasks).

Used by:
- rlvr/resources_server.py (FastAPI endpoints)
- scripts/composition_agent.py (legacy all-in-one, unchanged)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# sapien is imported lazily — only needed when RobotPrimitives is instantiated.
# This allows agent_server.py to import TOOLS_OPENAI, TASK_REGISTRY, etc.
# without requiring ManiSkill/sapien installed.


# ============================================================================
# Task Registry — descriptions + max steps for all 14 FAEA benchmark tasks
# ============================================================================

TASK_REGISTRY = {
    # Easy
    "PickCube-v1": {
        "description": "Pick up the cube and hold it at the goal position. Success = cube at goal AND robot static. Do NOT release the cube.",
        "max_steps": 25,
        "difficulty": "easy",
        "objects": ["cube"],
    },
    "PushCube-v1": {
        "description": "Push the cube to the goal position. You cannot pick it up — push by making contact from the side.",
        "max_steps": 25,
        "difficulty": "easy",
        "objects": ["cube"],
    },
    "PullCube-v1": {
        "description": "Pull the cube to the goal region. Grasp the cube and drag it along the surface to the goal position.",
        "max_steps": 25,
        "difficulty": "easy",
        "objects": ["cube"],
    },
    # Medium
    "LiftPegUpright-v1": {
        "description": "Lift the peg from lying flat to standing upright on the table.",
        "max_steps": 25,
        "difficulty": "medium",
        "objects": ["peg"],
    },
    "StackCube-v1": {
        "description": "Pick up cubeA (red) and stack it on top of cubeB (green). CubeA must rest stably on cubeB.",
        "max_steps": 25,
        "difficulty": "medium",
        "objects": ["cubeA", "cubeB"],
    },
    "PokeCube-v1": {
        "description": "Use the peg to poke/push the cube to the goal position. The peg is the tool, the cube is the target.",
        "max_steps": 25,
        "difficulty": "medium",
        "objects": ["peg", "cube"],
    },
    "RollBall-v1": {
        "description": "Push and roll the ball to the goal region at the other end of the table.",
        "max_steps": 25,
        "difficulty": "medium",
        "objects": ["ball"],
    },
    "PlaceSphere-v1": {
        "description": "Pick up the sphere and place it into the shallow bin/container.",
        "max_steps": 25,
        "difficulty": "medium",
        "objects": ["sphere", "bin"],
    },
    "PickSingleYCB-v1": {
        "description": "Pick up the YCB object and move it to the goal position.",
        "max_steps": 25,
        "difficulty": "medium",
        "objects": ["ycb_object"],
    },
    "TurnFaucet-v1": {
        "description": "Rotate the faucet handle to turn it on. Grasp and rotate the handle.",
        "max_steps": 25,
        "difficulty": "medium",
        "objects": ["faucet"],
    },
    # Hard
    "StackPyramid-v1": {
        "description": "Build a pyramid: place cubeA next to cubeB on the table, then stack cubeC on top of both.",
        "max_steps": 25,
        "difficulty": "hard",
        "objects": ["cubeA", "cubeB", "cubeC"],
    },
    "PegInsertionSide-v1": {
        "description": "Pick up the peg and insert it into the hole in the box from the side. Requires precise alignment.",
        "max_steps": 25,
        "difficulty": "hard",
        "objects": ["peg", "box"],
    },
    "PlugCharger-v1": {
        "description": "Pick up the charger and insert it into the receptacle. Requires precise alignment and orientation.",
        "max_steps": 25,
        "difficulty": "hard",
        "objects": ["charger", "receptacle"],
    },
    "AssemblingKits-v1": {
        "description": "Pick up misplaced shapes and insert them into the correct slots on the board.",
        "max_steps": 25,
        "difficulty": "hard",
        "objects": ["shapes", "board"],
    },
    # Kitchen / Mobile
    "RoboCasaKitchen-v1": {
        "description": "Kitchen environment with mobile robot. Navigate to counters, pick up objects, open drawers.",
        "max_steps": 100,
        "difficulty": "hard",
        "objects": ["kitchen_fixtures", "counter_objects"],
    },
}


def get_task_description(env_id: str) -> str:
    """Get task description for system prompt."""
    if env_id in TASK_REGISTRY:
        return TASK_REGISTRY[env_id]["description"]
    return f"Complete the task: {env_id}"


def get_max_steps(env_id: str) -> int:
    """Get max steps for a task."""
    if env_id in TASK_REGISTRY:
        return TASK_REGISTRY[env_id]["max_steps"]
    return 50


# ============================================================================
# RobotPrimitives — 12 atomic actions for ManiSkill Panda robot
# ============================================================================

class RobotPrimitives:
    """12 atomic primitives for robot control in ManiSkill.

    Generalized to work with all 14 benchmark tasks by auto-discovering
    scene objects instead of hardcoding cube names.
    """

    def __init__(self, env, planner):
        import sapien  # lazy import — only needed at runtime with ManiSkill
        self._sapien = sapien
        self.env = env.unwrapped
        self.planner = planner
        self.gripper_open = True
        self._home_qpos = self.env.agent.robot.get_qpos().cpu().numpy()[0].copy()

    # ------ State ------

    def get_state(self) -> dict:
        """Return current environment state as a dict for the LLM.

        Uses a two-tier approach:
        1. Try known attribute names for each task type
        2. Fall back to generic scene actor discovery
        """
        tcp_pos = self.env.agent.tcp.pose.p.cpu().numpy().flatten().tolist()
        state = {
            "gripper_position": [round(x, 4) for x in tcp_pos],
            "gripper_open": self.gripper_open,
        }

        # Discover objects — try known attributes first
        known_attrs = [
            # Cube tasks
            ("obj", "object"),
            ("cube", "cube"),
            ("cubeA", "cubeA"),
            ("cubeB", "cubeB"),
            ("cubeC", "cubeC"),
            # Peg tasks
            ("peg", "peg"),
            # Ball tasks
            ("ball", "ball"),
            # Sphere tasks
            ("sphere", "sphere"),
            # Charger tasks
            ("charger", "charger"),
            ("receptacle", "receptacle"),
            # YCB
            ("source_obj", "object"),
            ("manipulate_obj", "object"),
        ]

        found_objects = set()
        for attr, label in known_attrs:
            if hasattr(self.env, attr):
                obj = getattr(self.env, attr)
                if hasattr(obj, 'pose') and label not in found_objects:
                    pos = obj.pose.p.cpu().numpy().flatten().tolist()
                    state[f"{label}_position"] = [round(x, 4) for x in pos]
                    found_objects.add(label)
                    # Add size if available
                    size_attr = f"{attr}_half_size" if attr != "obj" else "cube_half_size"
                    if hasattr(self.env, size_attr):
                        hs = getattr(self.env, size_attr)
                        if hasattr(hs, 'cpu'):
                            hs = hs.cpu().numpy().flatten().tolist()
                        if hasattr(hs, '__len__'):
                            state[f"{label}_half_size"] = [round(float(x), 4) for x in hs]
                        else:
                            state[f"{label}_half_size"] = round(float(hs), 4)

        # Goal position — try multiple attribute names
        for attr in ["goal_site", "goal_pos", "goal_region", "goal_pose"]:
            if hasattr(self.env, attr):
                val = getattr(self.env, attr)
                if hasattr(val, 'pose'):
                    pos = val.pose.p.cpu().numpy().flatten().tolist()
                elif hasattr(val, 'cpu'):
                    pos = val.cpu().numpy().flatten().tolist()
                elif hasattr(val, 'p'):
                    pos = val.p.cpu().numpy().flatten().tolist()
                else:
                    continue
                state["goal_position"] = [round(x, 4) for x in pos]
                break

        # Task success + grasp status
        info = self.env.evaluate()
        if "success" in info:
            succ = info["success"]
            state["task_success"] = bool(succ.item()) if hasattr(succ, "item") else bool(succ)
        if "is_grasped" in info:
            ig = info["is_grasped"]
            state["is_grasped"] = bool(ig.item()) if hasattr(ig, "item") else bool(ig)

        # Articulation info (e.g., TurnFaucet joint angle)
        if hasattr(self.env, 'faucet'):
            try:
                qpos = self.env.faucet.get_qpos()
                if qpos is not None:
                    state["faucet_joint_angle"] = round(float(qpos.cpu().numpy().flatten()[0]), 4)
            except Exception as e:
                logger.warning(f"Failed to read faucet joint angle: {e}")

        return state

    # ------ Gripper (2) ------

    def close_gripper(self) -> dict:
        """Close the gripper fingers."""
        try:
            self.planner.close_gripper()
            self.gripper_open = False
            return {"success": True, "message": "Gripper closed."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def open_gripper(self) -> dict:
        """Open the gripper fingers."""
        try:
            self.planner.open_gripper()
            self.gripper_open = True
            return {"success": True, "message": "Gripper opened."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------ Movement (4) ------

    def move_to_position(self, x: float, y: float, z: float) -> dict:
        """Move the end-effector to the specified [x, y, z] world coordinate."""
        try:
            current_q = self.env.agent.tcp.pose.q.cpu().numpy().flatten()
            target_pose = self._sapien.Pose([x, y, z], current_q)
            res = self.planner.move_to_pose_with_screw(target_pose)

            if res == -1:
                return {"success": False, "error": "Motion planning failed — no valid path"}

            tcp_pos = self.env.agent.tcp.pose.p.cpu().numpy().flatten()
            return {
                "success": True,
                "reached_position": [round(float(v), 4) for v in tcp_pos],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def rotate_gripper(self, angle_degrees: float) -> dict:
        """Rotate the gripper around its vertical axis by the given angle in degrees."""
        try:
            current_pose = self.env.agent.tcp.pose
            current_p = current_pose.p.cpu().numpy().flatten()
            current_q = current_pose.q.cpu().numpy().flatten()

            angle_rad = np.radians(angle_degrees)
            rot_q = np.array([
                np.cos(angle_rad / 2), 0, 0, np.sin(angle_rad / 2)
            ])

            def qmul(q1, q2):
                w1, x1, y1, z1 = q1
                w2, x2, y2, z2 = q2
                return np.array([
                    w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2,
                ])

            new_q = qmul(current_q, rot_q)
            target_pose = self._sapien.Pose(current_p, new_q)
            res = self.planner.move_to_pose_with_screw(target_pose)

            if res == -1:
                return {"success": False, "error": "Motion planning failed — cannot rotate to target angle"}

            return {"success": True, "message": f"Gripper rotated {angle_degrees} degrees."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def move_base(self, x: float, y: float, theta: float) -> dict:
        """Move the robot base. Not available for Panda (fixed base)."""
        return {"success": False, "error": "move_base not available — Panda has a fixed base. Use move_to_position."}

    def go_home(self) -> dict:
        """Move the robot arm back to its default home position."""
        try:
            home_qpos = self._home_qpos[:len(self.planner.planner.joint_vel_limits)]
            current_qpos = self.env.agent.robot.get_qpos().cpu().numpy()[0][:len(self.planner.planner.joint_vel_limits)]

            result = self.planner.planner.plan_qpos(
                [home_qpos],
                current_qpos,
                time_step=self.env.control_timestep,
            )

            if result["status"] != "Success":
                tcp_pos = self.env.agent.tcp.pose.p.cpu().numpy().flatten()
                current_q = self.env.agent.tcp.pose.q.cpu().numpy().flatten()
                safe_pose = self._sapien.Pose([tcp_pos[0], tcp_pos[1], 0.3], current_q)
                res = self.planner.move_to_pose_with_screw(safe_pose)
                if res == -1:
                    return {"success": False, "error": "Motion planning failed — cannot return home"}
                return {"success": True, "message": "Moved to safe height (home planning failed)."}

            self.planner.follow_path(result)
            tcp_pos = self.env.agent.tcp.pose.p.cpu().numpy().flatten()
            return {
                "success": True,
                "message": "Returned to home position.",
                "position": [round(float(v), 4) for v in tcp_pos],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------ Perception (3) ------

    def detect_object(self, target: str) -> dict:
        """Detect an object by name and return its position and size."""
        try:
            obj = self._get_object(target)
            if obj is None:
                return {"success": False, "error": f"Object '{target}' not found in scene."}

            pos = obj.pose.p.cpu().numpy().flatten()
            result = {
                "success": True,
                "object_name": target,
                "position": [round(float(v), 4) for v in pos],
            }

            if hasattr(self.env, "cube_half_size"):
                hs = self.env.cube_half_size
                if hasattr(hs, 'cpu'):
                    hs = hs.cpu().numpy().flatten().tolist()
                if hasattr(hs, '__len__'):
                    result["half_size"] = [round(float(x), 4) for x in hs]
                else:
                    result["half_size"] = round(float(hs), 4)

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_camera_image(self) -> dict:
        """List all visible objects and their positions."""
        try:
            visible = []
            for attr, name in [
                ("obj", "object"), ("cube", "cube"),
                ("cubeA", "cubeA"), ("cubeB", "cubeB"), ("cubeC", "cubeC"),
                ("peg", "peg"), ("ball", "ball"), ("sphere", "sphere"),
                ("charger", "charger"), ("receptacle", "receptacle"),
                ("source_obj", "object"), ("manipulate_obj", "object"),
            ]:
                if hasattr(self.env, attr):
                    obj_val = getattr(self.env, attr)
                    if hasattr(obj_val, 'pose'):
                        pos = obj_val.pose.p.cpu().numpy().flatten()
                        visible.append({
                            "name": name,
                            "position": [round(float(v), 4) for v in pos],
                        })

            # Goal
            for attr in ["goal_site", "goal_pos", "goal_region"]:
                if hasattr(self.env, attr):
                    val = getattr(self.env, attr)
                    if hasattr(val, 'pose'):
                        pos = val.pose.p.cpu().numpy().flatten()
                    elif hasattr(val, 'cpu'):
                        pos = val.cpu().numpy().flatten()
                    else:
                        continue
                    visible.append({
                        "name": "goal",
                        "position": [round(float(v), 4) for v in pos],
                    })
                    break

            return {
                "success": True,
                "visible_objects": visible,
                "gripper_position": [round(float(v), 4) for v in
                                     self.env.agent.tcp.pose.p.cpu().numpy().flatten()],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def tilt_camera(self, angle_degrees: float) -> dict:
        """Tilt the camera. In simulation, this is a no-op."""
        return {
            "success": True,
            "message": f"Camera tilt adjusted by {angle_degrees} degrees. (Simulation: camera follows gripper.)",
        }

    # ------ Alignment-Aware Primitives (3) ------

    def grasp_object(self, target: str) -> dict:
        """Smart grasp using OBB (oriented bounding box) analysis.

        Instead of the LLM guessing coordinates, this primitive:
        1. Finds the object by name
        2. Computes optimal grasp pose using OBB + approaching direction
        3. Reaches above, lowers, and closes gripper

        Uses the same algorithm as ManiSkill's official solutions.
        """
        try:
            from mani_skill.examples.motionplanning.base_motionplanner.utils import (
                compute_grasp_info_by_obb, get_actor_obb,
            )

            obj = self._get_object(target)
            if obj is None:
                return {"success": False, "error": f"Object '{target}' not found in scene."}

            FINGER_LENGTH = 0.025

            # Compute OBB and grasp frame
            obb = get_actor_obb(obj)
            approaching = np.array([0, 0, -1])  # approach from above
            target_closing = self.env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

            grasp_info = compute_grasp_info_by_obb(
                obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH
            )
            closing, center = grasp_info["closing"], grasp_info["center"]
            grasp_pose = self.env.agent.build_grasp_pose(approaching, closing, center)

            # Step 1: Reach (approach from above)
            reach_pose = grasp_pose * self._sapien.Pose([0, 0, -0.05])
            res = self.planner.move_to_pose_with_screw(reach_pose)
            if res == -1:
                return {"success": False, "error": "Motion planning failed — cannot reach above object"}

            # Step 2: Lower to grasp position
            res = self.planner.move_to_pose_with_screw(grasp_pose)
            if res == -1:
                return {"success": False, "error": "Motion planning failed — cannot lower to grasp"}

            # Step 3: Close gripper
            self.planner.close_gripper()
            self.gripper_open = False

            tcp_pos = self.env.agent.tcp.pose.p.cpu().numpy().flatten()
            return {
                "success": True,
                "message": f"Grasped '{target}' using OBB-aligned grasp.",
                "grasp_position": [round(float(v), 4) for v in tcp_pos],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def align_object_to_goal(self) -> dict:
        """After grasping, align the held object with the goal pose.

        Uses pose algebra: goal_pose * object_pose.inv() * tcp_pose
        This computes where the gripper needs to be so that the held object
        matches the goal orientation/position. Includes iterative refinement.

        Must be called after grasp_object (gripper must be holding something).
        """
        try:
            if self.gripper_open:
                return {"success": False, "error": "Gripper is open — grasp an object first."}

            # Find the goal pose
            goal_pose = None
            for attr in ["goal_pose", "goal_site", "goal_pos"]:
                if hasattr(self.env, attr):
                    val = getattr(self.env, attr)
                    if hasattr(val, 'sp'):
                        goal_pose = val.sp if hasattr(val.sp, 'inv') else val
                    elif hasattr(val, 'inv'):
                        goal_pose = val
                    break

            if goal_pose is None:
                return {"success": False, "error": "No goal pose found in environment."}

            # Find the grasped object pose — try common object attributes
            obj_pose = None
            for attr in ["peg", "charger", "obj", "cube", "cubeA", "sphere", "ball",
                         "source_obj", "manipulate_obj"]:
                if hasattr(self.env, attr):
                    obj_val = getattr(self.env, attr)
                    if hasattr(obj_val, 'pose'):
                        obj_pose = obj_val.pose
                        if hasattr(obj_pose, 'sp'):
                            obj_pose = obj_pose.sp
                        break

            if obj_pose is None:
                return {"success": False, "error": "Cannot find grasped object pose."}

            tcp_pose = self.env.agent.tcp.pose
            if hasattr(tcp_pose, 'sp'):
                tcp_pose = tcp_pose.sp

            # Compute pre-insertion pose (offset back along x-axis for clearance)
            offset = self._sapien.Pose([-0.03, 0, 0])
            pre_insert_pose = goal_pose * offset * obj_pose.inv() * tcp_pose

            # Move to pre-insertion alignment
            res = self.planner.move_to_pose_with_screw(pre_insert_pose)
            if res == -1:
                return {"success": False, "error": "Motion planning failed — cannot align to pre-insertion pose"}

            # Iterative refinement (3 rounds, from ManiSkill official solutions)
            for i in range(3):
                # Re-read current object pose (it moves with the gripper)
                obj_pose_current = None
                for attr in ["peg", "charger", "obj", "cube", "cubeA", "sphere", "ball",
                             "source_obj", "manipulate_obj"]:
                    if hasattr(self.env, attr):
                        obj_val = getattr(self.env, attr)
                        if hasattr(obj_val, 'pose'):
                            obj_pose_current = obj_val.pose
                            if hasattr(obj_pose_current, 'sp'):
                                obj_pose_current = obj_pose_current.sp
                            break

                if obj_pose_current is not None:
                    delta_pose = goal_pose * offset * obj_pose_current.inv()
                    pre_insert_pose = delta_pose * pre_insert_pose
                    res = self.planner.move_to_pose_with_screw(pre_insert_pose)
                    if res == -1:
                        return {"success": False, "error": f"Refinement step {i+1} failed"}

            tcp_pos = self.env.agent.tcp.pose.p.cpu().numpy().flatten()
            return {
                "success": True,
                "message": "Object aligned with goal pose (3 refinement steps).",
                "aligned_position": [round(float(v), 4) for v in tcp_pos],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def insert_object(self, depth: float = 0.05) -> dict:
        """Push the held object forward along the insertion axis.

        After align_object_to_goal, the insertion axis is the local x-axis.
        This moves the gripper forward by `depth` meters to complete insertion.

        Args:
            depth: How far to push in meters (default 0.05 = 5cm)
        """
        try:
            if self.gripper_open:
                return {"success": False, "error": "Gripper is open — grasp and align first."}

            # Push forward along local x-axis (insertion direction)
            insert_offset = self._sapien.Pose([depth, 0, 0])
            tcp_pose = self.env.agent.tcp.pose
            if hasattr(tcp_pose, 'sp'):
                tcp_pose = tcp_pose.sp

            insert_pose = tcp_pose * insert_offset
            res = self.planner.move_to_pose_with_screw(insert_pose)

            if res == -1:
                return {"success": False, "error": "Motion planning failed — cannot insert (collision or unreachable)"}

            tcp_pos = self.env.agent.tcp.pose.p.cpu().numpy().flatten()

            # Check success
            info = self.env.evaluate()
            success = False
            if "success" in info:
                succ = info["success"]
                success = bool(succ.item()) if hasattr(succ, "item") else bool(succ)

            return {
                "success": True,
                "message": f"Inserted {depth*100:.1f}cm forward.",
                "position": [round(float(v), 4) for v in tcp_pos],
                "task_success": success,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------ Object Resolution ------

    def _get_object(self, name: str):
        """Resolve object name to SAPIEN actor. Supports all 14 task types."""
        name = name.lower().strip()

        # Cube variants
        if name in ("cube", "the cube", "red cube", "obj", "object"):
            for attr in ["obj", "cube", "cubeA", "source_obj", "manipulate_obj"]:
                if hasattr(self.env, attr):
                    return getattr(self.env, attr)
        if name in ("cubea", "cube_a", "cube a", "first cube", "red cube"):
            return getattr(self.env, "cubeA", None)
        if name in ("cubeb", "cube_b", "cube b", "second cube", "green cube", "target cube"):
            return getattr(self.env, "cubeB", None)
        if name in ("cubec", "cube_c", "cube c", "third cube", "blue cube"):
            return getattr(self.env, "cubeC", None)

        # Peg
        if name in ("peg", "the peg", "stick"):
            return getattr(self.env, "peg", None)

        # Ball
        if name in ("ball", "the ball"):
            return getattr(self.env, "ball", None)

        # Sphere
        if name in ("sphere", "the sphere"):
            return getattr(self.env, "sphere", None)

        # Charger
        if name in ("charger", "the charger", "plug"):
            return getattr(self.env, "charger", None)

        # Receptacle
        if name in ("receptacle", "the receptacle", "socket"):
            return getattr(self.env, "receptacle", None)

        # Faucet
        if name in ("faucet", "the faucet", "handle", "tap"):
            return getattr(self.env, "faucet", None)

        # Box (PegInsertionSide)
        if name in ("box", "the box", "box_with_hole"):
            for attr in ["box", "box_with_hole"]:
                if hasattr(self.env, attr):
                    return getattr(self.env, attr)

        # Bin (PlaceSphere)
        if name in ("bin", "the bin", "container", "bowl"):
            for attr in ["bin", "container", "bowl"]:
                if hasattr(self.env, attr):
                    return getattr(self.env, attr)

        # Generic fallback — try the name as attribute directly
        if hasattr(self.env, name):
            obj = getattr(self.env, name)
            if hasattr(obj, 'pose'):
                return obj

        return None


# ============================================================================
# Tool Definitions — OpenAI-compatible format for LLM API
# ============================================================================

TOOLS = [
    # --- Gripper ---
    {
        "name": "close_gripper",
        "description": "Close the gripper fingers. Use for grasping objects or forming a solid surface for pushing.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "open_gripper",
        "description": "Open the gripper fingers. Use to release held objects or prepare for grasping.",
        "input_schema": {"type": "object", "properties": {}},
    },
    # --- Movement ---
    {
        "name": "move_to_position",
        "description": "Move the end-effector (gripper) to the specified [x, y, z] position in world coordinates. Keeps current orientation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "X coordinate"},
                "y": {"type": "number", "description": "Y coordinate"},
                "z": {"type": "number", "description": "Z coordinate"},
            },
            "required": ["x", "y", "z"],
        },
    },
    {
        "name": "rotate_gripper",
        "description": "Rotate the gripper around its vertical axis by the given angle in degrees.",
        "input_schema": {
            "type": "object",
            "properties": {
                "angle_degrees": {"type": "number", "description": "Rotation angle in degrees. Positive = counterclockwise."},
            },
            "required": ["angle_degrees"],
        },
    },
    {
        "name": "move_base",
        "description": "Move the robot base to position (x, y) with rotation theta. Only for mobile robots. Not available for Panda.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "X coordinate for base"},
                "y": {"type": "number", "description": "Y coordinate for base"},
                "theta": {"type": "number", "description": "Rotation in degrees"},
            },
            "required": ["x", "y", "theta"],
        },
    },
    {
        "name": "go_home",
        "description": "Move the robot arm back to its default home position.",
        "input_schema": {"type": "object", "properties": {}},
    },
    # --- Perception ---
    {
        "name": "detect_object",
        "description": "Detect an object by name and return its position [x, y, z] and size.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Name of the object to detect (e.g. 'cube', 'cubeA', 'peg', 'ball', 'charger', 'faucet')"},
            },
            "required": ["target"],
        },
    },
    {
        "name": "get_camera_image",
        "description": "Capture the current camera view and return a list of all visible objects with their positions.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "tilt_camera",
        "description": "Tilt the camera by the given angle in degrees.",
        "input_schema": {
            "type": "object",
            "properties": {
                "angle_degrees": {"type": "number", "description": "Tilt angle in degrees. Negative = look down."},
            },
            "required": ["angle_degrees"],
        },
    },
    # --- Alignment-Aware ---
    {
        "name": "grasp_object",
        "description": "Smart grasp: automatically computes optimal grasp pose using the object's shape (OBB). Approaches from above, lowers, and closes gripper. Use this instead of manually positioning and closing gripper for precision tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Name of the object to grasp (e.g. 'peg', 'charger', 'cube')"},
            },
            "required": ["target"],
        },
    },
    {
        "name": "align_object_to_goal",
        "description": "After grasping an object, align it with the goal pose using precise pose algebra. Includes 3 iterative refinement steps. Must call grasp_object first. Use for insertion/alignment tasks (peg insertion, plug charger, etc.).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "insert_object",
        "description": "After aligning, push the held object forward along the insertion axis to complete insertion. Must call align_object_to_goal first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "depth": {"type": "number", "description": "How far to push in meters (default 0.05 = 5cm). Increase for deeper insertion."},
            },
        },
    },
]

# OpenAI function-calling format (for OpenRouter)
TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in TOOLS
]


# ============================================================================
# Tool Dispatch
# ============================================================================

def _coerce_float(val):
    """Coerce value to float. LLMs sometimes output strings instead of numbers."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            raise ValueError(f"Invalid numeric value: {val!r} (expected a number, not a string expression)")
    raise ValueError(f"Invalid numeric value: {type(val).__name__}")


def execute_tool(primitives: RobotPrimitives, tool_name: str, tool_input: dict) -> dict:
    """Execute an atomic primitive tool by name."""
    if tool_name == "close_gripper":
        return primitives.close_gripper()
    elif tool_name == "open_gripper":
        return primitives.open_gripper()
    elif tool_name == "move_to_position":
        try:
            x = _coerce_float(tool_input.get("x"))
            y = _coerce_float(tool_input.get("y"))
            z = _coerce_float(tool_input.get("z"))
        except ValueError as e:
            return {"success": False, "error": str(e)}
        return primitives.move_to_position(x, y, z)
    elif tool_name == "rotate_gripper":
        try:
            angle = _coerce_float(tool_input.get("angle_degrees"))
        except ValueError as e:
            return {"success": False, "error": str(e)}
        return primitives.rotate_gripper(angle)
    elif tool_name == "move_base":
        try:
            x = _coerce_float(tool_input.get("x"))
            y = _coerce_float(tool_input.get("y"))
            theta = _coerce_float(tool_input.get("theta"))
        except ValueError as e:
            return {"success": False, "error": str(e)}
        return primitives.move_base(x, y, theta)
    elif tool_name == "go_home":
        return primitives.go_home()
    elif tool_name == "detect_object":
        target = tool_input.get("target")
        if target is None or (isinstance(target, str) and not target.strip()):
            return {"success": False, "error": "Missing or empty 'target' argument"}
        return primitives.detect_object(str(target).strip())
    elif tool_name == "get_camera_image":
        return primitives.get_camera_image()
    elif tool_name == "tilt_camera":
        try:
            angle = _coerce_float(tool_input.get("angle_degrees"))
        except ValueError as e:
            return {"success": False, "error": str(e)}
        return primitives.tilt_camera(angle)
    elif tool_name == "grasp_object":
        target = tool_input.get("target")
        if target is None or (isinstance(target, str) and not target.strip()):
            return {"success": False, "error": "Missing or empty 'target' argument"}
        return primitives.grasp_object(str(target).strip())
    elif tool_name == "align_object_to_goal":
        return primitives.align_object_to_goal()
    elif tool_name == "insert_object":
        depth = tool_input.get("depth", 0.05)
        try:
            depth = _coerce_float(depth)
        except ValueError as e:
            return {"success": False, "error": str(e)}
        return primitives.insert_object(depth)
    else:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}


# ============================================================================
# System Prompt Builder
# ============================================================================

def build_system_prompt(env_id: str, include_patterns: bool = True) -> str:
    """Build system prompt for the composition agent."""
    task_desc = get_task_description(env_id)

    prompt = f"""You are a robot control agent. You must complete this task:

**Task: {task_desc}**

You have 12 atomic tools:

**Gripper:**
- close_gripper() — close fingers (for grasping or pushing)
- open_gripper() — open fingers (for releasing)

**Movement:**
- move_to_position(x, y, z) — move end-effector to world coordinate
- rotate_gripper(angle_degrees) — rotate gripper around vertical axis
- move_base(x, y, theta) — move robot base (not available for Panda)
- go_home() — return arm to default position

**Alignment (for precision tasks):**
- grasp_object(target) — smart grasp using shape analysis (no coordinate guessing)
- align_object_to_goal() — after grasping, align held object with goal pose
- insert_object(depth) — after aligning, push forward to complete insertion

**Perception:**
- detect_object(target) — find object, returns position and size
- get_camera_image() — list all visible objects and positions
- tilt_camera(angle_degrees) — adjust camera angle
"""

    if include_patterns:
        prompt += """
## Composition Patterns

### Pick and Hold
```
1. detect_object(target) → find position
2. move_to_position(obj.x, obj.y, obj.z + 0.05) → above object
3. move_to_position(obj.x, obj.y, obj.z) → lower to object
4. close_gripper() → grasp
5. move_to_position(goal.x, goal.y, goal.z) → carry to goal
```

### Pick and Place
```
1. detect_object(A) → find position
2. move_to_position(A.x, A.y, A.z + 0.05) → above A
3. move_to_position(A.x, A.y, A.z) → lower to A
4. close_gripper() → grasp A
5. move_to_position(A.x, A.y, 0.15) → LIFT straight up
6. move_to_position(target.x, target.y, 0.15) → TRANSLATE at safe height
7. move_to_position(target.x, target.y, target.z + 0.04) → LOWER onto target
8. open_gripper() → release
```
CRITICAL: Never move directly between objects. Always: lift → translate → lower.

### Insert (peg insertion, plug charger, etc.)
```
1. grasp_object(target) → smart grasp with shape-aligned approach
2. align_object_to_goal() → align held object with goal (iterative refinement)
3. insert_object(depth=0.05) → push forward to insert
```
Use this pattern for ANY task requiring precise alignment (peg, charger, assembly).

### Push
```
1. detect_object(target) → find object and goal
2. close_gripper() → form solid pushing surface
3. move_to_position(behind object, at object height) → approach from behind
4. move_to_position(past goal, at object height) → push through
5. open_gripper() → reset
```
"""

    prompt += """
## Approach
1. **Perceive**: Use detect_object or get_camera_image to understand the scene
2. **Find closest pattern**: Which pattern above is most similar?
3. **Adapt**: Plug in actual coordinates from perception
4. **Execute one tool at a time**, checking state after each step
5. **If unexpected**: Re-perceive, re-analyze, adapt

Always explain your reasoning before each tool call.
"""
    return prompt
