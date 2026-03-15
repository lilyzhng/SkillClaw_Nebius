"""TidyVerse robot agent for ManiSkill — Panda arm + TidyBot base + Robotiq 85 gripper."""
import os
from copy import deepcopy

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor

# --- Monkey-patch scene builders to recognize 'tidyverse' ---
def _patch_scene_builders():
    """Add tidyverse init to TableSceneBuilder and RoboCasaSceneBuilder."""

    # Patch TableSceneBuilder (PickCube, etc.)
    from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
    _orig_table_init = TableSceneBuilder.initialize

    def _patched_table_initialize(self, env_idx):
        _orig_table_init(self, env_idx)
        if self.env.robot_uids == "tidyverse":
            b = len(env_idx)
            qpos = self.env.agent.keyframes["rest"].qpos
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

    TableSceneBuilder.initialize = _patched_table_initialize

    # Patch RoboCasaSceneBuilder (RoboCasaKitchen, etc.)
    try:
        from mani_skill.utils.scene_builder.robocasa.scene_builder import (
            RoboCasaSceneBuilder,
            ROBOT_FRONT_FACING_SIZE,
        )

        # Register tidyverse front-facing size so the robot is placed
        # far enough from the counter to avoid initial collisions.
        ROBOT_FRONT_FACING_SIZE["tidyverse"] = 1.1

        _orig_robocasa_init = RoboCasaSceneBuilder.initialize

        def _patched_robocasa_initialize(self, env_idx, init_config_idxs=None):
            _orig_robocasa_init(self, env_idx, init_config_idxs)
            if self.env.robot_uids == "tidyverse" and self.env.agent is not None:
                # Use set_pose for world positioning, base qpos stays at 0.
                # Then reset the controller so its target matches qpos.
                qpos = self.env.agent.keyframes["rest"].qpos
                self.env.agent.robot.set_pose(self.robot_poses[env_idx])
                self.env.agent.robot.set_qpos(qpos)
                self.env.agent.reset(qpos)

        RoboCasaSceneBuilder.initialize = _patched_robocasa_initialize
    except ImportError:
        pass  # RoboCasa not available

_patch_scene_builders()

ASSET_DIR = os.path.dirname(os.path.abspath(__file__))


@register_agent()
class TidyVerse(BaseAgent):
    uid = "tidyverse"
    urdf_path = os.path.join(ASSET_DIR, "tidyverse.urdf")
    disable_self_collisions = True
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    # base: x, y, yaw
                    0.0, 0.0, 0.0,
                    # arm: 7 joints (home position)
                    0.0, -0.785, 0.0, -2.356, 0.0, 1.913, 0.785,
                    # robotiq: left_outer_knuckle, left_inner_knuckle,
                    #          left_inner_finger, right_outer_knuckle,
                    #          right_inner_knuckle, right_inner_finger
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ]
            ),
            pose=sapien.Pose(),
        )
    )

    # Arm joints (Panda)
    arm_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]

    # Robotiq 85 gripper joints
    gripper_joint_names = [
        "right_outer_knuckle_joint",
        "left_outer_knuckle_joint",
    ]
    passive_finger_joint_names = [
        "left_inner_knuckle_joint",
        "right_inner_knuckle_joint",
        "left_inner_finger_joint",
        "right_inner_finger_joint",
    ]

    ee_link_name = "eef"

    # Mobile base joints
    base_joint_names = [
        "base_x_joint",
        "base_y_joint",
        "base_yaw_joint",
    ]

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e5
    gripper_damping = 2000
    gripper_force_limit = 0.1
    gripper_friction = 1

    base_stiffness = 1e3
    base_damping = 520  # ~critical damping for ~67kg total mass: 2*sqrt(Kp*m)
    base_force_limit = 600

    @property
    def _controller_configs(self):
        # ---- Arm controllers ----
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        # ---- Robotiq 85 gripper ----
        passive_finger_joints = PassiveControllerConfig(
            joint_names=self.passive_finger_joint_names,
            damping=0,
            friction=0,
        )
        mimic_config = dict(
            left_outer_knuckle_joint=dict(
                joint="right_outer_knuckle_joint", multiplier=1.0, offset=0.0
            ),
        )
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=None,
            upper=None,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=False,
            mimic=mimic_config,
        )
        gripper_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            joint_names=self.gripper_joint_names,
            lower=-0.15,
            upper=0.15,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=True,
            use_delta=True,
            mimic=mimic_config,
        )

        # ---- Base controllers ----
        base_pd_joint_vel = PDJointVelControllerConfig(
            self.base_joint_names,
            lower=-1.0,
            upper=1.0,
            damping=self.base_damping,
            force_limit=self.base_force_limit,
        )
        base_pd_joint_pos = PDJointPosControllerConfig(
            self.base_joint_names,
            lower=None,
            upper=None,
            stiffness=self.base_stiffness,
            damping=self.base_damping,
            force_limit=self.base_force_limit,
            normalize_action=False,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper_active=gripper_pd_joint_delta_pos,
                gripper_passive=passive_finger_joints,
                base=base_pd_joint_vel,
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper_active=gripper_pd_joint_pos,
                gripper_passive=passive_finger_joints,
                base=base_pd_joint_vel,
            ),
            # Whole-body position control: arm + base both position-controlled
            whole_body=dict(
                arm=arm_pd_joint_pos,
                gripper_active=gripper_pd_joint_pos,
                gripper_passive=passive_finger_joints,
                base=base_pd_joint_pos,
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                gripper_active=gripper_pd_joint_pos,
                gripper_passive=passive_finger_joints,
                base=base_pd_joint_vel,
            ),
            pd_ee_pose=dict(
                arm=arm_pd_ee_pose,
                gripper_active=gripper_pd_joint_pos,
                gripper_passive=passive_finger_joints,
                base=base_pd_joint_vel,
            ),
        )

        return deepcopy_dict(controller_configs)

    def _after_loading_articulation(self):
        """Set up Robotiq closed-loop finger drives (same as xarm6_robotiq)."""
        outer_finger = self.robot.active_joints_map["right_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["right_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        p_f_right = [-1.6048949e-08, 3.7600022e-02, 4.3000020e-02]
        p_p_right = [1.3578170e-09, -1.7901104e-02, 6.5159947e-03]
        p_f_left = [-1.8080145e-08, 3.7600014e-02, 4.2999994e-02]
        p_p_left = [-1.4041154e-08, -1.7901093e-02, 6.5159872e-03]

        right_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_right), pad, sapien.Pose(p_p_right)
        )
        right_drive.set_limit_x(0, 0)
        right_drive.set_limit_y(0, 0)
        right_drive.set_limit_z(0, 0)

        outer_finger = self.robot.active_joints_map["left_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["left_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        left_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_left), pad, sapien.Pose(p_p_left)
        )
        left_drive.set_limit_x(0, 0)
        left_drive.set_limit_y(0, 0)
        left_drive.set_limit_z(0, 0)

        # Disable self-collisions between gripper links
        gripper_links = [
            "right_inner_knuckle",
            "right_outer_knuckle",
            "left_inner_knuckle",
            "left_outer_knuckle",
            "right_inner_finger_pad",
            "left_inner_finger_pad",
            "right_outer_finger",
            "left_outer_finger",
            "robotiq_arg2f_base_link",
            "right_inner_finger",
            "left_inner_finger",
            "panda_link8",
        ]
        for link_name in gripper_links:
            link = self.robot.links_map[link_name]
            link.set_collision_group_bit(group=2, bit_idx=31, bit=1)

    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_inner_finger_pad"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_inner_finger_pad"
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-6]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose
