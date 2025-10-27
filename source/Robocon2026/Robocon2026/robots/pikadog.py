from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from Robocon2026.utils.utils import euler2quaternion

import isaaclab.sim as sim_utils


"""Configuration for the Pikadog robot: two SO101 arms fixed on a Unitree Go2."""

PIKADOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/ArmDog/pikadog.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        # pos=(5.45, 5.575, 0.4),
        # rot=euler2quaternion([-90, 0, 0]),
        joint_pos={
            # Go2腿部
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            # 前臂
            "front_shoulder_pan": 0.0,
            "front_shoulder_lift": 0.0,
            "front_elbow_flex": 0.0,
            "front_wrist_flex": 0.0,
            "front_wrist_roll": 0.0,
            "front_gripper": 0.0,
            # 后臂
            "back_shoulder_pan": 0.0,
            "back_shoulder_lift": 0.0,
            "back_elbow_flex": 0.0,
            "back_wrist_flex": 0.0,
            "back_wrist_roll": 0.0,
            "back_gripper": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Go2腿部
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        # 前臂
        "front_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "front_shoulder_pan",
                "front_shoulder_lift",
                "front_elbow_flex",
                "front_wrist_flex",
                "front_wrist_roll",
            ],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
        "front_gripper": ImplicitActuatorCfg(
            joint_names_expr=["front_gripper"],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
        # 后臂
        "back_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "back_shoulder_pan",
                "back_shoulder_lift",
                "back_elbow_flex",
                "back_wrist_flex",
                "back_wrist_roll",
            ],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
        "back_gripper": ImplicitActuatorCfg(
            joint_names_expr=["back_gripper"],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
    },
)

# joint limit written in USD (degree)
SO101_FOLLOWER_USD_JOINT_LIMLITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10, 100.0),
}

# motor limit written in real device (normalized to related range)
SO101_FOLLOWER_MOTOR_LIMITS = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}


SO101_FOLLOWER_REST_POSE_RANGE = {
    "shoulder_pan": (0 - 30.0, 0 + 30.0),  # 0 degree
    "shoulder_lift": (-100.0 - 30.0, -100.0 + 30.0),  # -100 degree
    "elbow_flex": (90.0 - 30.0, 90.0 + 30.0),  # 90 degree
    "wrist_flex": (50.0 - 30.0, 50.0 + 30.0),  # 50 degree
    "wrist_roll": (0.0 - 30.0, 0.0 + 30.0),  # 0 degree
    "gripper": (-10.0 - 30.0, -10.0 + 30.0),  # -10 degree
}
