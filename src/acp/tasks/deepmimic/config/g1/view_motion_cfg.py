from isaaclab.utils import configclass

from acp.assets.robots.unitree import G1_REFERENCE_CFG
from acp.tasks.deepmimic.config.g1.deepmimic_tracking_cfg import G1PerceptiveShadowingEnvCfg


@configclass
class ViewMotionEnvCfg(G1PerceptiveShadowingEnvCfg):
    """只用来看数据集：影子机器人播放动作，站在配套的地形上。"""

    def __post_init__(self):
        super().__post_init__()

        # ① 放一个"影子机器人"，并打开动作显示
        self.scene.robot_reference = G1_REFERENCE_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference")
        self.scene.motion_reference = self.scene.motion_reference.replace(debug_vis=True)

        # ② 影子放在动作的真实位置；真机器人挪到旁边 1 米、上方 2 米，不挡视线
        self.scene.motion_reference.visualizing_robot_offset = (0.0, 0.0, 0.0)
        self.events.reset_robot.params["position_offset"] = [0.0, 1.0, 2.0]

        # ③ 每段动作从头播到尾，中途不重置
        self.episode_length_s = 60.0
        self.curriculum.beyond_adaptive_sampling = None
        self.events.bin_fail_counter_smoothing = None
        motion = self.scene.motion_reference.motion_buffers["TerrainMotion"]
        motion.motion_start_from_middle_range = [0.0, 0.0]
        motion.motion_bin_length_s = None
        motion.env_starting_stub_sampling_strategy = "independent"
        self.terminations.base_pos_too_far = None
        self.terminations.base_pg_too_far = None
        self.terminations.link_pos_too_far = None