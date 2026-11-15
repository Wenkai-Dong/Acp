"""
把 Isaac Lab 录制器导出的 hdf5 转换成 InstinctLab 能读的 *_retargeted.npz。
唯一的规则：基座离地形边缘不到 EDGE_DIST 的帧不要（去掉开头从站立到起步的那一段）。

用法：
    python hdf5_to_retargeted.py dataset.hdf5 输出文件夹 --prefix ame_stones_d095

每个 demo 输出一个 <prefix>_<demo名>_retargeted.npz。
"""

import argparse
import os

import h5py
import numpy as np

# robot.joint_names（你打印出来的），顺序和 joint_position 的 29 列一一对应
JOINT_NAMES = [
    "waist_yaw_joint", "left_hip_pitch_joint", "right_hip_pitch_joint",
    "waist_roll_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "waist_pitch_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
TILE_SIZE = (9.0, 12.0)  # 一块地形的大小 (x, y)，坐标原点在地形中心
EDGE_DIST = 0.5          # 基座离地形边缘不到这么远的帧不要


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="ame")
    args = parser.parse_args()

    id = "3"
    hdf5_dir = f"D:/acp_data/motion_data/vault/vault_over_R_00{id}__A533/dataset.hdf5"
    out_dir = f"D:/acp_data/motion_data/vault/vault_over_R_00{id}__A533"

    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(hdf5_dir, "r") as f:
        for name, demo in f["data"].items():
            robot = demo["states/articulation/robot"]
            root_pose = robot["root_pose"][:]  # (N, 7)：x, y, z, qx, qy, qz, qw
            joint_pos = robot["joint_position"][:]  # (N, 29)：列顺序 = JOINT_NAMES
            framerate = float(demo["framerate"][0, 0])  # 50.0

            # 规则：基座离地形边缘不到 EDGE_DIST 的帧不要（去掉开头从站立到起步的那一段）
            # 基座到最近一条边的距离 = min(x 方向到边的距离, y 方向到边的距离)
            dist = np.minimum(TILE_SIZE[0] / 2 - np.abs(root_pose[:, 0]), TILE_SIZE[1] / 2 - np.abs(root_pose[:, 1]))
            far = dist >= EDGE_DIST
            if not far.any():
                print(f"{name}: 基座一直离边缘不到 {EDGE_DIST} m，跳过")
                continue
            start = int(np.argmax(far))  # 第一次离开边缘区域的帧
            back = np.flatnonzero(~far[start:])  # 之后再次靠近边缘的帧
            end = start + int(back[0]) if len(back) else len(far)  # 只保留中间连续的一段
            root_pose, joint_pos = root_pose[start:end], joint_pos[start:end]

            base_pos_w = root_pose[:, 0:3]  # 位置：录的时候用了 is_relative=True，已经是地形自己的坐标
            base_quat_w = root_pose[:, [6, 3, 4, 5]]  # 四元数：Isaac Lab 3 的 xyzw 换成 wxyz

            out_path = os.path.join(out_dir, f"vault_over_R_00{id}__A533_retargeted")
            np.savez(
                out_path,
                framerate=np.array(framerate),
                joint_names=np.array(JOINT_NAMES),
                joint_pos=joint_pos,
                base_pos_w=base_pos_w,
                base_quat_w=base_quat_w,
                source_frames=np.array([start, end]),  # 保留的是原始数据的第 start 到 end-1 帧，接触标签按这个截
            )
            print(f"{name}: 保留原始第 {start}~{end - 1} 帧，共 {end - start} 帧 -> {out_path}")


if __name__ == "__main__":
    main()