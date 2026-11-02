"""
把 Isaac Lab 录下来的 hdf5 接触数据，转成每个 episode 一个 npz 文件。

hdf5 里每个 episode 的结构（由 PostStepContactRecorder 录下）：
    data/demo_0/contact/<身体名>/contact_pos_w   (T, 3)  接触点世界坐标，没接触点时是 nan
    data/demo_0/contact/<身体名>/force_z         (T,)    和地形之间的竖直接触力

输出的每个 npz 里有三样东西：
    body_names   (9,)        9 个身体的名字，顺序和下面两个数组的第 1 维一一对应
    contact      (T, 9)      bool，第 t 步第 b 个身体是否接触
    contact_pos_w  (T, 9, 3) float32，接触点的世界坐标（米）；没接触的地方是 nan

判断接触的四步（见 make_contact）：
    1. 竖直力 > 1 N 算接触
    2. 补上很短的空洞（默认 <= 2 步）
    3. 去掉很短的接触段（默认 < 10 步）
    4. 整理位置：没接触 -> nan；接触但这一步没位置 -> 用同一段里相邻步的位置补

用法：
    python hdf5_to_contact_npz.py dataset.hdf5 out_dir
    python hdf5_to_contact_npz.py dataset.hdf5 out_dir --force_thresh 1.0 --max_gap 2 --min_len 10
"""

import argparse
import os

import h5py
import numpy as np

BODY_NAMES = [
    "pelvis",
    "left_knee", "right_knee",
    "left_foot", "right_foot",
    "left_elbow", "right_elbow",
    "left_hand", "right_hand",
]


# ---------------------------------------------------------------------------
# 小工具：处理一个身体的接触序列 contact (T,) bool
# ---------------------------------------------------------------------------

def find_segments(contact):
    """找出所有连续接触段，返回 [(开始, 结束), ...]，结束那一步不包含在内。"""
    segments = []
    T = len(contact)
    t = 0
    while t < T:
        if contact[t]:
            start = t
            while t < T and contact[t]:
                t += 1
            segments.append((start, t))
        else:
            t += 1
    return segments


def fill_short_gaps(contact, max_gap):
    """两段接触之间如果只断开 <= max_gap 步，就把中间补成接触。返回 (新序列, 补了几个空洞)。"""
    contact = contact.copy()
    segments = find_segments(contact)
    n_filled = 0
    for (_, end_a), (start_b, _) in zip(segments[:-1], segments[1:]):
        gap = start_b - end_a
        if gap <= max_gap:
            contact[end_a:start_b] = True
            n_filled += 1
    return contact, n_filled


def remove_short_segments(contact, min_len):
    """连续接触少于 min_len 步的段改成没接触。返回 (新序列, 去掉了几段)。"""
    contact = contact.copy()
    n_removed = 0
    for start, end in find_segments(contact):
        if end - start < min_len:
            contact[start:end] = False
            n_removed += 1
    return contact, n_removed


def fill_missing_pos(pos, contact):
    """接触段里如果有几步位置是 nan，用同一段里前一步（没有就用后一步）的位置补。返回补了几步。"""
    n_filled = 0
    for start, end in find_segments(contact):
        seg = pos[start:end]                      # 这是 pos 的一部分，改它就是改 pos
        valid = ~np.isnan(seg).any(axis=-1)
        if valid.all():
            continue
        if not valid.any():
            continue                              # 整段都没有位置，留给后面处理
        # 先往后填：用前一个有效位置
        for i in range(1, len(seg)):
            if not valid[i] and valid[i - 1]:
                seg[i] = seg[i - 1]
                valid[i] = True
                n_filled += 1
        # 再往前填：段开头如果是 nan，用后一个有效位置
        for i in range(len(seg) - 2, -1, -1):
            if not valid[i] and valid[i + 1]:
                seg[i] = seg[i + 1]
                valid[i] = True
                n_filled += 1
    return n_filled


# ---------------------------------------------------------------------------
# 主要逻辑
# ---------------------------------------------------------------------------

def make_contact(pos, force_z, force_thresh, max_gap, min_len):
    """一个身体：从 (T,3) 位置和 (T,) 竖直力，得到 (T,) 接触标志和整理好的 (T,3) 位置。"""
    pos = pos.copy()

    # 第 1 步：竖直力超过阈值就算接触
    contact = np.abs(force_z) > force_thresh

    # 第 2 步：补上很短的空洞（只取当前物理步的力，偶尔会断一两步）
    contact, n_gap = fill_short_gaps(contact, max_gap)

    # 第 3 步：去掉很短的接触段
    contact, n_short = remove_short_segments(contact, min_len)

    # 第 4 步：整理位置
    pos[~contact] = np.nan                        # 没接触 -> nan
    n_pos = fill_missing_pos(pos, contact)        # 接触但没位置 -> 用相邻步补

    # 万一某段整段都没有位置（理论上不会发生），这段就不算接触
    no_pos = contact & np.isnan(pos).any(axis=-1)
    contact[no_pos] = False

    stats = dict(gaps_filled=n_gap, short_removed=n_short, pos_filled=n_pos, dropped_no_pos=int(no_pos.sum()))
    return contact, pos, stats


def convert_demo(demo, force_thresh, max_gap, min_len):
    """把 hdf5 里的一个 demo（一个 episode）转成三个数组。"""
    if "contact" not in demo:
        raise KeyError("这个 demo 里没有 'contact'，检查 PostStepContactRecorder 有没有登记")
    group = demo["contact"]

    contact_list, pos_list, stats_list = [], [], []
    for name in BODY_NAMES:
        if name not in group:
            raise KeyError(f"hdf5 里找不到 'contact/{name}'")
        pos = np.asarray(group[name]["contact_pos_w"], dtype=np.float32)   # (T, 3)
        force_z = np.asarray(group[name]["force_z"], dtype=np.float32)    # (T,)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"'contact/{name}/contact_pos_w' 形状是 {pos.shape}，应该是 (T, 3)")

        c, p, s = make_contact(pos, force_z, force_thresh, max_gap, min_len)
        contact_list.append(c)
        pos_list.append(p)
        stats_list.append(s)

    contact = np.stack(contact_list, axis=1)        # (T, 9)
    contact_pos = np.stack(pos_list, axis=1)        # (T, 9, 3)
    return np.array(BODY_NAMES), contact, contact_pos, stats_list


def main():
    parser = argparse.ArgumentParser(description="hdf5 接触数据 -> 每个 episode 一个 npz")
    parser.add_argument("hdf5_path", help="Isaac Lab 录下来的 hdf5 文件")
    parser.add_argument("out_dir", help="输出 npz 的文件夹")
    parser.add_argument("--force_thresh", type=float, default=1.0, help="竖直力超过多少牛算接触")
    parser.add_argument("--max_gap", type=int, default=2, help="断开不超过这么多步的空洞会被补上；0 表示不补")
    parser.add_argument("--min_len", type=int, default=10, help="短于这么多步的接触段会被去掉；0 表示不去")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.hdf5_path, "r") as f:
        for demo_name in f["data"]:
            body_names, contact, contact_pos, stats = convert_demo(
                f["data"][demo_name], args.force_thresh, args.max_gap, args.min_len
            )
            out_path = os.path.join(args.out_dir, f"{demo_name}.npz")
            np.savez(out_path, body_names=body_names, contact=contact, contact_pos_w=contact_pos)

            print(f"{demo_name}: {contact.shape[0]} 步 -> {out_path}")
            for b, name in enumerate(body_names):
                n_seg = len(find_segments(contact[:, b]))
                s = stats[b]
                print(
                    f"    {name:12s} 接触 {int(contact[:, b].sum()):4d} 步, {n_seg} 段 | "
                    f"补空洞 {s['gaps_filled']}, 去短段 {s['short_removed']}, 补位置 {s['pos_filled']}"
                )


if __name__ == "__main__":
    main()