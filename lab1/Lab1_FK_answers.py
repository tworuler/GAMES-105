import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def parse_joint(lines, line_index, parent, joint_name, joint_parent, joint_offset):
    index = len(joint_name)
    joint_type, name = lines[line_index].strip().split()
    if joint_type == 'ROOT':
        name = 'RootJoint'
    elif joint_type == 'End':
        name = joint_name[parent] + 'End'
    joint_name.append(name)
    joint_parent.append(parent)
    joint_offset.append([float(x) for x in lines[line_index + 2].strip().split()[1:]])
    line_count = 3 if joint_type == "End" else 4
    while lines[line_index + line_count].strip() != '}':
        new_line_count = parse_joint(lines, line_index + line_count, index, joint_name, joint_parent, joint_offset)
        line_count += new_line_count + 1
    return line_count


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = None
    joint_parent = None
    joint_offset = None

    lines = []
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()

    joint_name = []
    joint_parent = []
    joint_offset = []
    parse_joint(lines, 1, -1, joint_name, joint_parent, joint_offset)
    joint_offset = np.array(joint_offset)

    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = None
    joint_orientations = None

    joint_positions = []
    joint_orientations = []
    motion = motion_data[frame_id]
    index = 0
    for name, parent, offset in zip(joint_name, joint_parent, joint_offset):
        if parent == -1:
            position = motion[index:index + 3]
            orientation = R.from_euler('XYZ', motion[index + 3:index + 6], degrees=True)
            joint_positions.append(position)
            joint_orientations.append(orientation)
            index += 6
        else:
            position = joint_orientations[parent].as_matrix() @ offset + joint_positions[parent]
            euler = motion[index : index + 3] if not name.endswith('End') else [0, 0, 0]
            orientation = joint_orientations[parent] * R.from_euler('XYZ', euler, degrees=True)
            joint_positions.append(position)
            joint_orientations.append(orientation)
            if not name.endswith('End'):
                index += 3

    joint_positions = np.array(joint_positions)
    joint_orientations = np.array([r.as_quat() for r in joint_orientations])

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = None
    A_motion_data = load_motion_data(A_pose_bvh_path)
    T_joint_name, _, _ = part1_calculate_T_pose(T_pose_bvh_path)
    A_joint_name, _, _ = part1_calculate_T_pose(A_pose_bvh_path)
    A_name_index = {}
    index = 0
    for name in A_joint_name:
        if not name.endswith('End'):
            A_name_index[name] = index
            index += 1
    motion_data = [[] for _ in range(len(A_motion_data))]
    for i in range(len(T_joint_name)):
        name = T_joint_name[i]
        if name.endswith('End'):
            continue
        if i == 0:
            for j in range(len(motion_data)):
                motion_data[j] += list(A_motion_data[j][A_name_index[name] * 3 : A_name_index[name] * 3 + 6])
        else:
            for j in range(len(motion_data)):
                motion_data[j] += list(A_motion_data[j][A_name_index[name] * 3 + 3 : A_name_index[name] * 3 + 6])
        if name == 'lShoulder':
            for motion in motion_data:
                motion[-1] -= 45
        elif name == 'rShoulder':
            for motion in motion_data:
                motion[-1] += 45
    motion_data = np.array(motion_data)
    return motion_data
