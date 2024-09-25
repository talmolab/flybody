"""For transforming dataset from STAC to DMC format"""

import h5py
import numpy as np
from vnl_ray import quaternions
from dm_control import mjcf

_RAT_MOCAP_JOINTS = [
    "vertebra_1_extend",
    "vertebra_2_bend",
    "vertebra_3_twist",
    "vertebra_4_extend",
    "vertebra_5_bend",
    "vertebra_6_twist",
    "hip_L_supinate",
    "hip_L_abduct",
    "hip_L_extend",
    "knee_L",
    "ankle_L",
    "toe_L",
    "hip_R_supinate",
    "hip_R_abduct",
    "hip_R_extend",
    "knee_R",
    "ankle_R",
    "toe_R",
    "vertebra_C1_extend",
    "vertebra_C1_bend",
    "vertebra_C2_extend",
    "vertebra_C2_bend",
    "vertebra_C3_extend",
    "vertebra_C3_bend",
    "vertebra_C4_extend",
    "vertebra_C4_bend",
    "vertebra_C5_extend",
    "vertebra_C5_bend",
    "vertebra_C6_extend",
    "vertebra_C6_bend",
    "vertebra_C7_extend",
    "vertebra_C9_bend",
    "vertebra_C11_extend",
    "vertebra_C13_bend",
    "vertebra_C15_extend",
    "vertebra_C17_bend",
    "vertebra_C19_extend",
    "vertebra_C21_bend",
    "vertebra_C23_extend",
    "vertebra_C25_bend",
    "vertebra_C27_extend",
    "vertebra_C29_bend",
    "vertebra_cervical_5_extend",
    "vertebra_cervical_4_bend",
    "vertebra_cervical_3_twist",
    "vertebra_cervical_2_extend",
    "vertebra_cervical_1_bend",
    "vertebra_axis_twist",
    "vertebra_atlant_extend",
    "atlas",
    "mandible",
    "scapula_L_supinate",
    "scapula_L_abduct",
    "scapula_L_extend",
    "shoulder_L",
    "shoulder_sup_L",
    "elbow_L",
    "wrist_L",
    "finger_L",
    "scapula_R_supinate",
    "scapula_R_abduct",
    "scapula_R_extend",
    "shoulder_R",
    "shoulder_sup_R",
    "elbow_R",
    "wrist_R",
    "finger_R",
]
_RAT_MOCAP_BODY = [
    "torso",
    "pelvis",
    "upper_leg_L",
    "lower_leg_L",
    "foot_L",
    "upper_leg_R",
    "lower_leg_R",
    "foot_R",
    "skull",
    "jaw",
    "scapula_L",
    "upper_arm_L",
    "lower_arm_L",
    "finger_L",
    "scapula_R",
    "upper_arm_R",
    "lower_arm_R",
    "finger_R",
]
_RAT_MOCAP_SITE = [
    "hip_L",
    "hip_R",
    "knee_L",
    "ankle_L",
    "toe_L",
    "sole_L",
    "knee_R",
    "ankle_R",
    "toe_R",
    "sole_R",
    "head",
    "shoulder_L",
    "elbow_L",
    "wrist_L",
    "finger_L",
    "palm_L",
    "shoulder_R",
    "elbow_R",
    "wrist_R",
    "finger_R",
    "palm_R",
]


def read_h5_file(filename):
    """Read everything in the h5 file and print only shapes"""
    with h5py.File(filename, "r") as hf:

        def print_attrs(name, obj):
            print(f"{name}:")
            for key, val in obj.attrs.items():
                print(f"    {key}: {val}")

        def print_dataset(name, obj):
            print(f"Dataset: {name}")
            print(f"    shape: {obj.shape}")
            print(f"    dtype: {obj.dtype}")

        def visit_items(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
                print_attrs(name, obj)
            elif isinstance(obj, h5py.Dataset):
                print_dataset(name, obj)

        hf.visititems(visit_items)


def read_id(filename):
    """Read only the ids in the h5 file and print raw data"""
    with h5py.File(filename, "r") as hf:

        def print_attrs(name, obj):
            print(f"{name}:")
            for key, val in obj.attrs.items():
                print(f"    {key}: {val}")

        def print_dataset(name, obj):
            print(f"Dataset: {name}")
            print(f"    shape: {obj.shape}")
            print(f"    dtype: {obj.dtype}")
            print("    data:")
            print(obj[...])

        def visit_items(name, obj):
            if isinstance(obj, h5py.Group):
                if "root2site" in name or "id2name" in name:
                    print(f"Group: {name}")
                    print_attrs(name, obj)
            elif isinstance(obj, h5py.Dataset):
                if "root2site" in name or "id2name" in name:
                    print_dataset(name, obj)

        hf.visititems(visit_items)


def extract_feature(
    input_path,
    output_path,
    xml_path="/root/talmolab-smb/kaiwen/vnl_ray/vnl_ray/fruitfly/assets_rodent/rodent.xml",
):
    """Main function for data conversion from STAC to DMC format"""

    xml = mjcf.from_path(xml_path)
    physics = mjcf.Physics.from_mjcf_model(xml)
    rat_joints = xml.find_all("joint")
    rat_sites = xml.find_all("site")

    with h5py.File(input_path, "r") as input_file:
        with h5py.File(output_path, "w") as output_file:

            id2name_group = output_file.create_group("id2name")
            id2name_group.create_dataset("joints", data=np.array(_RAT_MOCAP_JOINTS, dtype="S"))  # shape: (67)
            id2name_group.create_dataset("qpos", data=np.array(_RAT_MOCAP_JOINTS, dtype="S"))  # shape: (67)
            id2name_group.create_dataset("sites", data=np.array(_RAT_MOCAP_SITE, dtype="S"))  # shape: (21)

            output_file.create_dataset("timestep_seconds", data=0.02)
            trajectories_group = output_file.create_group("trajectories")
            clip_lst = list(input_file.keys())

            trajectory_lengths = []
            for clip_key in clip_lst:
                if clip_key in input_file:
                    walkers_group = input_file[f"{clip_key}/walkers"]
                    key = str(clip_key)[5:]

                    position = walkers_group[f"walker_0/position"][()]  # shape: (3, 250)
                    quaternion = walkers_group[f"walker_0/quaternion"][()]  # shape: (4, 250)
                    center_of_mass = walkers_group[f"walker_0/center_of_mass"][()]  # shape: (3, 250)
                    angular_velocity = walkers_group[f"walker_0/angular_velocity"][()]  # shape: (3, 250)
                    velocity = walkers_group[f"walker_0/velocity"][()]  # shape: (3, 250)

                    joints = walkers_group[f"walker_0/joints"][()]  # shape: (67, 250)
                    joints_velocity = walkers_group[f"walker_0/joints_velocity"][()]  # shape: (67, 250)
                    body_positions = walkers_group[f"walker_0/body_positions"][()]  # shape: (54, 250)
                    body_quaternions = walkers_group[f"walker_0/body_quaternions"][()]  # shape: (72, 250)

                    # qpos root at index 0
                    qpos = np.hstack((quaternion.T, joints.T))
                    qvel = np.hstack((velocity.T, angular_velocity.T, joints_velocity.T))
                    sites = position.T
                    root2site = quaternions.get_egocentric_vec(qpos[:, :3], sites, qpos[:, 3:7])

                    # Joint quaternions in local egocentric reference frame, except root quaternion,
                    # which is in world reference frame

                    root_quat = quaternion.T  # shape: (250, 4)

                    # TODO: find the correct batched xaxis
                    # xaxis1 = physics.bind(rat_joints).xaxis[1:, :] # point directions for 66 joints, neglect root
                    # xaxis1 = quaternions.rotate_vec_with_quat(xaxis1, quaternions.reciprocal_quat(root_quat)) # rotate_vec take cares of tile

                    # joint_quat = quaternions.joint_orientation_quat(xaxis1, qpos[:,8:]) #shape: (250, 66)
                    # joint_quat = np.vstack((root_quat, joint_quat)) # stack root with rest of joints, shape: (250, 67)

                    joint_quat = np.arange(67 * 250).reshape(250, 67)

                    root_qpos = center_of_mass.T
                    root_qvel = angular_velocity.T

                    trajectory_length = body_positions.shape[1]
                    trajectory_lengths.append(trajectory_length)

                    traj_group = trajectories_group.create_group(key)
                    traj_group.create_dataset("qpos", data=qpos.astype(np.float32))
                    traj_group.create_dataset("qvel", data=qvel.astype(np.float32))
                    traj_group.create_dataset("joint_quat", data=joint_quat.astype(np.float32))
                    traj_group.create_dataset("root_qpos", data=root_qpos.astype(np.float32))
                    traj_group.create_dataset("root2site", data=root2site.astype(np.float32))
                    traj_group.create_dataset("root_qvel", data=root_qvel.astype(np.float32))

            output_file.create_dataset("trajectory_lengths", data=np.array(trajectory_lengths, dtype=np.int64))
