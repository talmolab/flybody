import h5py
import numpy as np

_RAT_MOCAP_JOINTS = [
    'vertebra_1_extend', 'vertebra_2_bend', 'vertebra_3_twist',
    'vertebra_4_extend', 'vertebra_5_bend', 'vertebra_6_twist',
    'hip_L_supinate', 'hip_L_abduct', 'hip_L_extend', 'knee_L', 'ankle_L',
    'toe_L', 'hip_R_supinate', 'hip_R_abduct', 'hip_R_extend', 'knee_R',
    'ankle_R', 'toe_R', 'vertebra_C1_extend', 'vertebra_C1_bend',
    'vertebra_C2_extend', 'vertebra_C2_bend', 'vertebra_C3_extend',
    'vertebra_C3_bend', 'vertebra_C4_extend', 'vertebra_C4_bend',
    'vertebra_C5_extend', 'vertebra_C5_bend', 'vertebra_C6_extend',
    'vertebra_C6_bend', 'vertebra_C7_extend', 'vertebra_C9_bend',
    'vertebra_C11_extend', 'vertebra_C13_bend', 'vertebra_C15_extend',
    'vertebra_C17_bend', 'vertebra_C19_extend', 'vertebra_C21_bend',
    'vertebra_C23_extend', 'vertebra_C25_bend', 'vertebra_C27_extend',
    'vertebra_C29_bend', 'vertebra_cervical_5_extend',
    'vertebra_cervical_4_bend', 'vertebra_cervical_3_twist',
    'vertebra_cervical_2_extend', 'vertebra_cervical_1_bend',
    'vertebra_axis_twist', 'vertebra_atlant_extend', 'atlas', 'mandible',
    'scapula_L_supinate', 'scapula_L_abduct', 'scapula_L_extend', 'shoulder_L',
    'shoulder_sup_L', 'elbow_L', 'wrist_L', 'finger_L', 'scapula_R_supinate',
    'scapula_R_abduct', 'scapula_R_extend', 'shoulder_R', 'shoulder_sup_R',
    'elbow_R', 'wrist_R', 'finger_R'
]

_RAT_MOCAP_BODY = [
    "torso","pelvis","upper_leg_L",
    "lower_leg_L","foot_L","upper_leg_R",
    "lower_leg_R","foot_R","skull","jaw",
    "scapula_L","upper_arm_L","lower_arm_L",
    "finger_L","scapula_R","upper_arm_R","lower_arm_R","finger_R"]

def read_h5_file(filename):
    '''Read everything in the h5 file'''
    
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
    '''Read only the ids in the h5 file'''
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


def extract_feature(input_path, output_path):
    '''Main function for data conversion from STAC to DMC format'''
    with h5py.File(input_path, 'r') as input_file:
        with h5py.File(output_path, 'w') as output_file:

            # id2name group
            id2name_group = output_file.create_group('id2name')
            id2name_group.create_dataset('joints', data=np.array(_RAT_MOCAP_JOINTS, dtype='S')) #shape: (67)
            id2name_group.create_dataset('qpos', data=np.array(_RAT_MOCAP_JOINTS, dtype='S')) #shape: (67)
            id2name_group.create_dataset('sites', data=np.array(_RAT_MOCAP_BODY, dtype='S')) #shape: (18)

            # Timestep_seconds dataset
            output_file.create_dataset('timestep_seconds', data=0.02)

            # Trajectories group
            trajectories_group = output_file.create_group('trajectories')
            clip_lst = list(input_file.keys())

            trajectory_lengths = []
            
            for clip_key in clip_lst:
                # print(f'Now processing {clip_key}')

                if clip_key in input_file:
                    walkers_group = input_file[f"{clip_key}/walkers"]
                    n_traj = len(walkers_group)
                    n_zeros = len(str(n_traj))
                    
                    key = str(clip_key)[5:]
                    
                    position = walkers_group[f"walker_0/position"][()] #shape: (3, 250)
                    quaternion = walkers_group[f"walker_0/quaternion"][()] #shape: (4, 250)
                    center_of_mass = walkers_group[f"walker_0/center_of_mass"][()] #shape: (3, 250)
                    joints = walkers_group[f"walker_0/joints"][()] #shape: (67, 250)

                    body_positions = walkers_group[f"walker_0/body_positions"][()] #shape: (54, 250)
                    body_quaternions = walkers_group[f"walker_0/body_quaternions"][()] #shape: (72, 250)
                    
                    velocity = walkers_group[f"walker_0/velocity"][()] #shape: (3, 250)
                    joints_velocity = walkers_group[f"walker_0/joints_velocity"][()] #shape: (67, 250)
                    angular_velocity = walkers_group[f"walker_0/angular_velocity"][()] #shape: (3, 250)
                    
                    
                    # qpos is just joints here
                    qpos = np.hstack((position.T, quaternion.T, joints.T))
                    qvel = np.hstack((velocity.T, angular_velocity.T, joints_velocity.T))
                    joint_quat = body_quaternions.T

                    # Dummy data for other not-sure data
                    root_qpos = center_of_mass.T
                    root_qvel = angular_velocity.T
                    root2site = np.zeros((body_positions.shape[1], 6, 3), dtype=np.float32)

                    trajectory_length = body_positions.shape[1]
                    trajectory_lengths.append(trajectory_length)

                    # Group for each trajectory
                    traj_group = trajectories_group.create_group(key)
                    traj_group.create_dataset("qpos", data=qpos.astype(np.float32))
                    traj_group.create_dataset("qvel", data=qvel.astype(np.float32))
                    traj_group.create_dataset("joint_quat", data=joint_quat.astype(np.float32))
                    traj_group.create_dataset("root_qpos", data=root_qpos.astype(np.float32))
                    traj_group.create_dataset("root2site", data=root2site.astype(np.float32))
                    traj_group.create_dataset("root_qvel", data=root_qvel.astype(np.float32))
            
            output_file.create_dataset('trajectory_lengths', data=np.array(trajectory_lengths, dtype=np.int64))