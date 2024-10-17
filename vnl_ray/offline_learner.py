import ray
from ray.util.scheduling_strategies import (
    PlacementGroupSchedulingStrategy,
    NodeAffinitySchedulingStrategy,
)
from ray.util.placement_group import placement_group
import logging
import os
import pandas as pd
import tables

os.environ["RAY_memory_usage_threshold"] = "1"

try:
    # Try connecting to existing Ray cluster.
    ray_context = ray.init(
        address="auto",
        include_dashboard=True,
        dashboard_host="0.0.0.0",
        logging_level=logging.INFO,
    )
except:
    # Spin up new Ray cluster.
    ray_context = ray.init(include_dashboard=True, dashboard_host="0.0.0.0", logging_level=logging.INFO)


import time
import os
import dataclasses
import uuid
import hydra
import functools
from omegaconf import DictConfig, OmegaConf
import numpy as np
from acme import specs
from acme import wrappers
from acme.tf import utils as tf2_utils
import sonnet as snt
import tensorflow as tf
from dm_control import composer
import imageio

import vnl_ray
from vnl_ray.agents.remote_as_local_wrapper import RemoteAsLocal
from vnl_ray.agents.counting import PicklableCounter
from vnl_ray.agents.network_factory import policy_loss_module_dmpo
from vnl_ray.agents.losses_mpo import PenalizationCostRealActions
from vnl_ray.tasks.basic_rodent_2020 import (
    rodent_run_gaps,
    rodent_maze_forage,
    rodent_escape_bowl,
    rodent_two_touch,
    walk_humanoid,
    rodent_walk_imitation,
)

from vnl_ray.tasks.mouse_reach import mouse_reach

from vnl_ray.fly_envs import (
    walk_on_ball,
    vision_guided_flight,
    walk_imitation as fly_walk_imitation,
)
from vnl_ray.default_logger import make_default_logger
from vnl_ray.single_precision import SinglePrecisionWrapper
from vnl_ray.agents.network_factory import make_network_factory_dmpo
from vnl_ray.agents.intention_network_factory import (
    make_network_factory_dmpo as make_network_factory_dmpo_intention,
)
from vnl_ray.tasks.task_utils import get_task_obs_size
from vnl_ray.utils import plot_reward
from collections import defaultdict

def render_with_rewards(env, policy, observation_spec, rollout_length=150):
    """
    render with the rewards progression graph concat alongside with the rendering
    """
    frames, reset_idx, reward_channels, activation_collection, kinematics_collection = render_with_rewards_info(env, policy, observation_spec, rollout_length=rollout_length)
    rewards = defaultdict(list)
    reward_keys = env.task._reward_keys
    for key in reward_keys:
        rewards[key] += [rcs[key] for rcs in reward_channels]
    concat_frames = []
    episode_start = 0
     # implement reset logics of the reward graph too.
    for idx, frame in enumerate(frames):
        if len(reset_idx) != 0 and idx == reset_idx[0]:
            reward_plot = plot_reward(idx, episode_start, rewards, terminated=True)
            for _ in range(50):
                concat_frames.append(np.hstack([frame, reward_plot])) # create stoppage when episode terminates
            reset_idx.pop(0)
            episode_start=idx
            continue
        concat_frames.append(np.hstack([frame, plot_reward(idx, episode_start, rewards)]))
    return concat_frames, activation_collection, kinematics_collection

def render_with_rewards_info(env, policy, observation_spec, rollout_length=150, render_vision_if_available=True):
    """
    Generate rollout with the reward-related information and collect kinematic data
    for all geoms, bodies, joint angles, velocities, and accelerations.
    """
    reward_channels = []
    frames = []
    reset_idx = []
    timestep = env.reset()
    activation_collection = []
    kinematics_collection = []
    
    prev_geom_positions = None  # To compute velocity as finite difference
    prev_body_positions = None  # To compute velocity for bodies
    
    render_kwargs = {"width": 640, "height": 480}
    
    # Get all geom names
    geom_names = [env.physics.model.id2name(i, 'geom') for i in range(env.physics.model.ngeom)]
    
    # Get all body names
    body_names = [env.physics.model.id2name(i, 'body') for i in range(env.physics.model.nbody)]
    
    # Get joint names
    joint_names = [env.physics.model.id2name(i, 'joint') for i in range(env.physics.model.njnt)]

    for i in range(rollout_length):
        pixels = env.physics.render(camera_id=1, **render_kwargs)
        frames.append(pixels)

        # Collect kinematic data for geoms
        geom_positions = {}
        geom_velocities = {}
        geom_accelerations = {}

        for geom_name in geom_names:
            try:
                geom_positions[geom_name] = env.physics.named.data.geom_xpos[geom_name].copy()
                
                # Calculate velocity as finite difference between positions
                if prev_geom_positions:
                    geom_velocities[geom_name] = geom_positions[geom_name] - prev_geom_positions[geom_name]
                else:
                    geom_velocities[geom_name] = np.zeros(3)  # No velocity for the first step

                # Acceleration as difference of velocities
                if prev_geom_positions:
                    geom_accelerations[geom_name] = geom_velocities[geom_name]  # Current velocity as acceleration for first difference
                else:
                    geom_accelerations[geom_name] = np.zeros(3)  # No acceleration for the first step
            except KeyError:
                print(f"Warning: Geom '{geom_name}' not found.")
        
        # Collect kinematic data for bodies (use position difference to estimate velocity)
        body_positions = {}
        body_velocities = {}
        body_accelerations = {}
        body_orientations = {}

        for body_name in body_names:
            try:
                body_positions[body_name] = env.physics.named.data.xpos[body_name].copy()
                
                # Calculate velocity as finite difference between positions
                if prev_body_positions:
                    body_velocities[body_name] = body_positions[body_name] - prev_body_positions[body_name]
                else:
                    body_velocities[body_name] = np.zeros(3)  # No velocity for the first step

                # Acceleration as difference of velocities
                if prev_body_positions:
                    body_accelerations[body_name] = body_velocities[body_name]  # Current velocity as acceleration for first difference
                else:
                    body_accelerations[body_name] = np.zeros(3)  # No acceleration for the first step

                # Orientation using quaternions or rotation matrix
                body_orientations[body_name] = env.physics.named.data.xquat[body_name].copy()  # Or use xmat for rotation matrix
            except KeyError:
                print(f"Warning: Body '{body_name}' not found.")
        
        # Collect joint angles
        joint_angles = {}
        for joint_name in joint_names:
            try:
                joint_angles[joint_name] = env.physics.named.data.qpos[joint_name].copy()
            except KeyError:
                print(f"Warning: Joint '{joint_name}' not found.")
        
        # Store positions, velocities, accelerations, orientations, and joint angles for this timestep
        kinematics_collection.append({
            "geoms": {
                "positions": geom_positions,
                "velocities": geom_velocities,
                "accelerations": geom_accelerations,
            },
            "bodies": {
                "positions": body_positions,
                "velocities": body_velocities,
                "accelerations": body_accelerations,
                "orientations": body_orientations,
            },
            "joints": joint_angles
        })
        
        # Update previous positions for velocity and acceleration calculation
        prev_geom_positions = geom_positions
        prev_body_positions = body_positions

        inputs = tf2_utils.add_batch_dim(timestep.observation)
        inputs = tf2_utils.batch_concat(inputs)

        action, activations = policy(inputs, True)
        activation_collection.append(activations)

        action = action.sample()
        timestep = env.step(action)
        reward_channels.append(env.task.last_reward_channels)
        if timestep.step_type == 2:
            reset_idx.append(i)
    
    return frames, reset_idx, reward_channels, activation_collection, kinematics_collection

def process_and_save_activation_collection(activation_collections, h5_file_path):
    """
    Processes the activation collections and saves them as two DataFrames in HDF5 format.
    - Rows: Timesteps.
    - Columns: Neurons and episode number.

    Parameters:
    - activation_collections: List of lists, where each sublist contains the activations for one episode.
    - h5_file_path: Path to the HDF5 file where the data will be saved.
    """
    layernorm_dfs = []
    mlp_elu_dfs = []
    
    for episode_idx, activation_collection in enumerate(activation_collections):
        # Prepare empty lists to collect activations for each timestep
        layernorm_activations = []
        mlp_elu_activations = []

        # Iterate over each timestep and extract activations
        for timestep in range(len(activation_collection)):
            activations = activation_collection[timestep]
            layernorm_activations.append(activations["layernorm_tanh"])
            mlp_elu_activations.append(activations["mlp_elu"])

        # Convert lists to 2D numpy arrays (timesteps x neurons)
        layernorm_array = np.squeeze(np.array(layernorm_activations))  # Ensure it's (timesteps, neurons)
        mlp_elu_array = np.squeeze(np.array(mlp_elu_activations))      # Ensure it's (timesteps, neurons)   

        # Add the episode number as a column (broadcasted for all timesteps)
        episode_column = np.full((layernorm_array.shape[0], 1), episode_idx + 1)

        # Combine neuron activations with the episode column
        layernorm_combined = np.hstack((layernorm_array, episode_column))
        mlp_elu_combined = np.hstack((mlp_elu_array, episode_column))

        # Create column labels (neuron_1, neuron_2, ..., episode)
        neuron_columns = [f"neuron_{i+1}" for i in range(layernorm_array.shape[1])]
        columns_with_episode = neuron_columns + ["episode"]

        # Create DataFrames with timesteps as rows, neurons + episode as columns
        df_layernorm = pd.DataFrame(layernorm_combined, columns=columns_with_episode)
        df_mlp_elu = pd.DataFrame(mlp_elu_combined, columns=columns_with_episode)

        # Store DataFrames for later concatenation
        layernorm_dfs.append(df_layernorm)
        mlp_elu_dfs.append(df_mlp_elu)

    # Concatenate the DataFrames from all episodes (append them in time direction)
    final_layernorm_df = pd.concat(layernorm_dfs, ignore_index=True)
    final_mlp_elu_df = pd.concat(mlp_elu_dfs, ignore_index=True)

    # Save the final DataFrames to HDF5
    with pd.HDFStore(h5_file_path) as store:
        store['layernorm_tanh'] = final_layernorm_df
        store['mlp_elu'] = final_mlp_elu_df

    print(f"Activation data saved to {h5_file_path}")

def process_and_save_kinematics_collection(kinematics_collections, h5_file_path):
    """
    Processes the kinematics collections and saves them as a DataFrame in HDF5 format.
    - Rows: Timesteps.
    - Columns: Geom positions, body positions, joint angles, velocities, accelerations, and episode number.

    Parameters:
    - kinematics_collections: List of lists, where each sublist contains the kinematics data for one episode.
    - h5_file_path: Path to the HDF5 file where the data will be saved.
    """
    kinematics_dfs = []

    for episode_idx, kinematics_collection in enumerate(kinematics_collections):
        # Prepare a list to collect kinematics data for each timestep
        timestep_data = []

        # Iterate over each timestep and extract positions, velocities, accelerations, and joint angles
        for timestep_idx, kinematics in enumerate(kinematics_collection):
            # Create a flat dictionary for this timestep
            timestep_dict = {'episode': episode_idx + 1, 'timestep': timestep_idx}

            # Add geom data (positions, velocities, accelerations) to the timestep dictionary
            for geom_name, position in kinematics['geoms']['positions'].items():
                timestep_dict[f"{geom_name}_geom_x"] = position[0]
                timestep_dict[f"{geom_name}_geom_y"] = position[1]
                timestep_dict[f"{geom_name}_geom_z"] = position[2]
            for geom_name, velocity in kinematics['geoms']['velocities'].items():
                timestep_dict[f"{geom_name}_geom_vel_x"] = velocity[0]
                timestep_dict[f"{geom_name}_geom_vel_y"] = velocity[1]
                timestep_dict[f"{geom_name}_geom_vel_z"] = velocity[2]
            for geom_name, acceleration in kinematics['geoms']['accelerations'].items():
                timestep_dict[f"{geom_name}_geom_acc_x"] = acceleration[0]
                timestep_dict[f"{geom_name}_geom_acc_y"] = acceleration[1]
                timestep_dict[f"{geom_name}_geom_acc_z"] = acceleration[2]

            # Add body data (positions, velocities, accelerations) to the timestep dictionary
            for body_name, position in kinematics['bodies']['positions'].items():
                timestep_dict[f"{body_name}_body_x"] = position[0]
                timestep_dict[f"{body_name}_body_y"] = position[1]
                timestep_dict[f"{body_name}_body_z"] = position[2]
            for body_name, velocity in kinematics['bodies']['velocities'].items():
                timestep_dict[f"{body_name}_body_vel_x"] = velocity[0]
                timestep_dict[f"{body_name}_body_vel_y"] = velocity[1]
                timestep_dict[f"{body_name}_body_vel_z"] = velocity[2]
            for body_name, acceleration in kinematics['bodies']['accelerations'].items():
                timestep_dict[f"{body_name}_body_acc_x"] = acceleration[0]
                timestep_dict[f"{body_name}_body_acc_y"] = acceleration[1]
                timestep_dict[f"{body_name}_body_acc_z"] = acceleration[2]

            # Add joint angles to the timestep dictionary
            for joint_name, angle in kinematics['joints'].items():
                timestep_dict[f"{joint_name}_joint_angle"] = angle

            timestep_data.append(timestep_dict)

        # Create a DataFrame for this episode
        df_kinematics = pd.DataFrame(timestep_data)
        kinematics_dfs.append(df_kinematics)

    # Concatenate the DataFrames from all episodes
    final_kinematics_df = pd.concat(kinematics_dfs, ignore_index=True)

    # Save the final DataFrame to HDF5
    with pd.HDFStore(h5_file_path) as store:
        store['kinematics'] = final_kinematics_df

    print(f"Kinematics data saved to {h5_file_path}")

def run_simulation(learner):
    # Begin simulation
    num_episodes = 3
    rollout_length = 15
    all_activations_per_episode = []
    all_kinematics_per_episode = []

    # Create the environment
    video_dir = "/root/vast/eric/vnl-ray/videos/"
    actuator_type = "torque"
    env = lambda: mouse_reach(actuator_type=actuator_type)
    env = env()

    # Apply wrappers
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env, clip=False)
    action_spec = env.action_spec()
    observation_spec = env.observation_spec()

    ts = env.reset()
    inputs = ts.observation
    inputs_add = tf2_utils.add_batch_dim(inputs)
    inputs_concat = tf2_utils.batch_concat(inputs_add)
    policy = learner._policy_network

    # Run the simulation for the specified number of episodes
    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1}")
        frames, activation_collection, kinematics_collection = render_with_rewards(
            env, policy, observation_spec, rollout_length=rollout_length
        )

        all_activations_per_episode.append(activation_collection)
        all_kinematics_per_episode.append(kinematics_collection)

        # Save the video for each episode
        video_path = f"{video_dir}mouse_reach_episode_{episode + 1}_checkpoint.mp4"
        with imageio.get_writer(video_path, fps=1 / env.control_timestep()) as video:
            for frame in frames:
                video.append_data(frame)

        print(f"Episode {episode + 1} finished and saved at {video_path}")
    return all_activations_per_episode, all_kinematics_per_episode

PYHTONPATH = os.path.dirname(os.path.dirname(vnl_ray.__file__))
LD_LIBRARY_PATH = os.environ["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in os.environ else ""

# Defer specifying CUDA_VISIBLE_DEVICES to sub-processes.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tasks = {
    "run-gaps": rodent_run_gaps,
    "maze-forage": rodent_maze_forage,
    "escape-bowl": rodent_escape_bowl,
    "two-taps": rodent_two_touch,
    "rodent_imitation": rodent_walk_imitation,
    "fly_imitation": fly_walk_imitation,
    "humanoid_imitation": walk_humanoid,
    "mouse_reach": mouse_reach,
}

@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="train_config_mouse_reach_offline",
)
def main(config: DictConfig) -> None:
    print("CONFIG:", config)

    from vnl_ray.agents.ray_distributed_dmpo import (
        DMPOConfig,
        ReplayServer,
        Learner,
        EnvironmentLoop,
    )

    print("\nRay context:")
    print(ray_context)

    ray_resources = ray.available_resources()
    print("\nAvailable Ray cluster resources:")
    print(ray_resources)

    # Create environment factory RL task.
    # Cannot parametrize it because it failed to serialize functions
    def environment_factory_mouse_reach() -> "composer.Environment":
            env = tasks["mouse_reach"](actuator_type=config.run_config.actuator_type)
            env = wrappers.SinglePrecisionWrapper(env)
            env = wrappers.CanonicalSpecWrapper(env)
            return env

    def environment_factory_run_gaps() -> "composer.Environment":
        env = tasks["run-gaps"]()
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.CanonicalSpecWrapper(env)
        return env

    def environment_factory_two_taps() -> "composer.Environment":
        env = tasks["two-taps"]()
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.CanonicalSpecWrapper(env)
        return env

    def environment_factory_maze_forage() -> "composer.Environment":
        env = tasks["maze-forage"]()
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.CanonicalSpecWrapper(env)
        return env

    def environment_factory_bowl_escape() -> "composer.Environment":
        env = tasks["escape-bowl"]()
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.CanonicalSpecWrapper(env)
        return env

    def environment_factory_imitation_humanoid() -> "composer.Environment":
        """Creates replicas of environment for the agent."""
        env = tasks["humanoid_imitation"](config["ref_traj_path"])
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.CanonicalSpecWrapper(env)
        return env

    def environment_factory_imitation_rodent(
        termination_error_threshold=0.12, always_init_at_clip_start=False
    ) -> "composer.Environment":
        """
        Creates replicas of environment for the agent. random range controls the
        range of the uniformed distributed termination logics
        """
        env = tasks["rodent_imitation"](
            config["ref_traj_path"],
            reward_term_weights=config["reward_term_weights"] if "reward_term_weights" in config else None,
            termination_error_threshold=termination_error_threshold,
            always_init_at_clip_start=always_init_at_clip_start,
        )
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.CanonicalSpecWrapper(env)
        return env

    environment_factories = {
        "run-gaps": environment_factory_run_gaps,
        "maze-forage": environment_factory_maze_forage,
        "escape-bowl": environment_factory_bowl_escape,
        "two-taps": environment_factory_two_taps,
        "general": environment_factory_run_gaps,
        "imitation_humanoid": environment_factory_imitation_humanoid,
        "imitation_rodent": functools.partial(
            environment_factory_imitation_rodent,
            # termination_error_threshold=config["termination_error_threshold"], # TODO modify the config yaml for the imitation learning too.
        ),
        "mouse_reach": environment_factory_mouse_reach,
    }

    # Dummy environment and network for quick use, deleted later. # create this earlier to access the obs
    dummy_env = environment_factories[config.run_config["task_name"]]()

    # Create network factory for RL task. Config specify different ANN structures
    if config.learner_network["use_intention"]:
        network_factory = make_network_factory_dmpo_intention(
            task_obs_size=get_task_obs_size(
                dummy_env.observation_spec(), config.run_config["agent_name"], config.obs_network["visual_feature_size"]
            ),
            encoder_layer_sizes=config.learner_network["encoder_layer_sizes"],
            decoder_layer_sizes=config.learner_network["decoder_layer_sizes"],
            critic_layer_sizes=config.learner_network["critic_layer_sizes"],
            intention_size=config.learner_network["intention_size"],
            use_tfd_independent=True,  # for easier KL calculation
            use_visual_network=config.obs_network["use_visual_network"],
            visual_feature_size=config.obs_network["visual_feature_size"],
            mid_layer_sizes=(
                config.learner_network["mid_layer_sizes"] if config.learner_network["use_multi_decoder"] else None
            ),
            high_level_intention_size=(
                config.learner_network["high_level_intention_size"]
                if config.learner_network["use_multi_decoder"]
                else None
            ),
        )
    else:
        # online settings
        network_factory = make_network_factory_dmpo(
            action_spec = dummy_env.action_spec(),
            policy_layer_sizes=config.learner_network["policy_layer_sizes"],
            critic_layer_sizes=config.learner_network["critic_layer_sizes"],
        )

    dummy_net = network_factory(dummy_env.action_spec())  # we should share this net for joint training
    # Get full environment specs.
    environment_spec = specs.make_environment_spec(dummy_env)

    # This callable will be calculating penalization cost by converting canonical
    # actions to real (not wrapped) environment actions inside DMPO agent.
    penalization_cost = None  # PenalizationCostRealActions(dummy_env.environment.action_spec())
    # Distributed DMPO agent configuration.
    dmpo_config = DMPOConfig(
        num_actors=config.env_params["num_actors"],
        batch_size=config.learner_params["batch_size"],
        discount=config.learner_params["discount"],
        prefetch_size=1024,  # aggresive prefetch param, because we have large amount of data
        num_learner_steps=1000,
        min_replay_size=50_000,
        max_replay_size=4_000_000,
        samples_per_insert=None,  # allow less sample per insert to allow more data in # None is only min limiter
        n_step=50,
        num_samples=20,
        policy_loss_module=policy_loss_module_dmpo(
            epsilon=0.1,
            epsilon_mean=0.0025,
            epsilon_stddev=1e-7,
            action_penalization=True,
            epsilon_penalty=0.1,
            penalization_cost=penalization_cost,
        ),
        policy_optimizer=snt.optimizers.Adam(config.learner_params["policy_optimizer_lr"]),  # reduce the lr
        critic_optimizer=snt.optimizers.Adam(config.learner_params["critic_optimizer_lr"]),
        dual_optimizer=snt.optimizers.Adam(config.learner_params["dual_optimizer_lr"]),
        target_critic_update_period=107,
        target_policy_update_period=101,
        actor_update_period=5_000,
        log_every=30,
        logger=make_default_logger,
        logger_save_csv_data=False,
        checkpoint_max_to_keep=None,
        checkpoint_directory=f"./training/ray-{config.run_config['agent_name']}-{config.run_config['task_name']}-ckpts/",
        checkpoint_to_load=config.learner_params["checkpoint_to_load"],
        print_fn=None,  # print # this causes issue pprint does not work
        userdata=dict(),
        kickstart_teacher_cps_path=(
            config.learner_params["kickstart_teacher_cps_path"]
            if "kickstart_teacher_cps_path" in config.learner_params
            else None
        ),  # specify the location of the kickstarter teacher policy's cps
        kickstart_epsilon=(
            config.learner_params["kickstart_epsilon"] if "kickstart_epsilon" in config.learner_params else 0
        ),
        time_delta_minutes=30,
        eval_average_over=config.eval_params["eval_average_over"],
        KL_weights=(0, 0),
        # specify the KL with intention & action output layer # do not penalize the output layer # disabled it for now.
        load_decoder_only=(
            config.learner_params["load_decoder_only"] if "load_decoder_only" in config.learner_params else False
        ),
        froze_decoder=config.learner_params["froze_decoder"] if "froze_decoder" in config.learner_params else False,
    )

    dmpo_dict_config = dataclasses.asdict(dmpo_config)
    merged_config = dmpo_dict_config | OmegaConf.to_container(config)  # merged two config

    logger_kwargs = {"config": merged_config}
    dmpo_config.userdata["logger_kwargs"] = logger_kwargs

    # Print full job config and full environment specs.
    print("\n", dmpo_config)
    print("\n", dummy_net)
    print("\nobservation_spec:\n", dummy_env.observation_spec())
    print("\naction_spec:\n", dummy_env.action_spec())
    print("\ndiscount_spec:\n", dummy_env.discount_spec())
    print("\nreward_spec:\n", dummy_env.reward_spec(), "\n")


    # Environment variables for learner, actor, and replay buffer processes.
    runtime_env_learner = {
        "env_vars": {
            "MUJOCO_GL": "osmesa",
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "PYTHONPATH": PYHTONPATH,
            "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
        }
    }
    runtime_env_actor = {
        "env_vars": {
            "MUJOCO_GL": "osmesa",
            "CUDA_VISIBLE_DEVICES": "-1",  # CPU-actors don't use CUDA.
            "PYTHONPATH": PYHTONPATH,
            "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
        }
    }

    # Define resources and placement group
    gpu_pg = placement_group([{"GPU": 1, "CPU": 10}], strategy="STRICT_PACK")

    # === Create Replay Server.
    runtime_env_replay = {
        "env_vars": {
            "PYTHONPATH": PYHTONPATH,  # Also used for counter.
        }
    }

    ReplayServer = ray.remote(
        num_gpus=0,
        runtime_env=runtime_env_replay,
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=gpu_pg),  # test out performance w/o
    )(ReplayServer)

    replay_servers = dict()  # {task_name: addr} # TODO: Probably could simplify this logic quite a bit
    servers = []
    if "actors_envs" in config:
        dmpo_config.max_replay_size = dmpo_config.max_replay_size // 4  # reduce each replay buffer size by 4.
        for name, num_actors in config.actors_envs.items():
            if num_actors != 0:
                if not config["separate_replay_servers"]:
                    # mixed experience replay buffers
                    replay_server = ReplayServer.remote(
                        dmpo_config, environment_spec
                    )  # each envs will share the same environment spec and dmpo_config
                    addr = ray.get(replay_server.get_server_address.remote())
                    replay_server = RemoteAsLocal(replay_server)
                    servers.append(replay_server)
                    replay_servers["general"] = addr
                    print("SINGLE: Started Single replay server for this task.")
                    break
                elif "num_replay_servers" in config and config["num_replay_servers"] != 0:
                    dmpo_config.max_replay_size = (
                        dmpo_config.max_replay_size // config["num_replay_servers"]
                    )  # shrink down the replay size correspondingly
                    for i in range(config["num_replay_servers"]):
                        _name = f"{name}-{i+1}"
                        # multiple replay server for load balancing
                        replay_server = ReplayServer.remote(
                            dmpo_config, environment_spec
                        )  # each envs will share the same environment spec and dmpo_config
                        addr = ray.get(replay_server.get_server_address.remote())
                        print(f"MULTIPLE: Started Replay Server for task {_name} on {addr}")
                        replay_servers[_name] = addr
                        replay_server = RemoteAsLocal(replay_server)
                        # this line is essential to keep a refernce to the replay server
                        # otherwise the object will be garbage collected and clean out
                        servers.append(replay_server)
                        time.sleep(0.1)
                        # multiple replay server setup
                else:
                    replay_server = ReplayServer.remote(
                        dmpo_config, environment_spec
                    )  # each envs will share the same environment spec and dmpo_config
                    addr = ray.get(replay_server.get_server_address.remote())
                    print(f"MULTIPLE: Started Replay Server for task {name} on {addr}")
                    replay_servers[name] = addr
                    replay_server = RemoteAsLocal(replay_server)
                    # this line is essential to keep a refernce to the replay server
                    # otherwise the object will be garbage collected and clean out
                    servers.append(replay_server)
                    time.sleep(0.1)
    else:
        if "num_replay_servers" in config.env_params and config.env_params["num_replay_servers"] != 0:
            dmpo_config.max_replay_size = (
                dmpo_config.max_replay_size // config.env_params["num_replay_servers"]
            )  # shrink down the replay size correspondingly
            for i in range(config.env_params["num_replay_servers"]):
                name = f"{config.run_config['task_name']}-{i+1}"
                # multiple replay server for load balancing
                replay_server = ReplayServer.remote(
                    dmpo_config, environment_spec
                )  # each envs will share the same environment spec and dmpo_config
                addr = ray.get(replay_server.get_server_address.remote())
                print(f"MULTIPLE: Started Replay Server for task {name} on {addr}")
                replay_servers[name] = addr
                replay_server = RemoteAsLocal(replay_server)
                # this line is essential to keep a refernce to the replay server
                # otherwise the object will be garbage collected and clean out
                servers.append(replay_server)
                time.sleep(0.5)
        else:
            # single replay server
            replay_server = ReplayServer.remote(dmpo_config, environment_spec)
            addr = ray.get(replay_server.get_server_address.remote())
            print(f"Started Replay Server on {addr}")
            replay_servers[config.run_config["task_name"]] = addr

    # === Create Counter.
    counter = ray.remote(PicklableCounter)  # This is class (direct call to ray.remote decorator).
    counter = counter.remote()  # Instantiate.
    counter = RemoteAsLocal(counter)

    # === Create Learner.

    learner = Learner(
        replay_servers,
        counter,
        environment_spec,
        dmpo_config,
        network_factory,
    )

    print(learner)

    # Run the simulation and collect data
    all_activations_per_episode, all_kinematics_per_episode = run_simulation(learner)

    # Paths to save the data
    activations_h5_file_path = "/root/vast/eric/vnl-ray/training/activations/activations_test.h5"
    kinematics_h5_file_path = "/root/vast/eric/vnl-ray/training/kinematics/kinematics_test.h5"

    # Process and save activations
    process_and_save_activation_collection(all_activations_per_episode, activations_h5_file_path)

    # Process and save kinematics
    process_and_save_kinematics_collection(all_kinematics_per_episode, kinematics_h5_file_path)

if __name__ == "__main__":
    main()