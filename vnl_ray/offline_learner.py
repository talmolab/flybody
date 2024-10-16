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
    frames, reset_idx, reward_channels, activation_collection = render_with_rewards_info(env, policy, observation_spec, rollout_length=rollout_length)
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
    return concat_frames, activation_collection

def render_with_rewards_info(env, policy, observation_spec, rollout_length=150, render_vision_if_available=True):
    """
    Generate rollout with the reward related information.
    """
    reward_channels = []
    frames = []
    reset_idx = []
    timestep = env.reset()
    activation_collection = []

    render_kwargs = {"width": 640, "height": 480}
    for i in range(rollout_length):
        pixels = env.physics.render(camera_id=1, **render_kwargs)
        # optionally also render the vision.
        frames.append(pixels)

        inputs = tf2_utils.add_batch_dim(timestep.observation)
        inputs = tf2_utils.batch_concat(inputs)

        action, activations = policy(inputs, True)
        
        activation_collection.append(activations)

        action = action.sample()
        timestep = env.step(action)
        reward_channels.append(env.task.last_reward_channels)
        if timestep.step_type == 2:
            reset_idx.append(i)
    
    return frames, reset_idx, reward_channels, activation_collection

def run_simulation(learner):
        # Begin simulation
        num_episodes=10
        rollout_length=150

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
        
        #learner._policy_network(inputs_concat, True)
        policy = learner._policy_network

        # Run the simulation for the specified number of episodes
        for episode in range(num_episodes):
            print(f"Starting Episode {episode + 1}")
            frames, activation_collection = render_with_rewards(env, policy, observation_spec, rollout_length=rollout_length)

            # Save the video for each episode
            video_path = f"{video_dir}mouse_reach_episode_{episode + 1}_checkpoint.mp4"
            with imageio.get_writer(video_path, fps=1 / env.control_timestep()) as video:
                for frame in frames:
                    video.append_data(frame)

            print(f"Episode {episode + 1} finished and saved at {video_path}")

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

    run_simulation(learner)

if __name__ == "__main__":
    main()