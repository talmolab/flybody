"""
Script for distributed reinforcement learning training with Ray.

This script trains the fly-on-ball RL task using a distributed version of the
DMPO agent. The training runs in an infinite loop until terminated.

For lightweight testing, run this script with --test argument. It will run
training with a single actor and print training statistics every 10 seconds.

This script is not task-specific and can be used with other fly RL tasks by
swapping in other environments in the environment_factory function. The single
main configurable component below is the DMPO agent configuration and
training hyperparameters specified in the DMPOConfig data structure.
"""

# ruff: noqa: F821, E722, E402

# Start Ray cluster first, before imports.
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import placement_group
import logging

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
    ray_context = ray.init(
        include_dashboard=True, dashboard_host="0.0.0.0", logging_level=logging.INFO
    )

import argparse
import time
import os
import dataclasses
import uuid
import hydra
import functools
from omegaconf import DictConfig, OmegaConf

from acme import specs
from acme import wrappers
from acme.tf import utils as tf2_utils
import sonnet as snt
import tensorflow as tf

import flybody
from flybody.agents.remote_as_local_wrapper import RemoteAsLocal
from flybody.agents.counting import PicklableCounter
from flybody.agents.network_factory import policy_loss_module_dmpo
from flybody.agents.losses_mpo import PenalizationCostRealActions
from flybody.basic_rodent_2020 import (
    rodent_run_gaps,
    rodent_maze_forage,
    rodent_escape_bowl,
    rodent_two_touch,
)
from flybody.fly_envs import walk_on_ball, vision_guided_flight
from flybody.agents.network_factory import make_network_factory_dmpo
from flybody.default_logger import make_default_logger
from flybody.single_precision import SinglePrecisionWrapper

PYHTONPATH = os.path.dirname(os.path.dirname(flybody.__file__))
LD_LIBRARY_PATH = (
    os.environ["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in os.environ else ""
)
# Defer specifying CUDA_VISIBLE_DEVICES to sub-processes.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--test",
    help="Run job in test mode with one actor and output to current terminal.",
    action="store_true",
)
args = parser.parse_args()
is_test = args.test
if parser.parse_args().test:
    print("\nRun job in test mode with one actor.")
    test_num_actors = 1
    test_log_every = 10
    test_min_replay_size = 40
else:
    test_num_actors = None
    test_log_every = None
    test_min_replay_size = None

tasks = {
    "run-gaps": rodent_run_gaps,
    "maze-forage": rodent_maze_forage,
    "escape-bowl": rodent_escape_bowl,
    "two-taps": rodent_two_touch,
    "general": rodent_run_gaps,
}


@hydra.main(
    version_base=None, config_path="./config", config_name="train_config_bowl"
)
def main(config: DictConfig) -> None:
    print("CONFIG:", config)
    from flybody.agents.ray_distributed_dmpo import (
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

    environment_factories = {
        "run-gaps": environment_factory_run_gaps,
        "maze-forage": environment_factory_maze_forage,
        "escape-bowl": environment_factory_bowl_escape,
        "two-taps": environment_factory_two_taps,
        "general": environment_factory_run_gaps,
    }

    # Create network factory for RL task.
    network_factory = make_network_factory_dmpo(
        policy_layer_sizes=config["policy_layer_sizes"],
        critic_layer_sizes=config["critic_layer_sizes"],
    )

    # Dummy environment and network for quick use, deleted later.
    dummy_env = environment_factories[config["task_name"]]()
    dummy_net = network_factory(
        dummy_env.action_spec()
    )  # we should share this net for joint training
    # Get full environment specs.
    environment_spec = specs.make_environment_spec(dummy_env)

    # This callable will be calculating penalization cost by converting canonical
    # actions to real (not wrapped) environment actions inside DMPO agent.
    penalization_cost = (
        None  # PenalizationCostRealActions(dummy_env.environment.action_spec())
    )
    # Distributed DMPO agent configuration.
    dmpo_config = DMPOConfig(
        num_actors=test_num_actors or config["num_actors"],
        batch_size=config["batch_size"],
        prefetch_size=4,  # maybe unlimited?
        num_learner_steps=200,
        min_replay_size=test_min_replay_size or 10_000,
        max_replay_size=4_000_000,  # increase reply size
        samples_per_insert=15,
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
        policy_optimizer=snt.optimizers.Adam(1e-4),
        critic_optimizer=snt.optimizers.Adam(1e-4),
        dual_optimizer=snt.optimizers.Adam(1e-3),
        target_critic_update_period=107,
        target_policy_update_period=101,
        actor_update_period=1000,
        log_every=test_log_every or 60,
        logger=make_default_logger,
        logger_save_csv_data=False,
        checkpoint_max_to_keep=None,
        checkpoint_directory=f"./training/ray-{config['agent_name']}-{config['task_name']}-ckpts/",
        checkpoint_to_load=config["checkpoint_to_load"],
        print_fn=None,  # print # this causes issue pprint does not work
        userdata=dict(),
        kickstart_teacher_cps_path=config[
            "kickstart_teacher_cps_path"
        ],  # specify the location of the kickstarter teacher policy's cps
        kickstart_epsilon=config["kickstart_epsilon"],
    )

    dmpo_dict_config = dataclasses.asdict(dmpo_config)
    merged_config = dmpo_dict_config | OmegaConf.to_container(
        config
    )  # merged two config

    logger_kwargs = {"config": merged_config}
    dmpo_config.userdata["logger_kwargs"] = logger_kwargs

    # Print full job config and full environment specs.
    print("\n", dmpo_config)
    print("\n", dummy_net)
    print("\nobservation_spec:\n", dummy_env.observation_spec())
    print("\naction_spec:\n", dummy_env.action_spec())
    print("\ndiscount_spec:\n", dummy_env.discount_spec())
    print("\nreward_spec:\n", dummy_env.reward_spec(), "\n")
    del dummy_env
    del dummy_net

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
    gpu_pg = placement_group([{"GPU": 1, "CPU": 12}], strategy="STRICT_PACK")

    # === Create Replay Server.
    runtime_env_replay = {
        "env_vars": {
            "PYTHONPATH": PYHTONPATH,  # Also used for counter.
        }
    }

    ReplayServer = ray.remote(
        num_gpus=0,
        runtime_env=runtime_env_replay,
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=gpu_pg),
    )(ReplayServer)

    replay_servers = dict()  # {task_name: addr}
    servers = []
    if "actors_envs" in config:
        dmpo_config.max_replay_size = dmpo_config.max_replay_size // 2  # reduce each replay buffer size by 4.
        for name, num_actors in config.actors_envs.items():
            if num_actors != 0:
                # multiple replay server setup
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
        if "num_replay_servers" in config and config["num_replay_servers"] != 0:
            for i in range(config["num_replay_servers"]):
                name = f"{config['task_name']}-{i+1}"
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
            replay_servers[config["task_name"]] = addr

    # === Create Counter.
    counter = ray.remote(PicklableCounter)  # This is class (direct call to ray.remote decorator).
    counter = counter.remote()  # Instantiate.
    counter = RemoteAsLocal(counter)

    # === Create Learner.
    Learner = ray.remote(
        num_gpus=1,
        runtime_env=runtime_env_learner,
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=gpu_pg),
    )(Learner)

    learner = Learner.remote(
        replay_servers,
        counter,
        environment_spec,
        dmpo_config,
        network_factory,
    )
    learner = RemoteAsLocal(learner)

    print("Waiting until learner is ready...")
    learner.isready(block=True)

    checkpointer_dir, snapshotter_dir = learner.get_checkpoint_dir()
    print("Checkpointer directory:", checkpointer_dir)
    print("Snapshotter directory:", snapshotter_dir)

    # === Create Actors and Evaluator.

    EnvironmentLoop = ray.remote(num_gpus=0, runtime_env=runtime_env_actor)(
        EnvironmentLoop
    )

    n_actors = dmpo_config.num_actors

    def create_actors(n_actors, environment_factory, replay_server_addr):  # callalbe env factory
        """Return list of requested number of actor instances."""
        actors = []
        for _ in range(n_actors):
            actor = EnvironmentLoop.remote(
                replay_server_address=replay_server_addr,
                variable_source=learner,
                counter=counter,
                network_factory=network_factory,
                environment_factory=environment_factory,
                dmpo_config=dmpo_config,
                actor_or_evaluator="actor",
            )
            actor = RemoteAsLocal(actor)
            actors.append(actor)
            time.sleep(0.2)
        return actors

    def create_evaluator(task_name, replay_server_addr):
        evaluator = EnvironmentLoop.remote(
            replay_server_address=replay_server_addr,
            variable_source=learner,
            counter=counter,
            network_factory=network_factory,
            environment_factory=environment_factories[task_name],
            dmpo_config=dmpo_config,
            actor_or_evaluator="evaluator",
            task_name=task_name,
        )
        return evaluator

    actors = []
    evaluators = []
    if "actors_envs" in config:
        # if the config file specify diverse actor envs
        print(config.actors_envs)
        for name, num_actors in config.actors_envs.items():
            if num_actors != 0:
                actors += create_actors(num_actors, environment_factories[name], replay_servers[name])
                print(f"ACTOR Creation: {name}, has #{num_actors} of actors.")
                if num_actors != 0: # only create evaluator if we decided to run that task.
                    evaluators.append(RemoteAsLocal(create_evaluator(name, replay_servers[name])))
                    print(f"EVALUTATOR Creation for task: {name}")
    else:
        # Get actors.
        print(f"ACTOR Creation: {n_actors}")
        if "num_replay_servers" in config and config["num_replay_servers"] != 0:
            for i in range(config["num_replay_servers"]):
                name = f"{config['task_name']}-{i+1}"
                # multiple replay servers, equally direct replay servers
                num_actor_per_replay = n_actors // config["num_replay_servers"]
                actors += create_actors(num_actor_per_replay, environment_factories[config["task_name"]], replay_servers[name])
        else:
            # single replay server
            actors = create_actors(n_actors, environment_factories[config["task_name"]], replay_servers[config["task_name"]])

        evaluator = EnvironmentLoop.remote(
            replay_server_address=replay_servers[config["task_name"]],
            variable_source=learner,
            counter=counter,
            network_factory=network_factory,
            environment_factory=environment_factories[config["task_name"]],
            dmpo_config=dmpo_config,
            actor_or_evaluator="evaluator",
        )
        evaluators.append(RemoteAsLocal(evaluator))

    print("Waiting until actors are ready...")
    # Block until all actors and evaluator are ready and have called `get_variables`
    # in learner with variable_client.update_and_wait() from _make_actor. Otherwise
    # they will be blocked and won't be inserting data to replay table, which in
    # turn will cause learner to be blocked.
    for actor in actors:
        actor.isready(block=True)
    for evaluator in evaluators:
        evaluator.isready(block=True)

    print("Actors ready, issuing run command to all")

    # === Run all.
    if hasattr(counter, "run"):
        counter.run(block=False)
    for actor in actors:
        actor.run(block=False)
    for evaluator in evaluators:
        evaluator.run(block=False)

    while True:
        # Single call to `run` makes a fixed number of learning steps.
        # Here we need to block, otherwise `run` calls pile up and spam the queue.
        learner.run(block=True)


if __name__ == "__main__":
    main()
