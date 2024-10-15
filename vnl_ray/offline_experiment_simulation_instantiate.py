import os
import tensorflow as tf
import imageio
from dm_control import composer
from acme import wrappers
from pathlib import Path
from vnl_ray.agents.utils_tf import TestPolicyWrapper
from vnl_ray.tasks.mouse_reach_task import MouseReachTask
from vnl_ray.tasks.arenas.mouse_arena import MouseReachArena
from vnl_ray.mouse_forelimb.mouse_entity import MouseEntity
from vnl_ray.tasks.mouse_reach import mouse_reach
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import Independent
from vnl_ray.utils import plot_reward
from collections import defaultdict
import numpy as np
from vnl_ray.agents.policy_network_activations import IntermediateActivationsPolicyNetwork, custom_actor
from acme.tf import utils as tf2_utils
import sonnet as snt


# Simulation and Physics Constants
_CONTROL_TIMESTEP = 0.02
_PHYSICS_TIMESTEP = 0.001

def render_with_rewards(env, policy, observation_spec, rollout_length=150):
    """
    render with the rewards progression graph concat alongside with the rendering
    """
    frames, reset_idx, reward_channels = render_with_rewards_info(env, policy, observation_spec, rollout_length=rollout_length)
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
    return concat_frames
    

def render_with_rewards_info(env, policy, observation_spec, rollout_length=150, render_vision_if_available=True):
    """
    Generate rollout with the reward related information.
    """
    reward_channels = []
    frames = []
    reset_idx = []
    timestep = env.reset()

    render_kwargs = {"width": 640, "height": 480}
    for i in range(rollout_length):
        pixels = env.physics.render(camera_id=1, **render_kwargs)
        # optionally also render the vision.
        frames.append(pixels)
        action, activation = policy(timestep.observation, return_activations=True)
        timestep = env.step(action)
        reward_channels.append(env.task.last_reward_channels)
        if timestep.step_type == 2:
            reset_idx.append(i)
    return frames, reset_idx, reward_channels

    # Assuming IntermediateActivationsPolicyNetwork is already defined or imported
def instantiate_policy_network(action_spec,
                               observation_spec,
                               policy_layer_sizes=[512,512,512,512],
                               tanh_mean=False,
                               init_sigma=0.1,
                               fixed_sigma=True,
                               use_tfd_independent=True):
    """Instantiate the policy network."""
    action_size = np.prod(action_spec.shape, dtype=int)
    observation_size = sum(
        np.prod(spec.shape, dtype=int) for spec in observation_spec.values()
    )

    # Instantiate the policy network with the desired specifications
    policy_network = IntermediateActivationsPolicyNetwork(
        layer_sizes=policy_layer_sizes,
        num_dimensions=action_size,
        tanh_mean=tanh_mean,
        init_scale=init_sigma,
        fixed_scale=fixed_sigma,
        use_tfd_independent=use_tfd_independent
    )

    # Call the network with a dummy input to initialize variables
    dummy_input = tf.zeros([1, observation_size])  # Assuming a batch of 1 and correct input size
    _ = policy_network(dummy_input)

    return policy_network



def print_variable_names(loaded_policy, policy_network):
    """
    Prints the variable names for both loaded_policy and policy_network.
    """
    loaded_policy_vars = loaded_policy.trainable_variables
    policy_network_vars = policy_network.trainable_variables

    print("Loaded Policy Variables:")
    for var in loaded_policy_vars:
        print(var.name, var.shape)

    print("\nPolicy Network Variables:")
    for var in policy_network_vars:
        print(var.name, var.shape)

def transfer_weights(loaded_policy, policy_network):
    """
    Transfers weights from the loaded policy (from a SavedModel) to the Sonnet policy network.
    Returns the updated policy network with the new weights.
    """
    loaded_policy_vars = loaded_policy.trainable_variables
    policy_network_vars = policy_network.trainable_variables

    # Print variable names to help identify the matching ones
    print_variable_names(loaded_policy, policy_network)

    # Now match variables by their names (stripping prefixes/suffixes if needed)
    loaded_policy_dict = {var.name.split('/')[-2]: var for var in loaded_policy_vars}
    policy_network_dict = {var.name.split('/')[-2]: var for var in policy_network_vars}

    # Check if both have the same keys
    if loaded_policy_dict.keys() != policy_network_dict.keys():
        raise ValueError(f"Mismatch in variable names. Loaded policy variables: {loaded_policy_dict.keys()}, policy network variables: {policy_network_dict.keys()}")

    # Assign the variables by matching their names
    for key in loaded_policy_dict.keys():
        loaded_var = loaded_policy_dict[key]
        policy_var = policy_network_dict[key]
        if loaded_var.shape == policy_var.shape:
            policy_var.assign(loaded_var)
        else:
            raise ValueError(f"Shape mismatch for {key}: loaded {loaded_var.shape}, policy {policy_var.shape}")

    print("Weights have been successfully transferred.")
    return policy_network

def run_simulation(env_factory, snapshot_path, video_dir, num_episodes=10, rollout_length=150):
    """Run the simulation for a number of episodes and render them."""
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    # Create the environment
    env = env_factory()

    # Apply wrappers
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env, clip=False)
    action_spec = env.action_spec()
    observation_spec = env.observation_spec()

    loaded_policy = tf.saved_model.load(snapshot_path)

    # instantiate 
    init_policy = instantiate_policy_network(action_spec, observation_spec)
    
    # transfer the weights
    policy = transfer_weights(loaded_policy, init_policy)
    
    # Test policy wrapper
    policy = TestPolicyWrapper(policy)

    policy = custom_actor(policy)

    # Run the simulation for the specified number of episodes
    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1}")
        frames = render_with_rewards(env, policy, observation_spec, rollout_length=rollout_length)

        # Save the video for each episode
        video_path = video_dir / f"mouse_reach_episode_{episode + 1}_instantiate.mp4"
        with imageio.get_writer(video_path, fps=1 / env.control_timestep()) as video:
            for frame in frames:
                video.append_data(frame)

        print(f"Episode {episode + 1} finished and saved at {video_path}")

if __name__ == "__main__":
    # Configure the paths
    #snapshot_path = "/root/vast/eric/vnl-ray/training/ray-mouse-mouse_reach-ckpts/325f0606-8a96-11ef-882c-2e7de45e22fb/snapshots/policy-5/"  # Update with the correct snapshot path
    snapshot_path = "/root/vast/eric/vnl-ray/training/ray-mouse-mouse_reach-ckpts/325f0606-8a96-11ef-882c-2e7de45e22fb/snapshots/policy-5/"
    video_dir = "/root/vast/eric/vnl-ray/videos/"
    
    # Configure the actuator type (e.g., 'muscle', 'torque', or 'position')
    actuator_type = "torque"  # Update as needed

    # Run the simulation for 10 episodes
    run_simulation(lambda: mouse_reach(actuator_type=actuator_type), snapshot_path, video_dir, num_episodes=10)
    