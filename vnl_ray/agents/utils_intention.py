import tensorflow as tf
import sonnet as snt
from acme.tf import utils as tf2_utils
from acme import types
from typing import List
import numpy as np


def separate_observation(observation: types.NestedTensor) -> tf.Tensor:
    """
    Function similar to tf2_utils.batch_concat, but returns a 2D tensor
    specifically for the intention network to take input into the encoder
    and decoder differently.

    It separates egocentric observation (e.g., joint angles, velocities) from
    task-specific observations (e.g., target positions) while respecting batch size.
    """
    observation = observation.copy()

    # Separate reference and non-reference observations
    egocentric_obs_keys = get_mouse_egocentric_obs_key()  # Use specific keys for mouse
    task_obs_keys = [k for k in observation.keys() if k not in egocentric_obs_keys]

    egocentric_obs = {k: observation.pop(k) for k in egocentric_obs_keys}
    task_obs = {k: observation.pop(k) for k in task_obs_keys}

    # Flatten each observation tensor while keeping the batch dimension intact
    task_obs_tensors = [tf.reshape(v, [tf.shape(v)[0], -1]) for v in task_obs.values()]  # Preserve batch size
    egocentric_obs_tensors = [tf.reshape(v, [tf.shape(v)[0], -1]) for v in egocentric_obs.values()]  # Preserve batch size

    # Concatenate task and egocentric observations along the last axis
    task_obs_tensor = tf.concat(task_obs_tensors, axis=-1) if task_obs_tensors else tf.constant([])
    egocentric_obs_tensor = tf.concat(egocentric_obs_tensors, axis=-1) if egocentric_obs_tensors else tf.constant([])

    concatenated_obs = tf.concat([task_obs_tensor, egocentric_obs_tensor], axis=-1)
    return concatenated_obs
    
def get_rodent_egocentric_obs_key() -> List[str]:
    """
    return the egocentric observation key of the rodent
    """
    return [
        "walker/actuator_activation",
        "walker/appendages_pos",
        "walker/body_height",
        "walker/end_effectors_pos",
        "walker/joints_pos",
        "walker/joints_vel",
        "walker/sensors_accelerometer",
        "walker/sensors_force",
        "walker/sensors_gyro",
        "walker/sensors_torque",
        "walker/sensors_touch",
        "walker/sensors_velocimeter",
        "walker/tendons_pos",
        "walker/tendons_vel",
        "walker/world_zaxis",
    ]

def get_mouse_egocentric_obs_key() -> List[str]:
    """
    Returns the observation keys for the mouse template.
    """
    return [
        "mouse/joint_angles",
        "mouse/joint_velocities",
    ]