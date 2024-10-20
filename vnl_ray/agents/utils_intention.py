import tensorflow as tf
import sonnet as snt
from acme.tf import utils as tf2_utils
from acme import types
from typing import List


def separate_observation(observation: types.NestedTensor) -> tf.Tensor:
    """
    function similar to tf2_utils.batch_concat, but returns a 2D tensor
    specifically for the intention network to take input into the encoder
    and decoder differently

    Observations are separated by egocentric observation (Walker's sensors, actuator activations etc,
    see the list for details) and tasks specific observations (for imitation, it is reference trajectory,
    for online RL, it is the vision inputs.) This function separate and sequence egocentric observation,
    to support routing to the intention network.
    """

    observation = observation.copy()
    # separate reference and non-reference observations
    egocentric_obs_keys = get_rodent_egocentric_obs_key() # sequence matters. Currently sorted in alphabetical order.

    task_obs_keys = [k for k in observation.keys() if k not in egocentric_obs_keys]

    egocentric_obs = {k: observation.pop(k) for k in egocentric_obs_keys}
    task_obs = {k: observation.pop(k) for k in task_obs_keys}
    
    # print(f"DEBUG: Tasks Obs: {task_obs}")
    # print(f"DEBUG: Ego Obs: {egocentric_obs}")

    # concatenate the observations
    task_obs_tensor = tf2_utils.batch_concat(task_obs)  # 1520 # this should be 1558
    egocentric_obs_tensor = tf2_utils.batch_concat(egocentric_obs)  # 196 # Scott: this should be 158

    return tf.concat([task_obs_tensor, egocentric_obs_tensor], axis=-1)  # hard coded for now


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
