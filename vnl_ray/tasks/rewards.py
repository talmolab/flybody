"""Define reward function for imitation tasks."""

from typing import Dict
import numpy as np

from vnl_ray import quaternions


def compute_diffs(
    walker_features: Dict[str, np.ndarray],
    reference_features: Dict[str, np.ndarray],
    n: int = 2,
) -> Dict[str, float]:
    """Computes sums of absolute values of differences between components of
    model and reference features.

    Args:
        model_features, reference_features: Dictionaries of features to compute
            differences of.
        n: Exponent for differences. E.g., for squared differences use n = 2.

    Returns:
        Dictionary of differences, one value for each entry of input dictionary.
    """
    diffs = {}
    for k in walker_features:
        if "quat" not in k:
            # Regular vector differences.
            diffs[k] = np.sum(np.abs(walker_features[k] - reference_features[k]) ** n)
        else:
            # Quaternion differences (always positive, no need to use np.abs).
            diffs[k] = np.sum(quaternions.quat_dist_short_arc(walker_features[k], reference_features[k]) ** n)
    return diffs


def get_walker_features(physics, mocap_joints, mocap_sites):
    """Returns model pose features."""

    qpos = physics.bind(mocap_joints).qpos
    qvel = physics.bind(mocap_joints).qvel
    sites = physics.bind(mocap_sites).xpos
    root2site = quaternions.get_egocentric_vec(qpos[:3], sites, qpos[3:7])

    # Joint quaternions in local egocentric reference frame,
    # (except root quaternion, which is in world reference frame).
    root_quat = qpos[3:7]
    xaxis1 = physics.bind(mocap_joints).xaxis[1:, :]
    xaxis1 = quaternions.rotate_vec_with_quat(xaxis1, quaternions.reciprocal_quat(root_quat))
    qpos7 = qpos[7:]
    joint_quat = quaternions.joint_orientation_quat(xaxis1, qpos7)
    joint_quat = np.vstack((root_quat, joint_quat))

    model_features = {
        "com": qpos[:3],
        "qvel": qvel,
        "root2site": root2site,
        "joint_quat": joint_quat,
    }

    return model_features


def get_reference_features(reference_data, step):
    """Returns reference pose features."""

    qpos_ref = reference_data["qpos"][step, :]
    qvel_ref = reference_data["qvel"][step, :]
    root2site_ref = reference_data["root2site"][step, :]  # [step, :, :]
    joint_quat_ref = reference_data["joint_quat"][step, :]  # [step, :, :]
    joint_quat_ref = np.vstack((qpos_ref[3:7], joint_quat_ref))

    reference_features = {
        "com": reference_data["qpos"][step, :3],
        "qvel": qvel_ref,
        "root2site": root2site_ref,
        "joint_quat": joint_quat_ref,
    }

    return reference_features


def reward_factors_deep_mimic(walker_features, reference_features, std=None, weights=(1, 1, 1, 1)):
    """Returns four reward factors, each of which is a product of individual
    (unnormalized) Gaussian distributions evaluated for the four model
    and reference data features:
        1. Cartesian center-of-mass position, qpos[:3].
        2. qvel for all joints, including the root joint.
        3. Egocentric end-effector vectors.
        4. All joint orientation quaternions (in egocentric local reference
          frame), and the root quaternion.

    The reward factors are equivalent to the ones in the DeepMimic:
    https://arxiv.org/abs/1804.02717
    """
    if std is None:
        # Default values for fruitfly walking imitation task.
        std = {
            "com": 0.078487,
            "qvel": 53.7801,
            "root2site": 0.0735,
            "joint_quat": 1.2247,
        }

    diffs = compute_diffs(walker_features, reference_features, n=2)
    reward_factors = []
    for k in walker_features.keys():
        reward_factors.append(np.exp(-0.5 / std[k] ** 2 * diffs[k]))
    reward_factors = np.array(reward_factors)
    reward_factors *= np.asarray(weights)

    return reward_factors


# concat reward
# Copyright 2020 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Define reward function options for reference pose tasks."""

import collections

import numpy as np

RewardFnOutput = collections.namedtuple("RewardFnOutput", ["reward", "debug", "reward_terms"])


def bounded_quat_dist(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Computes a quaternion distance limiting the difference to a max of pi/2.

    This function supports an arbitrary number of batch dimensions, B.

    Args:
      source: a quaternion, shape (B, 4).
      target: another quaternion, shape (B, 4).

    Returns:
      Quaternion distance, shape (B, 1).
    """
    source /= np.linalg.norm(source, axis=-1, keepdims=True)
    target /= np.linalg.norm(target, axis=-1, keepdims=True)
    # "Distance" in interval [-1, 1].
    dist = 2 * np.einsum("...i,...i", source, target) ** 2 - 1
    # Clip at 1 to avoid occasional machine epsilon leak beyond 1.
    dist = np.minimum(1.0, dist)
    # Divide by 2 and add an axis to ensure consistency with expected return
    # shape and magnitude.
    return 0.5 * np.arccos(dist)[..., np.newaxis]


def sort_dict(d):
    return collections.OrderedDict(sorted(d.items()))


def compute_squared_differences(walker_features, reference_features, exclude_keys=()):
    """Computes squared differences of features."""
    squared_differences = {}
    for k in walker_features:
        if k not in exclude_keys:
            if "quaternion" not in k:
                squared_differences[k] = np.sum((walker_features[k] - reference_features[k]) ** 2)
            elif "quaternions" in k:
                quat_dists = bounded_quat_dist(walker_features[k], reference_features[k])
                squared_differences[k] = np.sum(quat_dists**2)
            else:
                squared_differences[k] = bounded_quat_dist(walker_features[k], reference_features[k]) ** 2

    return squared_differences


def termination_reward_fn(termination_error, termination_error_threshold, **unused_kwargs):
    """Termination error.

    This reward is intended to be used in conjunction with the termination error
    calculated in the task. Due to terminations if error > error_threshold this
    reward will be in [0, 1].

    Args:
      termination_error: termination error computed in tracking task
      termination_error_threshold: task termination threshold
      unused_kwargs: unused_kwargs

    Returns:
      RewardFnOutput tuple containing reward, debug information and reward terms.
    """
    debug_terms = {
        "termination_error": termination_error,
        "termination_error_threshold": termination_error_threshold,
    }
    termination_reward = 1 - termination_error / termination_error_threshold
    return RewardFnOutput(
        reward=termination_reward,
        debug=debug_terms,
        reward_terms=sort_dict({"termination": termination_reward}),
    )


def debug(reference_features, walker_features, **unused_kwargs):
    debug_terms = compute_squared_differences(walker_features, reference_features)
    return RewardFnOutput(reward=0.0, debug=debug_terms, reward_terms=None)


def multi_term_pose_reward_fn(walker_features, reference_features, **unused_kwargs):
    """A reward based on com, body quaternions, joints velocities & appendages."""
    differences = compute_squared_differences(walker_features, reference_features)
    # Original:
    # com = .1 * np.exp(-10 * differences['center_of_mass'])
    # joints_velocity = 1.0 * np.exp(-0.1 * differences['joints_velocity'])
    # appendages = 0.15 * np.exp(-40. * differences['appendages'])
    # body_quaternions = 0.65 * np.exp(-2 * differences['body_quaternions'])
    com = 1 * np.exp(-10 * differences["center_of_mass"])
    joints_velocity = 0.1 * np.exp(-differences["joints_velocity"])
    appendages = 0.15 * np.exp(-40.0 * differences["appendages"])
    body_quaternions = 0.65 * np.exp(-2 * differences["body_quaternions"])
    terms = {
        "center_of_mass": com,
        "joints_velocity": joints_velocity,
        "appendages": appendages,
        "body_quaternions": body_quaternions,
    }
    reward = sum(terms.values())
    return RewardFnOutput(reward=reward, debug=terms, reward_terms=sort_dict(terms))


def comic_reward_fn(
    termination_error,
    termination_error_threshold,
    walker_features,
    reference_features,
    reward_term_weights: dict = {
        "appendages": 1.0,
        "body_quaternions": 1.0,
        "center_of_mass": 1.0,
        "termination": 1.0,
        "joints_velocity": 1.0,
    },
    **unused_kwargs
):
    """A reward that mixes the termination_reward and multi_term_pose_reward.

    This reward function was used in
      Hasenclever et al.,
      CoMic: Complementary Task Learning & Mimicry for Reusable Skills,
      International Conference on Machine Learning, 2020.
      [https://proceedings.icml.cc/static/paper_files/icml/2020/5013-Paper.pdf]

    Args:
      termination_error: termination error as described
      termination_error_threshold: threshold to determine whether to terminate
        episodes. The threshold is used to construct a reward between [0, 1]
        based on the termination error.
      walker_features: Current features of the walker
      reference_features: features of the current reference pose
      reward_term_weights:  controls the weights of each reward terms in the multi_term_pose_reward_fn.
      unused_kwargs: unused addtional keyword arguments.

    Returns:
      RewardFnOutput tuple containing reward, debug terms and reward terms.
    """
    termination_reward, debug_terms, termination_reward_terms = termination_reward_fn(
        termination_error, termination_error_threshold
    )
    mt_reward, mt_debug_terms, mt_reward_terms = multi_term_pose_reward_fn(walker_features, reference_features)
    debug_terms.update(mt_debug_terms)
    reward_terms = {k: 0.5 * v / 5 for k, v in termination_reward_terms.items()}
    reward_terms.update({k: 0.5 * v for k, v in mt_reward_terms.items()})
    return RewardFnOutput(
        reward=0.5 * termination_reward / 5 + 0.5 * mt_reward,
        debug=debug_terms,
        reward_terms=sort_dict(reward_terms),
    )


_REWARD_FN = {
    "termination_reward": termination_reward_fn,
    "multi_term_pose_reward": multi_term_pose_reward_fn,
    "comic": comic_reward_fn,
}

_REWARD_CHANNELS = {
    "termination_reward": ("termination",),
    "multi_term_pose_reward": (
        "appendages",
        "body_quaternions",
        "center_of_mass",
        "joints_velocity",
    ),
    "comic": (
        "appendages",
        "body_quaternions",
        "center_of_mass",
        "termination",
        "joints_velocity",
    ),
}


def get_reward(reward_key):
    if reward_key not in _REWARD_FN:
        raise ValueError("Requested loss %s, which is not a valid option." % reward_key)

    return _REWARD_FN[reward_key]


def get_reward_channels(reward_key):
    if reward_key not in _REWARD_CHANNELS:
        raise ValueError("Requested loss %s, which is not a valid option." % reward_key)

    return _REWARD_CHANNELS[reward_key]
