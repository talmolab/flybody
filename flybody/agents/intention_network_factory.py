"""Network factories for distributed D4PG and DMPO agents."""

from typing import Optional, Callable

from acme.tf import utils as tf2_utils
from acme.tf import networks

import numpy as np
import sonnet as snt

from flybody.agents import losses_mpo
from flybody.agents.utils_intention import separate_observation
from flybody.agents.intention_network_base import IntentionNetwork
from flybody.agents.vis_net import VisNetRodent


def network_factory_dmpo(
    action_spec,
    task_obs_size: int,
    encoder_layer_sizes=(512, 512, 512),
    decoder_layer_sizes=(512, 512, 512),
    critic_layer_sizes=(512, 512, 256),
    intention_size=60,
    vmin=-150.0,
    vmax=150.0,
    num_atoms=51,
    min_scale=1e-6,
    tanh_mean=False,
    init_scale=1.0,
    action_dist_scale=0.1,
    use_tfd_independent=True,
    use_visual_network=False,
    visual_feature_size:int=0,
):
    """Networks for DMPO agent."""
    action_size = np.prod(action_spec.shape, dtype=int)

    policy_network = IntentionNetwork(
        action_size=action_size,
        intention_size=intention_size,
        task_obs_size=task_obs_size,
        min_scale=min_scale,
        tanh_mean=tanh_mean,
        init_scale=init_scale,
        action_dist_scale=action_dist_scale,
        use_tfd_independent=use_tfd_independent,
        encoder_layer_sizes=encoder_layer_sizes,
        decoder_layer_sizes=decoder_layer_sizes,
    )

    # The multiplexer concatenates the (maybe transformed) observations/actions.
    critic_network = networks.CriticMultiplexer(
        action_network=networks.ClipToSpec(action_spec),
        critic_network=networks.LayerNormMLP(layer_sizes=critic_layer_sizes, activate_final=True),
    )
    critic_network = snt.Sequential(
        [
            critic_network,
            networks.DiscreteValuedHead(vmin=vmin, vmax=vmax, num_atoms=num_atoms),
        ]
    )
    networks_out = {
        "policy": policy_network,
        "critic": critic_network,
    }
    if use_visual_network:
        if visual_feature_size == 0:
            raise ValueError("Use visual network but vis feature size is 0")
        networks_out["observation"] = VisNetRodent(vis_output_dim=visual_feature_size)
    else:
        networks_out["observation"] = separate_observation
    return networks_out 


def make_network_factory_dmpo(
    task_obs_size: int, # required for routing obs.
    encoder_layer_sizes=(512, 512, 512),
    decoder_layer_sizes=(512, 512, 512),
    critic_layer_sizes=(512, 512, 256),
    intention_size=60,
    vmin=-150.0,
    vmax=150.0,
    num_atoms=51,
    min_scale=1e-6,
    tanh_mean=False,
    init_scale=0.7,
    use_tfd_independent=True,
    use_visual_network:bool=False,
    visual_feature_size:int=0,
):
    """Returns network factory for distributed DMPO agent."""

    def network_factory(action_spec):
        return network_factory_dmpo(
            action_spec=action_spec,
            task_obs_size=task_obs_size,
            encoder_layer_sizes=encoder_layer_sizes,
            decoder_layer_sizes=decoder_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            intention_size=intention_size,
            vmin=vmin,
            vmax=vmax,
            num_atoms=num_atoms,
            min_scale=min_scale,
            tanh_mean=tanh_mean,
            init_scale=init_scale,
            use_tfd_independent=use_tfd_independent,
            use_visual_network=use_visual_network,
            visual_feature_size=visual_feature_size,
        )

    return network_factory


def policy_loss_module_dmpo(
    epsilon: float = 0.1,
    epsilon_penalty: float = 0.001,
    epsilon_mean: float = 0.0025,
    epsilon_stddev: float = 1e-7,
    init_log_temperature: float = 10.0,
    init_log_alpha_mean: float = 10.0,
    init_log_alpha_stddev: float = 1000.0,
    action_penalization: bool = True,
    per_dim_constraining: bool = True,
    penalization_cost: Optional[Callable] = None,
):
    """Returns policy loss module for DMPO agent."""
    return losses_mpo.MPO(
        epsilon=epsilon,
        epsilon_penalty=epsilon_penalty,
        epsilon_mean=epsilon_mean,
        epsilon_stddev=epsilon_stddev,
        init_log_temperature=init_log_temperature,
        init_log_alpha_mean=init_log_alpha_mean,
        init_log_alpha_stddev=init_log_alpha_stddev,
        action_penalization=action_penalization,
        per_dim_constraining=per_dim_constraining,
        penalization_cost=penalization_cost,
    )
