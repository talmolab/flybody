"""Network factories for distributed D4PG and DMPO agents."""

from typing import Optional, Callable

from acme.tf import utils as tf2_utils
from acme.tf import networks

import numpy as np
import sonnet as snt

from flybody.agents import losses_mpo

from flybody.agents.intention_split_net import separate_observation
from flybody.agents.intention_network_base import IntentionNetwork


def network_factory_dmpo(
    action_spec,
    encoder_layer_sizes=(512, 512, 512),
    decoder_layer_sizes=(512, 512, 512),
    critic_layer_sizes=(512, 512, 256),
    latent_layer_sizes=60,
    ref_size=1520,
    vmin=-150.0,
    vmax=150.0,
    num_atoms=51,
    min_scale=1e-6,
    tanh_mean=False,
    init_scale=0.7,
    fixed_scale=False,
    use_tfd_independent=True,
):
    """Networks for DMPO agent."""
    action_size = np.prod(action_spec.shape, dtype=int)

    policy_network = IntentionNetwork(
        action_size,
        latent_layer_sizes,
        ref_size,
        min_scale=min_scale,
        tanh_mean=tanh_mean,
        init_scale=init_scale,
        fixed_scale=fixed_scale,
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

    return {
        "policy": policy_network,
        "critic": critic_network,
        "observation": separate_observation,
        # can possibly return more stats for calculating the KL loss for the intention network? Can I?
    }


def make_network_factory_dmpo(
    encoder_layer_sizes=(512, 512, 512),
    decoder_layer_sizes=(512, 512, 512),
    critic_layer_sizes=(512, 512, 256),
    latent_layer_sizes=60,
    vmin=-150.0,
    vmax=150.0,
    num_atoms=51,
    min_scale=1e-6,
    tanh_mean=False,
    init_scale=0.7,
    fixed_scale=False,
    use_tfd_independent=True,
):
    """Returns network factory for distributed DMPO agent."""

    def network_factory(action_spec):
        return network_factory_dmpo(
            action_spec,
            encoder_layer_sizes=encoder_layer_sizes,
            decoder_layer_sizes=decoder_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            latent_layer_sizes=latent_layer_sizes,
            vmin=vmin,
            vmax=vmax,
            num_atoms=num_atoms,
            min_scale=min_scale,
            tanh_mean=tanh_mean,
            init_scale=init_scale,
            fixed_scale=fixed_scale,
            use_tfd_independent=use_tfd_independent,
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
