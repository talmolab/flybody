from typing import List
from acme.tf import utils as tf2_utils
from acme.tf import networks
import tensorflow as tf
import sonnet as snt


class Decoder(snt.Module):
    """
    separate decoder structure for skills reuses
    """

    def __init__(
        self,
        decoder_layer_sizes,
        action_size,
        min_scale,
        tanh_mean,
        init_scale,
        fixed_scale,
        use_tfd_independent,
    ):
        """
        decoder_layer_sizes: the size of the decoder layer
        action_size: the action output size
        min_scale: the minimal scale for the action space.

        returns a tfd.distribution
        """
        super().__init__()
        self.MLP = networks.LayerNormMLP(layer_sizes=decoder_layer_sizes, activate_final=True)
        self.stochastic_actions = networks.MultivariateNormalDiagHead(
            action_size,
            min_scale=min_scale,
            tanh_mean=tanh_mean,
            init_scale=init_scale,
            fixed_scale=fixed_scale,
            use_tfd_independent=use_tfd_independent,
        )

    def __call__(self, inputs, return_activations=False):
        """
        Tether the module together to be the decoder module. Can record the activation if
        `return_activations` is True
        """
        x = self.MLP(inputs)
        x = self.stochastic_actions(x)
        return x


class IntentionNetwork(snt.Module):
    """encoder decoder now have the same size from the policy layer argument, decoder + latent"""

    def __init__(
        self,
        action_size: int,
        intention_size: int,
        task_obs_size: int,
        min_scale: float,
        tanh_mean: bool,
        init_scale: float,
        action_dist_scale: float,
        use_tfd_independent: bool,
        encoder_layer_sizes: List[int],
        decoder_layer_sizes: List[int],
        mid_layer_sizes: List[int] = None,
        high_level_intention_size: int | None = None,
    ):
        """
        action_size: the action size for the output of the network
        intention_size: specify the size of the intention stochastic layer
        task_obs_size: the tasks specific observation size.
        min_scale: float: specify the minimal scale of the stochastic layer
        tanh_mean: bool, whether we apply tanh_mean layer
        init_scale: float, the scale of of the stochastic layer that we initialize to
        action_dist_scale: float, the scale of the action output layers
        use_tfd_independent: bool, whether we use tfd independent to model the stochastic layer
        encoder_layer_sizes: List[int], specifies the layer sizes of the encoder
        decoder_layer_sizes: List[int], specifies the layer sizes of the decoder
        mid_layer_sizes: List[int], if specified, will create an additional high level motor intention stochastic layer
            useful in skill transfer of the multi-tasks
        high_level_intention_size: int, specify the high level intention stochastic layer sizes.
        """

        super().__init__()
        self.task_obs_size = task_obs_size
        self.action_size = action_size
        self.intention_size = intention_size
        self.use_multi_encoder = high_level_intention_size is not None
        self.mid_layer_sizes = mid_layer_sizes
        self.high_level_intention_size = high_level_intention_size
        if mid_layer_sizes is not None:
            self.high_level_encoder = snt.Sequential(
                [
                    tf2_utils.batch_concat,
                    networks.LayerNormMLP(layer_sizes=encoder_layer_sizes, activate_final=True),
                    networks.MultivariateNormalDiagHead(
                        num_dimensions=high_level_intention_size,
                        min_scale=min_scale,
                        tanh_mean=tanh_mean,
                        init_scale=init_scale,
                        fixed_scale=False,
                        use_tfd_independent=use_tfd_independent,
                    ),
                ]
            )
            self.mid_level_encoder = snt.Sequential(
                [
                    networks.LayerNormMLP(layer_sizes=mid_layer_sizes, activate_final=True),
                    networks.MultivariateNormalDiagHead(
                        intention_size,
                        min_scale=min_scale,
                        tanh_mean=tanh_mean,
                        init_scale=init_scale,
                        fixed_scale=False,
                        use_tfd_independent=use_tfd_independent,
                    ),
                ]
            )
        else:
            self.encoder = snt.Sequential(
                [
                    tf2_utils.batch_concat,
                    networks.LayerNormMLP(layer_sizes=encoder_layer_sizes, activate_final=True),
                    networks.MultivariateNormalDiagHead(
                        intention_size,
                        min_scale=min_scale,
                        tanh_mean=tanh_mean,
                        init_scale=init_scale,
                        fixed_scale=False,
                        use_tfd_independent=use_tfd_independent,
                    ),  # stochastic layer for the encoder.
                ]
            )

        self.decoder = Decoder(
            decoder_layer_sizes=decoder_layer_sizes,
            action_size=action_size,
            min_scale=min_scale,
            tanh_mean=tanh_mean,
            init_scale=action_dist_scale,
            fixed_scale=True,
            use_tfd_independent=use_tfd_independent,
        )

    def __call__(self, observations, return_intentions_dist=False):
        """
        split the observation tensor to task obs and egocentric obs, and pass through 
        the encoder -> intention -> decoder
        """
        # split the observation
        task_obs = observations[..., : self.task_obs_size] 
        egocentric_obs = observations[..., self.task_obs_size :]
        # feed into the encoder
        if self.high_level_intention_size is not None:
            # sample through the high level intention
            high_level_intention_dist = self.high_level_encoder(task_obs)
            hl_intention = high_level_intention_dist.sample()
            intentions_dist = self.mid_level_encoder(hl_intention)
            intentions = intentions_dist.sample()
        else:
            # maybe this batch-concat can be taken off as it already in the encoder?
            intentions_dist = self.encoder(tf2_utils.batch_concat(task_obs))
            intentions = intentions_dist.sample()
        concatenated = tf.concat([intentions, egocentric_obs], axis=-1)
        actions = self.decoder(tf2_utils.batch_concat(concatenated))
        if return_intentions_dist:
            return actions, intentions_dist
        return actions
