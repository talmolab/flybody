from acme.tf import utils as tf2_utils
from acme.tf import networks
import tensorflow as tf
import sonnet as snt


class Decoder(snt.Module):
    """
    separate decoder structure for skills reuses
    """

    def __init__(
        self, decoder_layer_sizes, action_size, min_scale, tanh_mean, init_scale, fixed_scale, use_tfd_independent
    ):
        """
        decoder_layer_sizes: the size of the decoder layer
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
    '''encoder decoder now have the same size from the policy layer argument, decoder + latent'''
    def __init__(
        self,
        action_size,
        latent_layer_size,
        ref_size,
        min_scale,
        tanh_mean,
        init_scale,
        fixed_scale,
        use_tfd_independent,
        encoder_layer_sizes,
        decoder_layer_sizes,
    ):

        super().__init__()
        self.ref_size = ref_size
        self.latent_layer_size = latent_layer_size
        self.encoder = snt.Sequential(
            [
                tf2_utils.batch_concat,
                networks.LayerNormMLP(layer_sizes=encoder_layer_sizes, activate_final=True),
                networks.MultivariateNormalDiagHead(
                    latent_layer_size,
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
            init_scale=init_scale,
            fixed_scale=True,
            use_tfd_independent=use_tfd_independent,
        )

    def __call__(self, observations):
        reference_obs = observations[...,:self.ref_size]
        env_obs = observations[..., self.ref_size :]
        intentions_dist = self.encoder(tf2_utils.batch_concat(reference_obs))
        intentions = intentions_dist.sample()
        concatenated = tf.concat([intentions, env_obs], axis=-1)
        actions = self.decoder(tf2_utils.batch_concat(concatenated))
        return actions
