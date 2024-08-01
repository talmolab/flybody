from acme.tf import utils as tf2_utils
from acme.tf import networks
import tensorflow as tf
import sonnet as snt


class IntentionNetwork(snt.Module):
    '''encoder decoder now have the same size from the policy layer argument, decoder + latent'''
    def __init__(self,
                 action_size,
                 latent_size,
                 ref_size,
                 min_scale,
                 tanh_mean,
                 init_scale,
                 fixed_scale, 
                 use_tfd_independent,
                 policy_layer_sizes):
        
        super().__init__()
        self.ref_size = ref_size
        self.encoder = snt.Sequential([
            tf2_utils.batch_concat,
            networks.LayerNormMLP(layer_sizes=policy_layer_sizes[:-1], activate_final=True),
            snt.nets.MLP([latent_size], activate_final=False)
            ])
        
        self.decoder = snt.Sequential([
            networks.LayerNormMLP(layer_sizes=[latent_size] + policy_layer_sizes[-1:], activate_final=True),
            networks.MultivariateNormalDiagHead(
                action_size,
                min_scale=min_scale,
                tanh_mean=tanh_mean,
                init_scale=init_scale,
                fixed_scale=fixed_scale,
                use_tfd_independent=use_tfd_independent
            )
            ])

    def __call__(self, observations):
        reference_obs = observations[...,:self.ref_size]
        remaining_obs = observations[...,self.ref_size:]
        latent = self.encoder(tf2_utils.batch_concat(reference_obs))
        concatenated = tf.concat([latent,remaining_obs], axis=-1)
        policy_params = self.decoder(tf2_utils.batch_concat(concatenated))
        return policy_params