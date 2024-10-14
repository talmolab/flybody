import functools
from typing import Dict, Sequence, Tuple

from absl import app
from absl import flags
from acme import specs
from acme import types
from acme.agents.tf import mompo
from acme.tf import networks
from acme.tf import utils as tf2_utils
import numpy as np
import sonnet as snt
import tensorflow as tf
from dm_control import suite
from acme import wrappers

def _uniform_initializer():
  return tf.initializers.VarianceScaling(
      distribution='uniform', mode='fan_out', scale=0.333)

class IntermediateActivationsPolicyNetwork(snt.Module):
    def __init__(self, layer_sizes, num_dimensions, tanh_mean, init_scale, fixed_scale, use_tfd_independent):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.num_dimensions = num_dimensions
        self.tanh_mean = tanh_mean
        self.init_scale = init_scale
        self.fixed_scale = fixed_scale
        self.use_tfd_independent = use_tfd_independent

        self.linear = snt.Linear(self.layer_sizes[0], w_init=None or _uniform_initializer())
        self.layer_norm = snt.LayerNorm(
            axis=slice(1, None), create_scale=True, create_offset=True
        )
        
        self.MLP = snt.nets.MLP(
                self.layer_sizes[1:],
                w_init=None or _uniform_initializer(),
                activation=tf.nn.elu,
                activate_final=True)
        
        self.MultivariateNormalDiagHead = networks.MultivariateNormalDiagHead(
                self.num_dimensions,
                init_scale=self.init_scale,
                use_tfd_independent=self.use_tfd_independent,
                tanh_mean=self.tanh_mean,
                fixed_scale=self.fixed_scale,
            )
    
    def __call__(self, inputs, return_activations=False):
        activations = dict()

        x = self.linear(inputs)
        x = self.layer_norm(x)
        x = tf.nn.tanh(x)
        activations["layernorm_tanh"] = x

        x = self.MLP(x)
        activations["mlp_elu"] = x

        x = self.MultivariateNormalDiagHead(x)

        if return_activations:
            return x, activations
        return x