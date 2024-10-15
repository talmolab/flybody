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
from acme.tf import networks as network_utils
from dm_control import suite
from acme import wrappers
from vnl_ray.agents.utils_intention import separate_observation

def _uniform_initializer():
  return tf.initializers.VarianceScaling(
      distribution='uniform', mode='fan_out', scale=0.333)

class custom_actor(snt.Module):
    def __init__(self, policy_network):
        # Initialize the parent snt.Module class
        super(custom_actor, self).__init__(name="custom_actor")

        self._policy_network = policy_network
        self._observation_network = separate_observation
        self._samplehead = network_utils.StochasticSamplingHead()

    def __call__(self, inputs, return_activations=True):
        x = self._observation_network(inputs)
        #print(f"## ## ## INPUT TO POLICY NETWORK SHAPE # # # {x.shape}")  # Add this debug line
        x, activations = self._policy_network(x, return_activations=return_activations)
        #print(f"** ** ** activations: {activations}")
        #print(f"## ## ## OUTPUT OF POLICY NETWORK SHAPE # # # {x.shape}")  
        x = self._samplehead(x)
        #print(f"## ## ## MEANHEAD OUPUT SHAPE  # # # {x.shape}")  
        return x, activations
    
class custom_evaluator(snt.Module):
    def __init__(self, policy_network):
        # Initialize the parent snt.Module class
        super(custom_evaluator, self).__init__(name="custom_evaluator")

        self._policy_network = policy_network
        self._observation_network = separate_observation
        self._meanhead = network_utils.StochasticMeanHead()

    def __call__(self, inputs, return_activations=True):
        x = self._observation_network(inputs)
        print(f"## ## ## INPUT TO POLICY NETWORK SHAPE # # # {x.shape}")  # Add this debug line
        x, activations = self._policy_network(x, return_activations=return_activations)
        #print(f"** ** ** activations: {activations}")
        #print(f"## ## ## OUTPUT OF POLICY NETWORK SHAPE # # # {x.shape}")  
        x = self._meanhead(x)
        #print(f"## ## ## MEANHEAD OUPUT SHAPE  # # # {x.shape}")  
        return x, activations

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

        #print(f"Before Linear Layer: {inputs.shape}")
        x = self.linear(inputs)
        #print(f"After Linear Layer: {x.shape}")

        x = self.layer_norm(x)
        print(f"After LayerNorm: {x.shape}")

        x = tf.nn.tanh(x)
        activations["layernorm_tanh"] = x

        x = self.MLP(x)
        #print(f"After MLP: {x.shape}")
        activations["mlp_elu"] = x

        x = self.MultivariateNormalDiagHead(x)

        if return_activations:
            return x, activations
        return x