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
from vnl_ray.agents.actors import DelayedFeedForwardActor
import tensorflow_probability as tfp
import pandas as pd

def _uniform_initializer():
  return tf.initializers.VarianceScaling(
      distribution='uniform', mode='fan_out', scale=0.333)

class CustomActorNetwork(snt.Module):
    def __init__(self, policy_network):
        # Initialize the parent snt.Module class
        super(CustomActorNetwork, self).__init__(name="CustomActorNetwork")

        self._policy_network = policy_network
        self._observation_network = separate_observation

    def __call__(self, inputs, return_activations=True):
        x = self._observation_network(inputs)
        #print(f"## ## ## INPUT TO POLICY NETWORK SHAPE # # # {x.shape}")  # Add this debug line
        x, activations = self._policy_network(x, return_activations=return_activations)
        #print(f"** ** ** activations: {activations}")
        #print(f"## ## ## OUTPUT OF POLICY NETWORK SHAPE # # # {x.shape}")  
        #print(f"## ## ## MEANHEAD OUPUT SHAPE  # # # {x.shape}")  
        return x, activations
    
class ActivationsAwareFeedForwardActor(DelayedFeedForwardActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activations = None
        self.activation_collection = []
        self.activation_count = 0
        self.episode_count = 0
        self.df = None
    
    @tf.function
    def _policy(self, observation: types.NestedTensor, return_activations: bool = True) -> types.NestedTensor:
        
        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)

        policy, self.activations = self._policy_network(batched_observation, return_activations=True)

        # Sample from the policy if it is stochastic.
        action = policy.sample() if isinstance(policy, tfp.distributions.Distribution) else policy

        return policy, self.activations

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        action, self.activations = self._policy(observation, return_activations=True)
        if self.activation_count < 50:
            self.activation_collection.append(self.activations)
            self.activation_count += 1

        if self.activation_count == 50:
            self.save_activations()
            self.activation_count = 0
            self.activation_collection.clear()
            self.episode_count += 1

        return tf2_utils.to_numpy_squeeze(action)
    
    def save_activations(self):
        if self.df is None:
            self.df = pd.DataFrame(columns=['layernorm_tanh', 'mlp_elu'])
        
        for activation in self.activation_collection:
            self.df.loc[len(self.df)] = {'layernorm_tanh': activation['layernorm_tanh'].numpy(), 'mlp_elu': activation['mlp_elu'].numpy()}

        # Write the accumulated DataFrame to an HDF5 file at the end of an episode
        self.df.to_hdf('data.h5', key='activations', mode='a')

class CustomEvaluatorNetwork(snt.Module):
    def __init__(self, modules, name=None):
        super(CustomEvaluatorNetwork, self).__init__(name="CustomEvaluatorNetwork")
        self._modules = modules
        self.activations = None

    def __call__(self, inputs, return_activations=True):
        outputs = inputs
        for module in self._modules:
            print(type(module))

        for module in self._modules:
            if "IntermediateActivationsPolicyNetwork" in str(type(module)) and return_activations:
                print("Activations module found!")
                outputs, self.activations = module(outputs, return_activations=True)
            else:
                outputs = module(outputs)
        if return_activations:
            return outputs, self.activations
        return outputs

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

        #print(f"Input to policy network shape: {inputs.shape}, rank: {len(inputs.shape)}")
        x = self.linear(inputs)
        #print(f"After Linear Layer: {x.shape}")

        x = self.layer_norm(x)
        #print(f"After LayerNorm: {x.shape}")

        x = tf.nn.tanh(x)
        activations["layernorm_tanh"] = x

        x = self.MLP(x)
        #print(f"After MLP: {x.shape}")
        activations["mlp_elu"] = x

        x = self.MultivariateNormalDiagHead(x)

        if return_activations:
            return x, activations
        return x