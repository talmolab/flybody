from typing import List
from acme.tf import utils as tf2_utils
from acme.tf import networks
import tensorflow as tf
import sonnet as snt
from vnl_ray.agents.utils_sonnet import Sequential, ReLU, NormalTanhDistribution
import functools

import pickle
from typing import Any
from etils import epath


def load_params(path: str) -> Any:
  with epath.Path(path).open('rb') as fin:
    buf = fin.read()
  return pickle.loads(buf)


class EncoderJAX(Sequential):
    """
    Encoder Object Loaded from the JAX checkpoint, produced by MJX
    """

    def __init__(
        self,
        layer_sizes: List[int],
        layer_norm: bool,
        intention_size: int,
        min_scale: float,
        module_name: str = "EncoderJAX",
    ):
        # need to separate the fc layer and the mean/var layer
        self.num_layers = len(layer_sizes)
        self.layers: List[callable] = []  # will be a list of network_obj
        for i, size in enumerate(layer_sizes):
            name = f"hidden_{i}"
            linear = snt.Linear(size, name=name)
            self.layers.append(linear)
            if layer_norm:
                ln_name = f"LayerNorm_{i}"
                # if we want to freeze the decoder layernorm, use the checkpoint stats
                # in __call__ of the layer norm.
                ln = snt.LayerNorm(axis=slice(1, None), create_scale=False, create_offset=False, name=ln_name)
                self.layers.append(ln)
            self.layers.append(ReLU(name=f"relu_{i}"))

        # linear layer for the stochastic mean and std
        fc2_mean = snt.Linear(intention_size, name=f"fc2_mean")
        fc2_logvar = snt.Linear(intention_size, name=f"fc2_logvar")
        self.layers.append(fc2_mean)
        self.layers.append(fc2_logvar)
        stochastic_actions = NormalTanhDistribution(intention_size, min_scale)
        self.layers.append(stochastic_actions)
        self.name_to_layer = {l.name: l for l in self.layers}
        super().__init__(layers=self.layers, name=module_name, need_activation=True)

    def __call__(self, inputs, return_activations=False, *args, **kwargs):
        """
        Call method of the sonnet module. If a layer is a tuple, it will return two outputs correspondingly. Only the last layer is allowed with a tuple
        """
        activations = {}
        outputs = inputs
        for i in range(self.num_layers):
            mod = self.name_to_layer[f"hidden_{i}"]
            outputs = mod(outputs)
            activations[mod.name] = outputs
            mod = self.name_to_layer[f"relu_{i}"]  # hard coded relu for now
            outputs = mod(outputs)
            activations[mod.name] = outputs  # record the activations
        mean = self.name_to_layer["fc2_mean"](outputs)
        activations["fc2_mean"] = mean
        log_var = self.name_to_layer["fc2_logvar"](outputs)
        activations["fc2_logvar"] = log_var
        input_to_dist = tf.concat([mean, log_var], axis=-1)
        outputs = self.name_to_layer["NormalTanhDist"](input_to_dist)
        activations["NormalTanhDist"] = outputs
        if return_activations:
            return outputs, activations
        return outputs

    def load_from_JAX_checkpoint(self, path):
        params = load_params(path)
        network = params[1]["params"]
        encoder_cpts = network["encoder"]
        transfer_names = list(filter(lambda x: "relu" not in x and "NormalTanhDist" not in x, self._layer_names))

        if set(transfer_names) != set(encoder_cpts.keys()):
            raise ValueError(
                f"Key mismatch between target model: {set(transfer_names)} and the checkpoint: {set(encoder_cpts.keys())}"
            )

        for key in transfer_names:
            module = self.name_to_layer[key]
            if "hidden" in key or key in ["fc2_mean", "fc2_logvar"]:
                module.w.assign(tf.convert_to_tensor(encoder_cpts[key]["kernel"], dtype=tf.float32))
                module.b.assign(tf.convert_to_tensor(encoder_cpts[key]["bias"], dtype=tf.float32))
            if "LayerNorm" in key:
                idx = self._layer_names.index(key)
                # change the layers in place.
                par = functools.partial(module, scale=encoder_cpts[key]["scale"], offset=encoder_cpts[key]["bias"])
                self.layers[idx] = par
                self.name_to_layer[key] = par
            print(f"Layer: {key} transferred!")


class DecoderJAX(Sequential):
    """
    Decoder object loaded from the JAX checkpoint, produced by MuJoCo MJX
    """

    def __init__(
        self,
        layer_sizes: List[int],
        layer_norm: bool,
        action_size: int,
        min_scale: float,
        module_name: str = "DecoderJAX",
    ):
        self.layers: List[callable] = []  # will be a list of network_obj
        for i, size in enumerate(layer_sizes):
            name = f"hidden_{i}"
            linear = snt.Linear(size, name=name)
            self.layers.append(linear)
            if layer_norm:
                ln_name = f"LayerNorm_{i}"
                # if we want to freeze the decoder layernorm, use the checkpoint stats
                # in __call__ of the layer norm.
                ln = snt.LayerNorm(axis=slice(1, None), create_scale=False, create_offset=False, name=ln_name)
                self.layers.append(ln)
            self.layers.append(ReLU(name=f"relu_{i}"))
        # linear layer for the stochastic mean and std
        linear = snt.Linear(action_size * 2, name=f"hidden_{len(layer_sizes)}")
        self.layers.append(linear)
        stochastic_actions = NormalTanhDistribution(action_size, min_scale)
        self.layers.append(stochastic_actions)
        self.name_to_layer = {l.name: l for l in self.layers}
        super().__init__(layers=self.layers, name=module_name, need_activation=True)

    def load_from_JAX_checkpoint(self, path):
        params = load_params(path)
        network = params[1]["params"]
        decoder_cpts = network["decoder"]
        transfer_names = list(filter(lambda x: "relu" not in x and "NormalTanhDist" not in x, self._layer_names))

        if set(transfer_names) != set(decoder_cpts.keys()):
            raise ValueError(
                f"Key mismatch between target model: {set(transfer_names)} and the checkpoint: {set(decoder_cpts.keys())}"
            )

        for key in transfer_names:
            module = self.name_to_layer[key]
            if "hidden" in key:
                module.w.assign(decoder_cpts[key]["kernel"])
                module.b.assign(decoder_cpts[key]["bias"])
            if "LayerNorm" in key:
                idx = self._layer_names.index(key)
                # change the layers in place.
                par = functools.partial(module, scale=decoder_cpts[key]["scale"], offset=decoder_cpts[key]["bias"])
                self.layers[idx] = par
                self.name_to_layer[key] = par
            print(f"Layer: {key} transferred!")


class IntentionNetworkJAX(snt.Module):
    """encoder decoder now have the same size from the policy layer argument, decoder + latent"""

    def __init__(
        self,
        action_size: int,
        intention_size: int,
        task_obs_size: int,
        min_scale: float,
        encoder_layer_sizes: List[int],
        decoder_layer_sizes: List[int],
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
        self.encoder = EncoderJAX(
            encoder_layer_sizes, layer_norm=True, intention_size=intention_size, min_scale=min_scale
        )
        self.decoder = DecoderJAX(decoder_layer_sizes, layer_norm=True, action_size=action_size, min_scale=min_scale)
    
    def load_from_JAX_checkpoint(self, path):
        # initialize the network with appropriate size
        self.encoder(tf.ones((1, self.task_obs_size), dtype=tf.float32))
        self.decoder(tf.ones((1, self.intention_size + 147), dtype=tf.float32)) # hardcoded!
        # load the checkpoint
        self.encoder.load_from_JAX_checkpoint(path)
        self.decoder.load_from_JAX_checkpoint(path)

    def __call__(self, observations, return_intentions_dist=False):
        """
        split the observation tensor to task obs and egocentric obs, and pass through
        the encoder -> intention -> decoder
        """
        # split the observation
        task_obs = observations[..., : self.task_obs_size]
        egocentric_obs = observations[..., self.task_obs_size :]
        # feed into the encoder
        # maybe this batch-concat can be taken off as it already in the encoder?
        intentions_dist = self.encoder(tf2_utils.batch_concat(task_obs))
        intentions = intentions_dist.sample()
        concatenated = tf.concat([intentions, egocentric_obs], axis=-1)
        actions = self.decoder(tf2_utils.batch_concat(concatenated))
        if return_intentions_dist:
            return actions, intentions_dist
        return actions


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
