import sonnet as snt


class Sequential(snt.Sequential):
    """
    sub-class of `snt.Sequential` to make
    intermediate activation accessible
    """

    def __init__(self, layers, name=None, layer_names=()):
        """
        Initializer of the custom sequential methods

        layers: list of layers that the neural network is operating on
        name: name of the Sequential Module
        activation bool: whether it returns activations of each intermediate layers
        layer_names list: list of layer names within the sequential layer
        """
        super().__init__(layers, name)
        self._layer_names = layer_names
        if len(layer_names) != 0:
            assert len(layers) == len(layer_names), "length mismatch between layers and layer_names."

    def __call__(self, inputs, return_activations=False, *args, **kwargs):
        """ """
        if return_activations:
            assert len(self._layers) == len(self._layer_names), "length mismatch between layers and layer_names."
            activations = {}
        outputs = inputs
        for i, mod in enumerate(self._layers):
            if i == 0:
                # Pass additional arguments to the first layer.
                outputs = mod(outputs, *args, **kwargs)
            else:
                if isinstance(mod, Sequential) and return_activations:
                    outputs, act = mod(outputs, return_activations)  # need to think about more about this TODO:
                else:
                    outputs = mod(outputs)
            if return_activations:
                activations[self._layer_names[i]] = outputs
        if return_activations:
            return outputs, activations
        return outputs
