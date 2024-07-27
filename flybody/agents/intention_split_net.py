import tensorflow as tf
import sonnet as snt
from acme.tf import utils as tf2_utils

class SeparateObservations(snt.Module):
    """Module to separate and concatenate observations into reference and non-reference parts."""

    def __init__(self):
        super().__init__()

    def __call__(self, observation):
        # copy to prevent modifying observation in-place
        observation = observation.copy()

        # separate reference and non-reference observations
        reference_keys = [k for k in observation.keys() if 'reference' in k]
        non_reference_keys = [k for k in observation.keys() if 'reference' not in k]

        reference_observations = {k: observation.pop(k) for k in reference_keys}
        non_reference_observations = {k: observation.pop(k) for k in non_reference_keys}

        # concatenate the observations
        reference_tensor = tf2_utils.batch_concat(reference_observations)
        non_reference_tensor = tf2_utils.batch_concat(non_reference_observations)

        return [non_reference_tensor, reference_tensor]
