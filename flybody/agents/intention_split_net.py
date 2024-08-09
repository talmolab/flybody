import tensorflow as tf
import sonnet as snt
from acme.tf import utils as tf2_utils
from acme import types


def separate_observation(observation: types.NestedTensor) -> tf.Tensor:
    '''
    function similar to tf2_utils.batch_concat, but returns a 2D tensor 
    specifically for the intention network to take input into the encoder
    and decoder differently
    '''

    observation = observation.copy()
    # separate reference and non-reference observations
    reference_keys = [k for k in observation.keys() if 'reference' in k]
    non_reference_keys = [k for k in observation.keys() if 'reference' not in k]

    reference_observations = {k: observation.pop(k) for k in reference_keys}
    non_reference_observations = {k: observation.pop(k) for k in non_reference_keys}

    # concatenate the observations
    reference_tensor = tf2_utils.batch_concat(reference_observations) # 1520
    non_reference_tensor = tf2_utils.batch_concat(non_reference_observations) # 196

    return tf.concat([reference_tensor, non_reference_tensor], axis=-1) # hard coded for now