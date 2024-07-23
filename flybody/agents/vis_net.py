import tensorflow as tf
import sonnet as snt
from acme.tf import utils as tf2_utils


class VisNetFly(snt.Module):
    """Visual network for 'simple' eyes with no buffers."""

    def __init__(self, vis_output_dim=8):
        """Visual convolutional network. It separates the 'walker/left_eye' and
        'walker/right_eye' observables from the full input, processes them, and
        concatenates the result with the rest of the observations, which bypass
        the visual processing conv net.

        Args:
            vis_output_dim: Output dimension of the visual processing network.
                It will be concatenated with the rest of the observation, which
                just bypasses the visual processing.
        """

        super().__init__()

        # Mean and std from the "trench" task.
        self._mean = 77. # TODO: Need to change it to rodent run gaps
        self._std = 56. # TODO

        # Visual network.
        self._layers = [
            snt.Conv2D(output_channels=2, kernel_shape=(3, 3),
                       stride=1, padding='VALID', data_format='NHWC'),
            tf.keras.layers.ReLU(),
            snt.Conv2D(output_channels=4, kernel_shape=(3, 3),
                       stride=1, padding='VALID', data_format='NHWC'),
            tf.keras.layers.ReLU(),
            snt.Conv2D(output_channels=8, kernel_shape=(3, 3),
                       stride=2, padding='VALID', data_format='NHWC'),
            tf.keras.layers.ReLU(),
            snt.Conv2D(output_channels=16, kernel_shape=(3, 3),
                       stride=2, padding='VALID', data_format='NHWC'),
            tf.keras.layers.ReLU(),
            snt.Flatten(),
            snt.Linear(output_size=vis_output_dim),
        ]

    def __call__(self, observation):

        # Copy to prevent modifying observation in-place.
        # (the modification is done with .pop() below.)
        observation = observation.copy()

        if not hasattr(self, '_task_input'):
            # If task input is present in the observation, it will be popped
            # and concatenated at specific position in the output vector.
            self._task_input = 'walker/task_input' in observation.keys()

        # Pop eyes from `observation`.
        left_eye = tf.cast(observation.pop('walker/left_eye'), dtype=tf.float32)
        right_eye = tf.cast(observation.pop('walker/right_eye'), dtype=tf.float32)

        # If RGB, transform from RGB to 1-channel gray scale.
        if left_eye.shape[-1] == 3:  # Is RGB?
            left_eye = tf.reduce_mean(left_eye, axis=-1)
            right_eye = tf.reduce_mean(right_eye, axis=-1)
        # Normalize.
        left_eye = (left_eye - self._mean) / self._std
        right_eye = (right_eye - self._mean) / self._std
        # Stack the two eyes, shape (batch, height, width, channel=2).
        x = tf.stack((left_eye, right_eye), axis=-1)
        print("DEBUG: Fly left/right eye: ", left_eye.shape, right_eye.shape)
        print("DEBUG: Fly Vision x: ", x.shape)

        # Forward pass.
        for layer in self._layers:
            x = layer(x)

        if self._task_input:
            task_input = observation.pop('walker/task_input')
            # Concatenate the visual network output with the rest of
            # observations and task input.
            observation = tf2_utils.batch_concat(observation)
            out = tf.concat((task_input, x, observation), axis=-1)  # (batch, -1)
        else:
            # Concatenate the visual network output with the rest of observation.
            observation = tf2_utils.batch_concat(observation)
            out = tf.concat((x, observation), axis=-1)  # (batch, -1)

        return out


class VisNetRodent(snt.Module):
    """Visual network for 'simple' eyes with no buffers."""

    def __init__(self, vis_output_dim=8):
        """Visual convolutional network. It separates the 'walker/left_eye' and
        'walker/right_eye' observables from the full input, processes them, and
        concatenates the result with the rest of the observations, which bypass
        the visual processing conv net.

        Args:
            vis_output_dim: Output dimension of the visual processing network.
                It will be concatenated with the rest of the observation, which
                just bypasses the visual processing.
        """

        super().__init__()

        # Mean and std from the "trench" task.
        self._mean = 77.
        self._std = 56.

        # Visual network.
        self._layers = [
            snt.Conv2D(output_channels=2, kernel_shape=(3, 3),
                       stride=1, padding='VALID', data_format='NHWC'),
            tf.keras.layers.ReLU(),
            snt.Conv2D(output_channels=4, kernel_shape=(3, 3),
                       stride=1, padding='VALID', data_format='NHWC'),
            tf.keras.layers.ReLU(),
            snt.Conv2D(output_channels=8, kernel_shape=(3, 3),
                       stride=2, padding='VALID', data_format='NHWC'),
            tf.keras.layers.ReLU(),
            snt.Conv2D(output_channels=16, kernel_shape=(3, 3),
                       stride=2, padding='VALID', data_format='NHWC'),
            tf.keras.layers.ReLU(),
            snt.Flatten(),
            snt.Linear(output_size=vis_output_dim),
        ]

    def __call__(self, observation):

        # Copy to prevent modifying observation in-place.
        # (the modification is done with .pop() below.)
        observation = observation.copy()

        if not hasattr(self, '_task_input'):
            # If task input is present in the observation, it will be popped
            # and concatenated at specific position in the output vector.
            self._task_input = 'task_logic' in observation.keys()

        # Pop eyes from `observation`.
        egocentric_camera = tf.cast(observation.pop('walker/egocentric_camera'), dtype=tf.float32)

        # If RGB, transform from RGB to 1-channel gray scale.
        if egocentric_camera.shape[-1] == 3:  # Is RGB?
            egocentric_camera = tf.reduce_mean(egocentric_camera, axis=-1)
        # Normalize.
        egocentric_camera = (egocentric_camera - self._mean) / self._std
        # Stack the two eyes, shape (batch, height, width, channel=2).
        x = tf.expand_dims(egocentric_camera, axis=-1)

        # Forward pass.
        for layer in self._layers:
            x = layer(x)

        if self._task_input:
            task_input = observation.pop('task_logic')
            task_input = tf.cast(task_input, tf.float32)
            # Concatenate the visual network output with the rest of
            # observations and task input.
            observation = tf2_utils.batch_concat(observation)
            out = tf.concat((task_input, x, observation), axis=-1)  # (batch, -1)
        else:
            # Concatenate the visual network output with the rest of observation.
            observation = tf2_utils.batch_concat(observation)
            out = tf.concat((x, observation), axis=-1)  # (batch, -1)

        return out
