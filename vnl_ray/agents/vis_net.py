import tensorflow as tf
import sonnet as snt
from acme.tf import utils as tf2_utils
from collections import OrderedDict
from vnl_ray.agents.utils_intention import separate_observation


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
        self._mean = 77.0  # TODO: Need to change it to rodent run gaps
        self._std = 56.0  # TODO

        # Visual network.
        self._layers = [
            snt.Conv2D(
                output_channels=2,
                kernel_shape=(3, 3),
                stride=1,
                padding="VALID",
                data_format="NHWC",
            ),
            tf.keras.layers.ReLU(),
            snt.Conv2D(
                output_channels=4,
                kernel_shape=(3, 3),
                stride=1,
                padding="VALID",
                data_format="NHWC",
            ),
            tf.keras.layers.ReLU(),
            snt.Conv2D(
                output_channels=8,
                kernel_shape=(3, 3),
                stride=2,
                padding="VALID",
                data_format="NHWC",
            ),
            tf.keras.layers.ReLU(),
            snt.Conv2D(
                output_channels=16,
                kernel_shape=(3, 3),
                stride=2,
                padding="VALID",
                data_format="NHWC",
            ),
            tf.keras.layers.ReLU(),
            snt.Flatten(),
            snt.Linear(output_size=vis_output_dim),
        ]

    def __call__(self, observation):

        # Copy to prevent modifying observation in-place.
        # (the modification is done with .pop() below.)
        observation = observation.copy()

        if not hasattr(self, "_task_input"):
            # If task input is present in the observation, it will be popped
            # and concatenated at specific position in the output vector.
            self._task_input = "walker/task_input" in observation.keys()

        # Pop eyes from `observation`.
        left_eye = tf.cast(observation.pop("walker/left_eye"), dtype=tf.float32)
        right_eye = tf.cast(observation.pop("walker/right_eye"), dtype=tf.float32)

        # If RGB, transform from RGB to 1-channel gray scale.
        if left_eye.shape[-1] == 3:  # Is RGB?
            left_eye = tf.reduce_mean(left_eye, axis=-1)
            right_eye = tf.reduce_mean(right_eye, axis=-1)
        # Normalize.
        left_eye = (left_eye - self._mean) / self._std
        right_eye = (right_eye - self._mean) / self._std
        # Stack the two eyes, shape (batch, height, width, channel=2).
        x = tf.stack((left_eye, right_eye), axis=-1)
        # print("DEBUG: Fly left/right eye: ", left_eye.shape, right_eye.shape)
        # print("DEBUG: Fly Vision x: ", x.shape)

        # Forward pass.
        for layer in self._layers:
            x = layer(x)

        if self._task_input:
            task_input = observation.pop("walker/task_input")
            # Concatenate the visual network output with the rest of
            # observations and task input.
            observation = tf2_utils.batch_concat(observation)
            out = tf.concat((task_input, x, observation), axis=-1)  # (batch, -1)
        else:
            # Concatenate the visual network output with the rest of observation.
            observation = tf2_utils.batch_concat(observation)
            out = tf.concat((x, observation), axis=-1)  # (batch, -1)

        return out


class AlexNet(snt.Module):
    def __init__(self, vis_output_dim: int = 8) -> None:
        super().__init__()
        super().__init__()

        # Convolutional Layers
        self.conv1 = snt.Conv2D(output_channels=96, kernel_shape=11, stride=4, padding="SAME")
        self.relu1 = tf.keras.layers.ReLU()
        self.lrn1 = tf.nn.local_response_normalization  # Local Response Normalization
        self.pool1 = snt.MaxPool2D(kernel_shape=3, stride=2)

        self.conv2 = snt.Conv2D(output_channels=256, kernel_shape=5, padding="SAME")
        self.relu2 = tf.keras.layers.ReLU()
        self.lrn2 = tf.nn.local_response_normalization
        self.pool2 = snt.MaxPool2D(kernel_shape=3, stride=2)

        self.conv3 = snt.Conv2D(output_channels=384, kernel_shape=3, padding="SAME")
        self.relu3 = tf.keras.layers.ReLU()

        self.conv4 = snt.Conv2D(output_channels=384, kernel_shape=3, padding="SAME")
        self.relu4 = tf.keras.layers.ReLU()

        self.conv5 = snt.Conv2D(output_channels=256, kernel_shape=3, padding="SAME")
        self.relu5 = tf.keras.layers.ReLU()
        self.pool5 = snt.MaxPool2D(kernel_shape=3, stride=2)

        # Flatten
        self.flatten = snt.Flatten()

        # Fully Connected Layers
        self.linear1 = snt.Linear(output_size=4096)
        self.relu6 = tf.keras.layers.ReLU()
        self.dropout1 = tf.keras.layers.Dropout(0.5)

        self.linear2 = snt.Linear(output_size=4096)
        self.relu7 = tf.keras.layers.ReLU()
        self.dropout2 = tf.keras.layers.Dropout(0.5)

        self.linear3 = snt.Linear(output_size=vis_output_dim)

    def forward(self, inputs, record_activations=False):
        """Forward pass of the AlexNet.

        Args:
          inputs: Input tensor.
          record_activations: Boolean indicating whether to record intermediate activations.

        Returns:
          Output tensor, or a dictionary of activations if record_activations is True.
        """
        activations = {}

        x = self.conv1(inputs)
        activations["conv1"] = x
        x = self.relu1(x)
        activations["relu1"] = x
        x = self.lrn1(x)
        activations["lrn1"] = x
        x = self.pool1(x)
        activations["pool1"] = x

        x = self.conv2(x)
        activations["conv2"] = x
        x = self.relu2(x)
        activations["relu2"] = x
        x = self.lrn2(x)
        activations["lrn2"] = x
        x = self.pool2(x)
        activations["pool2"] = x

        x = self.conv3(x)
        activations["conv3"] = x
        x = self.relu3(x)
        activations["relu3"] = x

        x = self.conv4(x)
        activations["conv4"] = x
        x = self.relu4(x)
        activations["relu4"] = x

        x = self.conv5(x)
        activations["conv5"] = x
        x = self.relu5(x)
        activations["relu5"] = x
        x = self.pool5(x)
        activations["pool5"] = x

        x = self.flatten(x)
        activations["flatten"] = x

        x = self.linear1(x)
        activations["linear1"] = x
        x = self.relu6(x)
        activations["relu6"] = x
        x = self.dropout1(x)
        activations["dropout1"] = x

        x = self.linear2(x)
        activations["linear2"] = x
        x = self.relu7(x)
        activations["relu7"] = x
        x = self.dropout2(x)
        activations["dropout2"] = x

        x = self.linear3(x)
        activations["linear3"] = x

        if record_activations:
            return activations
        else:
            return x

    def __call__(self, observation, record_activations=False) -> tf.Tensor:
        # Copy to prevent modifying observation in-place.
        # (the modification is done with .pop() below.)
        observation = observation.copy()
        # sort the observation space to make sure that the observation is consistent
        observation = OrderedDict(sorted(observation.items()))

        if not hasattr(self, "_task_input"):
            # If task input is present in the observation, it will be popped
            # and concatenated at specific position in the output vector.
            self._task_input = "task_logic" in observation.keys()

        # Pop eyes from `observation`.
        egocentric_camera = tf.cast(observation.pop("walker/egocentric_camera"), dtype=tf.float32)

        # If RGB, transform from RGB to 1-channel gray scale.
        if egocentric_camera.shape[-1] == 3:  # Is RGB?
            egocentric_camera = tf.reduce_mean(egocentric_camera, axis=-1)  # to gray-scale image
            # Normalize. # TODO: figure out the actual means and std
            egocentric_camera = (egocentric_camera - self._mean) / self._std
            # Stack the two eyes, shape (batch, height, width, channel=2).
            x = tf.expand_dims(egocentric_camera, axis=-1)

            # Forward pass.
            for layer in self._layers:
                x = layer(x)
            observation["task_logic"] = tf.cast(observation["task_logic"], tf.float32)
            # Concatenate the visual network output with the rest of
            # observations and task input.
            observation["walker/visual_features"] = x
        return separate_observation(observation)


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
        self._mean = 77.0
        self._std = 56.0

        # Visual network.

        self._layers = [
            snt.Conv2D(
                output_channels=2,
                kernel_shape=(3, 3),
                stride=1,
                padding="VALID",
                data_format="NHWC",
            ),
            tf.keras.layers.ReLU(),
            snt.Conv2D(
                output_channels=4,
                kernel_shape=(3, 3),
                stride=1,
                padding="VALID",
                data_format="NHWC",
            ),
            tf.keras.layers.ReLU(),
            snt.Conv2D(
                output_channels=8,
                kernel_shape=(3, 3),
                stride=2,
                padding="VALID",
                data_format="NHWC",
            ),
            tf.keras.layers.ReLU(),
            snt.Conv2D(
                output_channels=16,
                kernel_shape=(3, 3),
                stride=2,
                padding="VALID",
                data_format="NHWC",
            ),
            tf.keras.layers.ReLU(),
            snt.Flatten(),
            snt.Linear(output_size=vis_output_dim),
        ]

    def __call__(self, observation):

        # Copy to prevent modifying observation in-place.
        # (the modification is done with .pop() below.)
        observation = observation.copy()
        # sort the observation space to make sure that the observation is consistent
        observation = OrderedDict(sorted(observation.items()))

        if not hasattr(self, "_task_input"):
            # If task input is present in the observation, it will be popped
            # and concatenated at specific position in the output vector.
            self._task_input = "task_logic" in observation.keys()

        # Pop eyes from `observation`.
        egocentric_camera = tf.cast(observation.pop("walker/egocentric_camera"), dtype=tf.float32)

        # If RGB, transform from RGB to 1-channel gray scale.
        if egocentric_camera.shape[-1] == 3:  # Is RGB?
            egocentric_camera = tf.reduce_mean(egocentric_camera, axis=-1)  # to gray-scale image
            # Normalize. # TODO: figure out the actual means and std
            egocentric_camera = (egocentric_camera - self._mean) / self._std
            # Stack the two eyes, shape (batch, height, width, channel=2).
            x = tf.expand_dims(egocentric_camera, axis=-1)

            # Forward pass.
            for layer in self._layers:
                x = layer(x)
            observation["task_logic"] = tf.cast(observation["task_logic"], tf.float32)
            # Concatenate the visual network output with the rest of
            # observations and task input.
            observation["walker/visual_features"] = x
        return separate_observation(observation)
