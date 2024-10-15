from dm_control import composer
from dm_control.composer import variation
from dm_control.utils import rewards
import numpy as np
import collections
import random
from dm_control.suite.utils import randomizers


class MouseReachTask(composer.Task):
    """
    A task where a mouse (loaded from an XML file) reaches for a target.

    This task sets up a mouse to perform reaching movements to a randomized
    target position. The task rewards the mouse for reaching the target and penalizes
    for disallowed contacts or failing to stay upright.

    Attributes:
        _target_size: Float representing the size of the target.
        _target_pos: List of available target positions.
        _randomize_pose: Boolean indicating whether to randomize the initial mouse pose.
        _randomize_target: Boolean indicating whether to randomize the target position.
        _failure_termination: Boolean indicating whether the episode should be terminated.
    """

    def __init__(self,
                 mouse,
                 arena,
                 target_positions=None,
                 randomize_pose=False,
                 randomize_target=False,
                 contact_termination=False,
                 physics_timestep=0.002,
                 control_timestep=0.025):
        """
        Initializes the MouseReachTask.

        Args:
            mouse: An instance of a custom mouse entity loaded from an XML file.
            arena: An instance of an environment's arena (the space where the agent operates).
            target_size: Size of the target.
            target_positions: Optional list of predefined target positions.
            randomize_pose: Boolean, whether to randomize the initial pose.
            randomize_target: Boolean, whether to randomize the target position.
            contact_termination: Whether to terminate the episode upon invalid contact.
            physics_timestep: Physics simulation timestep.
            control_timestep: Control input timestep.
        """
        self._arena = arena
        self._mouse = mouse
        self._randomize_pose = randomize_pose
        self._randomize_target = randomize_target
        self._contact_termination = contact_termination
        self._target_pos = target_positions or [
                                                [-0.0045,     -0.00740453,  0.01371433],
                                                [-0.00598663, -0.00475719,  0.01332173],
                                                [-0.00660241, -0.00295503,  0.01092473],
                                                [-0.00598663, -0.00305373,  0.00792745],
                                                [-0.0045,     -0.00499547,  0.00608567],
                                                [-0.00301337, -0.00764281,  0.00647827],
                                                [-0.00239759, -0.00944497,  0.00887527],
                                                [-0.00301337, -0.00934627,  0.01187255]]

        self._reward_keys = ['distance_reward']

        # Attach the mouse to the arena
        self._arena.attach(self._mouse)

        # Enable default observables
        enabled_observables = self._mouse.observables
        
        # Set physics and control timesteps
        self.set_timesteps(
            physics_timestep=physics_timestep, control_timestep=control_timestep)
        
        self.last_reward_channels = []  # Initialize the last_reward_channels attribute

    @property
    def root_entity(self):
        """
        Defines the root entity of the task.

        Returns:
            The arena instance that serves as the root entity of the task.
        """
        return self._arena

    def initialize_episode_mjcf(self, random_state):
        """
        Initializes the MJCF model for the episode.

        Args:
            random_state: A NumPy random state instance for consistent randomization.
        """
        self._arena.regenerate(random_state)
        self._arena.mjcf_model.visual.map.znear = 0.00025
        self._arena.mjcf_model.visual.map.zfar = 4.0

    def initialize_episode(self, physics, random_state):
        """
        Sets the initial state of the environment at the start of each episode.

        Args:
            physics: A Physics object used for simulating the environment.
            random_state: A NumPy random state instance for consistent randomization.
        """
        self._target_sizes = [0.001, 0.002, 0.003]

        self._target_size = random.choice([self._target_sizes])[0]

        physics.named.model.geom_size['mouse/target',0] = self._target_size

        # Randomize the target position
        shoulder_to_fingertip = np.array([-0.0129, -0.0076, -0.0024])

        # Normalize the vector so it has a length of 1
        shoulder_to_fingertip = shoulder_to_fingertip / np.linalg.norm(shoulder_to_fingertip)

        selected_target = random.choice(self._target_pos)

        # Convert selected_target to a numpy array for easier manipulation
        selected_target = np.array(selected_target)

        shift_distance = self._target_size - .001

        shift_vector = shift_distance * shoulder_to_fingertip

        adjusted_target = selected_target + shift_vector

        physics.named.model.geom_pos['mouse/target', 'x'] = adjusted_target[0]
        physics.named.model.geom_pos['mouse/target', 'y'] = adjusted_target[1]
        physics.named.model.geom_pos['mouse/target', 'z'] = adjusted_target[2]

        # Reinitialize the mouse's pose
        self._mouse.reinitialize_pose(physics, random_state)

        # Reset failure termination condition
        self._failure_termination = False

    def _is_disallowed_contact(self, contact):
        """
        Determines whether a disallowed contact has occurred between the mouse and the arena.

        Args:
            contact: A contact object describing the interaction between two geoms.

        Returns:
            Boolean indicating whether the contact is disallowed.
        """
        set1, set2 = self._mouse_nonfoot_geomids, self._ground_geomids
        return ((contact.geom1 in set1 and contact.geom2 in set2) or
                (contact.geom1 in set2 and contact.geom2 in set1))

    def before_step(self, physics, action, random_state):
        """
        Applies actions to the mouse before the simulation step. Actuator limits are set to [-1, 1].

        Args:
            physics: A Physics object used for simulating the environment.
            action: The control input to be applied to the mouse.
            random_state: A NumPy random state instance for consistent randomization.
        """
        clipped_action = np.clip(action, -1.0, 1.0)
        self._mouse.apply_action(physics, clipped_action, random_state)

    def after_step(self, physics, random_state):
        """
        Checks for any conditions after a simulation step, such as failure termination.

        Args:
            physics: A Physics object used for simulating the environment.
            random_state: A NumPy random state instance for consistent randomization.
        """
        self._failure_termination = False
        if self._contact_termination:
            for c in physics.data.contact:
                if self._is_disallowed_contact(c):
                    self._failure_termination = True
                    break

    def get_reward(self, physics):
        """
        Calculates and returns the reward for the current step based on the distance
        between the mouse's finger tip and the target.

        Args:
            physics: A Physics object used for simulating the environment.

        Returns:
            Float representing the calculated reward based on the proximity to the target.
        """
        # Check for valid physics object and geom_xpos attribute
        if not hasattr(physics.named.data, 'geom_xpos'):
            raise ValueError("Invalid physics object: missing geom_xpos attribute")
        finger_to_target_dist = np.linalg.norm(
            physics.named.data.geom_xpos['mouse/target'] - physics.named.data.geom_xpos['mouse/finger_tip'])
        reward = rewards.tolerance(finger_to_target_dist, bounds=(0, self._target_size), margin=0.006)
        self.last_reward_channels = {'distance_reward': reward}
        return reward

    def should_terminate_episode(self, physics):
        """
        Returns whether the episode should be terminated due to failure conditions.

        Args:
            physics: A Physics object used for simulating the environment.

        Returns:
            Boolean indicating whether the episode should terminate.
        """
        return self._failure_termination

    def get_discount(self, physics):
        """
        Returns the discount factor for the current episode step, indicating whether
        the episode is continuing or terminating.

        Args:
            physics: A Physics object used for simulating the environment.

        Returns:
            Float representing the discount factor. Returns 0 if the episode should terminate,
            otherwise returns 1.
        """
        return 0. if self._failure_termination else 1.