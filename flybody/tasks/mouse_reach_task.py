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
                 target_size=0.1,
                 target_positions=None,
                 randomize_pose=False,
                 randomize_target=False,
                 contact_termination=False,
                 physics_timestep=0.005,
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
        self._target_size = target_size
        self._randomize_pose = randomize_pose
        self._randomize_target = randomize_target
        self._contact_termination = contact_termination
        self._target_pos = target_positions or [[0.7, 0.47, 1.6], [0.5, 0.3, 1.6]]

        # Attach the mouse to the arena
        self._arena.attach(self._mouse)

        # Enable default observables
        enabled_observables = self._mouse.observables
        
        # Set physics and control timesteps
        self.set_timesteps(
            physics_timestep=physics_timestep, control_timestep=control_timestep)

    @property
    def root_entity(self):
        """Defines the root entity of the task."""
        return self._arena

    def initialize_episode_mjcf(self, random_state):
        """Initializes the MJCF model for the episode."""
        self._arena.regenerate(random_state)
        self._arena.mjcf_model.visual.map.znear = 0.00025
        self._arena.mjcf_model.visual.map.zfar = 4.0

    def initialize_episode(self, physics, random_state):
        """Sets the initial state of the environment at the start of each episode."""
        # Randomize the pose of the mouse if required
        if self._randomize_pose:
            randomizers.randomize_limited_and_rotational_joints(physics, random_state)

        # Randomize the target position
        selected_target = random.choice(self._target_pos)
        physics.named.model.geom_pos['target', 'x'] = selected_target[0]
        physics.named.model.geom_pos['target', 'y'] = selected_target[1]
        physics.named.model.geom_pos['target', 'z'] = selected_target[2]

        # Reinitialize the mouse's pose
        self._mouse.reinitialize_pose(physics, random_state)

        # Reset failure termination condition
        self._failure_termination = False

    def _is_disallowed_contact(self, contact):
        """Determines whether a disallowed contact has occurred."""
        set1, set2 = self._mouse_nonfoot_geomids, self._ground_geomids
        return ((contact.geom1 in set1 and contact.geom2 in set2) or
                (contact.geom1 in set2 and contact.geom2 in set1))

    def before_step(self, physics, action, random_state):
        """Applies the action to the mouse before stepping the physics."""
        self._mouse.apply_action(physics, action, random_state)

    def after_step(self, physics, random_state):
        """Checks for any conditions after a step, such as failure termination."""
        self._failure_termination = False
        if self._contact_termination:
            for c in physics.data.contact:
                if self._is_disallowed_contact(c):
                    self._failure_termination = True
                    break

    def get_reward(self, physics):
        """Calculates and returns the reward for the current step."""
        finger_to_target_dist = np.linalg.norm(
            physics.named.data.geom_pos['target'] - physics.named.data.geom_pos['finger_tip'])
        return rewards.tolerance(finger_to_target_dist, bounds=(0, self._target_size), margin=0.1)

    def should_terminate_episode(self, physics):
        """Returns whether the episode should be terminated."""
        return self._failure_termination

    def get_discount(self, physics):
        """Returns the discount factor."""
        return 0. if self._failure_termination else 1.