"""Escape locomotion tasks."""

import collections
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable as base_observable
from dm_control.composer.observation import observable as dm_observable
from dm_control.rl import control
from dm_control.utils import rewards
from dm_control.locomotion.tasks.escape import Escape
from dm_control.locomotion.tasks.corridors import RunThroughCorridor
from dm_control.locomotion.tasks.escape import _upright_reward, _HEIGHTFIELD_ID
from dm_control.locomotion.tasks.random_goal_maze import (
    DEFAULT_ALIVE_THRESHOLD,
    DEFAULT_CONTROL_TIMESTEP,
    DEFAULT_PHYSICS_TIMESTEP,
    ManyGoalsMaze,
)
from dm_control.locomotion.tasks.reach import (
    DEFAULT_CONTROL_TIMESTEP,
    DEFAULT_PHYSICS_TIMESTEP,
    TwoTouch,
)
from dm_control.locomotion.tasks.random_goal_maze import RepeatSingleGoalMazeAugmentedWithTargets

import numpy as np
from flybody.tasks.tracking_old import ReferencePosesTask


# add dummy task_logic observations
def dummy_task_logic(physics):
    del physics
    return np.array([0])


# add dummy origin observations
def dummy_origin(physics):
    del physics
    return np.array([0.0, 0.0, 0.0])


class EscapeSameObs(Escape):

    def __init__(
        self,
        walker,
        arena,
        walker_spawn_position=(0, 0, 0),
        walker_spawn_rotation=None,
        physics_timestep=0.005,
        control_timestep=0.025,
        reward_termination=False,  # controls whether we terminate the episode based on the history of the reward
        reward_threshold=0.5,  # controls what considered as a valid. (inspiration from basketball shot timer)
        reward_stale_timestep=150,  # controls the length of the timer
    ):
        super().__init__(
            walker,
            arena,
            walker_spawn_position,
            walker_spawn_rotation,
            physics_timestep,
            control_timestep,
        )
        self._task_observables = collections.OrderedDict()
        # add dummy task_logic observations
        self._task_observables["task_logic"] = dm_observable.Generic(dummy_task_logic)
        list(self._task_observables.values())[0].enabled = True
        self._reward_keys = ["escape_reward", "upright_reward", "escape_reward * upright_reward"]
        self._reward_termination = reward_termination
        self._reward_threshold = reward_threshold
        self._reward_stale_timestep = reward_stale_timestep
        self._reward_timer = -1
        self._failure_termination = False

    @property
    def task_observables(self):
        return self._task_observables

    def _reset_reward_channels(self):
        """Reset the reward channel fo"""
        if self._reward_keys:
            self.last_reward_channels = collections.OrderedDict([(k, 0.0) for k in self._reward_keys])
        else:
            self.last_reward_channels = None

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._reset_reward_channels()
        self._reward_timer = -1
        self._failure_termination = False

    def get_reward(self, physics):
        # Escape reward term.
        reward_channels = {}
        terrain_size = physics.model.hfield_size[_HEIGHTFIELD_ID, 0]
        escape_reward = rewards.tolerance(
            np.asarray(np.linalg.norm(physics.named.data.site_xpos[self._reward_body])),
            bounds=(terrain_size, float("inf")),
            margin=terrain_size,
            value_at_margin=0,
            sigmoid="linear",
        )
        upright_reward = _upright_reward(physics, self._walker, deviation_angle=30)
        reward_channels["escape_reward"] = float(escape_reward)
        reward_channels["upright_reward"] = float(upright_reward)
        reward_channels["escape_reward * upright_reward"] = float(upright_reward * escape_reward)
        self.last_reward_channels = reward_channels
        timestep_reward = float(upright_reward * escape_reward)
        if self._reward_termination:
            if timestep_reward < self._reward_threshold:
                self._reward_timer += 1  # increment the timer
            else:
                self._reward_timer = 0  # reset the timer
        return timestep_reward

    def after_step(self, physics, random_state):
        if self._reward_timer >= self._reward_stale_timestep:
            self._failure_termination = True

    def should_terminate_episode(self, physics):
        return self._failure_termination


class RunThroughCorridorSameObs(RunThroughCorridor):

    def __init__(
        self,
        walker,
        arena,
        walker_spawn_position=(0, 0, 0),
        walker_spawn_rotation=None,
        target_velocity=3,
        contact_termination=True,
        terminate_at_height=-0.5,
        physics_timestep=0.005,
        control_timestep=0.025,
        reward_termination=False,  # controls whether we terminate the episode based on the history of the reward
        reward_threshold=0.5,  # controls what considered as a valid. (inspiration from basketball shot timer)
        reward_stale_timestep=150,  # controls the length of the timer
    ):
        super().__init__(
            walker,
            arena,
            walker_spawn_position,
            walker_spawn_rotation,
            target_velocity,
            contact_termination,
            terminate_at_height,
            physics_timestep,
            control_timestep,
        )
        # add dummy task_logic observations
        self._task_observables = collections.OrderedDict()
        self._task_observables["task_logic"] = dm_observable.Generic(dummy_task_logic)
        list(self._task_observables.values())[0].enabled = True
        # add dummy origin observations
        self._walker.observables.add_observable("origin", base_observable.Generic(dummy_origin))
        self._reward_keys = ["walker_xvel", "upright_reward", "walker_xvel * upright_reward"]
        self._reward_termination = reward_termination
        self._reward_threshold = reward_threshold
        self._reward_stale_timestep = reward_stale_timestep
        self._reward_timer = -1

    @property
    def task_observables(self):
        return self._task_observables

    def _is_disallowed_contact(self, physics, contact):
        # Geoms that should trigger termination if they contact the ground
        specific_nonfoot_geom_names = {'pelvis', 'torso', 'vertebra_C1', 'vertebra_C3'}

        # Get geom ids for the specific non-foot geoms
        specific_nonfoot_geoms = [
            geom for geom in self._walker.mjcf_model.find_all('geom')
            if geom.name in specific_nonfoot_geom_names
        ]
        specific_nonfoot_geomids = set(physics.bind(specific_nonfoot_geoms).element_id)

        # Set to check contact with the ground
        set1, set2 = specific_nonfoot_geomids, self._ground_geomids
        return ((contact.geom1 in set1 and contact.geom2 in set2) or
                (contact.geom1 in set2 and contact.geom2 in set1))

    def after_step(self, physics, random_state):
        self._failure_termination = False
        if self._contact_termination:
            for c in physics.data.contact:
                if self._is_disallowed_contact(physics, c):
                    self._failure_termination = True
                    break
        if self._terminate_at_height is not None:
            if any(physics.bind(self._walker.end_effectors).xpos[:, -1] < self._terminate_at_height):
                self._failure_termination = True
        if self._reward_timer >= self._reward_stale_timestep:
            self._failure_termination = True

    def _reset_reward_channels(self):
        if self._reward_keys:
            self.last_reward_channels = collections.OrderedDict([(k, 0.0) for k in self._reward_keys])
        else:
            self.last_reward_channels = None

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._reset_reward_channels()
        self._reward_timer = -1
        self._failure_termination = False

    def get_reward(self, physics):
        """
        Custom reward function, with reward channel recording.
        """
        reward_channels = {}
        walker_xvel = physics.bind(self._walker.root_body).subtree_linvel[0]
        xvel_term = rewards.tolerance(
            walker_xvel, (self._vel, self._vel),
            margin=self._vel,
            sigmoid='linear',
            value_at_margin=0.0)
        upright_reward = _upright_reward(physics, self._walker, deviation_angle=30)
        reward_channels["walker_xvel"] = float(xvel_term)
        reward_channels["upright_reward"] = float(upright_reward)
        reward_channels["walker_xvel * upright_reward"] = float(xvel_term * upright_reward)  # de-reference them.
        self.last_reward_channels = reward_channels
        timestep_reward = float(xvel_term * upright_reward)
        if self._reward_termination:
            if timestep_reward < self._reward_threshold:
                self._reward_timer += 1  # increment the timer
            else:
                self._reward_timer = 0  # reset the timer
        return timestep_reward

# Aliveness in [-1., 0.].
DEFAULT_ALIVE_THRESHOLD = -0.5

DEFAULT_PHYSICS_TIMESTEP = 0.001
DEFAULT_CONTROL_TIMESTEP = 0.025


class ManyGoalsMazeSameObs(ManyGoalsMaze):

    def __init__(
        self,
        walker,
        maze_arena,
        target_builder,
        target_reward_scale=1.0,
        randomize_spawn_position=True,
        randomize_spawn_rotation=True,
        rotation_bias_factor=0.0,
        aliveness_reward=0.0,
        aliveness_threshold=DEFAULT_ALIVE_THRESHOLD,
        contact_termination=True,
        physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
        control_timestep=DEFAULT_CONTROL_TIMESTEP,
        reward_termination=False,  # controls whether we terminate the episode based on the history of the reward
        reward_threshold=1,  # controls what considered as a valid. (inspiration from basketball shot timer)
        reward_stale_timestep=300,  # controls the length of the timer
    ):
        super().__init__(
            walker,
            maze_arena,
            target_builder,
            target_reward_scale,
            randomize_spawn_position,
            randomize_spawn_rotation,
            rotation_bias_factor,
            aliveness_reward,
            aliveness_threshold,
            contact_termination,
            physics_timestep,
            control_timestep,
        )
        # add dummy task_logic observations
        self._task_observables = collections.OrderedDict()
        self._task_observables["task_logic"] = dm_observable.Generic(dummy_task_logic)
        # add dummy origin observations
        self._walker.observables.add_observable("origin", base_observable.Generic(dummy_origin))
        list(self._task_observables.values())[0].enabled = True
        self._reward_keys = ["aliveness_reward", "target_reward"]
        self._reward_termination = reward_termination
        self._reward_threshold = reward_threshold
        self._reward_stale_timestep = reward_stale_timestep
        self._reward_timer = -1
        self._failure_termination = False

    @property
    def task_observables(self):
        return self._task_observables

    def _reset_reward_channels(self):
        """Reset the reward channel fo"""
        if self._reward_keys:
            self.last_reward_channels = collections.OrderedDict([(k, 0.0) for k in self._reward_keys])
        else:
            self.last_reward_channels = None

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._reset_reward_channels()
        self._reward_timer = -1
        self._failure_termination = False
        
    def get_reward(self, physics):
        del physics
        reward_channels = {}
        reward_channels["aliveness_reward"] = float(self._aliveness_reward)
        reward = self._aliveness_reward
        for target_type, targets in enumerate(self._active_targets):
            for i, target in enumerate(targets):
                if target.activated and not self._target_rewarded[target_type][i]:
                    reward += self._target_type_rewards[target_type]
                    self._target_rewarded[target_type][i] = True
        reward_channels["target_reward"] = float(reward - self._aliveness_reward)
        self.last_reward_channels = reward_channels
        timestep_reward = float(reward)
        if self._reward_termination:
            if timestep_reward < self._reward_threshold:
                self._reward_timer += 1  # increment the timer
            else:
                self._reward_timer = 0  # reset the timer
        return timestep_reward

    def after_step(self, physics, random_state):
        if self._reward_timer >= self._reward_stale_timestep:
            self._failure_termination = True

    def should_terminate_episode(self, physics):
        return super().should_terminate_episode(physics) or self._failure_termination


class TwoTouchSamObs(TwoTouch):

    def __init__(
        self,
        walker,
        arena,
        target_builders,
        target_type_rewards,
        shuffle_target_builders=False,
        randomize_spawn_position=False,
        randomize_spawn_rotation=True,
        rotation_bias_factor=0,
        aliveness_reward=0,
        touch_interval=0.8,
        interval_tolerance=0.1,
        failure_timeout=1.2,
        reset_delay=0,
        z_height=0.14,
        target_area=(),
        physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
        control_timestep=DEFAULT_CONTROL_TIMESTEP,
        reward_termination=True,  # controls whether we terminate the episode based on the history of the reward
        reward_threshold=1,  # controls what considered as a valid. (inspiration from basketball shot timer)
        reward_stale_timestep=300,  # controls the length of the timer
    ):
        super().__init__(
            walker,
            arena,
            target_builders,
            target_type_rewards,
            shuffle_target_builders,
            randomize_spawn_position,
            randomize_spawn_rotation,
            rotation_bias_factor,
            aliveness_reward,
            touch_interval,
            interval_tolerance,
            failure_timeout,
            reset_delay,
            z_height,
            target_area,
            physics_timestep,
            control_timestep,
        )

        # add dummy origin observations
        self._walker.observables.add_observable("origin", base_observable.Generic(dummy_origin))
        self._reward_keys = ["aliveness_reward", "target_reward"]
        self._reward_termination = reward_termination
        self._reward_threshold = reward_threshold
        self._reward_stale_timestep = reward_stale_timestep

    def _reset_reward_channels(self):
        """Reset the reward channel fo"""
        if self._reward_keys:
            self.last_reward_channels = collections.OrderedDict([(k, 0.0) for k in self._reward_keys])
        else:
            self.last_reward_channels = None

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._reset_reward_channels()
        self._reward_timer = -1
        self._failure_termination = False

    def get_reward(self, physics):
        reward_channels = {}
        reward_channels["aliveness_reward"] = float(self._aliveness_reward)
        reward = super().get_reward(physics)
        reward_channels["target_reward"] = float(reward - self._aliveness_reward)
        self.last_reward_channels = reward_channels
        if self._reward_termination:
            if float(reward) < self._reward_threshold:
                self._reward_timer += 1  # increment the timer
            else:
                self._reward_timer = 0  # reset the timer
        return reward

    def after_step(self, physics, random_state):
        if self._reward_timer >= self._reward_stale_timestep:
            self._failure_termination = True

    def should_terminate_episode(self, physics):
        return super().should_terminate_episode(physics) or self._failure_termination
