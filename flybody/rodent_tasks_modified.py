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
from dm_control.locomotion.tasks.escape import _upright_reward

import numpy as np


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

    @property
    def task_observables(self):
        return self._task_observables


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

    @property
    def task_observables(self):
        return self._task_observables
    
    def get_reward(self, physics):
        walker_xvel = physics.bind(self._walker.root_body).subtree_linvel[0]
        xvel_term = rewards.tolerance(
            walker_xvel, (self._vel, self._vel),
            margin=self._vel,
            sigmoid='linear',
            value_at_margin=0.0)
        upright_reward = _upright_reward(physics, self._walker, deviation_angle=30)
        return xvel_term + 0.5 * upright_reward


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

    @property
    def task_observables(self):
        return self._task_observables


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
