"Modified Tracking for Rodent Imitation MultiClip Learning"

from typing import Any, Callable, Sequence, Text
from dm_control.mjcf.physics import Physics
from matplotlib.pylab import RandomState
import numpy as np
from absl import logging
from dm_control.composer.arena import Arena
from dm_control.locomotion.tasks.reference_pose.tracking import (
    DEFAULT_PHYSICS_TIMESTEP,
    MultiClipMocapTracking,
)
from dm_control.locomotion.walkers.legacy_base import Walker
from dm_control.suite.utils.randomizers import randomize_limited_and_rotational_joints
import tree
from dm_control.locomotion.tasks.reference_pose import utils
from dm_control.mujoco.wrapper import mjbindings

mjlib = mjbindings.mjlib


DEFAULT_PHYSICS_TIMESTEP = 0.005


class MultiClipMocapTracking(MultiClipMocapTracking):
    def __init__(
        self,
        walker: Callable[..., Walker],
        arena: Arena,
        ref_path: str,
        ref_steps: Sequence[int],
        dataset: str | Sequence[Any],
        termination_error_threshold: float = 0.3,
        prop_termination_error_threshold: float = 0.1,
        min_steps: int = 10,
        reward_type: str = "termination_reward",
        physics_timestep: float = DEFAULT_PHYSICS_TIMESTEP,
        always_init_at_clip_start: bool = False,
        proto_modifier: Any | None = None,
        prop_factory: Any | None = None,
        disable_props: bool = True,
        ghost_offset: Sequence[int | float] | None = None,
        body_error_multiplier: int | float = 1,
        actuator_force_coeff: float = 0.015,
        enabled_reference_observables: Sequence[str] | None = None,
    ):
        super().__init__(
            walker,
            arena,
            ref_path,
            ref_steps,
            dataset,
            termination_error_threshold,
            prop_termination_error_threshold,
            min_steps,
            reward_type,
            physics_timestep,
            always_init_at_clip_start,
            proto_modifier,
            prop_factory,
            disable_props,
            ghost_offset,
            body_error_multiplier,
            actuator_force_coeff,
            enabled_reference_observables,
        )

    def get_clip_id(self, physics: "mjcf.Physics"):
        """Observation of the clip id."""
        del physics  # physics unused by get_clip_id.
        return np.array([self._current_clip_index]).astype(float)

    def initialize_episode(self, physics: Physics, random_state: RandomState):
        """
        initialize the episode with randomization without any floor contact
        """
        penetrating = True
        while penetrating:
            randomize_limited_and_rotational_joints(physics, None)
            # Check for collisions.
            physics.after_reset()
            penetrating = physics.data.ncon > 0
        return super().initialize_episode(physics, random_state)

    def _set_walker(self, physics: Physics):
        timestep_features = tree.map_structure(lambda x: x[self._time_step], self._clip_reference_features)
        utils.set_walker_from_features(physics, self._walker, timestep_features, offset=(0, 0, 0.01))
        if self._props:
            utils.set_props_from_features(physics, self._props, timestep_features)
            mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)

    def _get_clip_to_track(self, random_state: np.random.RandomState):
        # Randomly select a starting point.
        index = random_state.choice(len(self._possible_starts), p=self._start_probabilities)
        clip_index, start_step = self._possible_starts[index]

        self._current_clip_index = clip_index
        clip_id = self._dataset.ids[self._current_clip_index]

        if self._all_clips[self._current_clip_index] is None:
            # fetch selected trajectory
            logging.info("Loading clip %s", clip_id)
            self._all_clips[self._current_clip_index] = self._loader.get_trajectory(
                clip_id,
                start_step=self._dataset.start_steps[self._current_clip_index],
                end_step=self._dataset.end_steps[self._current_clip_index],
                zero_out_velocities=True,
            )
            self._current_clip = self._all_clips[self._current_clip_index]
            self._clip_reference_features = self._current_clip.as_dict()
        self._strip_reference_prefix()

        # The reference features are already restricted to
        # clip_start_step:clip_end_step. However start_step is in
        # [clip_start_step:clip_end_step]. Hence we subtract clip_start_step to
        # obtain a valid index for the reference features.
        self._time_step = start_step - self._dataset.start_steps[self._current_clip_index]
        self._current_start_time = (
            start_step - self._dataset.start_steps[self._current_clip_index]
        ) * self._current_clip.dt
        self._last_step = len(self._clip_reference_features["joints"]) - self._max_ref_step - 1
        logging.info(
            "Mocap %s at step %d with remaining length %d.",
            clip_id,
            start_step,
            self._last_step - start_step,
        )
