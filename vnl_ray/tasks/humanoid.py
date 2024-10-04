# ruff: noqa: F821

import functools

import numpy as np

from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import bowl
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.arenas import labmaze_textures
from dm_control.locomotion.arenas import mazes
from dm_control.locomotion.props import target_sphere
from dm_control.locomotion.tasks import RunThroughCorridor

from dm_control.locomotion import walkers
from dm_control.utils import io as resources

from dm_control.utils import io as resources
from dm_control.locomotion.tasks.reference_pose import types
import os
import h5py

from vnl_ray.tasks import rodent_tasks_modified as T

# from vnl_ray import rodent_walker as rodent
from dm_control.locomotion.walkers import rodent

from vnl_ray.tasks import (
    tracking_old as tracking,
)  # TODO hacky tape, new tracking did not work yet

from vnl_ray.tasks.trajectory_loaders import (
    HDF5WalkingTrajectoryLoader,
    InferenceWalkingTrajectoryLoader,
)

_CONTROL_TIMESTEP = 0.02
_PHYSICS_TIMESTEP = 0.001
GHOST_OFFSET = np.array((0, 0, 0))


def walk_humanoid(
    random_state: np.random.RandomState | None = None,
):
    arena = floors.Floor()
    walker = walkers.CMUHumanoidPositionControlledV2020()
    task = T.RunThroughCorridorSameObs(
        walker=walker,
        arena=arena,
        walker_spawn_position=(5, 0, 0),
        walker_spawn_rotation=0,
        target_velocity=1.0,
        contact_termination=True,
        terminate_at_height=-0.3,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP,
        enabled_vision=False,
        reward_termination=True
    )
    task._walker.observables.egocentric_camera.enabled=False
    
    return composer.Environment(
        time_limit=30,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )


def walk_humanoid_imitation(
    ref_path: str | None = None,
    random_state: np.random.RandomState | None = None,
    termination_error_threshold: float = 0.3,
):
    """
    Rodent walking imitation, following similar calling with fruitfly imitation
    """
    arena = floors.Floor()
    walker = walkers.CMUHumanoidPositionControlledV2020

    TEST_FILE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../vnl_ray/clips"))
    TEST_FILE_PATH = os.path.join(TEST_FILE_DIR, ref_path)
    test_data = resources.GetResourceFilename(TEST_FILE_PATH)

    with h5py.File(TEST_FILE_PATH, "r") as f:
        dataset_keys = tuple(f.keys())
        dataset = types.ClipCollection(
            ids=dataset_keys,
        )

    # Set up the mocap tracking task
    task = tracking.MultiClipMocapTracking(
        walker=walker,
        arena=arena,
        ref_path=test_data,
        dataset=dataset,
        ref_steps=(1, 2, 3, 4, 5),
        min_steps=1,
        reward_type="comic",
        always_init_at_clip_start=True,
        ghost_offset=GHOST_OFFSET,
        termination_error_threshold=termination_error_threshold,  # lower threshold are harder to terminate
    )
    time_limit = 10.0

    return composer.Environment(
        time_limit=time_limit,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )
