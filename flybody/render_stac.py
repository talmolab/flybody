import functools

import numpy as np

from dm_control import composer
from dm_control.locomotion.arenas import floors
from dm_control.utils import io as resources

from dm_control.utils import io as resources
from dm_control.locomotion.tasks.reference_pose import types
import os
import h5py

from dm_control.locomotion.walkers import rodent

from flybody.tasks import tracking_old as tracking

import imageio

GHOST_OFFSET = np.array((0, 0, 0))


def rodent_walk_rendering(
    ref_path: str | None = None,
    random_state: np.random.RandomState | None = None,
):
    """
    Rodent walking imitation, following similar calling with fruitfly imitation
    """
    walker = functools.partial(rodent.Rat, foot_mods=True)
    arena = floors.Floor()

    TEST_FILE_PATH = ref_path

    with h5py.File(TEST_FILE_PATH, "r") as f:
        dataset_keys = tuple(f.keys())
        dataset = types.ClipCollection(
            ids=dataset_keys,
        )

    # Set up the mocap tracking task
    task = tracking.PlaybackTask(
        walker=walker,
        arena=arena,
        ref_path=resources.GetResourceFilename(TEST_FILE_PATH),
        dataset=dataset,
    )

    return composer.Environment(
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )


TEST_FILE_PATH = "/root/vast/scott-yang/flybody/clips/all_snippets.h5"

env = rodent_walk_rendering(TEST_FILE_PATH)
num_clips = len(env.task._all_clips)
render_kwargs = {"width": 640, "height": 480}


def render_STAC(env, n_steps=500, run_until_termination=True, camera_ids=[1, 3], **render_kwargs):
    """Rollout policy for n_steps or until termination, and render video.
    Rendering is possible from multiple cameras; in that case, each element in
    returned `frames` is a list of cameras."""
    if isinstance(camera_ids, int):
        camera_ids = [camera_ids]
    timestep = env.reset()
    frames = []
    i = 0
    while (i < n_steps and not run_until_termination) or (timestep.step_type != 2 and run_until_termination):
        i += 1
        frame = []
        for camera_id in camera_ids:
            frame.append(env.physics.render(camera_id=camera_id, **render_kwargs))
        frame = frame[0] if len(camera_ids) == 1 else frame  # Maybe squeeze.
        frames.append(frame)
        action = np.zeros((38,))
        timestep = env.step(action)
    return frames


for i in range(num_clips):
    print(f"Rendering #{i}/842 Clips...")
    frames = render_STAC(env, n_steps=500)
    frames = np.array(frames)
    frames = frames.reshape(248, 240 * 2, 320, 3, order="A")
    clip_id = env.task.get_clip_id(None)[0]
    with imageio.get_writer(f"/root/vast/scott-yang/STAC_renderings/clip_id_{int(clip_id)}.mp4", fps=30) as video:
        for f in frames:
            video.append_data(f)
