"""Utility functions."""

from typing import Sequence

from IPython.display import HTML
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def rollout_and_render(
    env,
    policy,
    n_steps=100,
    run_until_termination=False,
    camera_ids=[-1],
    **render_kwargs
):
    """Rollout policy for n_steps or until termination, and render video.
    Rendering is possible from multiple cameras; in that case, each element in
    returned `frames` is a list of cameras."""
    if isinstance(camera_ids, int):
        camera_ids = [camera_ids]
    timestep = env.reset()
    frames = []
    i = 0
    while (i < n_steps and not run_until_termination) or (
        timestep.step_type != 2 and run_until_termination
    ):
        i += 1
        frame = []
        for camera_id in camera_ids:
            frame.append(env.physics.render(camera_id=camera_id, **render_kwargs))
        frame = frame[0] if len(camera_ids) == 1 else frame  # Maybe squeeze.
        frames.append(frame)
        action = policy(timestep.observation)
        timestep = env.step(action)
    return frames


def any_substr_in_str(substrings: Sequence[str], string: str) -> bool:
    """Checks if any of substrings is in string."""
    return any(s in string for s in substrings)


def display_video(frames, framerate=30):
    """
    Args:
        frames (array): (n_frames, height, width, 3)
        framerate (int)
    """
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.close("all")  # Figure auto-closing upon backend switching is deprecated.
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(
        fig=fig, func=update, frames=frames, interval=interval, blit=True, repeat=False
    )
    return HTML(anim.to_html5_video())


def parse_mujoco_camera(s: str):
    """Parse `Copy camera` XML string from MuJoCo viewer to pos and xyaxes.

    Example input string:
    <camera pos="-4.552 0.024 3.400" xyaxes="0.010 -1.000 0.000 0.382 0.004 0.924"/>
    """
    split = s.split('"')
    pos = split[1]
    xyaxes = split[3]
    pos = [float(s) for s in pos.split()]
    xyaxes = [float(s) for s in xyaxes.split()]
    return pos, xyaxes


# Helper Function
def blow(x, repeats=2):
    """Repeat columns and rows requested number of times."""
    return np.repeat(np.repeat(x, repeats, axis=0), repeats, axis=1)


def vision_rollout_and_render(env, policy, camera_id=1,
                              eye_blow_factor=5, **render_kwargs):
    """Run vision-guided flight episode and render frames, including eyes."""
    frames = []
    timestep = env.reset()
    # Run full episode until it ends.
    i=0
    while timestep.step_type != 2 and i < 500:
        i+=1
        # Render eyes and scene.
        pixels = env.physics.render(camera_id=camera_id, **render_kwargs)
        eyes = eye_pixels_from_observation(
            timestep, blow_factor=eye_blow_factor)
        # Add eye pixels to scene.
        pixels[0:eyes.shape[0], 0:eyes.shape[1], :] = eyes
        frames.append(pixels)
        # Step environment.
        action = policy(timestep.observation)
        timestep = env.step(action)
    return frames


def eye_pixels_from_observation(timestep, blow_factor=4):
    """Get current eye view from timestep.observation."""
    # In the actual task, the averaging over axis=-1 is done by the visual
    # network as a pre-processing step, so effectively the visual observations
    # are gray-scale.
    egocentric_camera = timestep.observation['walker/egocentric_camera'].mean(axis=-1)

    pixels = egocentric_camera
    pixels = np.tile(pixels[:, :, None], reps=(1, 1, 3))
    pixels = blow(pixels, blow_factor)
    pixels = pixels.astype('uint8')
    return pixels


def eye_pixels_from_cameras(physics, **render_kwargs):
    """Render two-eye view, assuming eye cameras have particular names."""
    for i in range(physics.model.ncam):
        name = physics.model.id2name(i, 'camera')
        if 'eye_left' in name:
            left_eye = physics.render(camera_id=i, **render_kwargs)
        if 'eye_right' in name:
            right_eye = physics.render(camera_id=i, **render_kwargs)
    pixels = np.hstack((left_eye, right_eye))
    return pixels


# Frame width and height for rendering.
render_kwargs = {'width': 640, 'height': 480}