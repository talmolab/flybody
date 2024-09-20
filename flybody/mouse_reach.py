from dm_control import composer

# import mouse entity 
from flybody.mouse_forelimb.mouse_entity import MouseEntity
from flybody.tasks.arenas.mouse_arena import MouseReachArena
from flybody.tasks.mouse_reach_task import MouseReachTask
from dm_control import mjcf

_CONTROL_TIMESTEP = 0.02
_PHYSICS_TIMESTEP = 0.001

mouse_xml_path = "/root/vast/eric/flybody/flybody/flybody/mouse_forelimb/assets_mousereach/modified_working.xml"

def mouse_reach(random_state=None):
    # Initialize the MouseEntity without passing the physics object directly
    mouse = MouseEntity(mouse_xml_path, 0.01)

    # Create the arena
    arena = MouseReachArena()

    # Create the task
    task = MouseReachTask(
        mouse=mouse,
        arena=arena,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP,
    )

    # Return the environment
    return composer.Environment(
        time_limit=3, 
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )