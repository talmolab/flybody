from dm_control import mjcf
from dm_control import composer

class MouseReachArena(composer.Arena):
    """
    The arena where the MouseReachTask is performed.

    This class defines the environment space (arena) as a plane with a checkered grid texture
    and material applied to it.
    """

    def _build(self):
        """
        Initializes the arena by creating a plane (floor) with a checkered texture and grid material.
        """
        # Create the MJCF root element for the arena
        self._mjcf_model = mjcf.RootElement()

        # Add a checkered texture to the floor
        grid_texture = self._mjcf_model.asset.add(
            'texture', 
            name='grid_texture', 
            type='2d', 
            builtin='checker', 
            rgb1=[0.1, 0.2, 0.3], 
            rgb2=[0.2, 0.3, 0.4], 
            mark="edge", 
            markrgb=[0.2, 0.3, 0.4],
            width=300, 
            height=300
        )

        # Add a material using the checkered texture
        grid_material = self._mjcf_model.asset.add(
            'material', 
            name='grid_material', 
            texture=grid_texture, 
            texrepeat=[1, 1], 
            texuniform=True, 
            reflectance=0.2
        )

        # Add a plane geom to serve as the floor, with the grid material
        self._mjcf_model.worldbody.add(
            'geom', 
            name='floor', 
            type='plane', 
            size=[3, 3, 0.2], 
            material=grid_material, 
            pos=[0, 0, -0.02]
        )

    @property
    def mjcf_model(self):
        """Returns the MJCF model for the arena."""
        return self._mjcf_model

    def regenerate(self, random_state):
        """Regenerates the arena environment (optional for randomization)."""
        # This can be used for randomization or resetting the arena if needed.
        pass