"""Floating ball arena."""

from dm_control import composer


class BallFloor(composer.Arena):
    """A floating ball arena for tethered fruitfly walking tasks."""

    def _build(
        self,
        ball_pos=(0, 0, 0),
        ball_radius=1,
        ball_density=1.0,
        name="ball",
        skybox=False,
        reflectance=0.2,
        top_camera_y_padding_factor=1.1,
        top_camera_distance=100,
    ):

        super()._build(name=name)
        self._size = ball_radius

        self._mjcf_root.visual.headlight.set_attributes(
            ambient=[0.4, 0.4, 0.4],
            diffuse=[0.8, 0.8, 0.8],
            specular=[0.1, 0.1, 0.1],
        )

        self._ground_texture = self._mjcf_root.asset.add(
            "texture",
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
            type="2d",
            builtin="checker",
            name="groundplane",
            width=200,
            height=200,
        )
        self._ground_material = self._mjcf_root.asset.add(
            "material",
            name="groundplane",
            texrepeat=[2, 2],
            texuniform=False,
            reflectance=reflectance,
            texture=self._ground_texture,
        )
        if skybox:
            self._skybox_texture = self._mjcf_root.asset.add(
                "texture",
                type="skybox",
                name="skybox",
                builtin="gradient",
                rgb1=[0.4, 0.6, 0.8],
                rgb2=[0.0, 0.0, 0.0],
                markrgb=[1, 1, 1],
                mark="random",
                random=0.02,
                width=800,
                height=800,
            )

        # Build ball.
        ball = self._mjcf_root.worldbody.add("body", name="ball", pos=ball_pos)
        self._ground_geom = ball.add(
            "geom",
            type="sphere",
            size=(ball_radius, 0, 0),
            material=self._ground_material,
            density=ball_density,
        )
        self._ball_joint = ball.add("joint", name="ball", type="ball")

    @property
    def ground_geoms(self):
        return (self._ground_geom,)

    def regenerate(self, random_state):
        pass

    @property
    def size(self):
        return self._size
