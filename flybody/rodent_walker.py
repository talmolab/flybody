from dm_control.locomotion.walkers.rodent import Rat


class Rat(Rat):
    """
    subclass the Rat to add additional functionality to remove the skin.
    """

    def _build(
        self,
        params=None,
        name="walker",
        torque_actuators=False,
        foot_mods=False,
        initializer=None,
        remove_skin=False,
    ):
        super()._build(params, name, torque_actuators, foot_mods, initializer)
        if remove_skin:
            self._mjcf_root.remove("skin")
