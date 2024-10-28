from dm_control.locomotion.walkers.rodent import Rat
from dm_control import composer
_OBSERVABLE_JOINT_NAMES = [
    "vertebra_1_extend",
    "hip_L_supinate",
    "hip_L_abduct",
    "hip_L_extend",
    "knee_L",
    "ankle_L",
    "toe_L",
    "hip_R_supinate",
    "hip_R_abduct",
    "hip_R_extend",
    "knee_R",
    "ankle_R",
    "toe_R",
    "vertebra_C11_extend",
    "vertebra_cervical_1_bend",
    "vertebra_axis_twist",
    "atlas",
    "mandible",
    "scapula_L_supinate",
    "scapula_L_abduct",
    "scapula_L_extend",
    "shoulder_L",
    "shoulder_sup_L",
    "elbow_L",
    "wrist_L",
    "scapula_R_supinate",
    "scapula_R_abduct",
    "scapula_R_extend",
    "shoulder_R",
    "shoulder_sup_R",
    "elbow_R",
    "wrist_R",
    "finger_R",
]

class Rat(Rat):

    def _build(self, params=None, name="walker", torque_actuators=False, foot_mods=False, initializer=None):
        return super()._build(params, name, torque_actuators, foot_mods, initializer)
    
    @composer.cached_property
    def observable_joints(self):
        """Return observable joints."""
        return tuple(self._mjcf_root.find("joint", joint) for joint in _OBSERVABLE_JOINT_NAMES)