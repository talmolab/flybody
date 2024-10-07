from dm_control import composer
from dm_control.composer.observation import observable
from dm_control import mjcf
import numpy as np

class MouseEntity(composer.Entity):
    """
    A class representing the mouse entity loaded from an XML file.
    This class loads the mouse model from an XML file, manages its interaction with the physics engine,
    and defines observables for the mouse's state and task-specific quantities.
    """

    def _build(self, xml_path, target_size):
        """
        Initializes the mouse entity by loading the MJCF model from an XML file.

        Args:
            xml_path: Path to the mouse XML model file.
            target_size: Size of the target for the task.
        """
        # Load the MJCF model from the specified XML file
        self._mjcf_model = mjcf.from_path(xml_path)
        self._target_size = target_size

    def _build_observables(self):
        """Defines the observables for the MouseEntity."""
        self._observables = composer.Observables(self)
        #print("Available geoms:", self._mjcf_model.find_all('geom'))
        #print("Available sites:", self._mjcf_model.find_all('site'))

        # Add joint observables
        self._observables.add_observable(
            "joint_angles", 
            observable.MJCFFeature(kind='qpos', mjcf_element=self._mjcf_model.find_all('joint'))
        )
        self._observables.add_observable(
            "joint_velocities", 
            observable.MJCFFeature(kind='qvel', mjcf_element=self._mjcf_model.find_all('joint'))
)

        # Add custom observables using the methods directly
        self._observables.add_observable(
            "to_target", 
            observable.Generic(self._to_target_observable)
        )
        self._observables.add_observable(
            "target_size", 
            observable.Generic(self._target_size_observable)
        )

        self._observables.enable_all()
        
        return self._observables

    def _to_target_observable(self, physics):
        """Returns the normalized distance from the mouse's finger to the target."""
        # Check if 'target' exists in the current physics model
        try:
            #print(f"GEOM_XPOS: {physics.named.data.geom_xpos}")
            target_pos = physics.named.data.geom_xpos['mouse/target']
            finger_pos = physics.named.data.geom_xpos['mouse/finger_tip']
        except KeyError as e:
            raise ValueError(f"Error accessing geom position: {e}. Ensure that 'target' and 'finger_tip' geoms are correctly defined in the MJCF model.")

        return (target_pos - finger_pos) / 0.02

    def _target_size_observable(self, physics):
        """Returns the normalized target size."""
        return np.array([self._target_size / 0.008])

    @property
    def mjcf_model(self):
        """Returns the MJCF model for the mouse."""
        return self._mjcf_model

    @property
    def actuators(self):
        """Returns the actuators for the mouse."""
        return self._mjcf_model.find_all('actuator')

    @property
    def observables(self):
        """Returns the collection of observables for the mouse."""
        return self._build_observables()

    def reinitialize_pose(self, physics, random_state):
        """Reinitialize the mouse's pose at the start of each episode."""
        pass

    def apply_action(self, physics, action, random_state):
        """Applies control actions to the mouse in the physics simulation."""
        # Bind the actuator controls to the action.
        physics.bind(self._mjcf_model.find_all('actuator')).ctrl[:] = action