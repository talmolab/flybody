from dm_control import composer
from dm_control.composer.observation import observable
from dm_control import mjcf
import numpy as np
import tensorflow as tf


class MouseEntity(composer.Entity):
    """
    A class representing the mouse entity loaded from an XML file.
    This class loads the mouse model from an XML file, manages its interaction with the physics engine,
    and defines observables for the mouse's state and task-specific quantities.
    """

    def _build(self, xml_path):
        """
        Initializes the mouse entity by loading the MJCF model from an XML file.

        Args:
            xml_path: Path to the mouse XML model file.
            target_size: Size of the target for the task.
        """
        # Load the MJCF model from the specified XML file
        self._mjcf_model = mjcf.from_path(xml_path)
        # Set the target size in the MJCF model

    def _build_observables(self):
        """
        Defines the observables for the MouseEntity.

        This method sets up observables for joint angles, joint velocities,
        the distance to the target, and the target size. These observables
        are then enabled for use in tasks.

        Returns:
            composer.Observables: A collection of observables for the mouse.
        """
        self._observables = composer.Observables(self)
        # print("Available geoms:", self._mjcf_model.find_all('geom'))
        # print("Available sites:", self._mjcf_model.find_all('site'))

        # Add joint observables
        self._observables.add_observable(
            "joint_angles", observable.MJCFFeature(kind="qpos", mjcf_element=self._mjcf_model.find_all("joint"))
        )
        self._observables.add_observable(
            "joint_velocities", observable.MJCFFeature(kind="qvel", mjcf_element=self._mjcf_model.find_all("joint"))
        )

        # Add custom observables using the methods directly
        self._observables.add_observable("to_target", observable.Generic(self._to_target_observable))
        self._observables.add_observable("target_size", observable.Generic(self._target_size_observable))

        self._observables.enable_all()

        return self._observables

    def _to_target_observable(self, physics):
        """
        Returns the normalized distance from the mouse's finger to the target.

        Args:
            physics (mjcf.Physics): The physics simulation object.

        Returns:
            np.ndarray: The normalized distance between the finger and the target.

        Raises:
            ValueError: If the 'target' or 'finger_tip' geoms are not defined in the MJCF model.
        """
        # Check if 'target' exists in the current physics model
        try:
            # print(f"GEOM_XPOS: {physics.named.data.geom_xpos}")
            target_pos = physics.named.data.geom_xpos["mouse/target"]
            finger_pos = physics.named.data.geom_xpos["mouse/finger_tip"]
        except KeyError as e:
            raise ValueError(
                f"Error accessing geom position: {e}. Ensure that 'target' and 'finger_tip' geoms are correctly defined in the MJCF model."
            )

        return (target_pos - finger_pos) / 0.02

    def _target_size_observable(self, physics):
        """
        Returns the normalized size of the target.

        Args:
            physics (mjcf.Physics): The physics simulation object.

        Returns:
            np.ndarray: The normalized target size.
        """
        return np.array([physics.named.model.geom_size["mouse/target", 0]])

    @property
    def mjcf_model(self):
        """
        Returns the MJCF model for the mouse.

        Returns:
            mjcf.RootElement: The MJCF model of the mouse.
        """
        return self._mjcf_model

    @property
    def actuators(self):
        """
        Returns the actuators for the mouse.

        Returns:
            list: A list of actuator elements from the MJCF model.
        """
        return self._mjcf_model.find_all("actuator")

    @property
    def observables(self):
        """
        Returns the collection of observables for the mouse.

        This property triggers the `_build_observables()` method
        to define the necessary observables.

        Returns:
            composer.Observables: A collection of observables.
        """
        return self._build_observables()

    def reinitialize_pose(self, physics, random_state):
        """
        Reinitializes the mouse's pose at the start of each episode.

        Args:
            physics (mjcf.Physics): The physics simulation object.
            random_state (np.random.RandomState): A random state for initializing the pose.
        """
        # self._init_pose = {
        #     'ground_clavicle': -1.82e-24, #+ self.random.normal(scale=0.2), #-0.7853 + self.random.normal(scale=0.5),
        #     'ground_scapula': 7.58e-7, #+ self.random.normal(scale=0.2), #1.5706 + self.random.normal(scale=0.5),
        #     # 'sh_elv_angle': -.144, #+ self.random.normal(scale=0.0), #0.8726 + self.random.normal(scale=0.5),
        #     # 'sh_extension_angle': -.276,
        #     # 'sh_rotation_angle': -1,
        #     'elbow': -.00564,
        #     }

        # self.set_joints(physics, self._init_pose)
        pass

    def apply_action(self, physics, action, random_state):
        """
        Applies control actions to the mouse in the physics simulation.

        Args:
            physics (mjcf.Physics): The physics simulation object.
            action (np.ndarray): The control actions to apply to the mouse actuators.
            random_state (np.random.RandomState): A random state for adding stochasticity to the actions.
        """
        # Bind the actuator controls to the action.
        physics.bind(self._mjcf_model.find_all("actuator")).ctrl[:] = action

    def set_target_size(self, physics, target_size):
        """
        Sets the target size in the physics model.

        Args:
            physics (mjcf.Physics): The physics simulation object.
        """
        # Update the size of the 'target' geom in the physics model
        physics.named.model.geom_size["mouse/target", 0] = target_size

    def disable_contacts(self, physics):
        """
        Disables contacts in the physics model.

        Args:
            physics (mjcf.Physics): The physics simulation object.
        """
        # Update the size of the contacts in the physics model
        physics.data.ncon = 0  # disable contacts, joint limits will serve as constraints

    def set_joints(physics, pose):
        qpos = physics.named.data.qpos

        for joint_id in range(physics.model.njnt):
            joint_name = physics.model.id2name(joint_id, "joint")
            physics.named.data.qpos[joint_name] = pose[joint_name]
