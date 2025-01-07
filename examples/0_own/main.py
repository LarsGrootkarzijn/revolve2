"""Main script for the example."""

from revolve2.experimentation.logging import setup_logging
from revolve2.modular_robot import ModularRobot, ModularRobotControlInterface
from revolve2.modular_robot.body import RightAngles
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2
from revolve2.modular_robot.brain import Brain, BrainInstance
from revolve2.modular_robot.sensor_state import ModularRobotSensorState
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from revolve2.standards import fitness_functions, modular_robots_v2, terrains
import numpy as np
def make_body() -> (
    tuple[BodyV2, ActiveHingeV2, ActiveHingeV2, ActiveHingeV2, ActiveHingeV2]
):
    """
    Create a body for the robot.

    :returns: The created body and references to each hinge: first_left_active_hinge, second_left_active_hinge, first_right_active_hinge, second_right_active_hinge.
    """
    body = BodyV2()
    first_left_active_hinge = ActiveHingeV2(0.0)
    second_left_active_hinge = ActiveHingeV2(0.0)
    first_right_active_hinge = ActiveHingeV2(0.0)
    second_right_active_hinge = ActiveHingeV2(0.0)
    rear_active_hinge_1 = ActiveHingeV2(np.pi / 2.0)
    rear_active_hinge_2 = ActiveHingeV2(np.pi / 2.0)
    
    body.core_v2.left_face.bottom = first_right_active_hinge
    first_left_active_hinge.attachment = BrickV2(0.0)

    body.core_v2.right_face.bottom = first_left_active_hinge
    first_right_active_hinge.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom = rear_active_hinge_1
    rear_active_hinge_1.attachment = BrickV2(-np.pi / 2.0)

    body.core_v2.back_face.bottom.attachment.front = rear_active_hinge_2
    body.core_v2.back_face.bottom.attachment.front.attachment = BrickV2(-np.pi / 2.0)

    body.core_v2.back_face.bottom.attachment.front.attachment.left = second_left_active_hinge
    body.core_v2.back_face.bottom.attachment.front.attachment.right = second_right_active_hinge

    second_right_active_hinge.attachment = BrickV2(0.0)
    second_left_active_hinge.attachment = BrickV2(0.0)

    return (
        body,
        first_left_active_hinge,
        second_left_active_hinge,
        first_right_active_hinge,
        second_right_active_hinge,
        rear_active_hinge_1,
        rear_active_hinge_2,
    )


class CustomBrainInstance(BrainInstance):
    """
    The actual object that controls the robot.

    Created by the `CustomBrain` class.
    """

    first_left_active_hinge: ActiveHingeV2
    second_left_active_hinge: ActiveHingeV2
    first_right_active_hinge: ActiveHingeV2
    second_right_active_hinge: ActiveHingeV2
    rear_active_hinge_1: ActiveHingeV2
    rear_active_hinge_2: ActiveHingeV2

    def __init__(
        self,
        first_left_active_hinge: ActiveHingeV2,
        second_left_active_hinge: ActiveHingeV2,
        first_right_active_hinge: ActiveHingeV2,
        second_right_active_hinge: ActiveHingeV2,
        rear_active_hinge_1: ActiveHingeV2,
        rear_active_hinge_2: ActiveHingeV2,
        first_left_value = 0.9,
        second_left_value = -0.9,
        first_right_value = -0.9,
        second_right_value = 0.9,
        rear_value_1 = 0.2,
        rear_value_2 = -0.2,
    ) -> None:
        """
        Initialize the Custom Brain Instance.

        :param first_left_active_hinge: First left active Hinge.
        :param second_left_active_hinge: Second left active Hinge.
        :param first_right_active_hinge: First right active Hinge.
        :param second_right_active_hinge: Second right active Hinge.
        """
        self.first_left_active_hinge = first_left_active_hinge
        self.second_left_active_hinge = second_left_active_hinge
        self.first_right_active_hinge = first_right_active_hinge
        self.second_right_active_hinge = second_right_active_hinge
        self.rear_active_hinge_1 = rear_active_hinge_1
        self.rear_active_hinge_2 = rear_active_hinge_2
        self.first_left_value = 0.9
        self.second_left_value = -0.9
        self.first_right_value = -0.9
        self.second_right_value = 0.9
        self.rear_value_1 = 0.2
        self.rear_value_2 = -0.2
    def control(
        self,
        dt: float,
        sensor_state: ModularRobotSensorState,
        control_interface: ModularRobotControlInterface,
    ) -> None:
        """
        Control the modular robot using our custom brain.

        :param dt: Elapsed seconds since last call to this function.
        :param sensor_state: Interface for reading the current sensor state.
        :param control_interface: Interface for controlling the robot.
        """
        self.first_left_value *= -1
        self.second_left_value *= -1
        self.first_right_value *= -1
        self.second_right_value *= -1
        self.rear_value_1 *= -1
        self.rear_value_2 *= -1
        control_interface.set_active_hinge_target(self.first_left_active_hinge, self.first_left_value)
        control_interface.set_active_hinge_target(self.second_left_active_hinge, self.second_left_value)
        control_interface.set_active_hinge_target(self.first_right_active_hinge, self.first_right_value)
        control_interface.set_active_hinge_target(self.second_right_active_hinge, self.second_right_value)
        control_interface.set_active_hinge_target(self.rear_active_hinge_1, self.rear_value_1)
        control_interface.set_active_hinge_target(self.rear_active_hinge_2, self.rear_value_2)

        


class CustomBrain(Brain):
    """
    This is our custom brain.

    It stores references to each hinge of the robot body so they can be controlled individually.
    A brain has a function `make_instance`, which creates the actual object that controls a robot.
    """

    first_left_active_hinge: ActiveHingeV2
    second_left_active_hinge: ActiveHingeV2
    first_right_active_hinge: ActiveHingeV2
    second_right_active_hinge: ActiveHingeV2
    rear_active_hinge_1: ActiveHingeV2
    rear_active_hinge_2: ActiveHingeV2
    
    def __init__(
        self,
        first_left_active_hinge: ActiveHingeV2,
        second_left_active_hinge: ActiveHingeV2,
        first_right_active_hinge: ActiveHingeV2,
        second_right_active_hinge: ActiveHingeV2,
        rear_active_hinge_1: ActiveHingeV2,
        rear_active_hinge_2: ActiveHingeV2,
    ) -> None:
        """
        Initialize the Custom Brain.

        :param first_left_active_hinge: First left active Hinge.
        :param second_left_active_hinge: Second left active Hinge.
        :param first_right_active_hinge: First right active Hinge.
        :param second_right_active_hinge: Second right active Hinge.
        """
        self.first_left_active_hinge = first_left_active_hinge
        self.second_left_active_hinge = second_left_active_hinge
        self.first_right_active_hinge = first_right_active_hinge
        self.second_right_active_hinge = second_right_active_hinge
        self.rear_active_hinge_1 = rear_active_hinge_1
        self.rear_active_hinge_2 = rear_active_hinge_2

    def make_instance(self) -> BrainInstance:
        """
        Create an instance of this brain.

        :returns: The created instance.
        """
        return CustomBrainInstance(
            self.first_left_active_hinge,
            self.second_left_active_hinge,
            self.first_right_active_hinge,
            self.second_right_active_hinge,
            self.rear_active_hinge_1,
            self.rear_active_hinge_2,
        )


def main() -> None:
    """Run the simulation."""
    # Set up logging.
    setup_logging()

    # Create a body for the robot.
    (
        body,
        first_left_active_hinge,
        second_left_active_hinge,
        first_right_active_hinge,
        second_right_active_hinge,
        rear_active_hinge_1,
        rear_active_hinge_2,
    ) = make_body()

    # Create the custom brain for the robot.
    brain = CustomBrain(
        first_left_active_hinge,
        second_left_active_hinge,
        first_right_active_hinge,
        second_right_active_hinge,
        rear_active_hinge_1,
        rear_active_hinge_2,
    )

    # Combine the body and brain into a moduflar robot.
    robot = ModularRobot(body, brain)

    # Create the scene.
    scene = ModularRobotScene(terrain=terrains.flat())
    scene.add_robot(robot)

    # Simulate the scene.
    simulator = LocalSimulator()
    simulate_scenes(
        simulator=simulator,
        batch_parameters=make_standard_batch_parameters(),
        scenes=scene,
    )


if __name__ == "__main__":
    main()
