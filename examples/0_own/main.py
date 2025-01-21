"""Main script for the example."""

from revolve2.experimentation.logging import setup_logging
from revolve2.modular_robot import ModularRobot, ModularRobotControlInterface
from revolve2.modular_robot.body import RightAngles
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2
from revolve2.modular_robot.brain import Brain, BrainInstance
from revolve2.modular_robot.sensor_state import ModularRobotSensorState
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes, Terrain
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from revolve2.standards import fitness_functions, modular_robots_v2, terrains
from revolve2.standards.modular_robots_v2 import gecko_v2
import numpy as npx
from revolve2.modular_robot.body.sensors import CameraSensor
from revolve2.simulation.scene import AABB, Color, Pose
from revolve2.standards.interactive_objects import Ball      
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.simulation.scene.geometry import GeometryBox, GeometryPlane
from pyrr import Vector3, Quaternion
from revolve2.simulators.mujoco_simulator.textures import Checker, Flat, Gradient
from revolve2.simulation.scene.geometry.textures import MapType

def make_custom_terrain() -> Terrain:
    """
    Create a custom terrain.

    :returns: The created terrain.
    """
    # A terrain is a collection of static geometries.
    # Here we create a simple terrain uses some boxes.
    return Terrain(
        static_geometry=[
            GeometryPlane(
                pose=Pose(position=Vector3(), orientation=Quaternion()),
                mass=0.0,
                size=Vector3([20.0, 20.0, 0.0]),
                texture=Checker(
                    primary_color=Color(170, 170, 180, 255),
                    secondary_color=Color(150, 150, 150, 255),
                    map_type=MapType.MAP2D,
                ),
            ),
            GeometryPlane(
                pose=Pose(position=Vector3(), orientation=Quaternion()),
                mass=0.0,
                size=Vector3([20.0, 0.1, 0.0]),
                texture=Checker(
                    primary_color=Color(0, 0, 180, 255),
                    secondary_color=Color(0, 0, 150, 255),
                    map_type=MapType.MAP2D,
                ),
            ),
        ]
    )

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
    
    body.core_v2.left_face.bottom = first_left_active_hinge
    first_left_active_hinge.attachment = BrickV2(0.0)

    body.core_v2.right_face.bottom = first_right_active_hinge
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
    """ANN brain instance."""

    active_hinges: list[ActiveHinge]
    steer = False
    def __init__(
        self,
        active_hinges: list[ActiveHinge]
    ) -> None:
        """
        Initialize the Object.

        :param active_hinges: The active hinges to control.
        :param imu_sensor: The IMU sensor.
        """
        self.active_hinges = active_hinges

    def control(
        self,
        dt: float,
        sensor_state: ModularRobotSensorState,
        control_interface: ModularRobotControlInterface,
    ) -> None:
        """
        Control the modular robot.

        :param dt: Elapsed seconds since last call to this function.
        :param sensor_state: Interface for reading the current sensor state.
        :param control_interface: Interface for controlling the robot.
        """
        # To get data from you sensors you need the sensor itself, with which you can query the sensor stare from the ModularRobotSensorState object.
        # Get the sensors from the active hinges
        sensors = [
            active_hinge.sensors.active_hinge_sensor
            for active_hinge in self.active_hinges
            if active_hinge.sensors.active_hinge_sensor is not None
        ]
        assert len(sensors) == len(
            self.active_hinges
        ), "One of the active hinges does not have a sensor set."

        # Get the current angular positions of the active hinges
        current_positions = [
            sensor_state.get_active_hinge_sensor_state(sensor).position
            for sensor in sensors
        ]
        #logging.info(f"current positions: {current_positions}")

        # Get the imu sensor state
        #logging.info(f"orientation: {imu_state.orientation}")
        #logging.info(f"angular rate: {imu_state.angular_rate}")
        #logging.info(f"specific force: {imu_state.specific_force}")
    
        # Here you can implement your controller.
        # The current controller does nothing except for always settings the joint positions to 0.5.
        #rear1 = 0
        #rear2 = 1
        #rearright = 2
        #rearleft = 3
        #frontright = 4
        #frontleft = 5
        left = [self.active_hinges[3],self.active_hinges[5]]
        right = [self.active_hinges[2],self.active_hinges[4]]
        rear = [self.active_hinges[0],self.active_hinges[1]]

        if(self.steer):
            for active_hinge, sensor in zip(left, sensors):
                control_interface.set_active_hinge_target(active_hinge, 0.8)

            for active_hinge, sensor in zip(right, sensors):
                control_interface.set_active_hinge_target(active_hinge, 0.0)

            for active_hinge, sensor in zip(rear, sensors):
                control_interface.set_active_hinge_target(active_hinge, 0.7)
            self.steer = False
        else:
            for active_hinge, sensor in zip(right, sensors):
                control_interface.set_active_hinge_target(active_hinge, 0.8)

            for active_hinge, sensor in zip(left, sensors):
                control_interface.set_active_hinge_target(active_hinge, 0.0)
            
            for active_hinge, sensor in zip(rear, sensors):
                control_interface.set_active_hinge_target(active_hinge, -0.7)
            self.steer = True

        


class CustomBrain(Brain):
    """The ANN brain."""

    active_hinges: list[ActiveHinge]

    def __init__(
        self,
        active_hinges: list[ActiveHinge]
    ) -> None:
        """
        Initialize the Object.

        :param active_hinges: The active hinges to control.
        :param imu_sensor: The IMU sensor.
        """
        self.active_hinges = active_hinges

    def make_instance(self) -> BrainInstance:
        """
        Create an instance of this brain.

        :returns: The created instance.
        """
        return CustomBrainInstance(
            active_hinges=self.active_hinges
        )


def main() -> None:
    """Run the simulation."""
    # Set up logging.
    setup_logging()

    body = gecko_v2()

    body.core.add_sensor(
        CameraSensor(position=Vector3([0, 0, 0]), camera_size=(480, 640))
    )
    """Every module on the robot can have sensors, to add them you do the following: """
    # Add an IMU Sensor to the core.

    # Create a brain for the robot.
    active_hinges = body.find_modules_of_type(ActiveHinge)

    # Create the custom brain for the robot.
    brain = CustomBrain(
        active_hinges
    )

    # Combine the body and brain into a moduflar robot.
    robot = ModularRobot(body, brain)

    # Create the scene.
    scene = ModularRobotScene(terrain=make_custom_terrain())
    scene.add_robot(robot)
    scene.add_interactive_object(
            Ball(radius=0.1, mass=200, pose=Pose(Vector3([-5, 0, 0])))
        )
    # Simulate the scene.
    simulator = LocalSimulator()
    batch_param = make_standard_batch_parameters()
    batch_param.control_frequency = 2
    simulate_scenes(
        simulator=simulator,
        batch_parameters=batch_param,
        scenes=scene,
    )


if __name__ == "__main__":
    main()
