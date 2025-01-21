"""An example on how to remote control a physical modular robot."""

from pyrr import Vector3
import revolve2
import math
from simulation import Brain

from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body import RightAngles
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot import ModularRobot, ModularRobotControlInterface
from revolve2.modular_robot.sensor_state import ModularRobotSensorState
from revolve2.modular_robot.brain import Brain, BrainInstance
from revolve2.modular_robot.body.sensors import CameraSensor
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2
from revolve2.modular_robot_physical import Config, UUIDKey
from revolve2.modular_robot_physical.remote import run_remote
from revolve2.standards.modular_robots_v2 import gecko_v2
from revolve2.modular_robot.body.sensors import CameraSensor

class CustomBrainInstance(BrainInstance):
    """ANN brain instance."""

    active_hinges: list[ActiveHinge]
    steer = False
    camera: CameraSensor
    def __init__(
        self,
        active_hinges: list[ActiveHinge],
        camera: CameraSensor
    ) -> None:
        """
        Initialize the Object.

        :param active_hinges: The active hinges to control.
        :param imu_sensor: The IMU sensor.
        """
        self.camera = camera
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
                control_interface.set_active_hinge_target(active_hinge, 0.9)

            for active_hinge, sensor in zip(right, sensors):
                control_interface.set_active_hinge_target(active_hinge, -0.3)

            for active_hinge, sensor in zip(rear, sensors):
                control_interface.set_active_hinge_target(active_hinge, 0.9)
            self.steer = False
        else:
            for active_hinge, sensor in zip(right, sensors):
                control_interface.set_active_hinge_target(active_hinge, 0.0)

            for active_hinge, sensor in zip(left, sensors):
                control_interface.set_active_hinge_target(active_hinge, -0.3)
            
            for active_hinge, sensor in zip(rear, sensors):
                control_interface.set_active_hinge_target(active_hinge, -0.9)
            self.steer = True

        


class CustomBrain(Brain):
    """The ANN brain."""

    active_hinges: list[ActiveHinge]
    camera: CameraSensor
    def __init__(
        self,
        active_hinges: list[ActiveHinge],
        camera: CameraSensor
    ) -> None:
        """
        Initialize the Object.

        :param active_hinges: The active hinges to control.
        :param imu_sensor: The IMU sensor.
        """
        self.active_hinges = active_hinges
        self.camera = camera
    def make_instance(self) -> BrainInstance:
        """
        Create an instance of this brain.

        :returns: The created instance.
        """
        return CustomBrainInstance(
            camera=self.camera,
            active_hinges=self.active_hinges
        )


def on_prepared() -> None:
    """Do things when the robot is prepared and ready to start the controller."""
    print("Done. Press enter to start the brain.")
    input("here:")


def main() -> None:
    """Remote control a physical modular robot."""
    # rng = make_rng_time_seed()
    """
    Create a modular robot, similar to what was done in the 1a_simulate_single_robot example.
    Of course, you can replace this with your own robot, such as one you have optimized using an evolutionary algorithm.
    """
    body = gecko_v2()
    body.core.add_sensor(
        CameraSensor(position=Vector3([0, 0, 0]), camera_size=(480, 640))
    )
    # Create a brain for the robot.
    active_hinges = body.find_modules_of_type(ActiveHinge)
    camera = body.find_modules_of_type(CameraSensor)
    # Create the custom brain for the robot.
    brain = CustomBrain(
        active_hinges,
        camera
    )
    
    # Combine the body and brain into a moduflar robot.
    robot = ModularRobot(body, brain)

    #rear1 = 0
    #rear2 = 1
    #rearright = 2
        #rearleft = 3
        #frontright = 4
        #frontleft = 5  
    hinge_1, hinge_2, hinge_3, hinge_4, hinge_5, hinge_6 = active_hinges
    hinge_mapping = {
        UUIDKey(hinge_1): 2,
        UUIDKey(hinge_2): 24, 
        UUIDKey(hinge_3): 30, 
        UUIDKey(hinge_4): 1, 
        UUIDKey(hinge_5): 31, 
        UUIDKey(hinge_6): 0, 
    }

    #hinge_mapping = {
    #    UUIDKey(hinge_1): 31, #rf
    #    UUIDKey(hinge_2): 0, #lf
    #    UUIDKey(hinge_3): 7, #tail 1
    #    UUIDKey(hinge_4): 24, #tail 2
    #    UUIDKey(hinge_5): 1, #lb
    #    UUIDKey(hinge_6): 29, #rb
    #}

    """
    A configuration consists of the follow parameters:
    - modular_robot: The ModularRobot object, exactly as you would use it in simulation.py.
    - hinge_mapping: This maps active hinges to GPIO pins on the physical modular robot core.
    - run_duration: How long to run the robot for in seconds.
    - control_frequency: Frequency at which to call the brain control functions in seconds. If you also ran the robot in simulation.py, this must match your setting there.
    - initial_hinge_positions: Initial positions for the active hinges. In Revolve2 the simulator defaults to 0.0.
    - inverse_servos: Sometimes servos on the physical robot are mounted backwards by accident. Here you inverse specific servos in software. Example: {13: True} would inverse the servo connected to GPIO pin 13.
    """
    config = Config(
        modular_robot=robot,
        hinge_mapping=hinge_mapping,
        run_duration=100,
        control_frequency=2,
        initial_hinge_positions={UUIDKey(active_hinge): 0.0 for active_hinge in active_hinges},
        inverse_servos={0: True, 1: True},
    )

    """
    Create a Remote for the physical modular robot.
    Make sure to target the correct hardware type and fill in the correct IP and credentials.
    The debug flag is turned on. If the remote complains it cannot wkeep up, turning off debugging might improve performance.
    """
    print("Initializing robot..")
    run_remote(
        config=config,
        hostname="10.15.3.45",  # "Set the robot IP here.
        debug=False,
        on_prepared=on_prepared,
        display_camera_view=False,
    )
    """
    Note that theoretically if you want the robot to be self controlled and not dependant on a external remote, you can run this script on the robot locally.
    """

if __name__ == "__main__":
    main()