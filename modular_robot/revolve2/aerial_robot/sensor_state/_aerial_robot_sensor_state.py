from abc import ABC, abstractmethod

from ..body.base._motor_sensor import MotorSensor
from ._motor_sensor_state import MotorSensorState


class AerialRobotSensorState(ABC):
    """The state of modular robot's sensors."""

    @abstractmethod
    def get_active_hinge_sensor_state(
        self, sensor: MotorSensor
    ) -> MotorSensorState:
        """
        Get the state of the provided active hinge sensor.

        :param sensor: The sensor.
        :returns: The state.
        """
