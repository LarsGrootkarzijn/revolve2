from pyrr import Vector3

from ..base import Motor


class MotorV1(Motor):
    """
    An active hinge module for a modular robot.

    This is a rotary joint.
    """

    def __init__(self, position: float, orientation: float):
        """
        Initialize this object.

        :param orientation: The Modules orientation.
        """
        super().__init__(
            position=position,
            orientation=orientation,
            rotor_bounding_box=Vector3([0.018, 0.018, 0.018]),
            frame_bounding_box=Vector3([0.018, 0.018, 0.018]),
            mass=0.025,
            ctrlrange=[0,100],
            gear=0.1
        )
