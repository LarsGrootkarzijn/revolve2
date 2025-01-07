"""An example on how to remote control a physical modular robot."""

from pyrr import Vector3
import revolve2
import math
from simulation import Brain

from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body import RightAngles
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.body.sensors import CameraSensor
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighborRandom
from revolve2.modular_robot.brain.cpg import BrainCpgInstance, active_hinges_to_cpg_network_structure_neighbor
from revolve2.modular_robot_physical import Config, UUIDKey
from revolve2.modular_robot_physical.remote import run_remote
from revolve2.standards.modular_robots_v2 import gecko_v2
from main import CustomBrain,make_body

def creat_barin(first_left_active_hinge,
        second_left_active_hinge,
        first_right_active_hinge,
        second_right_active_hinge,
        rear_active_hinge_1,
        rear_active_hinge_2,hinges,body,internal_weights, external_weights):
    # get the hinges of the body we created
    # active_hinges = body.find_modules_of_type(ActiveHinge)
    # print("active",active_hinges)
    # print(len(active_hinges))
    # print("hinges",hinges)
    # print(len(hinges))
    # a = [i for i in active_hinges if i in hinges]
    # print(a)
    # print(len(a))


    active_hinges = hinges

    cpg_network_structure, output_mapping = active_hinges_to_cpg_network_structure_neighbor(active_hinges)

    # initialize the network as uniform state.
    initial_state = cpg_network_structure.make_uniform_state(
        # 12
        0.5 * math.sqrt(2)
    )

    '''connections = [
        (
            active_hinges[pair.cpg_index_lowest.index],
            active_hinges[pair.cpg_index_highest.index],
        )
        for pair in cpg_network_structure.connections
    ]'''

    # (internal_weights, external_weights) = _make_weights(active_hinges, connections, body)
    # internal_weights, external_weights =
    weight_matrix = cpg_network_structure.make_connection_weights_matrix(
        {
            cpg: weight
            for cpg, weight in zip(cpg_network_structure.cpgs, internal_weights)
        },
        {
            pair: weight
            for pair, weight in zip(
            cpg_network_structure.connections, external_weights
        )
        },
    )

    brain = Brain(initial_state=initial_state, weight_matrix=weight_matrix, output_mapping=output_mapping,first_left_active_hinge = first_left_active_hinge,
        second_left_active_hinge = second_left_active_hinge,
        first_right_active_hinge = first_right_active_hinge,
        second_right_active_hinge = second_right_active_hinge,
        rear_active_hinge_1 = rear_active_hinge_1,
        rear_active_hinge_2 = rear_active_hinge_2,)

    return brain,active_hinges


def make__body() -> (
        tuple[BodyV2, tuple[ActiveHinge, ActiveHinge, ActiveHinge, ActiveHinge]]
):
    """
    Create a body for the robot.

    :returns: The created body and a tuple of all ActiveHinge objects for mapping later on.
    """
    """
    A modular robot body follows a 'tree' structure.
    The 'Body' class automatically creates a center 'core'.
    From here, other modular can be attached.
    Modules can be attached in a rotated fashion.
    This can be any angle, although the original design takes into account only multiples of 90 degrees.
    """

    # A modular robot body follows a 'tree' structure.
    # The 'Body' class automatically creates a center 'core'.
    # From here, other modular can be attached.
    # Modules can be attached in a rotated fashion.
    # This can be any angle, although the original design takes into account only multiples of 90 degrees.
    # You should explore the "standards" module as it contains lots of preimplemented elements you can use!
    body = BodyV2()
    body.core_v2.left_face.bottom = ActiveHingeV2(RightAngles.DEG_0)
    body.core_v2.left_face.bottom.attachment = BrickV2(RightAngles.DEG_0)
    body.core_v2.right_face.bottom = ActiveHingeV2(RightAngles.DEG_0)
    body.core_v2.right_face.bottom.attachment = BrickV2(RightAngles.DEG_0)

    # front1
    #body.core_v2.front_face.bottom = ActiveHingeV2(RightAngles.DEG_90)

    # back1
    body.core_v2.back_face.bottom = ActiveHingeV2(RightAngles.DEG_90)
    body.core_v2.back_face.bottom.attachment = BrickV2(RightAngles.DEG_90)
    body.core_v2.back_face.bottom.attachment.front = ActiveHingeV2(RightAngles.DEG_90)
    body.core_v2.back_face.bottom.attachment.front.attachment = BrickV2(RightAngles.DEG_90)

    # back left
    body.core_v2.back_face.bottom.attachment.front.attachment.left = ActiveHingeV2(RightAngles.DEG_0)
    body.core_v2.back_face.bottom.attachment.front.attachment.left.attachment = BrickV2(RightAngles.DEG_0)

    # back right
    body.core_v2.back_face.bottom.attachment.front.attachment.right = ActiveHingeV2(
        RightAngles.DEG_0)

    body.core_v2.back_face.bottom.attachment.front.attachment.right.attachment = BrickV2(
        RightAngles.DEG_0)

    """Here we collect all ActiveHinges, to map them later onto the physical robot."""
    active_hinges = (
        body.core_v2.left_face.bottom,
        body.core_v2.right_face.bottom,
        body.core_v2.back_face.bottom,
        body.core_v2.back_face.bottom.attachment.front,
        body.core_v2.back_face.bottom.attachment.front.attachment.left,
        body.core_v2.back_face.bottom.attachment.front.attachment.right
    )
    return body, active_hinges


def on_prepared() -> None:
    """Do things when the robot is prepared and ready to start the controller."""
    print("Done. Press enter to start the brain.")
    input("here:")


#parameters
parameter = ([-0.14172827785373787, -0.7453174218438501, 0.8917092089465866, -0.4079208369348286, -0.5546040184858618, 0.7306872594423566, 0.014909991559707958, -0.635612746434195, -0.844577464689336, 0.6672191354552035, 0.003331966953915133, 0.2675903938300723, -0.5597274078152055, 0.16162258824077114], [-0.7829257499093809, 0.7512907919337186, 0.9376649860125887, -0.03161854960599442, -0.4721831659520046, -0.30853062914210816, -0.8899192467370087, 0.8093912845317912, 0.39646591851858237, 0.4940918133172725, 0.5194923520256478, 0.32516388095738424, 0.12272094697158176, 0.1758506788])
internal_weights = parameter[0]
external_weights = parameter[1]


def main() -> None:
    """Remote control a physical modular robot."""
    # rng = make_rng_time_seed()
    """
    Create a modular robot, similar to what was done in the 1a_simulate_single_robot example.
    Of course, you can replace this with your own robot, such as one you have optimized using an evolutionary algorithm.
    """
    (
        body,
        first_left_active_hinge,
        second_left_active_hinge,
        first_right_active_hinge,
        second_right_active_hinge,
        rear_active_hinge_1,
        rear_active_hinge_2,
    ) = make_body()

    hinges = (first_left_active_hinge, second_left_active_hinge, first_right_active_hinge, second_right_active_hinge,
              rear_active_hinge_1, rear_active_hinge_2)

    brain,active_hinges = creat_barin(first_left_active_hinge,
        second_left_active_hinge,
        first_right_active_hinge,
        second_right_active_hinge,
        rear_active_hinge_1,
        rear_active_hinge_2,
        hinges,
        body,internal_weights,external_weights)
    robot = ModularRobot(body, brain)

    """
    Some important notes to understand:
    - Hinge mappings are specific to each robot, so they have to be created new for each type of body. 
    - The pin`s id`s can be found on th physical robots HAT.
    - The order of the pin`s is crucial for a correct translation into the physical robot.
    - Each ActiveHinge needs one corresponding pin to be able to move. 
    - If the mapping is faulty check the simulators behavior versus the physical behavior and adjust the mapping iteratively.

    For a concrete implementation look at the following example of mapping the robots`s hinges:
    """


    hinge_1, hinge_2, hinge_3, hinge_4,hinge_5,hinge_6 = hinges
    hinge_mapping = {
        UUIDKey(hinge_1): 31, #rf
        UUIDKey(hinge_2): 0, #lf
        UUIDKey(hinge_3): 7, #tail 1
        UUIDKey(hinge_4): 24, #tail 2
        UUIDKey(hinge_5): 1, #lb
        UUIDKey(hinge_6): 29, #rb
    }

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
        control_frequency=100,
        initial_hinge_positions={UUIDKey(active_hinge): 0.0 for active_hinge in hinges},
        # inverse_servos={hinge_6:True},
    )

    """
    Create a Remote for the physical modular robot.
    Make sure to target the correct hardware type and fill in the correct IP and credentials.
    The debug flag is turned on. If the remote complains it cannot wkeep up, turning off debugging might improve performance.
    """
    print("Initializing robot..")
    run_remote(
        config=config,
        hostname="10.15.3.39",  # "Set the robot IP here.
        debug=True,
        on_prepared=on_prepared,
        display_camera_view=True,
    )
    """
    Note that theoretically if you want the robot to be self controlled and not dependant on a external remote, you can run this script on the robot locally.
    """





if __name__ == "__main__":
    main()