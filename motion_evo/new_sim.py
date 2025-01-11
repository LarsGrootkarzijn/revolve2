"""Main script for the example."""
import math
import logging
import random

from pyrr import Vector3

from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body import RightAngles
from revolve2.modular_robot.body.base import ActiveHinge, Body
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2
from revolve2.modular_robot.brain.cpg import BrainCpgInstance, active_hinges_to_cpg_network_structure_neighbor
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulation.scene import Pose
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import terrains, fitness_functions
from revolve2.standards.interactive_objects import Ball
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from main import CustomBrain
from main import make_body as new_body
from remote import creat_barin



class Brain(CustomBrain):
    def __init__(self, initial_state,weight_matrix,output_mapping,first_left_active_hinge,
        second_left_active_hinge,
        first_right_active_hinge,
        second_right_active_hinge,
        rear_active_hinge_1,
        rear_active_hinge_2,):
        super().__init__(first_left_active_hinge,
        second_left_active_hinge,
        first_right_active_hinge,
        second_right_active_hinge,
        rear_active_hinge_1,
        rear_active_hinge_2,initial_state,weight_matrix,output_mapping)
        self.initial_state = initial_state
        self.weight_matrix = weight_matrix
        self.output_mapping = output_mapping


    def make_instance(self):
        return CustomBrain(
         self.first_left_active_hinge,
         self.second_left_active_hinge,
         self.first_right_active_hinge,
         self.second_right_active_hinge,
         self.rear_active_hinge_1,
         self.rear_active_hinge_2,
         self.initial_state,
         self.weight_matrix,
         self.output_mapping
        )

def move_forward_fit( end_state,target_state = Vector3([-5, 0, 0])):
    #begin_position = begin_state.get_pose().position
    end_position = end_state.get_pose().position
    return math.sqrt(
        (end_position.x - target_state.x) ** 2
        + (end_position.y - target_state.y) ** 2
    )

def final_direction_fit( end_state,last_state10,target_state = Vector3([-5, 0, 0])):
    end_position = end_state.get_pose().position
    last_10state_position = last_state10.get_pose().position

    direction_vector = end_position - last_10state_position
    goal_vector = target_state - last_10state_position

    magnitude1 = math.sqrt((direction_vector.x) ** 2 + (direction_vector.y) ** 2)
    magnitude2 = math.sqrt((goal_vector.x) ** 2 + (goal_vector.y) ** 2)

    cos = Vector3.dot(goal_vector,direction_vector)/(magnitude1 * magnitude2)
    # cos is between -1 to 1
    # Here cos 0 is -1(in math should be 1) , no matter left or right are all negative values(with in 90 degree)
    # minimize cos makes the best(0 degree), consistent with final fitness so use cos as part of fit.
    return cos+1
    # make it all positive incase cancel each other's mistake

def displacement_direction_fit(end_state,target_state = Vector3([-5, 0, 0]),start_state = Vector3([0, 0, 0])):
    end_position = end_state.get_pose().position
    # last_10state_position = last_state10.get_pose().position

    direction_vector = end_position - start_state
    goal_vector = target_state - start_state

    magnitude1 = math.sqrt((direction_vector.x) ** 2 + (direction_vector.y) ** 2)
    magnitude2 = math.sqrt((goal_vector.x) ** 2 + (goal_vector.y) ** 2)

    cos = Vector3.dot(goal_vector, direction_vector) / (magnitude1 * magnitude2)
    # cos is between -1 to 1
    # Here cos 0 is -1(in math should be 1) , no matter left or right are all negative values(with in 90 degree)
    # minimize cos makes the best(0 degree), consistent with final fitness so use cos as part of fit.
    return cos+1
    # make it all positive incase cancel each other's mistake


def creat_barin(first_left_active_hinge,   #copy from remote due to import error
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


def simulation(weights_list,time,parallel_branches,evo = True) -> None:
    """Run the simulation.py."""
    # Set up logging to give output of your simulation.py into the command line interface (CLI).
    setup_logging()

    # Set up a random number generator, used later for the brain.
    #rng = make_rng_time_seed()

    # Create a body for the robot.


    robot_list = []
    for (internal_weights, external_weights) in weights_list:

        (
            body,
            first_left_active_hinge,
            second_left_active_hinge,
            first_right_active_hinge,
            second_right_active_hinge,
            rear_active_hinge_1,
            rear_active_hinge_2,
        ) = new_body()

        hinges = (
        first_left_active_hinge, second_left_active_hinge, first_right_active_hinge, second_right_active_hinge,
        rear_active_hinge_1, rear_active_hinge_2)

        brain, active_hinges = creat_barin(first_left_active_hinge,
                                           second_left_active_hinge,
                                           first_right_active_hinge,
                                           second_right_active_hinge,
                                           rear_active_hinge_1,
                                           rear_active_hinge_2,
                                           hinges,
                                           body, internal_weights, external_weights)
        robot = ModularRobot(body, brain)

        robot_list.append(robot)

        """
        To simulate our newly created robot, we create a modular robot scene.
        This scene is a combination of one or more modular robots positioned in a given terrain.
        """
    scene_list = []
    for i in robot_list:
        scene = ModularRobotScene(terrain=terrains.flat())
        scene.add_robot(i,pose=Pose(Vector3([0, 0, 0])),translate_z_aabb = True)
        scene.add_interactive_object(
            Ball(radius=0.1, mass=200, pose=Pose(Vector3([-5, 0, 0])))
        )
        scene_list.append(scene)

    # Additionally to robots you can also add interactive objects to the scene.

    """
    After we have the scene ready we create a simulator that will perform the simulation.py.
    This tutorial chooses to use Mujoco, but your version of revolve might contain other simulators as well.

    For mujoco we can select either the `native` mujoco viewer (more performance) or our `custom` viewer (which is more flexible for adjustments).
    """
    if evo:
        simulator = LocalSimulator(headless = True,num_simulators = parallel_branches,viewer_type="native")
    else:
        simulator = LocalSimulator(headless=False, num_simulators=1,viewer_type="native")
    # `batch_parameters` are important parameters for simulation.py.
    # Here, we use the parameters that are standard in CI Group.
    batch_parameters = make_standard_batch_parameters()
    #batch_parameters.simulation_time =  time # Here we update our simulation.py time.
    #batch_parameters.sampling_frequency = 0.1
    #batch_parameters.simulation_timestep = 0.001
    #batch_parameters.control_frequency = 1

    # Simulate the scene.
    # A simulator can run multiple sets of scenes sequentially; it can be reused.
    # However, in this tutorial we only use it once.
    '''simulate_scenes(
        simulator=simulator,
        batch_parameters=batch_parameters,
        scenes=scene_list,
    )'''

    scene_states = simulate_scenes(
        simulator=simulator,
        batch_parameters=batch_parameters,
        scenes=scene_list,
    )

    """
        Using the previously obtained scene_states we can now start to evaluate our robot.
        Note in this example we simply use x-y displacement, but you can do any other way of evaluation as long as the required data is in the scene states.
        """
    # Get the state at the beginning and end of the simulation.py.
    #scene_state_begin = scene_states[0]
    #scene_state_end = scene_states[-1]
    # print("lenth",len(scene_state_end))


    # Retrieve the state of the modular robot, which also contains the location of the robot.
    fitness_list = []
    for i,j in zip(scene_states,robot_list):
        #v2  final_cos * 5 + displacement_cos * 5 + xy_displacement*4
        # print(len(i))
        # robot_state_end = i[-1].get_modular_robot_simulation_state(j)
        # robot_state_last10 = i[-11].get_modular_robot_simulation_state(j)
        # # robot_state_half = i[]
        #
        # final_cos = final_direction_fit(robot_state_end,robot_state_last10)
        # displacement_cos = displacement_direction_fit(robot_state_end)
        # xy_displacement = move_forward_fit(robot_state_end)
        # fitness = final_cos * 5 + displacement_cos * 5 + xy_displacement*4
        # print(final_cos,displacement_cos,xy_displacement)

        #v3 final_cos * 5 + displacement_cos * 5 + xy_displacement * 2 + mid_displacement_cos * 5
        robot_state_end = i[-1].get_modular_robot_simulation_state(j)
        # robot_state_last10 = i[-11].get_modular_robot_simulation_state(j)
        check_point = random.sample(range(0,int(len(i)/2)), 4)
        # mid_displacement_cos = 0
        accumulate_y_displacement = 0
        scene_length = int(len(i)/2)
        for k in range(scene_length):
            robot_state = i[k].get_modular_robot_simulation_state(j)
            accumulate_y_displacement += abs(robot_state.get_pose().position.y)

        avg_y_displacement = accumulate_y_displacement/scene_length

        # final_cos = final_direction_fit(robot_state_end, robot_state_last10)
        # displacement_cos = displacement_direction_fit(robot_state_end)



        xy_displacement = move_forward_fit(robot_state_end)
        fitness = avg_y_displacement * 5 + xy_displacement  #+ mid_displacement_cos * 10
        print(avg_y_displacement * 10, xy_displacement)#,mid_displacement_cos*10)

        print(fitness)
        fitness_list.append(fitness)

    # Calculate the xy displacement, using the locations of the robot.
    #xy_displacement = fitness_functions.xy_displacement(robot_state_begin, robot_state_end)


    return fitness_list








if __name__ == "__main__":
    n = 20
    #simulation([-0.7828140346735355, -0.2383063058394761, -0.30214756569585544, 0.3835617415327921, -0.7838038598153994, 0.8472082226406816, -0.9165282132645329, 0.38765038101366867, 0.4168883945053148, -0.4046033304890575, 0.5585965626702873, -0.14095180014502917], [0.48258927386216177, -0.7256455385907834, -0.8086211362031719, 0.8496228486181712, 0.0012546912947057898, 0.2685642039609566, 0.5569387903255725, 0.1894636907138254, 0.12347936666927484, 0.4528980825446496, -0.7548460753673705, 0.7457373844677293],60)
    test1 = [([random.random() for i in range(n)], [random.random() for i in range(n)])]# for i in range(25)]
    test2 = [([random.random()-1 for i in range(n)], [random.random()-1 for i in range(n)])]# for i in range(25)]

    
    simulation([([0.01572449274191734, 0.13878631418498233, 0.14635445151397874, -0.10099314619463406, 0.9446012415816385, -0.24579815907860048], [-0.6318089729861862, 0.5273734035024289, -0.3536325809880265, -0.4658103159433835, 0.08921225054092874, 0.11935918787922])],300,1,False)
