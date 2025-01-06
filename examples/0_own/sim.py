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
from revolve2.standards import fitness_functions, modular_robots_v2, terrains

##body1 Gecko
# def make_body() -> BodyV2:
#
#     # A modular robot body follows a 'tree' structure.
#     # The 'Body' class automatically creates a center 'core'.
#     # From here, other modular can be attached.
#     # Modules can be attached in a rotated fashion.
#     # This can be any angle, although the original design takes into account only multiples of 90 degrees.
#     # You should explore the "standards" module as it contains lots of preimplemented elements you can use!
#     body = BodyV2()
#     body.core_v2.left_face.bottom = ActiveHingeV2(RightAngles.DEG_0)
#     body.core_v2.left_face.bottom.attachment = ActiveHingeV2(RightAngles.DEG_0)
#     body.core_v2.left_face.bottom.attachment.attachment = BrickV2(RightAngles.DEG_0)
#     body.core_v2.right_face.bottom = ActiveHingeV2(RightAngles.DEG_0)
#     body.core_v2.right_face.bottom.attachment = ActiveHingeV2(RightAngles.DEG_0)
#     body.core_v2.right_face.bottom.attachment.attachment = BrickV2(RightAngles.DEG_0)
#
#     #back1
#     body.core_v2.back_face.bottom = ActiveHingeV2(RightAngles.DEG_0)
#     # body.core_v2.back_face.bottom.attachment = ActiveHingeV2(RightAngles.DEG_0)
#     body.core_v2.back_face.bottom.attachment = BrickV2(RightAngles.DEG_0)
#
#     #back2
#     body.core_v2.back_face.bottom.attachment.front = ActiveHingeV2(RightAngles.DEG_0)
#     body.core_v2.back_face.bottom.attachment.front.attachment = BrickV2(RightAngles.DEG_0)
#
#     #back left
#     body.core_v2.back_face.bottom.attachment.front.attachment.left = ActiveHingeV2(RightAngles.DEG_0)
#     body.core_v2.back_face.bottom.attachment.front.attachment.left.attachment = ActiveHingeV2(RightAngles.DEG_0)
#     body.core_v2.back_face.bottom.attachment.front.attachment.left.attachment.attachment = BrickV2(RightAngles.DEG_0)
#
#     #back right
#     body.core_v2.back_face.bottom.attachment.front.attachment.right = ActiveHingeV2(
#         RightAngles.DEG_0)
#     body.core_v2.back_face.bottom.attachment.front.attachment.right.attachment = ActiveHingeV2(
#         RightAngles.DEG_0)
#     body.core_v2.back_face.bottom.attachment.front.attachment.right.attachment.attachment = BrickV2(
#         RightAngles.DEG_0)
#
#     return body

#body2 spider

def make_body() -> BodyV2:
    body = modular_robots_v2.gecko_v2();
    return body


class Brain(BrainCpgInstance):
    def __init__(self, initial_state,
        weight_matrix,
        output_mapping):
        super().__init__(initial_state,
        weight_matrix,
        output_mapping)
        self.initial_state = initial_state
        self.weight_matrix = weight_matrix
        self.output_mapping = output_mapping


    def make_instance(self):
        return BrainCpgInstance(
            initial_state=self.initial_state,
            weight_matrix=self.weight_matrix,
            output_mapping=self.output_mapping,
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





def simulation(weights_list,time,parallel_branches,evo = True) -> None:
    """Run the simulation.py."""
    # Set up logging to give output of your simulation.py into the command line interface (CLI).
    setup_logging()

    # Set up a random number generator, used later for the brain.
    #rng = make_rng_time_seed()

    # Create a body for the robot.


    robot_list = []
    for (internal_weights, external_weights) in weights_list:
        body = make_body()

        """
        Here we create a brain for the robot.
        We choose a 'CPG' brain with random parameters.
        If you want to know more about CPGs checkout the Methods section in: https://doi.org/10.1038/s41598-023-48338-4.
        """

        #get the hinges of the body wecreated
        active_hinges = body.find_modules_of_type(ActiveHinge)

        cpg_network_structure, output_mapping = active_hinges_to_cpg_network_structure_neighbor(active_hinges)

        #initialize the network as uniform state.
        initial_state = cpg_network_structure.make_uniform_state(
            0.5
            #0.5 * math.sqrt(2)
        )


        '''connections = [
            (
                active_hinges[pair.cpg_index_lowest.index],
                active_hinges[pair.cpg_index_highest.index],
            )
            for pair in cpg_network_structure.connections
        ]'''

        #(internal_weights, external_weights) = _make_weights(active_hinges, connections, body)
        #internal_weights, external_weights =
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

        brain = Brain(initial_state,weight_matrix,output_mapping)
        #print("test_4by4",brain.control())


        """Once we have a body and a brain we combine it into a ModularRobot."""
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
            Ball(radius=0.1, mass=100, pose=Pose(Vector3([-5, 0, 0])))
        )
        scene_list.append(scene)

    # Additionally to robots you can also add interactive objects to the scene.

    """
    After we have the scene ready we create a simulator that will perform the simulation.py.
    This tutorial chooses to use Mujoco, but your version of revolve might contain other simulators as well.

    For mujoco we can select either the `native` mujoco viewer (more performance) or our `custom` viewer (which is more flexible for adjustments).
    """
    if evo:
        simulator = LocalSimulator(headless = True,num_simulators = parallel_branches)#viewer_type="native")
    else:
        simulator = LocalSimulator(headless=False, num_simulators=1,viewer_type="native")
    # `batch_parameters` are important parameters for simulation.py.
    # Here, we use the parameters that are standard in CI Group.
    batch_parameters = make_standard_batch_parameters()
    batch_parameters.simulation_time =  time # Here we update our simulation.py time.
    #batch_parameters.sampling_frequency = time
    #batch_parameters.simulation_timestep = 10
    #batch_parameters.control_frequency = 10

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
    #simulation(test1,600,1,False)
    #4.611
    #simulation([([0.2967323376101735, 0.37743301890929726, 0.047870385780613756, -0.06264995100063864, -0.29199466319108436, -0.054953380093005544, 0.017471930787840484, 0.4890981556813675, -0.22341676981940983, 0.4193277865600762, -0.06122693101934018, 0.19688507571319736], [-0.11375982987968236, 0.19672270861355223, 0.025968172223339026, -0.24939061325176815, -0.06498068306235494, -0.1835913249963368, 0.3130329737038524, 0.4864958799815957, -0.22669561423551687, 0.0715599125068449, 0.433651170160672, 0.0667055825947459])],600,1,False)
    #3.710
    #simulation([([0.45131110550249276, 0.7615033900671102, 0.12060458086429104, 0.11191512448460506, -0.7296545960841891, 0.38040561704261155, 0.07671475539441808, -0.4154774323173527, -0.340148850329812, 0.8513260415859583, 0.8385356496979797, -0.8674764472144272], [0.8584306446951102, 0.2920389385719706, -0.7460582371039166, 0.906089114223561, -0.1865878924191986, 0.2567526989879516, -0.862597346461748, 0.4482281204584744, -0.30148127607602526, 0.11424933125783854, 0.15730404605301862, -0.7517294869354747])],600,1,False)
    #spider 3.854
    #simulation([([0.4578765139717419, 0.16404096470355145, -0.2721551690216303, -0.14766207662752662, 0.49212637997021014, -0.3893388286364948, 0.4862042150038496, -0.3247855161634876, 0.2718586819108034, -0.47693327988993606, -0.3071812444575568, -0.11216046570816918, 0.16250182006704672, -0.10753328202787116, 0.2511458313510243, 0.23293943491313052, 0.27063382614168907, 0.3986903175433144, -0.2816989565285857, 0.388451241917652], [0.03268186828389252, -0.47251769821170364, 0.16038295483451193, -0.020291822309509433, 0.4488916099202259, -0.208496224003758, 0.373848938280427, 0.3252151770173759, 0.3390128907507435, -0.7043094199266089, 0.19612954585021802, -0.013987654548601958, -0.029900311813147784, 0.039250561814440865, 0.24027517113574826, 0.4489760456788713, 0.19446042029634758, 0.2161447502975764, -0.21757160591253233, 0.4453571143962345])],600,1,False)
    #oct 3.443
    #simulation([([0.7908887793095396, -0.255730435805398, -0.6490823016020741, -0.03836393834513108, -0.5232312408659365, -0.9323402840240231, 0.5085344681357196, -0.4087758230770788, 0.18769193705241793, -0.4989495254324543, -0.9951238797431488, 0.5027682949391303, -0.6412867426970212, -0.7122738206607067, -0.6744692084284545, 0.8609916916495877, -0.9267184110850439, -0.36954125198686194, -0.6140858851328117, -0.8002494099961812], [0.30059248241895675, -0.9179017757663936, -0.6928979281425762, 0.25138095655080606, 0.7288384985254936, -0.22666483969188023, 0.7982243209593238, 0.8225943281717358, 0.1652115636211431, 0.5929900510695387, 0.3722838023907664, 0.67447908324562, -0.7411248722322339, 0.8671581554714076, 0.08113633333470571, 0.9408659661028027, 0.13660145089566678, 0.8084612769592483, 0.1117908850997491, 0.7548180902893307])],600,1,False)
    #oct 2.954
    #simulation([([-0.9068556364352436, 0.9604243633786593, 0.030044853904031, -0.6238667162810192, 0.5622464327958367, 0.08382268694487616, 0.24329022161643477, 0.011377410733188542, 0.012959370398982761, -0.8154033556863134, 0.30744819097548914, 0.901359637924698, 0.3466257983679726, -0.17689131845245587, -0.7320948298998247, 0.07442488928696878, -0.3723928791474387, -0.4735927827808728, 0.42261219599355093, -0.9888883408672264], [0.7290515389866923, -0.5111884431891236, -0.786767038510868, 0.9745251717325409, -0.043199002460293734, -0.25970769062557686, 0.47197051250588684, -0.4249316076598384, 0.16706823192968967, -0.6302690014086529, 0.1156503151552033, 0.1728660108960225, 0.54227116602965, 0.7210216723488136, -0.1949141756071957, 0.26629168596736696, -0.5812380168328415, -0.017461325549040785, -0.17754971648954676, -0.08167985181348159])],600,1,False)
    #oct 1.9541
    # simulation([([-0.9068556364352436, 0.9604243633786593, 0.030044853904031, -0.042144783321301116, 0.5622464327958367, 0.08382268694487616, 0.24329022161643477, 0.011377410733188542, 0.012959370398982761, -0.8154033556863134, 0.12895302586579982, 0.2801972972279836, 0.30633863635019787, 0.7953922271904574, -0.021092346545308116, -0.6918673552523993, 0.3101199639138885, 0.3502738944065429, -0.9983311803739621, -0.3917267091548873], [0.7290515389866923, -0.9815906871650439, -0.786767038510868, 0.9745251717325409, -0.043199002460293734, -0.25970769062557686, 0.47197051250588684, -0.4249316076598384, 0.44576376840607956, -0.6302690014086529, 0.10731816722563536, -0.7004524222840471, 0.13022984739198717, 0.24734884059115947, 0.21346740313701318, 0.5093521426712089, 0.2256302692962917, -0.1859562533428356, -0.03855778649434738, 0.446775577736])],600,1,False)
    # 1.5560
    #simulation([([-0.9068556364352436, 0.9604243633786593, 0.030044853904031, 0.24942085018398008, 0.5622464327958367, -0.26987635315951786, -0.002007051286778916, -0.4266275394655734, 0.012959370398982761, -0.8154033556863134, 0.6819712674772287, -0.21321133967892503, 0.31576750179265156, -0.7284580536888747, -0.6920005850427071, -0.06271499235276856, -0.2691069630535827, 0.7291512299729541, -0.318042194099456, 0.3353216495615272], [0.7290515389866923, -0.5111884431891236, -0.786767038510868, 0.9745251717325409, -0.043199002460293734, -0.25970769062557686, 0.6215615724020302, -0.4249316076598384, 0.16706823192968967, 0.9316086580239238, 0.06654529707094614, -0.9354834726715631, 0.47917029325708027, -0.6785989065688309, 0.23960796647723992, -0.4119584867146202, -0.5397230596284188, -0.8994229184853872, -0.47615645464355016, -0.354225])],600,1,False)

    #v2
    #simulation([([-0.8694967049980131, -0.8035823851143342, -0.9530196824376616, 0.6095240300043481, 0.15296105122170411, -0.7250925805094306, -0.12794606105112538, -0.3795216928729961, -0.5891700885518998, 0.20789123882314287, -0.4472177294471713, -0.6394617893327998, 0.1575203166125907, 0.7024519075995865, 0.14680350975093348, 0.051426928869343635, -0.5146569934801994, 0.3140472441412576, 0.27742271378029737, -0.2578380655980146], [-0.07050142750650856, 0.6934302785318072, -0.3758336372688833, 0.6652633765891955, -0.6254005751013838, 0.6959991222145152, -0.11742483045166319, -0.22579621761049085, 0.2176637315450891, 0.6776711685727865, -0.6448472015418101, -0.7853976996108227, -0.010885942510700186, -0.004665567131134418, 0.12629655740495305, 0.25097737767602735, 0.7754406886450136, 0.10061426182027766, 0.914428739080738, -0.8335511755265])],300,1,False)
    #simulation([([-0.8144645505173582, -0.8710289187566234, 0.483702769697236, 0.9733728819149621, -0.5455920375696208, 0.6753210907021412, 0.8002573071669508, -0.27327598348861426, -0.24952977495659479, -0.3434093543764376, 0.5492635396557115, 0.83817610427411, -0.3997238591865353, -0.6593388082785157, 0.04086600752722047, -0.8284016290289469, 0.744333707008084, -0.27039893743318766, 0.8591948985837217, -0.9920093864563224], [-0.9569869334280934, 0.4706120731640726, 0.5000903068963436, 0.8614715759283373, -0.04274487434489416, -0.43357318351476337, 0.4956448253539272, -0.8792489313998497, -0.9750884208870634, -0.7071312847254609, 0.9250674076960232, 0.4853200320297175, -0.026500934959187594, 0.5355408506405726, -0.11749023404479719, 0.07345936174297152, 0.25886924881258255, 0.051401996265031835, 0.21384569721667557, -0.04726436912158])],300,1,False)
    #simulation([([0.7937152724747145, 0.07413014217976666, 0.5054564113154298, -0.8397771380837913, 0.4120615317581251, -0.08150651363744821, 0.3601949801769144, 0.6524671441031571, -0.5638128151069519, -0.3336048639303417, -0.6220306665206259, 0.38472801361036746, -0.692295968717944, 0.881646217400506, 0.5811915452108511, -0.44977952428774937, -0.3976778575228914, -0.7463122404066294, 0.5665730523698242, 0.7483747834916334], [-0.9189337966846365, 0.7309828535182783, 0.7931388779115915, -0.6950479103678089, -0.18714760330696056, -0.7348617158464705, 0.39247617693607073, 0.26057473879769133, -0.96696779960483, 0.7709158979010418, -0.6198161752989246, 0.9720969318856569, -0.792372588719183, -0.5937516660969786, 0.5467781549052195, -0.9082590612299877, 0.4138927827096597, -0.1993627526855417, -0.9743104376421472, -0.4270648447085476])],300,1,False)
    #simulation([([-0.19293446964651406, 0.9359621440632007, 0.7152389518078388, 0.27161488978789006, 0.24128662485249142, 0.615144413055831, -0.985676090214654, -0.885553480145312, 0.36835584044784686, -0.483836040188256, 0.9672396413653082, -0.20430399858412773, -0.4986341395613667, 0.3865410713458419, 0.3891169163687114, -0.72256823388598, 0.916639110502566, 0.7440394695317809, 0.9384987851278301, -0.4929505834065879], [0.9214892213040318, 0.8154983548014794, -0.18130146540995695, -0.4198859754031683, 0.06286983253595979, -0.988173682492802, -0.6196311154851597, 0.6147529050456846, -0.4131789718286807, 0.5835827654019548, -0.9774644155655967, 0.5085495381013885, -0.34919191542872685, -0.7440602065429467, 0.4936444203141406, 0.8956455103322387, -0.9934199273512914, -0.21386655677767075, 0.6537765229543961, 0.3495782585911888])],300,1,False)

    #simulation([([0.816002062805131, 0.4353469928935958, 0.2118516270847115, -0.621570275559078, -0.8444706478559576, 0.2736896977536565, -0.8007414480806025, -0.7047053475657092, 0.1482386367291093, -0.3680757370422292, 0.6208080274316514, -0.9553958954579567, 0.5850293096889438, 0.5978412844320018], [0.4814190299013992, -0.9789783345349947, -0.4739805402353785, -0.5518335862862205, -0.6867416242074269, -0.95234226526636, -0.4014905246126286, -0.5871445846403809, -0.1285249622674871, 0.7902540512267304, -0.2666742152683277, -0.5904255818514315, -0.059665996223696505, -0.919175869647561])],300,1,False)

    #V3
    #simulation([([-0.04976292924609016, -0.008724116725703501, -0.7401463485876545, 0.43248756833343904, 0.7759648831140185, -0.6537053670636204, 0.04338862477524952, -0.594996721108846, -0.6537827477092351, -0.9131426434515788, 0.7556205997920529, 0.7200206126711672, -0.2237007743778341, 0.4970605052959205], [-0.9821296478649486, 0.8040966148387343, 0.9909274790237597, -0.03161854960599442, -0.4721831659520046, 0.6284661469556723, -0.8899192467370087, 0.8093912845317912, -0.2761572403868311, 0.012674057177946452, 0.7727940742798518, 0.5628421161551613, -0.4264608164795147, -0.7169140492391144])],300,1,False)
    #simulation([([0.8230967843932604, -0.7453174218438501, 0.8917092089465866, -0.4079208369348286, -0.5546040184858618, 0.7306872594423566, 0.014909991559707958, -0.635612746434195, -0.844577464689336, 0.6672191354552035, 0.003331966953915133, 0.2675903938300723, -0.5597274078152055, 0.16162258824077114], [-0.7829257499093809, 0.8040966148387343, 0.7525830897474568, -0.03161854960599442, -0.4721831659520046, -0.17409637874275852, 0.5768274685479786, 0.8093912845317912, 0.39646591851858237, -0.903115946067282, 0.5194923520256478, 0.32516388095738424, 0.12272094697158176, 0.17585067881])],300,1,False)
    simulation([([-0.14172827785373787, -0.7453174218438501, 0.8917092089465866, -0.4079208369348286, -0.5546040184858618, 0.7306872594423566, 0.014909991559707958, -0.635612746434195, -0.844577464689336, 0.6672191354552035, 0.003331966953915133, 0.2675903938300723, -0.5597274078152055, 0.16162258824077114], [-0.7829257499093809, 0.7512907919337186, 0.9376649860125887, -0.03161854960599442, -0.4721831659520046, -0.30853062914210816, -0.8899192467370087, 0.8093912845317912, 0.39646591851858237, 0.4940918133172725, 0.5194923520256478, 0.32516388095738424, 0.12272094697158176, 0.1758506788])],300,1,False)


    #            [random.random()-1 for i in range(n)],
    #            [random.random()-1 for i in range(n)])], 60, 1, False)
    # simulation(test1,60,25,False)
    # simulation(test2,60,25,False)
    # simulation(test1 + test2,60,25,False)
    #simulation([([-0.7828140346735355, -0.2383063058394761, -0.30214756569585544, 0.3835617415327921, -0.7838038598153994, 0.8472082226406816, -0.9165282132645329, 0.38765038101366867, 0.4168883945053148, -0.4046033304890575, 0.5585965626702873, -0.14095180014502917], [0.48258927386216177, -0.7256455385907834, -0.8086211362031719, 0.8496228486181712, 0.0012546912947057898, 0.2685642039609566, 0.5569387903255725, 0.1894636907138254, 0.12347936666927484, 0.4528980825446496, -0.7548460753673705, 0.7457373844677293])],60,1,False)