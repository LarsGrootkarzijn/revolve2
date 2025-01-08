import time


import random
from new_sim import simulation
import numpy as np

#config
pop_size = 200
max_generation = 1500
geno_length = 6
simulation_time = 300
survival_rate = 0.5
mutation_rate = 0.15
parallel_num = 50


# create population to evolve
def create_population():
        population = []
        for i in range(pop_size):
                internal = [random.random()*2-1 for n in range (geno_length)]
                external = [random.random()*2-1 for n in range (geno_length)]
                population.append((internal,external))
        return population



def parent_selection(population):
        fitness_list = []
        parent_list = []
        counter = 0
        for i in range(int(len(population)/parallel_num)):
                fitness = simulation(population[i * parallel_num:(i + 1) * parallel_num],simulation_time,parallel_num)
                for j in fitness:
                        fitness_list.append(j)
                counter += parallel_num
                print(f"this is {counter}-th individual.")
                print(len(population))
        pop_size = len(population)
        current_best = population[fitness_list.index(min(fitness_list))]
        if current_best not in parent_list:
            parent_list.append(current_best)
        while len(parent_list) < int(pop_size*survival_rate):
                now = len(parent_list)
                team = random.sample(fitness_list,10)
                winner_index = fitness_list.index(min(team))
                winner = population[winner_index]
                del population[winner_index]
                del fitness_list[winner_index]
                if winner not in parent_list:
                        parent_list.append(winner)
        #output the parents and the best one in this generation
        return parent_list, (min(fitness_list),current_best),sum(fitness_list)/len(fitness_list)



def create_children(parent_list):
        child_list = []
        for (internal,external) in parent_list:
                match_list = parent_list.copy()
                match_list.remove((internal,external))
                match = random.sample(match_list,1)[0]
                # if internal != match[0]:
                #internal
                child_1 = cross_over(internal,match[0])
                if random.random() < mutation_rate:
                        child_1 = mutation(child_1)
                #external
                child_2 = cross_over(external,match[1])
                if random.random() < mutation_rate:
                        child_2 = mutation(child_2)

                child = (child_1,child_2)


                child_list.append(child)

        return child_list



def mutation(geno):

        gene = random.sample(geno,1)[0]
        new_gene = random.random()*2-1
        geno[geno.index(gene)] = new_gene

        return geno

def cross_over(geno1,geno2):
        cross_position = int(geno_length/2)
        new_geno = geno1[:cross_position] + geno2[cross_position:]
        return new_geno


def load_pop():
    with open("save_pop", "r") as f:
        population = []
        a = f.readlines()
        for i in a:
            if "[" in i:
                k = i.replace("]), ([", "])\n([")[2:-1]
                # print(k)
                indis = k.split("\n")
                for num, j in enumerate(indis):
                    new1 = j[1:-1].replace("], [", "],?[").split("?")[0][1:-2].split(",")
                    new2 = j[1:-1].replace("], [", "],?[").split("?")[1][1:-2].split(",")
                    # new = new1 + new2
                    new1 = [float(item) for item in new1]
                    new2 = [float(item) for item in new2]

                    individual = (new1, new2)
                    population.append(individual)

            if "round" in i:
                    round = int(i[6:-1])

        # make is hashable
        unique_list = list({tuple((tuple(item[0]), tuple(item[1]))) for item in population})

        unique_length = len(unique_list)



        #check same individual
        if len(population) != unique_length:
            print(f"There are {len(population) - unique_length} repeated individual found")

            #back to original format
            unique_list = [([*item[0]], [*item[1]]) for item in unique_list]

            for i in range(pop_size - unique_length):
                sample = random.sample(unique_list, 1)[0]

                new_individual = ([],[])
                for para1,para2 in zip(sample[0],sample[1]):
                    new_para1 = np.random.normal(loc=para1, scale=1.0, size=None)
                    new_individual[0].append(new_para1)
                    new_para2 = np.random.normal(loc=para2, scale=1.0, size=None)
                    new_individual[1].append(new_para2)

        return population,round



if __name__ == "__main__":



    '''This parameter to decide whether we want to load population from local or not, 
    Change it to True if you want to load'''

    load = True

    if load:
        population,round = load_pop()
        i = round

    else:
        population = create_population()
        i = 0

    while i < max_generation:
    #for i in range (max_generation):
            i += 1
            parent_list,best,avg = parent_selection(population)
            with open("best", "a") as f:
                    f.write(f"round {i}   {str(best)}  \n")
            with open("best", "a") as f:
                    f.write(f"round {i}  average_fit {str(avg)}  \n")
            child_list = create_children(parent_list)
            population = parent_list + child_list
            # print(len(population))
            with open("save_pop", "w") as f:
                f.write(f"round {i} \n population:\n {population} ")

