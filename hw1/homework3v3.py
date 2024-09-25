import numpy as np
import time
import os

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description="Genetic Algorithm Parameters")
    parser.add_argument('--population_size', type=int, required=True, help='Population size')
    parser.add_argument('--max_generations', type=int, required=True, help='Max generations')
    parser.add_argument('--mutation_rate', type=float, required=True, help='Mutation rate')
    parser.add_argument('--tournament_size', type=int, required=True, help='Tournament size')
    parser.add_argument('--tournament_finalists', type=int, required=True, help='Tournament finalists')
    parser.add_argument('--selection_rate', type=float, required=True, help='Selection rate')
    parser.add_argument('--crossover_rate', type=float, required=True, help='Crossover rate')

    return parser.parse_args()

args = parse_arguments()

# Use these variables in place of the hardcoded values
POPULATION_SIZE = args.population_size
MAX_GENERATIONS = args.max_generations
MUTATION_RATE = args.mutation_rate
TOURNAMENT_SIZE = args.tournament_size
TOURNAMENT_FINALISTS = args.tournament_finalists
SELECTION_RATE = args.selection_rate
CROSSOVER_RATE = args.crossover_rate
######################################
# check for BUG -- when input size is just 1 city!!! or 2 cities also crashes program. 3 seems okay tho...
# should I put in a timer that outputs best so far at 280s to avoid timing cutoff?
# add exception handling?
# multiprocessing to parallelize distance evals?

# Parameters to tweak
# POPULATION_SIZE = 60
# MAX_GENERATIONS = 1000
# MUTATION_RATE = 0.101
# TOURNAMENT_SIZE = 10
# TOURNAMENT_FINALISTS = 6
REPLACE_RATE = 0.0 # sampling without replacement is usually preferred to maintain diversity.
# SELECTION_RATE = 0.6
# RANDOM_RATE = 0.2 # don't think we need -- crossover + mutation should be enough
# CROSSOVER_RATE = 0.8

# get data from input file
def read_input(file_name):
    with open(file_name) as input_file:
        N = int(input_file.readline())  # number of cities on the first line
        cities = []                     
        for _ in range(N):
            # each city's coordinates -> ints
            x, y, z = map(int, input_file.readline().split())
            cities.append((x, y, z))
        return cities, N                

# create initial population of individuals (tours/paths)
def create_initial_population(N):
    population = []
    for each_individual in range(POPULATION_SIZE):
        individual = np.arange(N)
        np.random.shuffle(individual)   # shuffle to create a random tour
        population.append(individual)
    return population

# Euclidean distance between two cities
def distance(city1, city2):
    return np.sqrt(
        (city1[0] - city2[0]) ** 2 + 
        (city1[1] - city2[1]) ** 2 + 
        (city1[2] - city2[2]) ** 2 
    )

# calculate total distance of a tour (an individual)
def total_distance(individual, cities, N):
    dist = 0                       
    for i in range(N):
        # % to avoid out of bounds (ie loop back to city 1)
        idx1 = individual[i]
        idx2 = individual[(i + 1) % N]

        city1 = cities[idx1]
        city2 = cities[idx2]
        dist += distance(city1, city2)
    return dist   

# enter the colloseum
def tournament_selection(selected_population, selected_distances):
    TOURNY_POP_SIZE = len(selected_population)
    new_population = []

    while len(new_population) < TOURNY_POP_SIZE:
      #   if np.random.rand() < REPLACE_RATE:
      #      replace_bool = True
      #   else:
      #      replace_bool = False
        participants_idxs = np.random.choice(TOURNY_POP_SIZE, TOURNAMENT_SIZE, replace=False)
        participants = [selected_population[idx] for idx in participants_idxs]
        participants_distances = [selected_distances[idx] for idx in participants_idxs]

        # since lower distances are better, sort in ascending order
        sorted_participants = sorted(zip(participants, participants_distances), key=lambda x: x[1])

        for individual, distance in sorted_participants[:TOURNAMENT_FINALISTS]:
            new_population.append(individual)
            if len(new_population) >= TOURNY_POP_SIZE:
                break  # stop if we hit population size

    # ensure nnew population is expected size
    new_population = new_population[:TOURNY_POP_SIZE]

    return new_population
 
# Heuristic crossover -- inspo from citation 5
########################### YOU SHOULD SELECT "ADJACENT PARENTS"
########################### NEEDS UN-GPT REWORK
def crossover(parent1, parent2, cities, N):

    # prioritize best parent in crossover
    parent1_distance = total_distance(parent1, cities, N)
    parent2_distance = total_distance(parent2, cities, N)
    if parent1_distance < parent2_distance:
        higher_fitness_parent = parent1.copy()
        lower_fitness_parent = parent2.copy()
    else:
        higher_fitness_parent = parent2.copy()
        lower_fitness_parent = parent1.copy()

    # find distances between each city
    city_distances = []
    for i in range(N):
        gene1 = higher_fitness_parent[i]
        gene2 = higher_fitness_parent[(i + 1) % N]
        city1 = cities[gene1]
        city2 = cities[gene2]
        dist = distance(city1, city2)
        city_distances.append(dist)

    # find spot in path w largest distance = starting point for crossover
    max_distance_idx = np.argmax(city_distances)
    s = (max_distance_idx + 1) % N

    # get a random end point
    if s < N - 1:
        possible_end_points = list(range(s + 1, N))
    else:
        possible_end_points = list(range(0, s))
    e = np.random.choice(possible_end_points) # MAKE SURE POSSIBLE_END POINTS IS NOT EMPTY, OR ELSE THIS BREAKS!

    # Ensure e != s
    while e == s: # MAKE SURE POSSIBLE_END POINTS IS NOT EMPTY, OR ELSE THIS BREAKS!
        e = np.random.choice(possible_end_points)

    # crossover the selected paths from s to e
    child1 = higher_fitness_parent.copy()
    child2 = lower_fitness_parent.copy()

    if s <= e:
        slice1 = higher_fitness_parent[s:e+1].copy()
        slice2 = lower_fitness_parent[s:e+1].copy()
        child1[s:e+1] = slice2
        child2[s:e+1] = slice1
    else:
        # wrap-around case
        idxs = list(range(s, N)) + list(range(0, e+1))
        slice1 = np.concatenate((higher_fitness_parent[s:], higher_fitness_parent[:e+1])).copy()
        slice2 = np.concatenate((lower_fitness_parent[s:], lower_fitness_parent[:e+1])).copy()
        child1[idxs] = slice2
        child2[idxs] = slice1

    # repair duplicates caused by crossover
    child1 = repair_duplicates(child1, s, e, N)
    child2 = repair_duplicates(child2, s, e, N)

    return child1, child2

########################### NEEDS UN-GPT REWORK
def repair_duplicates(child, s, e, N):
    # gather indexes of cities outside of crossover space
    # gather indexes within crossover space (protect these)
    if s <= e:
        inside_cities = set(child[s:e+1])
        outside_idxs = list(range(0, s)) + list(range(e+1, N))
    else:
        inside_cities = set(np.concatenate((child[s:], child[:e+1])))
        outside_idxs = list(range(e+1, s))

    all_cities = set(range(N))
    missinng_cities = list(all_cities - set(child))
    np.random.shuffle(missinng_cities)
    
    # gather duplicate cities
    duplicates = []
    for idx in outside_idxs:
        gene = child[idx]
        if gene in inside_cities:
            duplicates.append(idx)
    
    # replace duplicate cities w missing ones to make a valid path
    for dup_idx, missing_city in zip(duplicates, missinng_cities):
        child[dup_idx] = missing_city
    
    return child

def is_valid_path(individual, N):
    return set(individual) == set(range(N))

# swap mutation -- swap 2 cities in the route
def swap_mutation(child, N):
    idx1, idx2 = np.random.choice(N, 2, replace=False)
    child[idx1], child[idx2] = child[idx2], child[idx1]
    return child
 
# Random interval reverse mutation
def reverse_mutation(individual, N):
   idxs = np.random.choice((N), 2, replace=False)
   start_idx, end_idx = sorted(idxs)
   if start_idx + 4 > end_idx:
      return swap_mutation(individual, N)
   middle_idx = (start_idx + end_idx) // 2
   interval1 = individual[start_idx:middle_idx] 
   interval2 = individual[middle_idx:end_idx]
   chernobyl_child = individual.copy()
   chernobyl_child[start_idx:middle_idx] = np.flip(interval1)
   chernobyl_child[middle_idx:end_idx] = np.flip(interval2)
   return chernobyl_child
   
# Main genetic algorithm
# SHOULD WE HAVE CASES WHERE UNDER 3 CITIES, WE DON'T DO CROSSOVER OR MUTATION SINCE THEY NEED TO HAVE 
# A MIN NUMBER OF ELEMENTS IN THEIR ARRAYS TO WORK?
# IMPLEMENT A STOP TEST OTHER THAN MAX GENERATIONS? COULD BE A TIMER FUNCTION OR IF NO IMPROVEMENT FOR X NUM OF GENS?
# HUGE PROBLEM OF NOT INCLUDING STARTING POINT AS END POINT UNTIL THE FINAL STEP!!!!!
def genetic_algorithm(cities, N):
    population = create_initial_population(N) 
    best_distance = float('inf')    
    best_individual = None           

    for generation in range(MAX_GENERATIONS):
        distances_of_individuals = [total_distance(individual, cities, N) for individual in population]
        current_best_distance = min(distances_of_individuals)
         # individual with the minimum distance path in the current generation
        current_best_individual = population[np.argmin(distances_of_individuals)]
        
        # update best solution found so far
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_individual = current_best_individual.copy()
            gen_of_best_distance = generation
        
        # print progress every 10 generations
        if generation % 10 == 0:
            print(f"Generation {generation}: Best distance so far: {best_distance}")
        
########## selection
        num_selected = int(SELECTION_RATE * POPULATION_SIZE)

        selected_indices = np.random.choice(POPULATION_SIZE, num_selected, replace=False)
        unselected_indices = list(set(range(POPULATION_SIZE)) - set(selected_indices))

        selected_population = [population[idx] for idx in selected_indices]
        unselected_population = [population[idx] for idx in unselected_indices]
        selected_distances = [distances_of_individuals[idx] for idx in selected_indices]

        # tournament selection w the selected individuals
        selected_population = tournament_selection(selected_population, selected_distances)

        # population is now a population of champions that make it out of the tournament selection
        population = selected_population + unselected_population
        
######### crossover
        crossover_offspring = []

       # heuristic crossover implementation
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = population[i]
            parent2 = population[(i + 1) % POPULATION_SIZE]
            if np.random.rand() < CROSSOVER_RATE:
               child1, child2 = crossover(parent1, parent2, cities, N)
            else:
               child1, child2 = parent1, parent2
            if np.random.rand() < MUTATION_RATE:
                chernobyl_child1 = reverse_mutation(child1, N)
                chernobyl_child2 = reverse_mutation(child2, N)
                crossover_offspring.append(chernobyl_child1)
                crossover_offspring.append(chernobyl_child2)
            else:
                crossover_offspring.append(child1)
                crossover_offspring.append(child2)

        # mutation
      #   mutated_offspring = []
      #   for child in crossover_offspring:
      #       if np.random.rand() < MUTATION_RATE:
      #           chernobyl_child = reverse_mutation(child, N)
      #           mutated_offspring.append(chernobyl_child)
      #       else:
      #           mutated_offspring.append(child)

        # Elitism
        sacrificed_individual = np.random.randint(0, POPULATION_SIZE)
        crossover_offspring[sacrificed_individual] = best_individual.copy()

        # new population after selection, crossover, mutation, and elitism
        population = crossover_offspring

   # check I'm not crazy
    if is_valid_path(best_individual, N):
       print("best path is valid :)")
       
    best_path = []
    for idx in best_individual:
        best_path.append(cities[idx])
    best_path.append(cities[best_individual[0]]) # append starting city at the end to complete the tour

    return best_distance, best_path, gen_of_best_distance

# Function to write the output to 'output.txt'
def write_output(distance, path):
    with open('output.txt', 'w') as output_file:
        output_file.write(f"{distance}\n")        # write the total distance on the first line
        for city in path:
            # write each city's coordinates, separated by spaces
            output_file.write(f"{city[0]} {city[1]} {city[2]}\n")

def log_parameters(**kwargs):
    """
    Logs the parameters and results to 'parameters_output.txt' in a tabular format.

    Parameters:
    - kwargs: Dictionary containing parameter names and their values.
    """
    log_file = 'parameters_output.txt'
    file_exists = os.path.isfile(log_file)

    # Define the order of columns and their widths
    columns = [
        ('Input File', 15),
        ('Best Distance', 15),
        ('Total Time(s)', 15),
        ('GenOfBest Generation', 15),
        ('Max Generations', 15),
        ('Population Size', 15),
        ('Mutation Rate', 15),
        ('Crossover Rate', 15),
        ('Tournament Size', 15),
        ('Tournament Finalists', 15),
        ('Replace Rate', 15)
    ]

    # Create the header and format strings
    header_lines = [''] * max(len(col_name.split()) for col_name, _ in columns)
    separator = '  '
    format_str = ''

    for col_name, width in columns:
        words = col_name.split()
        for i, word in enumerate(words):
            header_lines[i] += f'{word:<{width}}'
        for j in range(len(words), len(header_lines)):
            header_lines[j] += ' ' * width

        separator += '-' * width
        format_str += f'{{{col_name}:<{width}}}'

    # Combine the header lines into a single string
    header = '\n'.join(header_lines)

    # Open the file in append mode
    with open(log_file, 'a') as file:
        # Write the header only if the file doesn't exist
        if not file_exists:
            file.write(header + '\n')
            file.write(separator + '\n')

        # Prepare the row data
        row_data = {
            'Input File': kwargs.get('input_file', ''),
            'Best Distance': f"{round(kwargs.get('best_distance', 0), 2)}",
            'Total Time(s)': f"{round(kwargs.get('total_time', 0), 2)}",
            'GenOfBest Generation': kwargs.get('gen_of_best_distance' ''),
            'Max Generations': kwargs.get('max_generations', ''),
            'Population Size': kwargs.get('population_size', ''),
            'Mutation Rate': kwargs.get('mutation_rate', ''),
            'Crossover Rate': kwargs.get('crossover_rate', ''),
            'Tournament Size': kwargs.get('tournament_size', ''),
            'Tournament Finalists': kwargs.get('tournament_finalists', ''),
            'Replace Rate': kwargs.get('replace_rate', '')
        }

        # Write the formatted row to the file
        file.write(format_str.format(**row_data) + '\n')

# Main operator function
def main():
    print("Reading input...")
    input_file = 'input4.txt'
    cities, N = read_input(input_file)
    print(f"Number of cities: {N}")
    print("Running algorithm...")
    # Record the start time
    start_time = time.time()
    best_distance, best_path, gen_of_best_distance = genetic_algorithm(cities, N)
    # Record the end time and calculate the total time taken
    end_time = time.time()
    total_time = end_time - start_time
    print("Writing output...")
    write_output(best_distance, best_path)
    print("Program finished successfully.")
    
    log_parameters(
        input_file=input_file,
        best_distance=best_distance,
        total_time=total_time,
        gen_of_best_distance=gen_of_best_distance,
        max_generations=MAX_GENERATIONS,
        population_size=POPULATION_SIZE,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        tournament_size=TOURNAMENT_SIZE,
        tournament_finalists=TOURNAMENT_FINALISTS,
        replace_rate=REPLACE_RATE        
    )

# catch-all
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Print any error that occurs during execution
        print(f"An error occurred: {e}")









# Citations / Inspo     
# 1 -- https://arxiv.org/pdf/2303.00614
# 2 -- https://arxiv.org/pdf/1812.09351
# 3 -- https://www.researchgate.net/publication/283231968_Optimal_path_planning_for_UAVs_using_Genetic_Algorithm
# 4 -- https://iopscience.iop.org/article/10.1088/1742-6596/2216/1/012035/pdf
# 5 -- https://www.mdpi.com/2226-4310/9/2/86
# 6 -- https://www.researchgate.net/publication/367680503_UAV_Path_Planning_using_Genetic_Algorithm_with_Parallel_Implementation
# 7 --
# The
# following common parameters are selected for all algorithms:
# population size is 50, crossover probability is 1.0 (i.e., 100%),
# mutation probability is 0.09 (i.e., 9%), and maximum of 1,000
# generations is the terminating condition

#  The parameters are as follows: population
# size, maximum generation, crossover, and mutation
# probabilities are 150, 500, 0.80, and 0.10, respectively, for less
# than 100 size instances, whereas population size and
# maximum generation are 200 and 1000, respectively for more
# than 100 size instances

# III. DESIGN OF OUR GENETIC ALGORITHMS
# A simple GA may be summarized as follows:
# Step 1: Create initial random population of chromosomes
# of size Ps and set generation = 0.
# Step 2: Evaluate the population.
# Step 3: Set generation = generation + 1 and select good
# chromosomes by selection procedure.
# Step 4: Perform crossover with crossover probability Pc
# .
# Step 5: Perform bit-wise mutation with mutation
# probability Pm.
# Step 6: Replace old population with new one.
# Step 7: Repeat Steps 2 to 6 until the terminating criterion
# is satisfied.