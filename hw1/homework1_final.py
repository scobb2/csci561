import numpy as np
import time

# Parameters to tweak
POPULATION_SIZE = 60
MAX_GENERATIONS = 1000
MUTATION_RATE = 0.05
TOURNAMENT_SIZE = 5
TOURNAMENT_FINALISTS = 1
SELECTION_RATE = 0.5
CROSSOVER_RATE = 0.7

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
        participants_idxs = np.random.choice(
            TOURNY_POP_SIZE, TOURNAMENT_SIZE, replace=False)
        participants = [selected_population[idx] for idx in participants_idxs]
        participants_distances = [selected_distances[idx]
                                  for idx in participants_idxs]

        # since lower distances are better, sort in ascending order
        sorted_participants = sorted(
            zip(participants, participants_distances), key=lambda x: x[1])

        for individual, distance in sorted_participants[:TOURNAMENT_FINALISTS]:
            new_population.append(individual)
            if len(new_population) >= TOURNY_POP_SIZE:
                break  # stop if we hit population size

    # ensure nnew population is expected size
    new_population = new_population[:TOURNY_POP_SIZE]

    return new_population

# Heuristic crossover -- inspo from citation 5
def heuristic_crossover(parent1, parent2, cities, N):

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
    # MAKE SURE POSSIBLE_END POINTS IS NOT EMPTY, OR ELSE THIS BREAKS!
    e = np.random.choice(possible_end_points)

    # Ensure e != s
    while e == s:  # MAKE SURE POSSIBLE_END POINTS IS NOT EMPTY, OR ELSE THIS BREAKS!
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
        slice1 = np.concatenate(
            (higher_fitness_parent[s:], higher_fitness_parent[:e+1])).copy()
        slice2 = np.concatenate(
            (lower_fitness_parent[s:], lower_fitness_parent[:e+1])).copy()
        child1[idxs] = slice2
        child2[idxs] = slice1

    # repair duplicates caused by crossover
    child1 = repair_duplicates(child1, s, e, N)
    child2 = repair_duplicates(child2, s, e, N)

    return child1, child2

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

# two-point crossover between two parents
def two_point_crossover(parent1, parent2, N):
    crossover_idxs = np.random.choice(N, 2, replace=False) # randomly select two crossover points and sort them
    start, end = sorted(crossover_idxs) # sort to avoid end point < start point
    child = -np.ones(N, dtype=int)   # -1 placeholders bc no city indice is negative
    child[start:end+1] = parent1[start:end+1]       # copy slice from parent1 into child
    ptr = 0                         
    for city in parent2:
        if city not in child: # can't repeat cities, wouldn't be proper route
            # find next empty position in the child (ie one with '-1')
            while child[ptr] != -1:
                ptr += 1
            child[ptr] = city          
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

# construct path for output
def get_best_path(best_individual, cities):
   best_path = []
   for idx in best_individual:
      best_path.append(cities[idx])
      # append starting city at the end to complete the tour
   best_path.append(cities[best_individual[0]])
   return best_path

# Main genetic algorithm
def genetic_algorithm(cities, N):
    population = create_initial_population(N)
    if N < 3:
       best_individual = population[0]
       best_distance = total_distance(best_individual, cities, N)
       best_path = get_best_path(best_individual, cities)
       return best_distance, best_path, 1

    best_distance = float('inf')
    best_individual = None

    for generation in range(MAX_GENERATIONS):
        distances_of_individuals = [total_distance(individual, cities, N) for individual in population]
        current_best_distance = min(distances_of_individuals)
        # individual with the minimum distance path in the current generation
        current_best_individual = population[np.argmin(distances_of_individuals)]
        global SELECTION_RATE
        global TOURNAMENT_SIZE

        # update best solution found so far
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_individual = current_best_individual.copy()
            gen_of_best_distance = generation
            MUTATION_RATE = 0.05
            CROSSOVER_RATE = 0.8
            SELECTION_RATE = 0.7
            TOURNAMENT_SIZE = 3
        else:
           MUTATION_RATE = 0.2
           CROSSOVER_RATE = 1
           SELECTION_RATE = 0.5
           TOURNAMENT_SIZE = 20


        # print progress every 10 generations
      #   if generation % 100 == 0:
      #       print(
      #           f"Generation {generation}: Best distance so far: {best_distance}")

# selection
        num_selected = int(SELECTION_RATE * POPULATION_SIZE)

        selected_indices = np.random.choice(POPULATION_SIZE, num_selected, replace=False)
        unselected_indices = list(set(range(POPULATION_SIZE)) - set(selected_indices))

        selected_population = [population[idx] for idx in selected_indices]
        unselected_population = [population[idx] for idx in unselected_indices]
        selected_distances = [distances_of_individuals[idx]for idx in selected_indices]

        # tournament selection w the selected individuals
        selected_population = tournament_selection(selected_population, selected_distances)

        # population is now a population of champions that make it out of the tournament selection
        population = selected_population + unselected_population

# crossover + mutation
        crossover_offspring = []

       # heuristic crossover implementation
            # for i in range(0, POPULATION_SIZE, 2):
            #       parent1 = population[i]
            #       parent2 = population[(i + 1) % POPULATION_SIZE]
            #       if np.random.rand() < CROSSOVER_RATE:
            #          child1, child2 = heuristic_crossover(parent1, parent2, cities, N)
            #       else:
            #          child1, child2 = parent1, parent2
            #       if np.random.rand() < MUTATION_RATE:
            #          child1 = reverse_mutation(child1, N)
            #       if np.random.rand() < MUTATION_RATE:
            #          child2 = reverse_mutation(child2, N)
            #       crossover_offspring.append(child1)
            #       crossover_offspring.append(child2)
                  
            # two-point crossover  
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = population[i]
            parent2 = population[(i+1) % POPULATION_SIZE] # % so we don't go out of range
            if np.random.rand() < CROSSOVER_RATE:
               child1 = two_point_crossover(parent1, parent2, N)
               child2 = two_point_crossover(parent2, parent1, N)
            else:
               child1, child2 = parent1, parent2
            if np.random.rand() < MUTATION_RATE:
               child1 = reverse_mutation(child1, N)
            if np.random.rand() < MUTATION_RATE:
               child2 = reverse_mutation(child2, N)
            crossover_offspring.append(child1)
            crossover_offspring.append(child2)

        # Elitism
        sacrificed_individual = np.random.randint(0, POPULATION_SIZE)
        crossover_offspring[sacrificed_individual] = best_individual.copy()

        # new population after selection, crossover, mutation, and elitism
        population = crossover_offspring

    # check I'm not crazy
    if is_valid_path(best_individual, N):
        print("best path is valid :)")

    best_path = get_best_path(best_individual, cities)
    return best_distance, best_path, gen_of_best_distance

# write the output
def write_output(distance, path):
    with open('output.txt', 'w') as output_file:
        # write the total distance on the first line
        output_file.write(f"{distance}\n")
        for city in path:
            # write each city's coordinates, separated by spaces
            output_file.write(f"{city[0]} {city[1]} {city[2]}\n")

def main():
    # print("Reading input...")
    input_file = 'input2.txt'
    cities, N = read_input(input_file)
    start_time = time.time()
    # print(f"Number of cities: {N}")
    # print("Running algorithm...")
    best_distance, best_path, gen_of_best_distance = genetic_algorithm(cities, N)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Best path: {best_distance} found at : {gen_of_best_distance} gen")
    print(f"ran in : {total_time}s")
    # print("Writing output...")
    write_output(best_distance, best_path)
    # print("Program finished successfully.")

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
