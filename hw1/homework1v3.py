import numpy as np


######################################
# check for BUG -- when inout size is just 1 city!!!
# should I put in a timer that outputs best so far at 280s to avoid timing cutoff?


# Parameters to tweak
POPULATION_SIZE = 100
MAX_GENERATIONS = 300
MUTATION_RATE = 0.05
TOURNAMENT_SIZE = 5
ELITISM_RATE = 0.6
# RANDOM_RATE = 0.2
CROSSOVER_RATE = 0.8

# get data from input file
def read_input(file_name):
    with open(file_name) as input_file:
        N = int(input_file.readline())  # number of cities on the first line
        cities = []                     
        for _ in range(N):
            # each city's coordinates -> ints
            x, y, z = map(int, input_file.readline().split())
            cities.append((x, y, z))    # city as a tuple of coordinates
        return cities, N                

# create initial population of individuals (tours/paths)
def create_initial_population(N):
    population = []
    for each_individual in range(POPULATION_SIZE):
        individual = np.arange(N)
        np.random.shuffle(individual)   # shuffle to create a random tour
        population.append(individual)
    return population   
 
# create a new population while keeping ELITISM_RATE of the top ones from previous population
def create_elitist_population(population, distances_of_individuals):
    num_elites = max(1, int(POPULATION_SIZE * ELITISM_RATE))
    # num_random = int (POPULATION_SIZE * RANDOM_RATE)
    sorted_distances = np.argsort(distances_of_individuals)
    elites = []
    for idx in sorted_distances[:num_elites]:
       elites.append(population[idx])
    
    
    new_population = elites.copy()
    # Add random individuals
   #  for _ in range(num_random):
   #      individual = np.arange(N)
   #      np.random.shuffle(individual)
   #      new_population.append(individual)

    # Fill the rest of the population using selection and reproduction
    while len(new_population) < POPULATION_SIZE:
        parent1 = tournament_selection(population, distances_of_individuals)[0]
        parent2 = tournament_selection(population, distances_of_individuals)[0]
        child = two_point_crossover(parent1, parent2)
        if np.random.rand() < MUTATION_RATE:
            child = swap_mutation(child)
        new_population.append(child)

    return new_population

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

# enter colloseum
def tournament_selection(population, distances_of_individuals):
    tourny_champs = []         
    for _ in range(POPULATION_SIZE):
        participants = np.random.choice(POPULATION_SIZE, TOURNAMENT_SIZE, replace=False) # randomly select 'TOURNAMENT_SIZE' # of individuals from the population
        best_idx = min(participants, key=lambda idx: distances_of_individuals[idx]) # find individual w minimum distance tour/path
        tourny_champs.append(population[best_idx])
    return tourny_champs                    

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

# swap mutation -- swap 2 cities in the route
def swap_mutation(child, N):
    # randomly select two cities to swap
    idx1, idx2 = np.random.choice(N, 2, replace=False)
    child[idx1], child[idx2] = child[idx2], child[idx1]
    return child               

# Main genetic algorithm
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
        
        # print progress every 10 generations
        if generation % 10 == 0:
            print(f"Generation {generation}: Best distance so far: {best_distance}")
        
        # selection
        # population is now a population of champions that make it out of the tournament selection
        population = tournament_selection(population, distances_of_individuals)
        
        # crossover
        crossover_offspring = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = population[i]
            parent2 = population[(i+1) % POPULATION_SIZE] # % so we don't go out of range
            child1 = two_point_crossover(parent1, parent2, N)
            crossover_offspring.append(child1)
            child2 = two_point_crossover(parent2, parent1, N)
            crossover_offspring.append(child2)
        
        # mutation
        mutated_offspring = []
        for child in crossover_offspring:
            if np.random.rand() < MUTATION_RATE:
                chernobyl_child = swap_mutation(child, N)
                mutated_offspring.append(chernobyl_child)
            else:
                mutated_offspring.append(child)

        # population now contains some mutations
        population = mutated_offspring

    # get city coordinates of best tour
    best_path = []
    for idx in best_individual:
        best_path.append(cities[idx])
    best_path.append(cities[best_individual[0]]) # append starting city at the end to complete the tour

    return best_distance, best_path

# Function to write the output to 'output.txt'
def write_output(distance, path):
    with open('output.txt', 'w') as output_file:
        output_file.write(f"{distance}\n")        # write the total distance on the first line
        for city in path:
            # write each city's coordinates, separated by spaces
            output_file.write(f"{city[0]} {city[1]} {city[2]}\n")

# Main operator function
def main():
    print("Reading input...")
    cities, N = read_input('input1.txt')
    print(f"Number of cities: {N}")
    print("Running algorithm...")
    best_distance, best_path = genetic_algorithm(cities, N)
    print("Writing output...")
    write_output(best_distance, best_path)
    print("Program finished successfully.")

# catch-all
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Print any error that occurs during execution
        print(f"An error occurred: {e}")


        
        
        
        
        
# 1 -- https://arxiv.org/pdf/2303.00614
# 2 -- https://arxiv.org/pdf/1812.09351
# 3 -- https://www.researchgate.net/publication/283231968_Optimal_path_planning_for_UAVs_using_Genetic_Algorithm
# 4 -- https://iopscience.iop.org/article/10.1088/1742-6596/2216/1/012035/pdf
# 5 -- https://www.mdpi.com/2226-4310/9/2/86
# 6 -- https://www.researchgate.net/publication/367680503_UAV_Path_Planning_using_Genetic_Algorithm_with_Parallel_Implementation
