import numpy as np

# Parameters to tweak
POPULATION_SIZE = 100      # Number of individuals in the population
MAX_GENERATIONS = 500      # Number of generations to run the algorithm
MUTATION_RATE = 0.1        # Probability of mutation for each individual
TOURNAMENT_SIZE = 5        # Number of individuals participating in tournament selection

# Function to read input from 'input.txt'
def read_input(file_name):
    with open(file_name) as input_file:
        N = int(input_file.readline())  # Read the number of cities from the first line
        cities = []                     # Initialize an empty list to store city coordinates
        for _ in range(N):
            # Read each city's coordinates and convert them to integers
            x, y, z = map(int, input_file.readline().split())
            cities.append((x, y, z))    # Append the city as a tuple to the cities list
        return cities, N                # Return the list of cities and the total number of cities

# Function to create the initial population of individuals (tours)
def create_initial_population(N):
    population = []  # Initialize an empty list for the population
    for each_individual in range(POPULATION_SIZE):
        # Create an individual as a random permutation of city indices
        individual = np.arange(N)       # Array of city indices from 0 to N-1
        np.random.shuffle(individual)   # Shuffle the city indices to create a random tour
        population.append(individual)   # Add the individual to the population list
    print(individual)
    return population                   # Return the list representing the population

# Function to calculate the Euclidean distance between two cities
def distance(city1, city2):
    # Compute the square root of the sum of squared differences in coordinates
    return np.sqrt(
        (city1[0] - city2[0]) ** 2 +    # Difference in x-coordinates squared
        (city1[1] - city2[1]) ** 2 +    # Difference in y-coordinates squared
        (city1[2] - city2[2]) ** 2      # Difference in z-coordinates squared
    )

# Function to calculate the total distance of a tour represented by an individual
def total_distance(individual, cities, N):
    dist = 0                            # Initialize total distance to zero
    for i in range(N):
        # Get the indices of the current city and the next city (with wrap-around)
        idx1 = individual[i]
        idx2 = individual[(i + 1) % N]
        # Get the coordinates of the current city and the next city
        city1 = cities[idx1]
        city2 = cities[idx2]
        # Add the distance between the current city and the next city
        dist += distance(city1, city2)
    return dist                         # Return the total distance of the tour

# Function to perform tournament selection
def tournament_selection(population, distances_of_individuals):
    tourny_champs = []                       # List to hold the selected individuals
    for _ in range(POPULATION_SIZE):
        # Randomly select 'TOURNAMENT_SIZE' individuals from the population
        participants = np.random.choice(POPULATION_SIZE, TOURNAMENT_SIZE, replace=False)
        # Find the participant with the best fitness (minimum distance)
        best_idx = min(participants, key=lambda idx: distances_of_individuals[idx])
        # Add the best individual to the selected list
        tourny_champs.append(population[best_idx])
    return tourny_champs                     # Return the list of selected individuals

# Function to perform two-point crossover between two parents
def two_point_crossover(parent1, parent2, N):
    # Randomly select two crossover points and sort them
    crossover_idxs = np.random.choice(N, 2, replace=False)
    start, end = sorted(crossover_idxs)
    child = -np.ones(N, dtype=int)   # Initialize the child with -1 placeholders
    child[start:end+1] = parent1[start:end+1]       # Copy a slice from parent1 into the child
    ptr = 0                             # Pointer for filling the rest of the child
    for city in parent2:
        if city not in child:
            # Find the next empty position in the child
            while child[ptr] != -1:
                ptr += 1
            child[ptr] = city           # Insert the gene from parent2 into the child
    return child                        # Return the new child individual

# Function to perform swap mutation on an individual
def swap_mutation(child, N):
    # Randomly select two positions in the individual to swap
    idx1, idx2 = np.random.choice(N, 2, replace=False)
    # Swap the cities at the selected positions
    child[idx1], child[idx2] = child[idx2], child[idx1]
    return child                   # Return the mutated individual

# Main function to perform the genetic algorithm
def genetic_algorithm(cities, N):
    population = create_initial_population(N)  # Create the initial population
    best_distance = float('inf')        # Initialize the best distance found so far
    best_individual = None              # Initialize the best individual

    # Loop over each generation
    for generation in range(MAX_GENERATIONS):
        # Evaluate the fitness of each individual in the population
        distances_of_individuals = [total_distance(individual, cities, N) for individual in population]
        # Find the individual with the best (lowest) fitness in the current generation
        current_best_distance = min(distances_of_individuals)
        current_best_individual = population[np.argmin(distances_of_individuals)]
        
        # Update the best solution found so far if current is better
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_individual = current_best_individual.copy()
        
        # Print progress every 10 generations
        if generation % 10 == 0:
            print(f"Generation {generation}: Best distance so far: {best_distance}")
        
        # Selection: select individuals to be parents for the next generation
        # population is now a population of champions that make it out of the tournament selection
        population = tournament_selection(population, distances_of_individuals)
        
        # Crossover: create a new population through crossover of selected parents
        crossover_offspring = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = population[i]
            # Ensure we don't go out of index range
            parent2 = population[(i+1) % POPULATION_SIZE]
            # Perform two-point crossover to produce two children
            child1 = two_point_crossover(parent1, parent2, N)
            crossover_offspring.append(child1)
            child2 = two_point_crossover(parent2, parent1, N)
            crossover_offspring.append(child2)
        
        # Mutation: apply mutation to the offspring
        mutated_offspring = []
        for child in crossover_offspring:
            if np.random.rand() < MUTATION_RATE:
                chernobyl_child = swap_mutation(child, N)
                mutated_offspring.append(chernobyl_child)
            else:
                mutated_offspring.append(child)
        
        # Local optimization is commented out for computational reasons
        # optimized_offspring = [local_optimization(child, cities) for child in mutated_offspring]
        # Update the population with the new generation
        # population = optimized_offspring

        # Update the population with the mutated offspring
        population = mutated_offspring

    # After all generations, prepare the best path for output
    # Get the coordinates of the cities in the best tour
    best_path = []
    for idx in best_individual:
        best_path.append(cities[idx])
    best_path.append(cities[best_individual[0]])
    
    # best_path_coordinates = [cities[idx] for idx in best_path_indices]
    # Append the starting city at the end to complete the tour
    # best_path_coordinates.append(cities[best_path_indices[0]])

    # Return the best distance and the best path
    return best_distance, best_path

# Function to write the output to 'output.txt'
def write_output(distance, path):
    with open('output.txt', 'w') as output_file:
        output_file.write(f"{distance}\n")        # Write the total distance on the first line
        for city in path:
            # Write each city's coordinates, separated by spaces
            output_file.write(f"{city[0]} {city[1]} {city[2]}\n")

# Main function to run the program
def main():
    print("Reading input...")
    # Read the cities from 'input_.txt'
    cities, N = read_input('input1.txt')
    print(f"Number of cities: {N}")
    print("Running algorithm...")
    # Run the genetic algorithm to find the best tour
    best_distance, best_path = genetic_algorithm(cities, N)
    print("Writing output...")
    # Write the best distance and path to 'output.txt'
    write_output(best_distance, best_path)
    print("Program finished successfully.")

# Entry point of the program
if __name__ == "__main__":
    try:
        main()  # Call the main function to execute the program
    except Exception as e:
        # Print any error that occurs during execution
        print(f"An error occurred: {e}")


        
        
        
        
        
# 1 -- https://arxiv.org/pdf/2303.00614
# 2 -- https://arxiv.org/pdf/1812.09351
# 3 -- https://www.researchgate.net/publication/283231968_Optimal_path_planning_for_UAVs_using_Genetic_Algorithm