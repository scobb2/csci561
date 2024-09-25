# use python version 3.7.5
import random  # Import the random module for generating random numbers and selections
import math    # Import the math module for mathematical functions like sqrt

# Function to read input from 'input.txt'
def read_input(file_name):
    with open(file_name) as input_file:
        N = int(input_file.readline())  # Read the number of cities N from the first line
        cities = []            # Initialize an empty list to store city coordinates
        for city in range(N):
            # Read each city's coordinates and split them into x, y, z integers
            x, y, z = map(int, input_file.readline().split())
            cities.append((x, y, z))  # Add the city as a tuple to the cities list
    return cities, N  # Return the list of city coordinates

# Function to calculate Euclidean distance between two cities
def distance(city1, city2):
    # Compute the Euclidean distance using the square root of the sum of squared differences
    return math.sqrt(
        (city1[0] - city2[0]) ** 2 +  # Difference in x-coordinates squared
        (city1[1] - city2[1]) ** 2 +  # Difference in y-coordinates squared
        (city1[2] - city2[2]) ** 2    # Difference in z-coordinates squared
    )

# Function to calculate the total distance of a path (tour)
def total_distance(path):
    dist = 0  # Initialize total distance to zero
    for i in range(len(path)):
        # Add the distance between the current city and the next city (wrap around using modulus)
        dist += distance(path[i], path[(i + 1) % len(path)])
    # % accounts for distance from final city back to starting city
    return dist  # Return the total distance of the path

# Function to create the initial population of random tours
def createInitialPopulation(cities, population_size):
    population = []  # Initialize an empty list for the population
    for each_individual in range(population_size):
        individual = cities[:]      # Make a copy of the cities list for each individual
        random.shuffle(individual)  # Shuffle the cities to create a random tour
        population.append(individual)  # Add the individual to the population list
    return population  # Return the list representing the population

# Function for selection using tournament selection method
def parentSelection(population):
    selected = []  # Initialize an empty list for selected individuals
    for _ in range(len(population)):
        # Randomly select two individuals (tours) for the tournament
        i, j = random.sample(range(len(population)), 2)
        # Compare their total distances to select the better one
        if total_distance(population[i]) < total_distance(population[j]):
            selected.append(population[i])  # Add the better individual to the selected list
        else:
            selected.append(population[j])
    return selected  # Return the list of selected individuals

# Function to build an edge map for Edge Recombination Crossover (ERX)
def build_edge_map(parent1, parent2):
    edge_map = {}  # Initialize an empty dictionary for the edge map
    # Combine both parents to build the edge map
    for parent in [parent1, parent2]:
        for i in range(len(parent)):
            city = parent[i]  # Get the current city
            left = parent[i - 1]  # Get the city to the left (previous city)
            right = parent[(i + 1) % len(parent)]  # Get the city to the right (next city)
            if city not in edge_map:
                edge_map[city] = set()  # Initialize an empty set for the city if not already present
            edge_map[city].update([left, right])  # Add neighboring cities to the edge map
    return edge_map  # Return the completed edge map

# Function for Edge Recombination Crossover (ERX)
def erx_crossover(parent1, parent2):
    edge_map = build_edge_map(parent1, parent2)  # Build the edge map from both parents
    child = []  # Initialize an empty list for the child tour
    current_city = random.choice(parent1)  # Randomly select a starting city from parent1
    while len(child) < len(parent1):
        child.append(current_city)  # Add the current city to the child tour
        for edges in edge_map.values():
            edges.discard(current_city)  # Remove the current city from all edge lists
        if edge_map[current_city]:
            # Choose the next city with the fewest edges (ties broken randomly)
            next_city = min(edge_map[current_city], key=lambda city: len(edge_map[city]))
        else:
            # If no edges left, select a random unvisited city
            unvisited = set(parent1) - set(child)
            if unvisited:
                next_city = random.choice(list(unvisited))
            else:
                break  # Break if all cities have been visited
        del edge_map[current_city]  # Remove the current city from the edge map
        current_city = next_city  # Move to the next city
    return child  # Return the child tour generated

# Function for Inversion Mutation
def invert_mutation(individual, mutation_rate):
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(len(individual)), 2))  # Randomly select two indices
        individual[i:j] = individual[i:j][::-1]  # Reverse the sublist between the two indices
        
# 1 - Element-Wise Modification:  This operator involves the random modification of individual elements
# within a chromosome. Specifically, for each element in the chromosome, one of the following actions
# is randomly chosen: sign change, element swap with the successor, or no action. The selection of
# actions is governed by random numbers in the range [0, 1]. If the randomly generated number falls
# within the range [0, 0.1], a sign change is applied to the element. In the range (0.1, 0.2], a swap
# with the succeeding element takes place. For numbers beyond 0.2, no action is executed, leaving
# the element unaltered (Lines 8-11).

# 1 - Sequence Modification: This operator introduces variability by modifying subsequences within the
# chromosome. Two indices, i and j, are randomly selected, and one of the following operations is
# performed on the sequence between i and j (inclusive): reverse sequence, sign change within the
# sequence, or shuffle sequence. The choice of operation is determined randomly (Lines 13,14).
# It is imperative to iterate through the aforementioned process until each subpopulation attains a
# population size of µ.

# 1 - Tour Mutation: In this mutation scheme, 20 percent of the indices in the chromosome are randomly
# selected, and the elements at these positions are shuffled. This operation is analogous to the
# Scramble Mutation

def diversification():
    return
# 1 - Diversification. When no improvement is achieved after ItDIV iterations, we employ a plan to enhance
# the population’s genetic value. From each subpopulation, nBEST individuals will be retained based on
# fitness, while the rest will be discarded. The size of each subpopulation will then reach µ using the same
# strategy as the initial population algorithm.

# 2 - The main role of the education step is to improve the quality of solutions
# by means of the local search procedure. We design a hill-climbing and firstimprovement local search for both min-cost and min-time objectives. Similar
# to [10], we also apply the technique proposed in [16] to restrict the search to
# h × n closest vertices, where h = 0.1 is the granular threshold. This technique
# allows to reduce significantly the computation time consumed by the education
# process...... Moreover, the truck
# and drone cumulative time and cost as well as the cost and time of all drone
# tuples in set P are pre-computed at the beginning of the HGA to effectively
# accelerate the algorithm.


# Main function for the Genetic Algorithm
def genetic_algorithm(cities, mutation_rate = 0.2, population_size=30, generations=300):
    population = createInitialPopulation(cities, population_size)  # Create the initial population
    best_distance = float('inf')  # Initialize the best distance found to infinity
    best_individual = None  # Initialize the best individual (tour) found

    for gen in range(generations):  # Iterate over the specified number of generations
        population = parentSelection(population)  # Perform selection to get the mating pool
        next_generation = []  # Initialize the next generation list

        for i in range(0, len(population), 2):  # Iterate over the population in steps of two
            parent1 = population[i]  # Select the first parent
            parent2 = population[(i+1)%len(population)]  # Select the second parent
            child1 = erx_crossover(parent1, parent2)  # Perform ERX crossover to produce the first child
            child2 = erx_crossover(parent2, parent1)  # Perform ERX crossover to produce the second child
            invert_mutation(child1, mutation_rate)  # Apply inversion mutation to the first child
            invert_mutation(child2, mutation_rate)  # Apply inversion mutation to the second child
            next_generation.extend([child1, child2])  # Add both children to the next generation list

        population = next_generation[:population_size]  # Update the population with the new generation
        current_best = min(population, key=total_distance)  # Find the best individual in the current population
        current_distance = total_distance(current_best)  # Calculate the distance of the best individual

        if current_distance < best_distance:  # If the current best is better than the best found so far
            best_distance = current_distance  # Update the best distance
            best_individual = current_best  # Update the best individual (tour)
            
        # Print progress every 10 generations
        if gen % 10 == 0:
            print(f"Generation {gen}: Best distance so far: {best_distance}")

    return best_distance, best_individual  # Return the best distance and the best tour found

# Function to write the output as per the specified format
def write_output(distance, path):
    with open('output.txt', 'w') as f:
        f.write(f"{distance}\n")  # Write the total computed distance on the first line
        for city in path:  # Iterate over each city in the path
            # Write the coordinates of the city, separated by spaces
            f.write(f"{city[0]} {city[1]} {city[2]}\n")
        start_city = path[0]  # Get the starting city to return to
        # Write the coordinates of the starting city to complete the tour
        f.write(f"{start_city[0]} {start_city[1]} {start_city[2]}\n")

def main():
    print("Reading input...")
    cities, N = read_input('input1.txt')
    print(f"Number of cities: {len(cities)}")
    print("Running genetic algorithm...")
    best_distance, best_path = genetic_algorithm(cities) # (cities, 1/N) -- 1/N for mutation_rate
    print("Writing output...")
    write_output(best_distance, best_path)
    print("Program finished successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
        
        
        
# 1 -- https://arxiv.org/pdf/2303.00614
# 2 -- https://arxiv.org/pdf/1812.09351