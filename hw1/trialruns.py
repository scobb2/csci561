import itertools
import subprocess
import time

# Define your parameter values
population_sizes = [60, 100]  # Example values, adjust as needed
max_generations = [500, 1000]  # Example values
mutation_rates = [0.1]  # Example values
tournament_sizes = [3, 5]  # Example values
tournament_finalists = [1]  # Example values
selection_rates = [0.6]  # Example values
crossover_rates = [0.8, 0.9]  # Example values
input_files = ['input1.txt', 'input2.txt', 'input3.txt', 'input4.txt']

# Define how many times to run each permutation to account for randomness
runs_per_permutation = 3

# Create all permutations
all_permutations = list(itertools.product(
    population_sizes,
    max_generations,
    mutation_rates,
    tournament_sizes,
    tournament_finalists,
    selection_rates,
    crossover_rates,
    input_files
))

print(f"Total permutations to run: {len(all_permutations) * runs_per_permutation}")

# Path to your main program
program_path = "homework1v5.py"  # Replace with your actual program's path

# Loop through each permutation
for idx, (pop_size, max_gen, mut_rate, tour_size, tour_finalists, sel_rate, cross_rate, input_file) in enumerate(all_permutations):
    for run in range(runs_per_permutation):
        print(f"Running permutation {idx + 1}/{len(all_permutations)}, run {run + 1}/{runs_per_permutation}")

        # Call your program with the parameters
        try:
            subprocess.run([
                "python", program_path,
                "--population_size", str(pop_size),
                "--max_generations", str(max_gen),
                "--mutation_rate", str(mut_rate),
                "--tournament_size", str(tour_size),
                "--tournament_finalists", str(tour_finalists),
                "--selection_rate", str(sel_rate),
                "--crossover_rate", str(cross_rate),
                "--input_file", input_file
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during execution: {e}")

        # Wait for 60 seconds between each run
        time.sleep(15)

print("All permutations completed.")