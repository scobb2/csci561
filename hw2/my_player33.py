import random
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import subprocess
from read import readInput
from write import writeOutput, writePass
from host import GO

# Q-Learning Parameters
ALPHA = 0.1       # Learning rate
GAMMA = 0.9       # Discount factor
EPSILON = 0.2     # Exploration rate
BOARD_SIZE = 5    # Size of the Go board
WIN_REWARD = 10   # Reward for winning the game
LOSS_PENALTY = -10 # Penalty for losing the game
STEP_REWARD = -0.1 # Small penalty for each step to encourage shorter games
PASS_PENALTY = -0.5 # Penalty for passing to prevent overuse

# Q-Table to store state-action values
# The key is a string representation of the board, the value is a list of action values
Q_TABLE = {}

# Load Q-table if it exists
try:
    with open("q_table.json", "r") as f:
        Q_TABLE = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    Q_TABLE = {}


def state_to_string(board):
    """
    Convert the board state to a string to use as a key in the Q-table.
    """
    return json.dumps(board)


def get_valid_actions(board):
    """
    Get all valid actions for the current board state.
    """
    valid_actions = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == 0:  # Empty spot
                # Check if placing a stone here violates the liberty rule
                if has_liberty(board, i, j):
                    valid_actions.append((i, j))
    if not valid_actions:
        valid_actions.append("PASS")  # Add the option to pass if no valid move is available
    return valid_actions


def has_liberty(board, x, y):
    """
    Check if placing a stone at (x, y) results in liberties for the stone or group.
    """
    # Simulate placing the stone
    board[x][y] = 1 if piece_type == 1 else 2
    visited = set()
    has_lib = dfs_check_liberty(board, x, y, visited)
    # Revert the move
    board[x][y] = 0
    return has_lib


def dfs_check_liberty(board, x, y, visited):
    """
    Use DFS to check if a stone or group has at least one liberty.
    """
    if (x, y) in visited:
        return False
    visited.add((x, y))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            if board[nx][ny] == 0:
                return True
            if board[nx][ny] == board[x][y] and dfs_check_liberty(board, nx, ny, visited):
                return True
    return False


def choose_action(state, valid_actions):
    """
    Choose the next action using an epsilon-greedy strategy.
    """
    if random.uniform(0, 1) < EPSILON:
        # Explore: choose a random valid action
        return random.choice(valid_actions)
    else:
        # Exploit: choose the best action from the Q-table
        if state not in Q_TABLE:
            Q_TABLE[state] = [0] * (BOARD_SIZE * BOARD_SIZE + 1)  # +1 for the PASS action
        
        # Find the action with the highest Q-value
        max_q_value = -float('inf')
        best_action = None
        for action in valid_actions:
            if action == "PASS":
                index = BOARD_SIZE * BOARD_SIZE  # Index for PASS action
            else:
                index = action[0] * BOARD_SIZE + action[1]
            if Q_TABLE[state][index] > max_q_value:
                max_q_value = Q_TABLE[state][index]
                best_action = action
        return best_action if best_action is not None else "PASS"


def update_q_value(previous_state, action, reward, next_state):
    """
    Update the Q-value for the given state and action.
    """
    if previous_state not in Q_TABLE:
        Q_TABLE[previous_state] = [0] * (BOARD_SIZE * BOARD_SIZE + 1)
    if next_state not in Q_TABLE:
        Q_TABLE[next_state] = [0] * (BOARD_SIZE * BOARD_SIZE + 1)
    
    if action == "PASS":
        action_index = BOARD_SIZE * BOARD_SIZE  # Index for PASS action
    else:
        action_index = action[0] * BOARD_SIZE + action[1]
    
    max_next_q = max(Q_TABLE[next_state])
    Q_TABLE[previous_state][action_index] = (1 - ALPHA) * Q_TABLE[previous_state][action_index] + \
                                            ALPHA * (reward + GAMMA * max_next_q)


def check_ko(previous_board, current_board):
    """
    Check for the KO rule.
    """
    return previous_board == current_board


def check_game_end(board):
    """
    Check if the game has ended based on board state.
    Returns True if the game is over, otherwise False.
    """
    # In this simplified version, the game ends when the board is full
    for row in board:
        if 0 in row:
            return False
    return True


def calculate_score(board, piece_type):
    """
    Calculate the score for the given piece type.
    """
    score = 0
    for row in board:
        score += row.count(piece_type)
    return score


def visualize_board(board):
    """
    Visualize the current state of the board using matplotlib.
    """
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(BOARD_SIZE+1)-0.5, minor=True)
    ax.set_yticks(np.arange(BOARD_SIZE+1)-0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    ax.imshow([[1 if cell == 1 else 0.5 if cell == 2 else 0 for cell in row] for row in board], cmap="gray_r")
    plt.pause(0.5)
    plt.clf()


def agent(training_mode=False):
    """
    Main agent function to read input, choose an action, visualize the game, and write output.
    If training_mode is True, the agent will not visualize the board to speed up training.
    """
    global piece_type
    piece_type, previous_board, current_board = readInput(BOARD_SIZE)
    current_state = state_to_string(current_board)
    valid_actions = get_valid_actions(current_board)
    action = choose_action(current_state, valid_actions)

    # Ensure that action is valid and properly formatted
    if action == "PASS":
        writePass()
        print("Agent chooses to PASS.")  # Debugging statement
    elif isinstance(action, tuple) and len(action) == 2:
        x, y = action
        if current_board[x][y] == 0:  # Ensure the chosen position is empty
            current_board[x][y] = piece_type
            writeOutput(f"{x},{y}")
            print(f"Agent places at position: {x},{y}")  # Debugging statement
        else:
            print("Invalid position chosen, defaulting to PASS.")  # Debugging statement
            writePass()  # Default to pass if the chosen position is invalid
    else:
        print("No valid action found, defaulting to PASS.")  # Debugging statement
        writePass()  # Default to pass if no valid action

    # Determine reward (this is simplistic and may need adjustment)
    reward = STEP_REWARD
    if action == "PASS":
        reward += PASS_PENALTY
    else:
        # Check if the game ended after the move (simplified)
        if check_game_end(current_board):
            black_score = calculate_score(current_board, 1)
            white_score = calculate_score(current_board, 2) + 2.5  # Komi for white
            if (piece_type == 1 and black_score > white_score) or (piece_type == 2 and white_score > black_score):
                reward = WIN_REWARD
            else:
                reward = LOSS_PENALTY
        elif check_ko(previous_board, current_board):
            reward = LOSS_PENALTY  # Penalize for repeating the same board state (KO rule)

    # Update Q-table
    previous_state = state_to_string(previous_board)
    update_q_value(previous_state, action, reward, current_state)

    # Save Q-table
    with open("q_table.json", "w") as f:
        json.dump(Q_TABLE, f, indent=4)

    # Show final board state if not in training mode
    if not training_mode:
        visualize_board(current_board)


def self_play(num_games):
    """
    Automate self-play for training the Q-learning agent.
    """
    for game in range(num_games):
        go_host = GO(BOARD_SIZE)
        go_host.init_board(BOARD_SIZE)
        move_number = 0

        while not check_game_end(go_host.board):
            agent(training_mode=True)
            # Execute the game using host.py to validate moves and update the board
            result = subprocess.run(["python3", "host.py", "--move", str(move_number), "--verbose", "False"], capture_output=True)
            output = result.stdout.decode()
            error_output = result.stderr.decode()

            # Debugging output to understand what's happening with the subprocess
            print(f"Game {game + 1}, Move {move_number + 1} Output:\n{output}")
            if error_output:
                print(f"Error Output:\n{error_output}")

            if "Game end" in output:
                print(f"Game {game + 1} ended after {move_number + 1} moves:\n{output}")
                break
            move_number
