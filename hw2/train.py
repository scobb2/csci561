# train.py
import os
import subprocess
from agent import QLearningAgent
from read import readInput
from write import writeNextInput
from host import GO
import math

def train(num_games, visualize=False, visualize_interval=1000):
    N = 5
    epsilon_start = 1
    epsilon_min = 0.1
    decay_rate = 0.0025  # Adjust this value based on the total number of games

    agent_black = QLearningAgent(1, epsilon=epsilon_start)
    agent_white = QLearningAgent(2, epsilon=epsilon_start)
    agents = {1: agent_black, 2: agent_white}
    
    win_counts = {1: 0, 2: 0, 0: 0}

    for game_num in range(num_games):
        # Decay epsilon
        epsilon = epsilon_min + (epsilon_start - epsilon_min) * math.exp(-decay_rate * game_num)
        agent_black.epsilon = epsilon
        agent_white.epsilon = epsilon

        # Initialize the game
        go = GO(N)
        go.init_board(N)
        previous_board = [[0]*N for _ in range(N)]
        board = [[0]*N for _ in range(N)]
        piece_type = 1  # Black starts first
        done = False
        move_number = 0

        # For visualization and logging
        game_moves = []

        while not done:
            move_number += 1

            # Prepare input.txt
            writeNextInput(piece_type, previous_board, board)

            # Agent makes a move
            agent = agents[piece_type]
            agent.play(previous_board, board)

            # Read the move from output.txt for logging/visualization
            try:
                with open('output.txt', 'r') as f:
                    move_str = f.read().strip()
            except FileNotFoundError:
                move_str = 'PASS'

            # Parse the move
            if move_str == 'PASS':
                move = 'PASS'
                current_move = None
                captured_pieces = []
                go.pass_move(piece_type)
            else:
                i, j = map(int, move_str.split(','))
                current_move = (i, j)
                move = (i, j)

                # Check validity and update the game state
                valid = go.place_chess(i, j, piece_type)
                if not valid:
                    # Invalid move; the agent loses
                    winner = 3 - piece_type
                    done = True
                    # Provide reward
                    agents[piece_type].learn(agent.previous_board, board, -1, True)
                    agents[3 - piece_type].learn(agent.previous_board, board, 1, True)
                    break
                else:
                    # Remove opponent's dead stones
                    captured_pieces = go.remove_died_pieces(3 - piece_type)

            # For visualization and logging
            if visualize and (game_num + 1) % visualize_interval == 0:
                print(f"Game {game_num + 1}, Move {move_number}: Player {'X' if agent.piece_type == 1 else 'O'} plays {move_str}")
                # Visualize the board after the move and captures
                go.visualize_board(board=go.board, current_move=current_move, captured_pieces=captured_pieces)
                print()

                # Append move to game log
                game_moves.append((agent.piece_type, move_str))

            # Check if the game has ended
            if go.game_end(piece_type, 'PASS' if move_str == 'PASS' else 'MOVE'):
                winner = go.judge_winner()
                done = True
                # Provide rewards
                for pt in [1, 2]:
                    agent = agents[pt]
                    reward = agent.get_reward(go, agent.previous_board, go.board, True, move_str)
                    agent.learn(agent.previous_board, go.board, reward, True)
                break

            # Agents learn from the result
            reward = agent.get_reward(go, agent.previous_board, go.board, False, move_str)
            agent.learn(agent.previous_board, go.board, reward, False)

            # Update agent's previous board
            agent.previous_board = [row[:] for row in previous_board]
            agent.board = [row[:] for row in board]

            # Update the previous and current board
            previous_board = [row[:] for row in go.previous_board]
            board = [row[:] for row in go.board]

            # Switch players
            piece_type = 3 - piece_type

        # Save the game log if visualized
      #   if visualize and (game_num + 1) % visualize_interval == 0:
      #       log_filename = f"game_{game_num + 1}_log.txt"
      #       with open(log_filename, 'w') as log_file:
      #           for move in game_moves:
      #               log_file.write(f"Player {'X' if move[0] == 1 else 'O'}: {move[1]}\n")
      #       print(f"Game {game_num + 1} log saved to {log_filename}")

        if winner == 1:
            win_counts[1] += 1
        elif winner == 2:
            win_counts[2] += 1
        else:
            win_counts[0] += 1

         # Periodically print win rates:
        if (game_num + 1) % 10 == 0:
            total_games = sum(win_counts.values())
            print(f"Win rates after {total_games} games:")
            print(f"Player 1 (Black) wins: {win_counts[1]}")
            print(f"Player 2 (White) wins: {win_counts[2]}")
            print(f"Ties: {win_counts[0]}")

        # Clean up files
        if os.path.exists('input.txt'):
            os.remove('input.txt')
        if os.path.exists('output.txt'):
            os.remove('output.txt')

        if (game_num + 1) % visualize_interval == 0:
            print(f"Game {game_num + 1}/{num_games} completed. Epsilon: {epsilon:.4f}")
        else:
            print(f"Game {game_num + 1}/{num_games} completed.", end='\r')

if __name__ == "__main__":
    num_games = 1000  # Adjust the number of games as needed
    train(num_games, visualize=True, visualize_interval=100)
