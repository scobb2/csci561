# agent.py
import json
import random
import os
from read import readInput
from write import writeOutput
from host import GO

class QLearningAgent:
    def __init__(self, piece_type, epsilon=1.0):
        self.piece_type = piece_type
        self.epsilon = epsilon  # Exploration rate
        self.alpha = 0.01        # Learning rate
        self.gamma = 0.8        # Discount factor
        self.N = 5  # Board size
        self.num_features = 12  # Adjusted number of features
        self.weights = self.load_weights()
        self.previous_state = None
        self.previous_action = None
        self.previous_features = None
        self.previous_board = None
        self.board = None

    def get_weights_filename(self):
        if self.piece_type == 1:
            return 'weights_black.json'
        else:
            return 'weights_white.json'

    def load_weights(self):
        filename = self.get_weights_filename()
        try:
            with open(filename, 'r') as f:
                weights = json.load(f)
                return weights
        except FileNotFoundError:
            # Initialize weights to zero
            return [0.0 for _ in range(self.num_features)]

    def save_weights(self):
        filename = self.get_weights_filename()
        with open(filename, 'w') as f:
            json.dump(self.weights, f)

    def get_valid_actions(self, go, piece_type):
        actions = []
        for i in range(self.N):
            for j in range(self.N):
                if go.board[i][j] == 0:
                    if go.valid_place_check(i, j, piece_type, test_check=True):
                        actions.append((i, j))
        actions.append('PASS')
        return actions

    def extract_features_from_board(self, board):
        go = GO(self.N)
        go.board = [row[:] for row in board]
        go.previous_board = [row[:] for row in board]
        
        # Existing features
        own_stones = sum(row.count(self.piece_type) for row in board) / (self.N * self.N)
        opponent_stones = sum(row.count(3 - self.piece_type) for row in board) / (self.N * self.N)
        
        # Additional features
        total_stones = own_stones + opponent_stones
        board_control = (own_stones - opponent_stones)
        mobility = len(self.get_valid_actions(go, self.piece_type)) / (self.N * self.N)
        
        # Center control (assuming N is odd)
        center = self.N // 2
        center_control = 1 if board[center][center] == self.piece_type else 0

        # New features
        own_groups_in_atari = self.count_groups_in_atari(go, self.piece_type)
        opponent_groups_in_atari = self.count_groups_in_atari(go, 3 - self.piece_type)
        own_avg_liberties = self.calculate_average_liberties(go, self.piece_type)
        opponent_avg_liberties = self.calculate_average_liberties(go, 3 - self.piece_type)
        own_connected_groups = self.count_connected_groups(go, self.piece_type)
        opponent_connected_groups = self.count_connected_groups(go, 3 - self.piece_type)
        capture_opportunities = self.count_capture_opportunities(go)
        own_stones_threatened = self.count_threatened_stones(go, self.piece_type)
        
        features = [
            own_stones,
            opponent_stones,
            board_control,
            mobility,
            center_control,
            own_groups_in_atari / self.N,  # Normalized
            opponent_groups_in_atari / self.N,
            own_avg_liberties / 4,  # Max liberties per group
            opponent_avg_liberties / 4,
            own_connected_groups / (self.N * self.N),
            opponent_connected_groups / (self.N * self.N),
            capture_opportunities / (self.N * self.N),
            own_stones_threatened / (self.N * self.N)
        ]
        return features

    def get_state(self, go):
        # Use the board to extract features
        features = self.extract_features_from_board(go.board)
        return tuple(features)

    def q_value(self, features):
        return sum(w * f for w, f in zip(self.weights, features))

    def choose_action(self, go):
        valid_actions = self.get_valid_actions(go, self.piece_type)
        if random.uniform(0, 1) < self.epsilon:
            # Exploration
            action = random.choice(valid_actions)
        else:
            # Exploitation
            q_values = []
            for action in valid_actions:
                # Simulate the action
                go_copy = go.copy_board()
                if action != 'PASS':
                    valid = go_copy.place_chess(action[0], action[1], self.piece_type)
                    if not valid:
                        q_values.append(float('-inf'))
                        continue
                    go_copy.remove_died_pieces(3 - self.piece_type)
                else:
                    go_copy.pass_move(self.piece_type)
                features = self.get_state(go_copy)
                q = self.q_value(features)
                q_values.append(q)
            max_q = max(q_values)
            max_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
            action = random.choice(max_actions)
        return action

    def count_total_liberties(self, go, piece_type):
        visited = set()
        total_liberties = 0
        for i in range(go.size):
            for j in range(go.size):
                if go.board[i][j] == piece_type and (i, j) not in visited:
                    group = go.ally_dfs(i, j)
                    visited.update(group)
                    liberties = go.count_liberty(group)
                    total_liberties += liberties
        return total_liberties

    def calculate_territory(self, go, piece_type):
        territory = 0
        for i in range(go.size):
            for j in range(go.size):
                if go.board[i][j] == 0:
                    owner = go.check_territory_owner(i, j)
                    if owner == piece_type:
                        territory += 1
        return territory
    
    def count_groups_in_atari(self, go, piece_type):
        groups = go.get_groups(piece_type)
        atari_groups = 0
        for group in groups:
            liberties = go.count_liberty(group)
            if liberties == 1:
                atari_groups += 1
        return atari_groups

    def calculate_average_liberties(self, go, piece_type):
        groups = go.get_groups(piece_type)
        total_liberties = 0
        for group in groups:
            total_liberties += go.count_liberty(group)
        if len(groups) > 0:
            return total_liberties / len(groups)
        else:
            return 0

    def count_connected_groups(self, go, piece_type):
        groups = go.get_groups(piece_type)
        return len(groups)

    def count_capture_opportunities(self, go):
        # Count opponent stones that can be captured in one move
        capture_opportunities = 0
        for i in range(self.N):
            for j in range(self.N):
                if go.board[i][j] == 3 - self.piece_type:
                    if go.is_in_atari(i, j):
                        capture_opportunities += 1
        return capture_opportunities

    def count_threatened_stones(self, go, piece_type):
        # Count own stones that can be captured in one move
        threatened_stones = 0
        for i in range(self.N):
            for j in range(self.N):
                if go.board[i][j] == piece_type:
                    if go.is_in_atari(i, j):
                        threatened_stones += 1
        return threatened_stones

    def get_reward(self, go, previous_board, board, done, action):
        if done:
            winner = go.judge_winner()
            if winner == self.piece_type:
                return 1  #  reward for winning
            elif winner == 0:
                return 0  # Neutral reward for a tie
            else:
                return -1  #  penalty for losing
        else:
            # Intermediate rewards
            reward = 0.0
            # Calculate territory control difference
            own_territory = self.calculate_territory(go, self.piece_type)
            opponent_territory = self.calculate_territory(go, 3 - self.piece_type)
            territory_diff = own_territory - opponent_territory

            # Calculate captures
            prev_opponent_stones = sum(row.count(3 - self.piece_type) for row in previous_board)
            current_opponent_stones = sum(row.count(3 - self.piece_type) for row in board)
            opponent_stones_captured = prev_opponent_stones - current_opponent_stones

            # Update reward
            reward = territory_diff * opponent_stones_captured  # Weight captures more
            reward = max(min(reward, 1.0), -1.0)
            
            total_stones = own_territory + opponent_territory
            if action == 'PASS' and total_stones < 20:
                reward -= 0.5  # Adjust penalty value as needed

            return reward

    def update_weights(self, features, target):
        prediction = self.q_value(features)
        error = target - prediction
        lambda_reg = 0.01  # Regularization strength
        for i in range(len(self.weights)):
            self.weights[i] += self.alpha * (error * features[i] - lambda_reg * self.weights[i])
        self.save_weights()

    def learn(self, previous_board, board, reward, done):
        if self.previous_state is not None and self.previous_action is not None:
            # Create GO instance for the new state
            go_new = GO(self.N)
            go_new.set_board(self.piece_type, previous_board, board)
            new_features = self.get_state(go_new)

            # Calculate target
            if done:
                target = reward
            else:
                # Estimate future rewards
                future_q = self.q_value(new_features)
                target = reward + self.gamma * future_q

            # Update weights
            self.update_weights(self.previous_features, target)

            if done:
                self.previous_state = None
                self.previous_action = None
                self.previous_features = None
            else:
                self.previous_state = new_features
                self.previous_features = new_features
        else:
            # First move, no update needed
            self.previous_state = self.get_state_from_board(previous_board)
            self.previous_features = self.previous_state

    def play(self, previous_board, board):
        # Create a GO instance to check move validity
        go = GO(self.N)
        go.set_board(self.piece_type, previous_board, board)

        # Set agent's previous and current boards
        self.previous_board = [row[:] for row in previous_board]
        self.board = [row[:] for row in board]

        # Get the current state features
        state_features = self.get_state(go)

        # Choose an action
        action = self.choose_action(go)

        # Write the action to output.txt
        if action == 'PASS':
            writeOutput('PASS')
        else:
            writeOutput(action)

        # Save the current features and action for the next update
        self.previous_state = state_features
        self.previous_action = action
        self.previous_features = state_features

    def get_state_from_board(self, board):
        go = GO(self.N)
        go.board = [row[:] for row in board]
        return self.get_state(go)

def main():
    N = 5
    piece_type, previous_board, board = readInput(N)
    agent = QLearningAgent(piece_type)
    agent.play(previous_board, board)

if __name__ == "__main__":
    main()
