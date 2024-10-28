import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from read import readInput
from write import writeOutput
from host import GO
import math
import random
from collections import deque
import copy

class PolicyValueNet(nn.Module):
    def __init__(self, board_size):
        super(PolicyValueNet, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)  # 4 input channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * board_size * board_size, 256)
        self.fc_p = nn.Linear(256, board_size * board_size)
        self.fc_v = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # x shape: (batch_size, 4, board_size, board_size)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * self.board_size * self.board_size)
        x = F.relu(self.fc1(x))
        policy_logits = self.fc_p(x)
        value = torch.tanh(self.fc_v(x))
        return policy_logits, value

class MCTSNode:
    def __init__(self, state, parent=None, prior_p=1.0):
        self.state = state  # Game state
        self.parent = parent  # Parent node
        self.children = {}  # Action -> child node
        self.N = 0  # Visit count
        self.W = 0  # Total action-value
        self.Q = 0  # Mean action-value
        self.P = prior_p  # Prior probability

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self.children:
                next_state = self.state.copy_board()
                piece_type = 1 if next_state.X_move else 2
                if action != "PASS":
                    next_state.place_chess(action[0], action[1], piece_type)
                    next_state.died_pieces = next_state.remove_died_pieces(3 - piece_type)
                next_state.X_move = not next_state.X_move
                self.children[action] = MCTSNode(next_state, parent=self, prior_p=prob)

    def select(self, c_puct):
        return max(
            self.children.items(),
            key=lambda act_node: act_node[1].get_value(c_puct)
        )

    def update(self, leaf_value):
        self.N += 1
        self.W += leaf_value
        self.Q = self.W / self.N

    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        u = c_puct * self.P * math.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + u

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, mcts_prob, value):
        self.buffer.append((state, mcts_prob, value))

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs = [data[1] for data in mini_batch]
        value_batch = [data[2] for data in mini_batch]
        return state_batch, mcts_probs, value_batch

    def __len__(self):
        return len(self.buffer)

class AlphaZeroAgent:
    def __init__(self, num_simulations=50, c_puct=1.0, buffer_size=10000, batch_size=64, lr=1e-3):
        self.type = 'alphazero'
        self.N = 5  # Board size
        self.model = PolicyValueNet(self.N)
        self.num_simulations = num_simulations  # Number of MCTS simulations
        self.c_puct = c_puct  # Exploration constant
        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_input(self, go, piece_type):
        # Use MCTS to get the best move
        action, _ = self.mcts_search(go, piece_type, temp=1e-3)
        return action

    def mcts_search(self, go, piece_type, temp=1.0):
        # Initialize the root node
        root_state = go.copy_board()
        root_state.X_move = (piece_type == 1)
        root_node = MCTSNode(root_state)

        for _ in range(self.num_simulations):
            node = root_node
            state = root_state.copy_board()
            states = []

            # Selection
            while not node.is_leaf():
                action, node = node.select(self.c_puct)
                if action != "PASS":
                    state.place_chess(action[0], action[1], piece_type)
                    state.died_pieces = state.remove_died_pieces(3 - piece_type)
                state.X_move = not state.X_move
                piece_type = 1 if state.X_move else 2

            # Expansion and Evaluation
            if not state.game_end(piece_type):
                action_probs, leaf_value = self.evaluate(state, piece_type)
                node.expand(action_probs)
            else:
                # If it's a terminal node
                winner = state.judge_winner()
                if winner == piece_type:
                    leaf_value = 1.0
                elif winner == 0:
                    leaf_value = 0.0
                else:
                    leaf_value = -1.0

            # Backpropagation
            node.update_recursive(-leaf_value)

        # Get action probabilities
        actions_visits = [(action, child.N) for action, child in root_node.children.items()]
        actions, visits = zip(*actions_visits)
        visits = np.array(visits, dtype=np.float32)
        probs = visits / np.sum(visits)

        # Select action based on temperature
        if temp == 0:
            best_action = actions[np.argmax(probs)]
            action_probs = np.zeros(len(probs))
            action_probs[np.argmax(probs)] = 1.0
        else:
            probs = probs ** (1 / temp)
            probs /= np.sum(probs)
            action_idx = np.random.choice(len(actions), p=probs)
            best_action = actions[action_idx]
            action_probs = probs

        return best_action, [(a, p) for a, p in zip(actions, action_probs)]

    def evaluate(self, state, piece_type):
        # Convert the board to tensor
        board_tensor = self.board_to_tensor(state, piece_type)
        board_tensor = board_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        with torch.no_grad():
            policy_logits, value = self.model(board_tensor)
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = value.item()

        # Get valid moves
        valid_moves = []
        for i in range(state.size):
            for j in range(state.size):
                if state.valid_place_check(i, j, piece_type, test_check=True):
                    valid_moves.append((i, j))

        action_probs = []
        idx = 0
        for i in range(state.size):
            for j in range(state.size):
                if (i, j) in valid_moves:
                    action_probs.append(((i, j), policy[idx]))
                idx += 1

        if not action_probs:
            action_probs = [("PASS", 1.0)]
        else:
            # Normalize probabilities
            total_prob = sum([prob for _, prob in action_probs])
            action_probs = [(act, prob / total_prob) for act, prob in action_probs]

        return action_probs, value

    def board_to_tensor(self, state, piece_type):
        # Convert the board to a PyTorch tensor with 4 channels
        # Channels: [current_player, opponent_player, ones, zeros]
        board_array = np.array(state.board)
        current_player = (board_array == piece_type).astype(np.float32)
        opponent_player = (board_array == 3 - piece_type).astype(np.float32)
        ones = np.ones((self.N, self.N), dtype=np.float32)
        zeros = np.zeros((self.N, self.N), dtype=np.float32)
        board_tensor = torch.FloatTensor(
            np.array([current_player, opponent_player, ones, zeros])
        )
        return board_tensor

    def play_self_play_game(self):
        go = GO(self.N)
        go.init_board(self.N)
        states, mcts_probs, current_players = [], [], []
        piece_type = 1  # 'X' starts first

        while True:
            # Perform MCTS and get action probabilities
            action, action_probs = self.mcts_search(go, piece_type, temp=1.0)
            # Store the data
            state_tensor = self.board_to_tensor(go, piece_type)
            policy = np.zeros(self.N * self.N)
            for act, prob in action_probs:
                idx = self.action_to_idx(act)
                policy[idx] = prob
            states.append(state_tensor)
            mcts_probs.append(policy)
            current_players.append(piece_type)

            # Execute the action
            if action != "PASS":
                go.place_chess(action[0], action[1], piece_type)
                go.died_pieces = go.remove_died_pieces(3 - piece_type)
            else:
                go.previous_board = copy.deepcopy(go.board)
            # Check for game end
            if go.game_end(piece_type, "PASS" if action == "PASS" else "MOVE"):
                winner = go.judge_winner()
                break

            # Switch player
            piece_type = 3 - piece_type
            go.X_move = (piece_type == 1)

        # Assign values to each state
        values = []
        for player in current_players:
            if winner == 0:
                values.append(0)
            elif winner == player:
                values.append(1)
            else:
                values.append(-1)

        # Store the data in replay buffer
        for state, mcts_prob, value in zip(states, mcts_probs, values):
            self.buffer.add(state, mcts_prob, value)

    def action_to_idx(self, action):
        if action == "PASS":
            return self.N * self.N  # Assign last index to "PASS"
        else:
            i, j = action
            return i * self.N + j

    def train(self, epochs=1):
        self.model.train()
        for _ in range(epochs):
            if len(self.buffer) < self.batch_size:
                continue
            state_batch, mcts_probs_batch, value_batch = self.buffer.sample(self.batch_size)
            state_batch = torch.stack(state_batch).to(self.device)
            mcts_probs_batch = torch.FloatTensor(mcts_probs_batch).to(self.device)
            value_batch = torch.FloatTensor(value_batch).to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()
            # Forward
            policy_logits, values = self.model(state_batch)
            values = values.view(-1)
            policy_loss = F.cross_entropy(policy_logits, torch.argmax(mcts_probs_batch, dim=1))
            value_loss = F.mse_loss(values, value_batch)
            loss = policy_loss + value_loss
            # Backward
            loss.backward()
            self.optimizer.step()

    def save_model(self, path="model.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="model.pth"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

if __name__ == "__main__":
    agent = AlphaZeroAgent()
    num_iterations = 1000  # Number of training iterations
    games_per_iteration = 10  # Number of self-play games per iteration

    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}")
        for _ in range(games_per_iteration):
            agent.play_self_play_game()
        agent.train(epochs=5)
        agent.save_model()
        
        
    # To play a game w model
   #  if __name__ == "__main__":
   #    agent = AlphaZeroAgent()
   #    agent.load_model("model.pth")
    

