import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import copy
import logging
import math

class GO:
    def __init__(self, n):
        self.size = n
        self.X_move = True
        self.died_pieces = []
        self.n_move = 0
        self.max_move = n * n - 1
        self.komi = 2.5
        self.pass_count = 0
        self.previous_board = None
        self.init_board(n)

    def init_board(self, n):
        self.board = np.zeros((n, n), dtype=int)
        self.previous_board = copy.deepcopy(self.board)

    def copy_board(self):
        return copy.deepcopy(self)

    def detect_neighbor(self, i, j):
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i - 1, j))
        if i < self.size - 1: neighbors.append((i + 1, j))
        if j > 0: neighbors.append((i, j - 1))
        if j < self.size - 1: neighbors.append((i, j + 1))
        return neighbors

    def detect_neighbor_ally(self, i, j):
        neighbors = self.detect_neighbor(i, j)
        group_allies = []
        for piece in neighbors:
            if self.board[piece[0]][piece[1]] == self.board[i][j]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, i, j):
        stack = [(i, j)]
        ally_members = []
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def find_liberty(self, i, j):
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                if self.board[piece[0]][piece[1]] == 0:
                    return True
        return False

    def find_died_pieces(self, piece_type):
        died_pieces = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == piece_type:
                    if not self.find_liberty(i, j):
                        died_pieces.append((i, j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        '''
        Remove the dead stones from the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        '''
        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces:
            return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def remove_certain_pieces(self, positions):
        for piece in positions:
            self.board[piece] = 0

    def place_chess(self, i, j, piece_type):
        valid_place = self.valid_place_check(i, j, piece_type)
        if not valid_place:
            return False
        self.previous_board = copy.deepcopy(self.board)
        self.board[i][j] = piece_type
        self.died_pieces = self.remove_died_pieces(3 - piece_type)
        return True

    def valid_place_check(self, i, j, piece_type):
        if not (0 <= i < self.size and 0 <= j < self.size) or self.board[i][j] != 0:
            return False
        
        # Simulate the placement to check for suicide and Ko
        test_go = self.copy_board()
        test_go.board[i][j] = piece_type
        test_go.remove_died_pieces(3 - piece_type)
        
        if not test_go.find_liberty(i, j):
            return False
        if self.died_pieces and np.array_equal(test_go.board, self.previous_board):
            return False

        return True

    def game_end(self, piece_type):
        return self.n_move >= self.max_move or self.pass_count >= 2

    def visualize_board(self):
        print('-' * self.size * 2)
        for i in range(self.size):
            for j in range(self.size):
                piece = self.board[i][j]
                if piece == 0:
                    print('.', end=' ')
                elif piece == 1:
                    print('X', end=' ')
                else:
                    print('O', end=' ')
            print()
        print('-' * self.size * 2)

    def score(self, piece_type):
        return np.sum(self.board == piece_type)

    def judge_winner(self):
        score_x = self.score(1)
        score_o = self.score(2) + self.komi
        if score_x > score_o:
            return 1 
        else:
            return 2

class PolicyValueNet(nn.Module):
    def __init__(self, board_size, action_size, num_channels=128, num_res_blocks=5):
        super(PolicyValueNet, self).__init__()
        self.board_size = board_size
        self.action_size = action_size
        self.num_channels = num_channels

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(16, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch_size, 4, board_size, board_size)
        x = F.relu(self.bn1(self.conv1(x)))

        # Pass through residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy_logits = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value
    # def __init__(self, board_size):
    #     super(PolicyValueNet, self).__init__()
    #     self.board_size = board_size
    #     self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
    #     self.bn1 = nn.BatchNorm2d(16)
    #     self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    #     self.bn2 = nn.BatchNorm2d(32)
    #     self.fc1 = nn.Linear(32 * board_size * board_size, 128)
    #     self.dropout = nn.Dropout(p=0.3)
    #     self.fc_p = nn.Linear(128, board_size * board_size + 1)
    #     self.fc_v = nn.Linear(128, 1)

    # def forward(self, x):
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = F.relu(self.bn2(self.conv2(x)))
    #     x = x.view(-1, 32 * self.board_size * self.board_size)
    #     x = F.relu(self.fc1(x))
    #     x = self.dropout(x)
    #     policy_logits = self.fc_p(x)
    #     value = torch.tanh(self.fc_v(x))
    #     return policy_logits, value

    # def __init__(self, board_size, action_size, num_channels=512, dropout=0.3):
    #     super(PolicyValueNet, self).__init__()
    #     self.board_size = board_size
    #     self.action_size = action_size
    #     self.num_channels = num_channels
    #     self.dropout = dropout

    #     # Convolutional layers
    #     self.conv1 = nn.Conv2d(4, num_channels, kernel_size=3, padding=1)
    #     self.bn1 = nn.BatchNorm2d(num_channels)
    #     self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
    #     self.bn2 = nn.BatchNorm2d(num_channels)
    #     self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
    #     self.bn3 = nn.BatchNorm2d(num_channels)
    #     self.conv4 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
    #     self.bn4 = nn.BatchNorm2d(num_channels)

    #     # Compute the size of the flattened features after convolution layers
    #     def conv2d_size_out(size, kernel_size=3, padding=0, stride=1):
    #         return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    #     # conv_w = conv2d_size_out(
    #     #     conv2d_size_out(
    #     #         conv2d_size_out(
    #     #             conv2d_size_out(self.board_size, 3, padding=1),  # After conv1
    #     #             3, padding=1),  # After conv2
    #     #         3, padding=0),  # After conv3
    #     #     3, padding=0)  # After conv4

    #     # conv_h = conv_w  # Assuming square board

    #     # Calculating the output size after each convolution layer
    #     conv1_size = conv2d_size_out(self.board_size, kernel_size=3, padding=1)
    #     conv2_size = conv2d_size_out(conv1_size, kernel_size=3, padding=1)
    #     conv3_size = conv2d_size_out(conv2_size, kernel_size=3, padding=1)
    #     conv4_size = conv2d_size_out(conv3_size, kernel_size=3, padding=1)

    #     # self.flattened_size = num_channels * conv_w * conv_h
    #     self.flattened_size = self.num_channels * conv4_size * conv4_size

    #     # Fully connected layers
    #     self.fc1 = nn.Linear(self.flattened_size, 1024)
    #     self.bn_fc1 = nn.BatchNorm1d(1024)
    #     self.fc2 = nn.Linear(1024, 512)
    #     self.bn_fc2 = nn.BatchNorm1d(512)

    #     # Output layers
    #     self.fc_pi = nn.Linear(512, self.action_size)
    #     self.fc_v = nn.Linear(512, 1)

    # def forward(self, x):
    #     # x: (batch_size, 4, board_size, board_size)
    #     x = F.relu(self.bn1(self.conv1(x)))  # Conv layer 1
    #     x = F.relu(self.bn2(self.conv2(x)))  # Conv layer 2
    #     x = F.relu(self.bn3(self.conv3(x)))  # Conv layer 3
    #     x = F.relu(self.bn4(self.conv4(x)))  # Conv layer 4
    #     x = x.view(-1, self.flattened_size)  # Flatten

    #     x = F.dropout(F.relu(self.bn_fc1(self.fc1(x))), p=self.dropout, training=self.training)
    #     x = F.dropout(F.relu(self.bn_fc2(self.fc2(x))), p=self.dropout, training=self.training)

    #     policy_logits = self.fc_pi(x)  # Policy head (logits)
    #     value = torch.tanh(self.fc_v(x))  # Value head

    #     return policy_logits, value

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class MCTSNode:
    def __init__(self, state, parent=None, prior_p=1.0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = prior_p

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self.children and prob >= 0:
                next_state = self.state.copy_board()
                piece_type = 1 if next_state.X_move else 2
                if action != "PASS":
                    next_state.place_chess(action[0], action[1], piece_type)
                    next_state.remove_died_pieces(3 - piece_type)
                else:
                    next_state.pass_count += 1
                next_state.X_move = not next_state.X_move
                self.children[action] = MCTSNode(next_state, self, prob)
        # logging.info(f'Node expanded with {len(self.children)} children.')

    def select(self, c_puct):
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, action, leaf_value):
        self.N += 1
        self.W += leaf_value
        self.Q = self.W / self.N

    def update_recursive(self, action, leaf_value):
        if self.parent:
            self.parent.update_recursive(action, -leaf_value)
        self.update(action, leaf_value)

    def get_value(self, c_puct):
        u = c_puct * self.P * math.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + u

class ReplayBuffer:
    def __init__(self, capacity=200000):
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

class RandomPlayer:
    def __init__(self):
        self.type = 'random'

    def get_input(self, go, piece_type):
        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type):
                    possible_placements.append((i,j))

        if not possible_placements:
            return "PASS"
        else:
            return random.choice(possible_placements)

class AlphaZeroAgent:
    def __init__(self, num_simulations=100, c_puct=1.0, buffer_size=200000, batch_size=512, lr=0.001,
                 num_channels=516, dropout=0.3):
        self.type = 'alphazero'
        self.N = 5
        self.action_size = self.N * self.N + 1  # Number of positions plus "PASS"
        self.num_channels = num_channels
        self.dropout = dropout
        self.model = PolicyValueNet(self.N, self.action_size, num_channels=self.num_channels) # , dropout=self.dropout
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.batch_size = batch_size
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.best_win_rate = 0.0
        self.all_actions = [(i, j) for i in range(self.N) for j in range(self.N)] + ["PASS"]

    def get_input(self, go, piece_type):
        action, _ = self.mcts_search(go, piece_type, temp=1e-3)
        return action

    def mcts_search(self, go, piece_type, temp=1.0, add_noise=False):
        initial_piece_type = piece_type
        root_state = go.copy_board()
        root_state.X_move = (piece_type == 1)
        root_node = MCTSNode(root_state)

        # Evaluate the root node to get initial priors
        action_probs, _ = self.evaluate(root_state, piece_type)
        root_node.expand(action_probs)

        # Add Dirichlet noise to the root node's priors if needed
        if add_noise:
            dir_alpha = 0.3  # Adjust alpha as needed
            epsilon = 0.25   # Adjust epsilon as needed
            valid_actions = [action for action, prob in action_probs]
            noise = np.random.dirichlet([dir_alpha] * len(valid_actions))
            for idx, (action, _) in enumerate(action_probs):
                root_node.children[action].P = \
                    (1 - epsilon) * root_node.children[action].P + epsilon * noise[idx]

        for _ in range(self.num_simulations):
            node = root_node
            state = root_state.copy_board()
            actions_selected = []

            while not node.is_leaf():
                action, node = node.select(self.c_puct)
                actions_selected.append(action)
                if action != "PASS":
                    state.place_chess(action[0], action[1], piece_type)
                else:
                    state.pass_count += 1
                state.X_move = not state.X_move
                piece_type = 1 if state.X_move else 2

            if state.game_end(piece_type):
                winner = state.judge_winner()
                if winner == piece_type:
                    leaf_value = 1.0
                else:
                    leaf_value = -1.0            
            else:
                action_probs, leaf_value = self.evaluate(state, piece_type)
                node.expand(action_probs)

            for action in reversed(actions_selected):
                node.update(action, leaf_value)
                node = node.parent
                leaf_value = -leaf_value

        actions_visits = [(action, child.N) for action, child in root_node.children.items()]
        # debug
        logging.debug(f"Actions and visit counts: {actions_visits}")

        if not actions_visits:
            # debug
            logging.debug("No actions were visited during MCTS simulations.")
            return "PASS", [(action, 1.0)]

        actions, visits = zip(*actions_visits)
        visits = np.array(visits, dtype=np.float32)

        if np.sum(visits) == 0:
            # print("All visits are zero, using uniform distribution.")
            probs = np.full(len(visits), 1.0 / len(visits))  # Uniform distribution
        else:
            probs = visits / np.sum(visits)
        
        if temp == 0:
            best_action = actions[np.argmax(probs)]
            action_probs = np.zeros(len(probs))
            action_probs[np.argmax(probs)] = 1.0
        else:
            probs = np.where(probs > 0, probs ** (1 / temp), 0)

            if np.sum(probs) > 0:
                probs /= np.sum(probs)
            else:
                probs = np.full(len(actions), 1.0 / len(actions))

            action_idx = np.random.choice(len(actions), p=probs)
            best_action = actions[action_idx]
            action_probs = probs

        full_policy = np.zeros(self.N * self.N + 1)
        for action, prob in zip(actions, action_probs):
            idx = self.action_to_idx(action)
            full_policy[idx] = prob

        # Log the probabilities after temperature adjustment
        logging.debug(f"Action probabilities after temperature adjustment: {list(zip(actions, probs))}")

        # Log the selected action
        logging.debug(f"Selected action: {best_action}, Probability: {probs[action_idx]}")

        return best_action, [(action, full_policy[self.action_to_idx(action)]) for action in actions]

    def evaluate(self, state, piece_type):
        board_tensor = self.board_to_tensor(state, piece_type)
        board_tensor = board_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(board_tensor)
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = value.item()
        self.model.train()

        # debug
        logging.debug(f"Raw policy probabilities: {policy}")
        
        if np.all(state.board == 0) and piece_type == 1:
            center = (2,2)
            center_idx = self.action_to_idx(center)
            boost_factor = 10.0  # Adjust as needed
            policy[center_idx] += boost_factor
            # Re-normalize the probabilities
            policy /= np.sum(policy)
            # debug
            logging.debug(f"Policy probabilities after center boost: {policy}")

        valid_moves = []
        for i in range(state.size):
            for j in range(state.size):
                if state.valid_place_check(i, j, piece_type):
                    valid_moves.append((i, j))
        if not valid_moves:
            valid_moves.append("PASS")

        action_probs = []
        for idx, action in enumerate(self.all_actions):
            if action in valid_moves:
                prob = policy[idx]
                if action == "PASS":
                    prob = 1
                action_probs.append((action, prob))
        
        total_prob = sum(prob for _, prob in action_probs)
        # debug
        logging.debug(f"Total probability before normalization: {total_prob}")
        
        if total_prob > 0:
            action_probs = [(act, prob / total_prob) for act, prob in action_probs]
        else:
            action_probs = [(act, 1.0 / len(action_probs)) for act, _ in action_probs]


        # debug
        logging.debug(f"Normalized action probabilities: {action_probs}")
        return action_probs, value

    def board_to_tensor(self, state, piece_type):
        history = []
        current_state = state
        for _ in range(8):  # Include last 8 states
            current_player = (current_state.board == piece_type).astype(np.float32)
            opponent_player = (current_state.board == 3 - piece_type).astype(np.float32)
            history.extend([current_player, opponent_player])
            if current_state.previous_board is not None:
                current_state = GO(self.N)
                current_state.board = current_state.previous_board
            else:
                break
        # Pad history if less than required
        while len(history) < 16:
            history.append(np.zeros((self.N, self.N), dtype=np.float32))
        board_tensor = torch.FloatTensor(np.array(history))
        return board_tensor

    def play_self_play_game(self, log_game=False):
        go = GO(self.N)
        states, mcts_probs, current_players = [], [], []
        piece_type = 1
        game_moves = []

        while True:
            if go.n_move == 0 and piece_type == 1:
                # Hardcode the first move for Black to the center position
                center = (2,2)
                valid = go.place_chess(center[0], center[1], piece_type)
                if not valid:
                    go.pass_count += 1
                else:
                    # Record the state, policy, and current player for training
                    state_tensor = self.board_to_tensor(go, piece_type)
                    policy = np.zeros(self.N * self.N + 1)
                    center_idx = self.action_to_idx(center)
                    policy[center_idx] = 1.0  # 100% probability for the center move
                    states.append(state_tensor)
                    mcts_probs.append(policy)
                    current_players.append(piece_type)
                    game_moves.append((copy.deepcopy(go.board), center))
                go.n_move += 1
                piece_type = 3 - piece_type  # Switch player
                continue  # Proceed to the next iteration
            
            
            else:
                temp = max(1.0 - (go.n_move / 24), 1e-3)  # Dynamic temperature scaling
                # logging.info(f"Move {go.n_move}, Temperature: {temp}")
                action, action_probs = self.mcts_search(go, piece_type, temp=temp, add_noise=True)
                state_tensor = self.board_to_tensor(go, piece_type)
        ############################################################### why 26 below?            
                policy = np.zeros(self.N * self.N + 1)
                for act, prob in action_probs:
                    idx = self.action_to_idx(act)
                    policy[idx] = prob
                states.append(state_tensor)
                mcts_probs.append(policy)
                current_players.append(piece_type)

                # if log_game:
                #     logging.info(f"Action Probabilities: {policy}")

                if action != "PASS":
                    if not go.place_chess(action[0], action[1], piece_type):
                        go.pass_count += 1
                else:
                    go.pass_count += 1

                game_moves.append((copy.deepcopy(go.board), action))
        ###############################################################
                go.n_move += 1

                # debug
                # if log_game:
                #     logging.info(f"Player {'X' if piece_type == 1 else 'O'} took action: {action}")

                if go.game_end(piece_type):
                    winner = go.judge_winner()
                    # debug
                    logging.info(f"Game ended. Winner: {'X' if winner == 1 else 'O'}")
                    break

                # go.n_move += 1
                piece_type = 3 - piece_type  # Switch player

        values = [1 if winner == player else -1 for player in current_players]
        for state, mcts_prob, value in zip(states, mcts_probs, values):
            self.buffer.add(state, mcts_prob, value)


        if log_game:
            logging.info("Game moves:")
            for move_num, (board_state, action) in enumerate(game_moves):
                board_str = self.board_to_string(board_state)
                logging.info(f"Move {move_num + 1}, Player {'X' if current_players[move_num] == 1 else 'O'}, Action: {action}")
                logging.info("\n" + board_str)
                logging.info(f"Replay buffer size: {len(self.buffer)}")

    def board_to_string(self, board):
        board_str = ''
        for row in board:
            row_str = ' '.join(['.' if x == 0 else 'X' if x == 1 else 'O' for x in row])
            board_str += row_str + '\n'
        return board_str

    def action_to_idx(self, action):
        if action == "PASS":
            return self.N * self.N  # Assign last index to "PASS"
        else:
            i, j = action
            return i * self.N + j

    def idx_to_action(self, idx):
        if idx == self.N * self.N:
            return "PASS"
        else:
            i = idx // self.N
            j = idx % self.N
            return (i, j)

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            if len(self.buffer) < self.batch_size:
                # debug
                logging.info("Not enough samples in the buffer to start training.")
                continue
            state_batch, mcts_probs_batch, value_batch = self.buffer.sample(self.batch_size)
            state_batch = torch.stack(state_batch).to(self.device)
            mcts_probs_batch = np.array(mcts_probs_batch)
            mcts_probs_batch = torch.FloatTensor(mcts_probs_batch).to(self.device)
            value_batch = torch.FloatTensor(value_batch).to(self.device)

            self.optimizer.zero_grad()

            policy_logits, values = self.model(state_batch)
            values = values.view(-1)

            value_loss = F.mse_loss(values, value_batch)

            policy_loss = -torch.mean(torch.sum(mcts_probs_batch * F.log_softmax(policy_logits, dim=1), dim=1))
            loss = value_loss + policy_loss

            loss.backward()
            self.optimizer.step()
            logging.info(f'Epoch {epoch}, Loss: {loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}')

            # debug
            total_norm = 0
            for p in self.model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            logging.info(f'Gradient Norm: {total_norm}')
        # debug
        logging.info("Training completed for this iteration.")

    def save_model(self, path="modelx.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="modelx.pth"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)

    def evaluate_against_random_player(self, num_games):
        random_player = RandomPlayer()
        win_count = 0
        black_wins = 0
        white_wins = 0
        black_games = 0 
        white_games = 0
        for game_num in range(num_games):
            if game_num >= num_games/2:
                self_color = 1
            else:
                self_color = 2
            if self_color == 1:
                black_games += 1
            else:
                white_games += 1
            winner = self.play_game_against_random(random_player, self_color=self_color)
            if winner == self_color:
                win_count += 1
                if self_color == 1:
                    black_wins += 1
                else:
                    white_wins += 1
        win_rate = win_count / num_games
        logging.info(f"Win rate against Random Player: {win_rate:.2f}")
        logging.info(f"As black, won {black_wins} out of {black_games} games")
        logging.info(f"As white, won {white_wins} out of {white_games} games")
        return win_rate

    def play_game_against_random(self, opponent, self_color):
        go = GO(self.N)
        go.init_board(self.N)
        piece_type = 1  # Black ('X') starts first
        go.X_move = True
        go.pass_count = 0

        while True:
            if piece_type == self_color:
                # Agent's turn: use MCTS to select action
                action, _ = self.mcts_search(go, piece_type, temp=1e-3, add_noise=False)
            else:
                # Opponent's turn: use opponent's get_input method
                action = opponent.get_input(go, piece_type)
            
            if action != "PASS":
                valid = go.place_chess(action[0], action[1], piece_type)
                if not valid:
                    # Invalid move, pass instead
                    go.pass_count += 1
            else:
                go.pass_count += 1
            
            go.n_move += 1

            # Check for game end
            if go.game_end(piece_type):
                winner = go.judge_winner()
                return winner
            
            # Switch player
            piece_type = 3 - piece_type
            go.X_move = (piece_type == 1)

    def play_game_against(self, opponent_agent, self_color):
        """
        Play a game against another agent.

        :param opponent_agent: Instance of another agent to play against.
        :param self_color: 1 for black ('X'), 2 for white ('O').
        :return: Winner (1 or 2).
        """
        go = GO(self.N)
        go.init_board(self.N)
        go.X_move = True  # Black starts first
        go.pass_count = 0
        piece_type = 1  # Black

        while True:
            if piece_type == self_color:
                action, _ = self.mcts_search(go, piece_type, temp=1e-3, add_noise=False)
            else:
                action = opponent_agent.get_input(go, piece_type)
            
            if action != "PASS":
                valid = go.place_chess(action[0], action[1], piece_type)
                if not valid:
                    # Invalid move, treat as pass
                    go.pass_count += 1
            else:
                go.pass_count += 1

            go.n_move += 1

            if go.game_end(piece_type):
                winner = go.judge_winner()
                return winner

            # Switch player
            piece_type = 3 - piece_type
            go.X_move = (piece_type == 1)

    def evaluate_against_best(self, best_agent, num_games):
        win_count = 0
        for game_num in range(num_games):
            # Randomly decide who plays as black
            if game_num >= num_games/2:
                self_color = 1
            else:
                self_color = 2
            winner = self.play_game_against(best_agent, self_color=self_color)
            if winner == self_color:
                win_count += 1
        win_rate = win_count / num_games
        logging.info(f"Win rate against previous model: {win_rate}")
        return win_rate

if __name__ == "__main__":
    logging.basicConfig(filename='training.log', level=logging.INFO)
    agent = AlphaZeroAgent()

    num_iterations = 1000
    games_per_iteration = 200

    # Initialize best_model as a separate agent
    best_model = AlphaZeroAgent()
    best_model_path = "best_modelx.pth"

    if os.path.exists(best_model_path):
        best_model.load_model(best_model_path)
        best_model_i = -1  # Indicates that the best model was loaded from a previous run
        logging.info("Loaded existing best model for evaluation.")
    else:
        best_model_i = -1  # Will be set after the first iteration
        logging.info("No existing best model found. Will initialize after first iteration.")

    for i in range(num_iterations):
        print(f"Iteration {i + 1}/{num_iterations}")
        for j in range(games_per_iteration):
            if j == 0:
                agent.play_self_play_game(log_game=True)  # Log the first game
            else:
                agent.play_self_play_game()
            print(f"  Completed game {j + 1}/{games_per_iteration} in iteration {i + 1}")

        agent.train(epochs=10)
        agent.save_model(f"modelx_{i + 1}.pth")

        # if win_rate > agent.best_win_rate:
        #     agent.best_win_rate = win_rate
        #     agent.save_model(f"best_modelx_{i + 1}.pth")
        #     logging.info(f"Best model updated at iteration {i + 1} with win rate {win_rate:.2f}")

        # To evaluate against prior model:
        ################################################################
        if i == 0:
            agent.save_model(best_model_path)
            best_model.load_model(best_model_path)
            best_model_i = i
            logging.info(f"First model from iteration {i+1} accepted.")
            win_rate = agent.evaluate_against_random_player(num_games=20)
            logging.info(f"Win rate against Random Player after iteration {i + 1}: {win_rate}")
        if i > 0:
            # Evaluate against best model
            win_rate_against_best = agent.evaluate_against_best(best_model, num_games=40)
            if win_rate_against_best < 0.55:
                # Revert to best model
                agent.load_model(best_model_path)
                logging.info(f"Model reverted to iteration {best_model_i + 1} due to insufficient win rate.")
            else:
                logging.info(f"Model from iteration {i+1} accepted.")
                agent.save_model(best_model_path)
                best_model.load_model(best_model_path)
                best_model_i = i
                win_rate = agent.evaluate_against_random_player(num_games=20)
                logging.info(f"Win rate against Random Player after iteration {i + 1}: {win_rate}")

    agent.save_model("final_modelx.pth")
    logging.info("Training completed. Final model saved.")