import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from collections import deque
import copy
import logging

class GO:
    def __init__(self, n):
        """
        Go game.

        :param n: size of the board n*n
        """
        self.size = n
        self.X_move = True  # X chess plays first
        self.died_pieces = []  # Initialize died pieces to be empty
        self.n_move = 0  # Trace the number of moves
        self.max_move = n * n - 1  # The max movement of a Go game
        self.komi = 2.5  # Komi rule
        self.verbose = False  # Verbose only when there is a manual player
        self.pass_count = 0  # Count consecutive passes
        self.previous_board = None
        self.board = None

    def init_board(self, n):
        '''
        Initialize a board with size n*n.

        :param n: width and height of the board.
        :return: None.
        '''
        board = [[0 for x in range(n)] for y in range(n)]  # Empty space marked as 0
        # 'X' pieces marked as 1
        # 'O' pieces marked as 2
        self.board = board
        self.previous_board = copy.deepcopy(board)

    def copy_board(self):
        '''
        Copy the current board for potential testing.

        :param: None.
        :return: the copied board instance.
        '''
        return copy.deepcopy(self)

    def detect_neighbor(self, i, j):
        '''
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        '''
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i - 1, j))
        if i < self.size - 1: neighbors.append((i + 1, j))
        if j > 0: neighbors.append((i, j - 1))
        if j < self.size - 1: neighbors.append((i, j + 1))
        return neighbors

    def detect_neighbor_ally(self, i, j):
        '''
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbor allies (row, column) of position (i, j).
        '''
        neighbors = self.detect_neighbor(i, j)
        group_allies = []
        for piece in neighbors:
            if self.board[piece[0]][piece[1]] == self.board[i][j]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, i, j):
        '''
        Use DFS to search for all allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing all allies (row, column) of position (i, j).
        '''
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
        '''
        Find liberty of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        '''
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                if self.board[piece[0]][piece[1]] == 0:
                    return True
        return False

    def find_died_pieces(self, piece_type):
        '''
        Find the dead stones that have no liberty.

        :param piece_type: 1('X') or 2('O').
        :return: a list containing the dead pieces (row, column).
        '''
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
        '''
        Remove stones from certain positions.

        :param positions: a list of positions (row, column).
        :return: None.
        '''
        for piece in positions:
            self.board[piece[0]][piece[1]] = 0

    def place_chess(self, i, j, piece_type):
        '''
        Place a stone on the board.

        :param i: row number.
        :param j: column number.
        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the placement is valid.
        '''
        valid_place = self.valid_place_check(i, j, piece_type)
        if not valid_place:
            return False
        self.previous_board = copy.deepcopy(self.board)
        self.board[i][j] = piece_type
        self.died_pieces = self.remove_died_pieces(3 - piece_type)
        return True

    def valid_place_check(self, i, j, piece_type):
        '''
        Check whether a placement is valid.

        :param i: row number.
        :param j: column number.
        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the placement is valid.
        '''
        # Check if the place is in the board range
        if not (0 <= i < self.size and 0 <= j < self.size):
            return False
        # Check if the place already has a piece
        if self.board[i][j] != 0:
            return False
        # Copy the board for testing
        test_go = self.copy_board()
        test_go.board[i][j] = piece_type
        # Remove opponent's dead pieces
        test_go.remove_died_pieces(3 - piece_type)
        # Check if the placement causes suicide
        if not test_go.find_liberty(i, j):
            return False
        # Check for Ko rule
        if self.died_pieces and test_go.board == self.previous_board:
            return False
        return True

    def game_end(self, piece_type, action="MOVE"):
        '''
        Check if the game should end.

        :param piece_type: 1('X') or 2('O').
        :param action: "MOVE" or "PASS".
        :return: boolean indicating whether the game should end.
        '''
        # Max move reached
        if self.n_move >= self.max_move:
            return True
        # Two consecutive passes
        if self.pass_count >= 2:
            return True
        return False

    def visualize_board(self):
        '''
        Visualize the board.

        :return: None
        '''
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

    def update_board(self, new_board):
        '''
        Update the board with new_board

        :param new_board: new board.
        :return: None.
        '''
        self.board = new_board

    def score(self, piece_type):
        '''
        Calculate the score for a player.

        :param piece_type: 1('X') or 2('O').
        :return: score.
        '''
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == piece_type:
                    cnt += 1
        return cnt

    def judge_winner(self):
        '''
        Judge the winner of the game.

        :return: 1('X') or 2('O') or 0 (tie).
        '''
        cnt_1 = self.score(1)
        cnt_2 = self.score(2) + self.komi
        if cnt_1 > cnt_2:
            return 1
        elif cnt_1 < cnt_2:
            return 2
        else:
            return 0

class PolicyValueNet(nn.Module):
    def __init__(self, board_size):
        super(PolicyValueNet, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)  # Input channels: 4
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * board_size * board_size, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.fc_p = nn.Linear(256, board_size * board_size + 1)
        self.fc_v = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # x shape: (batch_size, 4, board_size, board_size)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * self.board_size * self.board_size)
        x = F.relu(self.fc1(x))
        policy_logits = self.fc_p(x)  # Output size: board_size * board_size + 1
        value = torch.tanh(self.fc_v(x))
        return policy_logits, value

class MCTSNode:
    def __init__(self, state, parent=None, prior_p=1.0):
        self.state = state  # GO game state
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
                # Ensure prob is not NaN or negative
                if np.isnan(prob) or prob < 0:
                    print(f"Invalid probability for action {action}: {prob}")
                    continue  # Skip this action
                next_state = self.state.copy_board()
                piece_type = 1 if next_state.X_move else 2
                if action != "PASS":
                    next_state.place_chess(action[0], action[1], piece_type)
                    next_state.died_pieces = next_state.remove_died_pieces(3 - piece_type)
                else:
                    next_state.pass_count += 1
                next_state.X_move = not next_state.X_move
                self.children[action] = MCTSNode(next_state, parent=self, prior_p=prob)

    def select(self, c_puct):
        return max(
            self.children.items(),
            key=lambda act_node: act_node[1].get_value(c_puct)
        )

    def update(self, action, leaf_value):
        self.N += 1
         # Apply a penalty if the action is "PASS"
        if action == "PASS":
            self.W += (leaf_value * 0.5)  # Penalize "PASS" by scaling the value
        else:
            self.W += leaf_value
        self.Q = self.W / self.N

    def update_recursive(self, action, leaf_value):
        if self.parent:
            self.parent.update_recursive(action, -leaf_value)
        self.update(leaf_value, action)

    def get_value(self, c_puct):
        u = c_puct * self.P * math.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + u

class ReplayBuffer:
    def __init__(self, capacity=50000):
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
        '''
        Get one input.

        :param go: GO instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''
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
    def __init__(self, num_simulations=50, c_puct=1.0, buffer_size=50000, batch_size=64, lr=1e-4):
        self.type = 'alphazero'
        self.N = 5  # Board size
        self.model = PolicyValueNet(self.N)
        self.num_simulations = num_simulations  # Number of MCTS simulations
        self.c_puct = c_puct  # Exploration constant
        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.best_win_rate = 0.0  # For saving best models

        actions = []
        for i in range(5):
            for j in range(5):
                actions.append((i, j))
        
        actions.append("PASS")
        
        self.all_actions = actions

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

            # Selection
            actions_selected = []
            while not node.is_leaf():
                action, node = node.select(self.c_puct)
                actions_selected.append(action)
                if action != "PASS":
                    valid = state.place_chess(action[0], action[1], piece_type)
                    if not valid:
                        break  # Invalid move, skip
                else:
                    state.pass_count += 1
                state.X_move = not state.X_move
                piece_type = 1 if state.X_move else 2

            # Evaluation
            if state.game_end(piece_type):
                winner = state.judge_winner()
                if winner == 0:
                    leaf_value = 0.0
                elif winner == piece_type:
                    leaf_value = 1.0
                else:
                    leaf_value = -1.0
            else:
                # Expansion and Evaluation
                action_probs, leaf_value = self.evaluate(state, piece_type)
                node.expand(action_probs)

            # Backpropagation
            for action in reversed(actions_selected):
                node.update(action, leaf_value)
                node = node.parent
                leaf_value = -leaf_value

        # Get action probabilities
        actions_visits = [(action, child.N) for action, child in root_node.children.items()]
        if not actions_visits:
            # If no moves were made, pass
            return "PASS", [("PASS", 1.0)]

        actions, visits = zip(*actions_visits)
        visits = np.array(visits, dtype=np.float32)

        # Debugging: Check for NaN values in visits
        # print("Visits:", visits)  # Check the contents of visits
        # if np.isnan(visits).any():
        #     print("NaN detected in visits:", visits)

        # Handle case where all visits are zero
        # print("Sum of visits:", np.sum(visits))
        if np.sum(visits) == 0:
            # print("All visits are zero, using uniform distribution.")
            probs = np.full(len(visits), 1.0 / len(visits))  # Uniform distribution
        else:
            probs = visits / np.sum(visits)

        # Print normalized probabilities for debugging
        # print("Normalized Probabilities before selection:", probs)  # Debugging

        # Ensure valid probabilities
        # if np.any(probs < 0):
        #     print("Negative probabilities detected:", probs)

        # Select action based on temperature
        if temp == 0:
            best_action = actions[np.argmax(probs)]
            action_probs = np.zeros(len(probs))
            action_probs[np.argmax(probs)] = 1.0
        else:
            probs = np.where(probs > 0, probs ** (1 / temp), 0)
            if np.sum(probs) > 0:
                probs /= np.sum(probs)
            else:
                # print("After adjustment, all probabilities are zero.")
                probs = np.full(len(actions), 1.0 / len(actions))

            # print("Probabilities before selection:", probs)

            action_idx = np.random.choice(len(actions), p=probs)
            best_action = actions[action_idx]
            action_probs = probs

        # Map action_probs to full policy vector
        full_policy = np.zeros(self.N * self.N + 1)
        for action, prob in zip(actions, action_probs):
            idx = self.action_to_idx(action)
            full_policy[idx] = prob

        return best_action, [(action, full_policy[self.action_to_idx(action)]) for action in actions]

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
                if state.valid_place_check(i, j, piece_type):
                    valid_moves.append((i, j))

        # Only append "PASS" if there are no valid moves
        if not valid_moves:
            valid_moves.append("PASS")

        action_probs = []
        for idx, action in enumerate(self.all_actions):
            if action in valid_moves:
                prob = policy[idx]
                if action == "PASS":
                    prob = 1  # Since PASS only allowed when no valid moves exist
                action_probs.append((action, prob))

        # Normalize probabilities
        total_prob = sum([prob for _, prob in action_probs])
        if total_prob > 0:
            action_probs = [(act, prob / total_prob) for act, prob in action_probs]
        else:
            # Handle case where total probability is 0
            action_probs = [(act, 1.0 / len(action_probs)) for act, prob in action_probs]

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

    def old_play_self_play_game(self, log_game=False):
        go = GO(self.N)
        go.init_board(self.N)
        states, mcts_probs, current_players = [], [], []
        piece_type = 1  # 'X' starts first
        go.X_move = True
        go.pass_count = 0
        game_moves = []

        while True:
            # Adjust temperature
            temp = max(1.0 - (go.n_move / 24), 1e-3)  # Dynamic temperature scaling
            # Perform MCTS and get action probabilities
            action, action_probs = self.mcts_search(go, piece_type, temp=temp)
            # Store the data
            state_tensor = self.board_to_tensor(go, piece_type)
            policy = np.zeros(self.N * self.N + 1)
            for act, prob in action_probs:
                idx = self.action_to_idx(act)
                policy[idx] = prob
            states.append(state_tensor)
            mcts_probs.append(policy)
            current_players.append(piece_type)

            if log_game:
                logging.info(f"Action Probabilities: {policy}")
            game_moves.append((copy.deepcopy(go.board), action))

            # Execute the action
            if action != "PASS":
                valid = go.place_chess(action[0], action[1], piece_type)
                if not valid:
                    # Invalid move, pass instead
                    go.pass_count += 1
            else:
                go.pass_count += 1
            # Check for game end
            if go.game_end(piece_type, "PASS" if action == "PASS" else "MOVE"):
                winner = go.judge_winner()
                break

            go.n_move += 1
            # Switch player
            piece_type = 3 - piece_type
            go.X_move = (piece_type == 1)

        # Assign values to each state
        values = []
        for idx, player in enumerate(current_players):
            if winner == 0:
                values.append(0)
            elif winner == player:
                # Penalize for passing if it's not necessary
                if game_moves[idx][1] == "PASS":
                    values.append(-0.5)  # Assign a negative reward for passing
                else:
                    values.append(1)
            else:
                values.append(-1)

        # Store the data in replay buffer
        for state, mcts_prob, value in zip(states, mcts_probs, values):
            self.buffer.add(state, mcts_prob, value)

         # Log the game moves if log_game is True
        if log_game:
            logging.info("Game moves:")
            for move_num, (board_state, action) in enumerate(game_moves):
                board_str = self.board_to_string(board_state)
                logging.info(f"Move {move_num + 1}, Player {'X' if current_players[move_num] == 1 else 'O'}, Action: {action}")
                logging.info("\n" + board_str)

    def play_self_play_game(self, log_game=False):
        go = GO(self.N)
        go.init_board(self.N)
        states, mcts_probs, current_players = [], [], []
        piece_type = 1  # 'X' starts first
        go.X_move = True
        go.pass_count = 0
        game_moves = []

        while True:
            # Dynamic temperature scaling
            max_moves = 24  # Maximum moves for a 5x5 board
            temp = max(1.0 - (go.n_move / max_moves), 1e-3)
            # Perform MCTS and get action probabilities
            action, action_probs = self.mcts_search(go, piece_type, temp=temp)
            # Store the data
            state_tensor = self.board_to_tensor(go, piece_type)
            policy = np.zeros(self.N * self.N + 1)
            for act, prob in action_probs:
                idx = self.action_to_idx(act)
                policy[idx] = prob
            states.append(state_tensor)
            mcts_probs.append(policy)
            current_players.append(piece_type)

            if log_game:
                logging.info(f"Action Probabilities: {policy}")
            # Append to game_moves regardless of logging
            game_moves.append((copy.deepcopy(go.board), action))

            # Execute the action
            if action != "PASS":
                valid = go.place_chess(action[0], action[1], piece_type)
                if not valid:
                    # Invalid move, pass instead
                    go.pass_count += 1
            else:
                go.pass_count += 1

            go.n_move += 1

            # Check for game end
            if go.game_end(piece_type, "PASS" if action == "PASS" else "MOVE"):
                winner = go.judge_winner()
                break

            # Switch player
            piece_type = 3 - piece_type
            go.X_move = (piece_type == 1)

        # Assign values to each state
        values = []
        for idx, player in enumerate(current_players):
            if winner == 0:
                values.append(0)
            elif winner == player:
                values.append(1)
            else:
                values.append(-1)

        # Store the data in replay buffer
        for state, mcts_prob, value in zip(states, mcts_probs, values):
            self.buffer.add(state, mcts_prob, value)

        # Log the game moves if log_game is True
        if log_game:
            logging.info("Game moves:")
            for move_num, (board_state, action) in enumerate(game_moves):
                board_str = self.board_to_string(board_state)
                logging.info(f"Move {move_num + 1}, Player {'X' if current_players[move_num] == 1 else 'O'}, Action: {action}")
                logging.info("\n" + board_str)

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

    # def get_all_actions(self):
    #     actions = []
    #     for i in range(state.size):
    #         for j in range(state.size):
    #             actions.append((i, j))
        
    #     actions.append("PASS")
        
    #     return actions

    def get_all_valid_actions(self, state, piece_type):
        '''
        Retrieves all valid actions for the current state and player.

        :param state: Current GO game state.
        :param piece_type: 1 ('X') or 2 ('O').
        :return: List of valid actions, including "PASS" if applicable.
        '''
        valid_actions = []
        for i in range(state.size):
            for j in range(state.size):
                if state.valid_place_check(i, j, piece_type):
                    valid_actions.append((i, j))
        
        # "PASS" is always a valid move unless the game has ended
        if not state.has_any_valid_moves(piece_type):
            valid_actions.append("PASS")
        
        return valid_actions

    def train(self, epochs=5):
        self.model.train()
        for epoch in range(epochs):
            if len(self.buffer) < self.batch_size:
                continue
            state_batch, mcts_probs_batch, value_batch = self.buffer.sample(self.batch_size)
            state_batch = torch.stack(state_batch).to(self.device)
            mcts_probs_batch = np.array(mcts_probs_batch)
            mcts_probs_batch = torch.FloatTensor(mcts_probs_batch).to(self.device)
            value_batch = torch.FloatTensor(value_batch).to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()
            # Forward
            policy_logits, values = self.model(state_batch)
            values = values.view(-1)
            # Compute losses
            value_loss = F.mse_loss(values, value_batch)
            # For policy loss, use negative log likelihood
            policy_loss = -torch.mean(torch.sum(mcts_probs_batch * F.log_softmax(policy_logits, dim=1), dim=1))
            loss = value_loss + policy_loss
            # Backward
            loss.backward()
            self.optimizer.step()
            logging.info(f'Epoch {epoch}, Loss: {loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}')

    def save_model(self, path="modelx.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="modelx.pth"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def evaluate_against_previous(self, previous_agent, num_games=20):
        win_count = 0
        for game_num in range(num_games):
            # Randomly decide who plays as black
            self_color = random.choice([1, 2])
            winner = self.play_game_against(previous_agent, self_color=self_color)
            if winner == self_color:
                win_count += 1
        win_rate = win_count / num_games
        logging.info(f"Win rate against previous model: {win_rate}")
        return win_rate
    
    def evaluate_against_random_player(self, num_games=20):
        random_player = RandomPlayer()
        win_count = 0
        for game_num in range(num_games):
            winner = self.play_game_against_random(random_player, self_color=random.choice([1, 2]))
            if winner == 1:  # Assuming 'X' is your agent
                win_count += 1
            elif winner == 2:
                pass  # 'O' wins
            else:
                pass  # Tie
        win_rate = win_count / num_games
        logging.info(f"Win rate against Random Player: {win_rate}")
        return win_rate

    def play_game_against(self, opponent_agent, self_color=1):
        go = GO(self.N)
        go.init_board(self.N)
        piece_type = 1  # Black starts first
        go.X_move = True
        go.pass_count = 0

        while True:
            if piece_type == self_color:
                action, _ = self.mcts_search(go, piece_type, temp=1e-3)
            else:
                action, _ = opponent_agent.mcts_search(go, piece_type, temp=1e-3)
            if action != "PASS":
                valid = go.place_chess(action[0], action[1], piece_type)
                if not valid:
                    go.pass_count += 1
            else:
                go.pass_count += 1
                
            go.n_move += 1
            if go.game_end(piece_type, "PASS" if action == "PASS" else "MOVE"):
                winner = go.judge_winner()
                return winner
            # Switch player
            piece_type = 3 - piece_type
            go.X_move = (piece_type == 1)

    def play_game_against_random(self, opponent, self_color=1):
        go = GO(self.N)
        go.init_board(self.N)
        piece_type = 1  # Black starts first
        go.X_move = True
        go.pass_count = 0

        while True:
            if piece_type == self_color:
                # Agent's turn: use MCTS to select action
                action, _ = self.mcts_search(go, piece_type, temp=1e-3)
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
            
            # Check for game end
            if go.game_end(piece_type, "PASS" if action == "PASS" else "MOVE"):
                winner = go.judge_winner()
                return winner
            
            go.n_move += 1
            # Switch player
            piece_type = 3 - piece_type
            go.X_move = (piece_type == 1)



    def test_agent(self):
        # Test function to visualize game plays
        go = GO(self.N)
        go.init_board(self.N)
        piece_type = 1  # 'X' starts first
        go.X_move = True
        go.pass_count = 0

        while True:
            go.visualize_board()
            print(f"Player {'X' if piece_type == 1 else 'O'}'s turn.")
            action = self.get_input(go, piece_type)
            print(f"Action chosen: {action}")
            if action != "PASS":
                valid = go.place_chess(action[0], action[1], piece_type)
                if not valid:
                    print("Invalid move. Passing instead.")
                    go.pass_count += 1
            else:
                go.pass_count += 1
            # Check for game end
            if go.game_end(piece_type, "PASS" if action == "PASS" else "MOVE"):
                winner = go.judge_winner()
                go.visualize_board()
                if winner == 0:
                    print("The game is a tie.")
                else:
                    print(f"The winner is {'X' if winner == 1 else 'O'}.")
                break

            go.n_move += 1
            # Switch player
            piece_type = 3 - piece_type
            go.X_move = (piece_type == 1)

if __name__ == "__main__":
    logging.basicConfig(filename='training.log', level=logging.INFO)
    agent = AlphaZeroAgent()
    # For testing before training
    # print("Testing the agent's game logic and visualization:")
    # agent.test_agent()

    # Uncomment the following for training
    num_iterations = 10  # Adjust as needed
    games_per_iteration = 50  # Adjust as needed

    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}")
        for j in range(games_per_iteration):
            if j == 0:
                agent.play_self_play_game(log_game=True)  # Log the first game
            else:
                agent.play_self_play_game()
            print(f"  Completed game {j+1}/{games_per_iteration} in iteration {i+1}")

        agent.train(epochs=5)
        # Save current model
        agent.save_model(f"modelx_{i+1}.pth")

        # Evaluate against Random Player
        #################################################################
        win_rate = agent.evaluate_against_random_player(num_games=20)
        logging.info(f"Win rate against Random Player after iteration {i+1}: {win_rate}")
        # Save the best model
        if win_rate > agent.best_win_rate -0.02:
            agent.best_win_rate = win_rate
            agent.save_model(f"best_modelx_{i+1}.pth")
            logging.info(f"Best model updated at iteration {i+1} with win rate {win_rate:.2f}")
        
        # To evaluate against prior model:
        #################################################################
        # if i > 0:
        #     # Load previous model
        #     previous_agent = AlphaZeroAgent(num_simulations=50, c_puct=1.0)
        #     previous_agent.load_model(f"modelx_{i}.pth")
        #     # Evaluate against previous model
        #     win_rate = agent.evaluate_against_previous(previous_agent, num_games=20)
        #     if win_rate < 0.55:
        #         # Revert to previous model
        #         agent.load_model(f"modelx_{i}.pth")
        #         logging.info(f"Model reverted to iteration {i} due to insufficient win rate.")
        #     else:
        #         logging.info(f"Model from iteration {i+1} accepted.")
        # else:
        #     logging.info(f"First model from iteration {i+1} accepted.")




    # Save final model
    agent.save_model("final_modelx.pth")


# Passing issues
# epochs 10
# batch size 64
# alpha (learning rate) 0.001
# 10 iterations
# 100 games per iteration

# dropout 0.2 or 0.4??
# c_puct 1.2??

# Passing issues
# prev
# epochs 5
# batch size 256
# alpha (learning rate) 0.0001
# 10 iterations
# 50 games per iteration




# Incorporate batch normalization layers after convolutional layers to stabilize and accelerate training.
# python
# Copy code
# self.bn1 = nn.BatchNorm2d(32)
# self.bn2 = nn.BatchNorm2d(64)
# self.bn3 = nn.BatchNorm2d(128)



# Implement Annealing Schedule:
# Example: Decrease temperature linearly or exponentially over moves.
# Implementation Example:
# python
# Copy code
# max_moves = 24  # Maximum moves for a 5x5 board
# temp = max(1.0 - (go.n_move / max_moves), 1e-3)