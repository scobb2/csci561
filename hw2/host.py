import sys
import random
import timeit
import math
import argparse
from collections import Counter
from copy import deepcopy

from read import *
from write import writeNextInput

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
        self.komi = n / 2  # Komi rule
        self.verbose = False  # Verbose only when there is a manual player
        self.pass_count = 0

    def init_board(self, n):
        '''
        Initialize a board with size n*n.

        :param n: width and height of the board.
        :return: None.
        '''
        board = [[0 for x in range(n)] for y in range(n)]  # Empty space marked as 0
        self.board = board
        self.previous_board = deepcopy(board)

    def set_board(self, piece_type, previous_board, board):
        '''
        Initialize board status.
        :param previous_board: previous board state.
        :param board: current board state.
        :return: None.
        '''
        self.previous_board = [row[:] for row in previous_board]
        self.board = [row[:] for row in board]

    def compare_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def copy_board(self):
        '''
        Create a deep copy of the game instance.

        :return: A new GO instance with the same state.
        '''
        new_go = GO(self.size)
        new_go.board = deepcopy(self.board)
        new_go.previous_board = deepcopy(self.previous_board)
        new_go.n_move = self.n_move
        return new_go

    def detect_neighbor(self, i, j):
        '''
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i - 1, j))
        if i < len(board) - 1: neighbors.append((i + 1, j))
        if j > 0: neighbors.append((i, j - 1))
        if j < len(board) - 1: neighbors.append((i, j + 1))
        return neighbors

    def detect_neighbor_ally(self, i, j):
        '''
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = self.detect_neighbor(i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, i, j):
        '''
        Using DFS to search for all allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the all allies row and column (row, column) of position (i, j).
        '''
        stack = [(i, j)]  # stack for DFS search
        ally_members = []  # record allies positions during the search
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
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        '''
        board = self.board
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in an allied group has an empty space, it has no liberty
        return False

    def find_died_pieces(self, piece_type):
        '''
        Find the dead stones that have no liberty in the board for a given piece type.

        :param piece_type: 1('X') or 2('O').
        :return: a list containing the dead pieces row and column(row, column).
        '''
        board = self.board
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece dies if it has no liberty
                    if not self.find_liberty(i, j):
                        died_pieces.append((i, j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        '''
        Remove the dead stones in the board.

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
        Remove the stones of certain locations.

        :param positions: a list containing the pieces to be removed row and column(row, column)
        :return: None.
        '''
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)

    def place_chess(self, i, j, piece_type):
        '''
        Place a chess stone on the board.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the placement is valid.
        '''
        board = self.board

        valid_place = self.valid_place_check(i, j, piece_type)
        if not valid_place:
            return False
        self.previous_board = deepcopy(board)
        board[i][j] = piece_type
        self.update_board(board)
        self.n_move += 1  # Increment move counter
        self.pass_count = 0
        return True

    def pass_move(self, piece_type):
        '''
        Handle a pass move.

        :param piece_type: 1('X') or 2('O').
        '''
        self.previous_board = deepcopy(self.board)
        self.n_move += 1  # Increment move counter
        self.pass_count += 1

    def valid_place_check(self, i, j, piece_type, test_check=False):
        '''
        Check whether a placement is valid.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1(white piece) or 2(black piece).
        :param test_check: boolean if it's a test check.
        :return: boolean indicating whether the placement is valid.
        '''
        board = self.board
        verbose = self.verbose
        if test_check:
            verbose = False

        # Check if the place is in the board range
        if not (i >= 0 and i < len(board)):
            if verbose:
                print(('Invalid placement. row should be in the range 0 to {}.').format(len(board) - 1))
            return False
        if not (j >= 0 and j < len(board)):
            if verbose:
                print(('Invalid placement. column should be in the range 0 to {}.').format(len(board) - 1))
            return False

        # Check if the place already has a piece
        if board[i][j] != 0:
            if verbose:
                print('Invalid placement. There is already a chess in this position.')
            return False

        # Copy the board for testing
        test_go = self.copy_board()
        test_board = test_go.board

        # Check if the place has liberty
        test_board[i][j] = piece_type
        test_go.update_board(test_board)
        if test_go.find_liberty(i, j):
            return True

        # If not, remove the dead pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            if verbose:
                print('Invalid placement. No liberty found in this position.')
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_board(self.previous_board, test_go.board):
                if verbose:
                    print('Invalid placement. A repeat move not permitted by the KO rule.')
                return False
        return True

    def update_board(self, new_board):
        '''
        Update the board with new_board

        :param new_board: new board.
        :return: None.
        '''
        self.board = new_board

    def visualize_board(self, board=None, current_move=None, captured_pieces=None):
        '''
        Visualize the board, highlighting the current move and captured pieces.

        :param board: The board state to visualize. If None, use self.board.
        :param current_move: Tuple (i, j) indicating the current move.
        :param captured_pieces: List of tuples [(i1, j1), (i2, j2), ...] indicating captured pieces.
        '''
        if board is None:
            board = self.board
        print('   ' + ' '.join(f"{i:2}" for i in range(self.size)))
        print('  +' + '---' * self.size + '+')
        for i in range(self.size):
            row_str = ''
            for j in range(self.size):
                cell = board[i][j]
                symbol = '.'
                if cell == 1:
                    symbol = 'X'
                elif cell == 2:
                    symbol = 'O'

                # Highlight the current move
                if current_move == (i, j):
                    symbol = f'[{symbol}]'
                # Indicate captured pieces
                elif captured_pieces and (i, j) in captured_pieces:
                    symbol = f'({symbol})'
                else:
                    symbol = f' {symbol} '

                row_str += symbol
            print(f"{i:2}|{row_str}|")
        print('  +' + '---' * self.size + '+')

    def game_end(self, piece_type, action="MOVE"):
        '''
        Check if the game should end.

        :param piece_type: 1('X') or 2('O').
        :param action: "MOVE" or "PASS".
        :return: boolean indicating whether the game should end.
        '''
        # Case 1: max move reached
        if self.n_move >= self.max_move:
            return True
        # Case 2: two players pass consecutively
        if self.pass_count >= 2:
            return True
        return False

    def score(self, piece_type):
        '''
        Get score of a player by counting the number of stones.

        :param piece_type: 1('X') or 2('O').
        :return: score of the player.
        '''
        board = self.board
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type:
                    cnt += 1
        return cnt

    def judge_winner(self):
        '''
        Judge the winner of the game by number of pieces for each player.

        :param: None.
        :return: piece type of winner of the game (0 if it's a tie).
        '''
        cnt_1 = self.score(1)
        cnt_2 = self.score(2) + self.komi  # Add Komi to White's score
        if cnt_1 > cnt_2:
            return 1
        elif cnt_1 < cnt_2:
            return 2
        else:
            return 0  # Tie

    def count_liberty(self, group):
        liberties = set()
        for (i, j) in group:
            neighbors = self.detect_neighbor(i, j)
            for (x, y) in neighbors:
                if self.board[x][y] == 0:
                    liberties.add((x, y))
        return len(liberties)

    def check_territory_owner(self, i, j):
        """
        Determine which player controls the empty point at (i, j).

        Returns:
            1 if controlled by Black,
            2 if controlled by White,
            0 if neutral or undefined.
        """
        visited = set()
        queue = [(i, j)]
        owners = set()
        while queue:
            x, y = queue.pop()
            visited.add((x, y))
            neighbors = self.detect_neighbor(x, y)
            for (nx, ny) in neighbors:
                if self.board[nx][ny] == 0 and (nx, ny) not in visited:
                    queue.append((nx, ny))
                elif self.board[nx][ny] != 0:
                    owners.add(self.board[nx][ny])
        if len(owners) == 1:
            return owners.pop()
        else:
            return 0  # Neutral territory
        
        
    def get_groups(self, piece_type):
        board = self.board
        visited = set()
        groups = []
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type and (i, j) not in visited:
                    group = self.ally_dfs(i, j)
                    groups.append(group)
                    visited.update(group)
        return groups

    def is_in_atari(self, i, j):
        group = self.ally_dfs(i, j)
        liberties = self.count_liberty(group)
        return liberties == 1

    def count_liberty(self, group):
        liberties = set()
        for (i, j) in group:
            neighbors = self.detect_neighbor(i, j)
            for (x, y) in neighbors:
                if self.board[x][y] == 0:
                    liberties.add((x, y))
        return len(liberties)

def judge(n_move, verbose=False):
    N = 5

    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.verbose = verbose
    go.set_board(piece_type, previous_board, board)
    go.n_move = n_move
    try:
        action, x, y = readOutput()
    except:
        print("output.txt not found or invalid format")
        sys.exit(1)  # Current player loses

    if action == "MOVE":
        if not go.place_chess(x, y, piece_type):
            print('Game end.')
            print('The winner is {}'.format('X' if 3 - piece_type == 1 else 'O'))
            sys.exit(1)  # Current player loses

        go.died_pieces = go.remove_died_pieces(3 - piece_type)

    elif action == "PASS":
        go.pass_move(piece_type)
    else:
        print("Invalid action.")
        sys.exit(1)  # Current player loses

    if verbose:
        go.visualize_board()
        print()

    if go.game_end(piece_type, action):
        result = go.judge_winner()
        if verbose:
            print('Game end.')
            if result == 0:
                print('The game is a tie.')
            else:
                print('The winner is {}'.format('X' if result == 1 else 'O'))
        if result == 0:
            sys.exit(2)  # Game ends in a tie
        elif result == piece_type:
            sys.exit(0)  # Current player wins
        else:
            sys.exit(1)  # Current player loses

    piece_type = 2 if piece_type == 1 else 1

    if action == "PASS":
        go.previous_board = deepcopy(go.board)
    writeNextInput(piece_type, go.previous_board, go.board)

    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--move", "-m", type=int, help="number of total moves", default=0)
    parser.add_argument("--verbose", "-v", type=bool, help="print board", default=False)
    args = parser.parse_args()

    judge(args.move, args.verbose)
