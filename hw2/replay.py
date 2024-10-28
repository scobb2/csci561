from host import GO
import time

def replay_game(log_filename, delay=1):
    N = 5
    go = GO(N)
    go.init_board(N)
    moves = []

    with open(log_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Player'):
                parts = line.split(':')
                player = parts[0][-1]
                move = parts[1].strip()
                moves.append((player, move))

    for idx, (player, move) in enumerate(moves):
        piece_type = 1 if player == 'X' else 2
        if move == 'PASS':
            action = "PASS"
            print(f"Move {idx + 1}: Player {player} passes.")
        else:
            i, j = map(int, move.split(','))
            action = (i, j)
            print(f"Move {idx + 1}: Player {player} places at ({i}, {j}).")
            go.place_chess(i, j, piece_type)
            go.remove_died_pieces(3 - piece_type)
        go.visualize_board()
        time.sleep(delay)

if __name__ == "__main__":
    log_filename = input("Enter the game log filename: ")
    replay_game(log_filename)
