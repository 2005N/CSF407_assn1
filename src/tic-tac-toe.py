import random
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

class TicTacToe:
    def __init__(self, size):
        self.size = size
        self.board = [[' ' for _ in range(size)] for _ in range(size)]
        self.last_move = None

    def print_board(self):
        for row in self.board:
            print("|".join(row))
            print("-" * (2 * self.size - 1))

    def is_valid_move(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == ' '

    def apply_move(self, row, col, player):
        if self.is_valid_move(row, col):
            self.board[row][col] = player
            self.last_move = (row, col)
            return True
        return False

    def is_full(self):
        return all(self.board[r][c] != ' ' for r in range(self.size) for c in range(self.size))

    def check_winner(self):
        # Check rows for win
        for row in self.board:
            if row[0] != ' ' and all(cell == row[0] for cell in row):
                return row[0]
        # Check columns for win
        for col in range(self.size):
            if self.board[0][col] != ' ' and all(self.board[row][col] == self.board[0][col] for row in range(self.size)):
                return self.board[0][col]
        # Check main diagonal
        if self.board[0][0] != ' ' and all(self.board[i][i] == self.board[0][0] for i in range(self.size)):
            return self.board[0][0]
        # Check anti-diagonal
        if self.board[0][self.size - 1] != ' ' and all(self.board[i][self.size - 1 - i] == self.board[0][self.size - 1] for i in range(self.size)):
            return self.board[0][self.size - 1]
        return None

def llm_agent_move(game, player, opponent_move, agent_name):
    """
    Simulate an LLM agent move.
    The agent "receives" the current board and details of the opponent's last move.
    It then chooses a move by:
      1. Checking for a winning move.
      2. Blocking an opponent winning move.
      3. Otherwise selecting a random available move.
    """
    size = game.size
    # Get list of available moves
    moves = [(i, j) for i in range(size) for j in range(size) if game.board[i][j] == ' ']

    # 1. Check if a winning move is available
    for move in moves:
        row, col = move
        game_copy = copy.deepcopy(game)
        game_copy.apply_move(row, col, player)
        if game_copy.check_winner() == player:
            return move

    # 2. Check if opponent is about to win and block
    opponent = 'O' if player == 'X' else 'X'
    for move in moves:
        row, col = move
        game_copy = copy.deepcopy(game)
        game_copy.apply_move(row, col, opponent)
        if game_copy.check_winner() == opponent:
            return move

    # 3. Otherwise choose a random move
    return random.choice(moves) if moves else None

def human_agent_move(game):
    """
    Get move input from a human player.
    Input should be in the format row,col (e.g., "0,1")
    """
    while True:
        try:
            move_str = input("Enter your move as row,col (e.g., 0,1): ")
            row, col = map(int, move_str.split(','))
            if game.is_valid_move(row, col):
                return (row, col)
            else:
                print("Invalid move. That cell is either occupied or out of bounds.")
        except Exception:
            print("Invalid input format. Please enter your move as row,col.")

def play_game(board_size, mode="LLM_vs_LLM"):
    """
    Play one game of tic-tac-toe on an NxN board.
    
    mode: "LLM_vs_LLM" for two simulated agents or "LLM_vs_Human" where player 'O' is human.
    In LLM_vs_LLM mode the agents are labeled ChatGPT (X) and Claude (O).
    In simulation mode (used later) a draw is resolved via a coin toss.
    """
    game = TicTacToe(board_size)
    current_player = 'X'
    last_move = None

    # Set agent names depending on mode
    if mode == "LLM_vs_LLM":
        agent_X = "ChatGPT"
        agent_O = "Claude"
    else:
        # In human mode we assume player 'X' is the LLM and player 'O' is the human.
        agent_X = "LLM"
        agent_O = "Human"

    while True:
        if mode == "LLM_vs_LLM" or (mode == "LLM_vs_Human" and current_player == 'X'):
            # Simulate LLM agent move
            agent_name = agent_X if current_player == 'X' else agent_O
            print(f"\nCurrent board state:")
            game.print_board()
            print(f"{agent_name} ({current_player}) is thinking... "
                  f"(Opponent's last move: {last_move})")
            move = llm_agent_move(game, current_player, last_move, agent_name)
        else:
            # Human move
            print("\nCurrent board state:")
            game.print_board()
            move = human_agent_move(game)

        if move is None:
            break

        game.apply_move(move[0], move[1], current_player)
        winner = game.check_winner()
        if winner:
            print("\nFinal board state:")
            game.print_board()
            return winner
        if game.is_full():
            # In simulation mode we force a winner via coin toss to create a Bernoulli outcome.
            if mode == "LLM_vs_LLM":
                coin = random.random()
                winner = 'X' if coin < 0.5 else 'O'
                print("\nBoard is full. Game ended in a draw. Deciding winner by coin toss...")
                print("\nFinal board state:")
                game.print_board()
                return winner
            else:
                print("\nFinal board state:")
                game.print_board()
                return "Draw"
        # Switch players
        last_move = move
        current_player = 'O' if current_player == 'X' else 'X'

def simulate_games(num_games, board_size):
    """
    Run 500 games automatically between two LLM agents (LLM_vs_LLM mode).
    Record the outcome for each game as a Bernoulli trial:
      1 -> win for LLM1 (ChatGPT playing as 'X')
      0 -> win for LLM2 (Claude playing as 'O')
    Saves the outcomes to "Exercise1.json" and plots the binomial distribution to "Exercise1.png".
    """
    outcomes = []  # 1 if LLM1 wins, 0 if LLM2 wins

    for i in range(num_games):
        winner = play_game(board_size, mode="LLM_vs_LLM")
        if winner == 'X':
            outcomes.append(1)
        elif winner == 'O':
            outcomes.append(0)
        

    # Save outcomes and counts to a JSON file
    results = {
        "LLM1_wins": outcomes.count(1),
        "LLM2_wins": outcomes.count(0),
        "total_games": num_games,
        "outcomes": outcomes
    }
    with open("Exercise1.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSimulation complete. Results saved to Exercise1.json")

    # Plot the binomial distribution based on the outcomes
    n = num_games
    p_empirical = outcomes.count(1) / n  # empirical probability of LLM1 win
    x = np.arange(0, n + 1)
    binom_pmf = binom.pmf(x, n, p_empirical)

    plt.figure(figsize=(10, 6))
    plt.plot(x, binom_pmf, 'bo', ms=2, label='Binomial PMF')
    plt.vlines(x, 0, binom_pmf, colors='b', lw=1, alpha=0.5)
    plt.title('Binomial Distribution of LLM1 Wins in 500 Games')
    plt.xlabel('Number of LLM1 Wins')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig("Exercise1.png")
    plt.close()
    print("Binomial distribution plot saved to Exercise1.png")
    return results

def main():
    try:
        board_size = int(input("Enter board size (N for NxN board, e.g., 3 for 3x3): "))
    except ValueError:
        print("Invalid board size input. Exiting.")
        return

    print("\nSelect game mode:")
    print("1. LLM vs LLM")
    print("2. LLM vs Human")
    print("3. Simulation mode (500 automatic LLM vs LLM games)")
    mode_input = input("Enter mode number (1/2/3): ")

    if mode_input == "1":
        winner = play_game(board_size, mode="LLM_vs_LLM")
        if winner in ['X', 'O']:
            print(f"\nWinner is: {'ChatGPT' if winner == 'X' else 'Claude'}")
        else:
            print("Game ended in a draw.")
    elif mode_input == "2":
        winner = play_game(board_size, mode="LLM_vs_Human")
        if winner == "Draw":
            print("Game ended in a draw.")
        else:
            print(f"\nWinner is: {winner}")
    elif mode_input == "3":
        simulate_games(500, board_size)
    else:
        print("Invalid mode selected. Exiting.")

if __name__ == "__main__":
    main()
