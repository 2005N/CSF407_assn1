import random
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
import os
from mistralai import Mistral

# Import Gemini and Anthropic libraries
from google import genai
import anthropic

# Create a Gemini client (agent for player X)
gemini_client = genai.Client(api_key="AIzaSyArkKhE3_VargUQs3HRKDC9N8OqvywVDBQ")


# ---------------------
# Board and Game Classes
# ---------------------

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
        # Check rows
        for row in self.board:
            if row[0] != ' ' and all(cell == row[0] for cell in row):
                return row[0]
        # Check columns
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


# ---------------------
# LLM Utility Function
# ---------------------

def board_to_string(game):
    """Converts the current board into a string for the prompt."""
    lines = []
    for row in game.board:
        lines.append(" | ".join(row))
    return "\n".join(lines)


# ---------------------
# LLM Agent Move Functions
# ---------------------

def get_llm_move_gemini(board_state, last_move, player):
    """
    Uses the official Gemini Python client to get the next move.
    Expects the Gemini model to respond with a move in the format "row,col".
    """
    prompt = (
        f"You are playing tic-tac-toe. The board state is:\n{board_state}\n"
        f"The opponent's last move was: {last_move}. Your mark is '{player}'.\n"
        "Respond with your move in the format row,col. Only include the coordinates."
    )
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",  # Adjust if you have access to a different Gemini model
            contents=prompt,
        )
        move_str = response.text.strip()
        row, col = map(int, move_str.split(','))
        return (row, col)
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None


def get_llm_move_claude(board_state, last_move, player):
    """
    Uses the official Anthropic Python client to get the next move.
    Expects the model to return a move in the format "row,col".
    """
    prompt_text = (
        f"You are playing tic-tac-toe. The board state is:\n{board_state}\n"
        f"The opponent's last move was: {last_move}. Your mark is '{player}'.\n"
        "Respond with your move in the format row,col. Only include the coordinates."
    )
    try:
        client = anthropic.Client(api_key="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")
        # Create a message with the correct messages parameter structure
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=20,
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]
        )
        
        # Extract the move from the response
        move_str = message.content[0].text.strip()
        row, col = map(int, move_str.split(','))
        return (row, col)
    except Exception as e:
        print(f"Anthropic API error: {e}")
        return None

def get_llm_move_mistral(board_state, last_move, player):
    """
    Uses the Mistral AI Python client to get the next move.
    Expects the model to return a move in the format "row,col".
    """
    prompt_text = (
        f"You are playing tic-tac-toe. The board state is:\n{board_state}\n"
        f"The opponent's last move was: {last_move}. Your mark is '{player}'.\n"
        "Respond with your move in the format row,col. Only include the coordinates."
    )
    try:
        client = Mistral(api_key="T8VEqzz4oYeK3affFYmu09iiBpAZJgvQ")
        # Create a chat completion
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]
        )
        
        # Extract the move from the response
        move_str = response.choices[0].message.content.strip()
        row, col = map(int, move_str.split(','))
        return (row, col)
    except Exception as e:
        print(f"Mistral API error: {e}")
        return None

def llm_agent_move(game, player, last_move, agent_name):
    """
    Determines which LLM to call based on the agent_name.
    Falls back to a random move if the LLM call fails or produces an invalid move.
    """
    board_str = board_to_string(game)
    move = None
    if agent_name == "Gemini":
        move = get_llm_move_gemini(board_str, last_move, player)
    elif agent_name == "Mistral":
        move = get_llm_move_mistral(board_str, last_move, player)
    
    # Fallback: if LLM move is None or invalid, choose a random valid move
    if move is None or not game.is_valid_move(move[0], move[1]):
        valid_moves = [(r, c) for r in range(game.size) for c in range(game.size) if game.board[r][c] == ' ']
        if valid_moves:
            move = random.choice(valid_moves)
    return move


def human_agent_move(game):
    """Prompts the human user to enter a move in 'row,col' format."""
    while True:
        move_str = input("Enter your move as row,col (e.g., 0,1): ")
        try:
            row, col = map(int, move_str.split(','))
            if game.is_valid_move(row, col):
                return (row, col)
            else:
                print("Invalid move. That cell is either occupied or out of bounds.")
        except:
            print("Invalid input format. Please enter your move as row,col.")


# ---------------------
# Main Game Logic
# ---------------------

def play_game(board_size, mode="LLM_vs_LLM"):
    """
    Plays a single game of tic-tac-toe.
      - mode="LLM_vs_LLM": Gemini (X) vs Mistral (O)
      - mode="LLM_vs_Human": Gemini (X) vs Human (O)
    """
    game = TicTacToe(board_size)
    current_player = 'X'
    last_move = None

    if mode == "LLM_vs_LLM":
        agent_X = "Gemini"
        agent_O = "Mistral"
    else:
        agent_X = "Gemini"  # LLM agent as Gemini for player X
        agent_O = "Human"

    while True:
       # print("\nCurrent board state:")
       # game.print_board()
        if mode == "LLM_vs_LLM" or (mode == "LLM_vs_Human" and current_player == 'X'):
            agent_name = agent_X if current_player == 'X' else agent_O
           # print(f"{agent_name} ({current_player}) is thinking... (Opponent's last move: {last_move})")
            move = llm_agent_move(game, current_player, last_move, agent_name)
        else:
            move = human_agent_move(game)

        if move is None:
            print("No valid moves available!")
            return "Draw"

        game.apply_move(move[0], move[1], current_player)
        winner = game.check_winner()
        if winner:
            print("\nFinal board state:")
            game.print_board()
            return winner
        if game.is_full():
            # For LLM vs LLM, resolve draws with a coin toss
            if mode == "LLM_vs_LLM":
                coin = random.random()
                winner = 'X' if coin < 0.5 else 'O'
                print("\nBoard is full. Game ended in a draw. Deciding winner by coin toss...")
               # print("\nFinal board state:")
               # game.print_board()
                return winner
            else:
                # print("\nFinal board state:")
               # game.print_board()
                return "Draw"

        last_move = move
        current_player = 'O' if current_player == 'X' else 'X'


def simulate_games(num_games, board_size):
    """
    Runs multiple games automatically between Gemini (X) and Mistral (O).
    Records outcomes (1 for Gemini win, 0 for Mistral win), saves results to Exercise1.json,
    and plots a binomial distribution saved as Exercise1.png.
    """
    outcomes = []
    for i in range(num_games):
        winner = play_game(board_size, mode="LLM_vs_LLM")
        if winner == 'X':
            outcomes.append(1)
        elif winner == 'O':
            outcomes.append(0)
        if (i+1) % 50 == 0:
            print(f"Completed {i+1} games.")

    results = {
        "Gemini_wins": outcomes.count(1),
        "Mistral_wins": outcomes.count(0),
        "total_games": num_games,
        "outcomes": outcomes
    }
    with open("Exercise1.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSimulation complete. Results saved to Exercise1.json")

    n = num_games
    p_empirical = outcomes.count(1) / n
    x = np.arange(0, n + 1)
    binom_pmf = binom.pmf(x, n, p_empirical)

    plt.figure(figsize=(10, 6))
    plt.plot(x, binom_pmf, 'bo', ms=2, label='Binomial PMF')
    plt.vlines(x, 0, binom_pmf, colors='b', lw=1, alpha=0.5)
    plt.title('Binomial Distribution of Gemini (X) Wins')
    plt.xlabel('Number of Gemini Wins')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig("Exercise1.png")
    plt.close()
    print("Binomial distribution plot saved to Exercise1.png")


def main():
    """
    Main entry point for running the game or simulation.
    """
    print("Welcome to the Tic Tac Toe AI Assignment!")
    try:
        board_size = int(input("Enter board size (N for NxN board, e.g., 3 for 3x3): "))
    except ValueError:
        print("Invalid board size input. Exiting.")
        return

    print("\nSelect game mode:")
    print("1. LLM vs LLM (Gemini vs Mistral)")
    print("2. LLM vs Human (Gemini vs You)")
    print("3. Simulation mode (500 automatic games)")
    mode_input = input("Enter mode number (1/2/3): ")

    if mode_input == "1":
        winner = play_game(board_size, mode="LLM_vs_LLM")
        if winner in ['X', 'O']:
            print(f"\nWinner is: {'Gemini' if winner == 'X' else 'Mistral'}")
        else:
            print("Game ended in a draw.")
    elif mode_input == "2":
        winner = play_game(board_size, mode="LLM_vs_Human")
        if winner == "Draw":
            print("Game ended in a draw.")
        else:
            print(f"\nWinner is: {winner}")
    elif mode_input == "3":
        simulate_games(100, board_size)
    else:
        print("Invalid mode selected. Exiting.")


if __name__ == "__main__":
    main()
