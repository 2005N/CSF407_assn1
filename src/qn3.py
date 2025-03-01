import logging
from tic_tac_toe import TicTacToe, play_game  # Tic-Tac-Toe implementation
from wumpus_world_system import WumpusWorld, BayesianInference, Agent  # Wumpus World implementation
import random
import matplotlib.pyplot as plt
import numpy as np

class IntegratedSimulation:
    def __init__(self, wumpus_size=4, ttt_size=3):
        self.wumpus_world = WumpusWorld(N=wumpus_size)
        self.bayes_infer = BayesianInference(N=wumpus_size, pit_prob=0.2)
        self.agent = Agent(start=self.wumpus_world.agent_start)
        self.ttt_size = ttt_size
        self.step = 0
        self.evidence = {}

    def play_ttt_game(self):
        """Play a single Tic-Tac-Toe game between Gemini (X) and Mistral (O)"""
        winner = play_game(self.ttt_size, mode="LLM_vs_LLM")
        return winner

    def make_wumpus_move(self, use_best_move):
        """Make a move in the Wumpus World based on the specified strategy"""
        r, c = self.agent.position
        breeze = self.wumpus_world.sense_breeze(r, c)
        stench = self.wumpus_world.sense_stench(r, c)
        
        self.evidence[f"Breeze_{r}_{c}"] = 1 if breeze else 0
        self.evidence[f"Stench_{r}_{c}"] = 1 if stench else 0
        
        for pos in self.agent.visited:
            self.evidence[f"Pit_{pos[0]}_{pos[1]}"] = 0
            if pos != (0, 0):  # Only add wumpus evidence for non-starting positions
                self.evidence[f"Wumpus_{pos[0]}_{pos[1]}"] = 0
        
        pit_probs, wumpus_probs = self.bayes_infer.update_inference(self.evidence)
        
        danger_probs = pit_probs + wumpus_probs - (pit_probs * wumpus_probs)
        
        if use_best_move:
            next_move = self.agent.choose_best_move(danger_probs)
            move_type = "best"
        else:
            moves = []
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.wumpus_world.N and 0 <= nc < self.wumpus_world.N:
                    moves.append((nr, nc))
            
            if not moves:  
                next_move = self.agent.position
            else:
                next_move = random.choice(moves)
            move_type = "random"
        
        self.agent.position = next_move
        self.agent.visited.add(next_move)
        
        self.plot_current_state(pit_probs, move_type)
        
        if self.wumpus_world.is_gold(*self.agent.position):
            return True
        
        if self.wumpus_world.is_pit(*self.agent.position) or self.wumpus_world.is_wumpus(*self.agent.position):
            print(f"Agent died at position {self.agent.position}!")
            self.agent.position = self.wumpus_world.agent_start
            return False
        
        return False

    def plot_current_state(self, pit_probs, move_type):
        """Plot the current state of both the world and probabilities"""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        for r in range(self.wumpus_world.N):
            for c in range(self.wumpus_world.N):
                cell_content = self.wumpus_world.grid[r][c]
                plt.text(c, r, cell_content, ha='center', va='center', fontsize=14)
        
        plt.plot(self.agent.position[1], self.agent.position[0], 'ro', markersize=10, label='Agent')
        
        for pos in self.agent.visited:
            if pos != self.agent.position:  
                plt.plot(pos[1], pos[0], 'go', alpha=0.3, markersize=8)
        
        plt.grid(True)
        plt.title(f"Wumpus World (Step {self.step}, {move_type} move)")
        plt.xlim(-0.5, self.wumpus_world.N - 0.5)
        plt.ylim(self.wumpus_world.N - 0.5, -0.5)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.imshow(pit_probs, cmap='RdBu_r', origin='upper', vmin=0, vmax=1)
        plt.colorbar(label='Pit Probability')
        
        for r in range(pit_probs.shape[0]):
            for c in range(pit_probs.shape[1]):
                plt.text(c, r, f"{pit_probs[r,c]:.2f}",
                        ha="center", va="center", color="black")
    
        plt.title(f"Pit Probability Heatmap\nIn Step-{self.step}")
        plt.tight_layout()
        plt.savefig(f"Integrated_Simulation_Step_{self.step}.png")
        plt.close()

    def run_simulation(self, max_steps=20):
        """Run the full integrated simulation"""
        print(f"Starting simulation with max {max_steps} steps")
        print(f"Initial agent position: {self.agent.position}")
        
        while self.step < max_steps:
            print(f"\nStep {self.step + 1}")
            
            ttt_winner = self.play_ttt_game()
            use_best_move = (ttt_winner == 'X')  # Gemini (X) wins -> use best move
            
            print(f"TTT Winner: {ttt_winner}, Using {'best' if use_best_move else 'random'} move")
            
            print(f"Agent position before move: {self.agent.position}")
            found_gold = self.make_wumpus_move(use_best_move)
            print(f"Agent position after move: {self.agent.position}")
            
            if found_gold:
                print(f"Found gold at step {self.step + 1}!")
                break
                
            self.step += 1
        
        if self.step >= max_steps:
            print(f"Reached maximum steps ({max_steps}) without finding gold.")



def main():
    logging.getLogger().setLevel(logging.WARNING)  
    
    board_size = int(input("Enter tic-tac-toe board size (N for NxN board): "))
    N = int(input("Enter the size of the Wumpus World (N>=4): "))
    
    sim = IntegratedSimulation(wumpus_size=N, ttt_size=board_size)
    sim.run_simulation(max_steps=300)

if __name__ == "__main__":
    main()