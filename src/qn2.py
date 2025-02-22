import numpy as np
import random
import matplotlib.pyplot as plt

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

class WumpusWorld:
    def __init__(self, N=4, pit_probability=0.15, seed=42):
        self.N = N
        self.rng = np.random.RandomState(seed)
        self.grid = [['E' for _ in range(N)] for _ in range(N)]
        
        # Place Wumpus randomly (avoiding (0,0))
        while True:
            wr, wc = self.rng.randint(0, N), self.rng.randint(0, N)
            if (wr, wc) != (0, 0):
                self.grid[wr][wc] = 'W'
                break
        
        # Place Gold randomly (avoiding (0,0) and Wumpus)
        while True:
            gr, gc = self.rng.randint(0, N), self.rng.randint(0, N)
            if (gr, gc) != (0, 0) and self.grid[gr][gc] == 'E':
                self.grid[gr][gc] = 'G'
                break
        
        # Place Pits randomly (avoiding (0,0), Wumpus and Gold)
        for r in range(N):
            for c in range(N):
                if (r, c) != (0, 0) and self.grid[r][c] == 'E':
                    if self.rng.rand() < pit_probability:
                        self.grid[r][c] = 'P'
        
        self.agent_start = (0, 0)
        
    def in_bounds(self, r, c):
        return 0 <= r < self.N and 0 <= c < self.N
    
    def sense_breeze(self, r, c):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc) and self.grid[nr][nc] == 'P':
                return True
        return False
    
    def sense_stench(self, r, c):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc) and self.grid[nr][nc] == 'W':
                return True
        return False
    
    def is_pit(self, r, c):
        return self.grid[r][c] == 'P'
    
    def is_wumpus(self, r, c):
        return self.grid[r][c] == 'W'
    
    def is_gold(self, r, c):
        return self.grid[r][c] == 'G'

    def plot_world(self, step):
        """Plot the actual world state"""
        plt.figure(figsize=(5, 4))
        plt.grid(True)
        for r in range(self.N):
            for c in range(self.N):
                if self.grid[r][c] == 'P':
                    plt.text(c, r, 'P', ha='center', va='center')
                elif self.grid[r][c] == 'W':
                    plt.text(c, r, 'W', ha='center', va='center')
                elif self.grid[r][c] == 'G':
                    plt.text(c, r, 'G', ha='center', va='center')
                else:
                    plt.text(c, r, '0', ha='center', va='center')
        
        plt.xlim(-0.5, self.N-0.5)
        plt.ylim(self.N-0.5, -0.5)
        plt.title(f"Wumpus World (Step-{step})")
        plt.savefig(f"Wumpus_World_Step_{step}.png")
        plt.close()

class BayesianPitInference:
    def __init__(self, N, pit_prob=0.15):
        self.N = N
        self.pit_prob = pit_prob
        self.max_depth = N//2 - 1  # Maximum depth for inference
        self.model = BayesianNetwork()
        
        # Create nodes only for cells within max_depth of any cell
        self.pit_vars = []
        self.breeze_vars = []
        
        for r in range(N):
            for c in range(N):
                p_var = f"Pit_{r}_{c}"
                b_var = f"Breeze_{r}_{c}"
                self.pit_vars.append(p_var)
                self.breeze_vars.append(b_var)
        
        self.model.add_nodes_from(self.pit_vars + self.breeze_vars)
        
        # Add edges with depth limitation
        for r in range(N):
            for c in range(N):
                b_var = f"Breeze_{r}_{c}"
                neighbors = self.get_neighbors_within_depth(r, c)
                for nr, nc in neighbors:
                    p_var = f"Pit_{nr}_{nc}"
                    self.model.add_edge(p_var, b_var)
        
        # Define CPDs
        self._define_cpds()
        
        # Create inference object
        self.inference = VariableElimination(self.model)
    
    def get_neighbors_within_depth(self, r, c):
        """Get neighbors within max_depth Manhattan distance"""
        neighbors = []
        for dr in range(-self.max_depth, self.max_depth + 1):
            for dc in range(-self.max_depth, self.max_depth + 1):
                if abs(dr) + abs(dc) <= self.max_depth and (dr != 0 or dc != 0):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.N and 0 <= nc < self.N:
                        neighbors.append((nr, nc))
        return neighbors
    
    def _define_cpds(self):
        # Define pit CPDs
        pit_cpds = []
        for r in range(self.N):
            for c in range(self.N):
                p_var = f"Pit_{r}_{c}"
                cpd = TabularCPD(
                    variable=p_var,
                    variable_card=2,
                    values=[[1 - self.pit_prob], [self.pit_prob]]
                )
                pit_cpds.append(cpd)
        
        # Define breeze CPDs
        breeze_cpds = []
        for r in range(self.N):
            for c in range(self.N):
                b_var = f"Breeze_{r}_{c}"
                neighbors = self.get_neighbors_within_depth(r, c)
                parent_vars = [f"Pit_{nr}_{nc}" for nr, nc in neighbors]
                
                if not parent_vars:  # No parents case
                    cpd = TabularCPD(
                        variable=b_var,
                        variable_card=2,
                        values=[[1.0], [0.0]]
                    )
                else:
                    num_parents = len(parent_vars)
                    cpd_size = 2 ** num_parents
                    false_row = []
                    true_row = []
                    
                    for combo in range(cpd_size):
                        bits = [(combo >> i) & 1 for i in range(num_parents)]
                        if any(bits):
                            false_row.append(0.0)
                            true_row.append(1.0)
                        else:
                            false_row.append(1.0)
                            true_row.append(0.0)
                    
                    cpd = TabularCPD(
                        variable=b_var,
                        variable_card=2,
                        values=[false_row, true_row],
                        evidence=parent_vars,
                        evidence_card=[2]*num_parents
                    )
                breeze_cpds.append(cpd)
        
        # Add all CPDs to model
        self.model.add_cpds(*(pit_cpds + breeze_cpds))
        self.model.check_model()
    
    def update_inference(self, evidence_dict):
        pit_prob_matrix = np.zeros((self.N, self.N))
        
        for r in range(self.N):
            for c in range(self.N):
                p_var = f"Pit_{r}_{c}"
                query_res = self.inference.query(
                    variables=[p_var],
                    evidence=evidence_dict,
                    show_progress=False
                )
                pit_prob_matrix[r, c] = query_res.values[1]
        
        return pit_prob_matrix

def plot_pit_probabilities(prob_matrix, step):
    """Plot pit probabilities matching the example format"""
    plt.figure(figsize=(5, 4))
    plt.imshow(prob_matrix, cmap='RdBu_r', origin='upper', vmin=0, vmax=1)
    plt.colorbar(label='Probability of Pit')
    
    for r in range(prob_matrix.shape[0]):
        for c in range(prob_matrix.shape[1]):
            plt.text(c, r, f"{prob_matrix[r,c]:.2f}",
                     ha="center", va="center", color="black")
    
    plt.title(f"Pit Probability Heatmap\nIn Step-{step}")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.savefig(f"Pit_Probability_Heatmap_Step_{step}.png")
    plt.close()

class Agent:
    def __init__(self, start=(0,0)):
        self.position = start
        self.visited = {start}
        self.last_safe_position = start
    
    def get_possible_moves(self, N):
        r, c = self.position
        moves = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N:
                moves.append((nr, nc))
        return moves
    
    def choose_random_move(self, N):
        moves = self.get_possible_moves(N)
        return random.choice(moves) if moves else self.position
    
    def choose_best_move(self, pit_prob_matrix):
        moves = self.get_possible_moves(pit_prob_matrix.shape[0])
        if not moves:
            return self.position
        
        # Choose move with lowest pit probability
        best_prob = float('inf')
        best_moves = []
        for mr, mc in moves:
            prob = pit_prob_matrix[mr, mc]
            if prob < best_prob:
                best_prob = prob
                best_moves = [(mr, mc)]
            elif abs(prob - best_prob) < 1e-9:
                best_moves.append((mr, mc))
        
        # Prefer unvisited moves with equal probability
        unvisited_best = [m for m in best_moves if m not in self.visited]
        if unvisited_best:
            return random.choice(unvisited_best)
        return random.choice(best_moves)

def main():
    N = int(input("Enter the size of the Wumpus World (N>=4): "))
    world = WumpusWorld(N=N, pit_probability=0.15, seed=42)
    bayes_infer = BayesianPitInference(N=N, pit_prob=0.15)
    agent = Agent(start=world.agent_start)
    
    step = 0
    max_steps = 6*N
    evidence = {}
    done = False
    
    while step < max_steps and not done:
        r, c = agent.position
        
        # Sense environment
        breeze = world.sense_breeze(r, c)
        stench = world.sense_stench(r, c)
        
        # Update evidence
        evidence[f"Breeze_{r}_{c}"] = 1 if breeze else 0
        
        # Plot current state
        world.plot_world(step)
        
        # Update and plot probabilities
        pit_probs = bayes_infer.update_inference(evidence)
        plot_pit_probabilities(pit_probs, step)
        
        # # Try both random and best moves
        # if step % 2 == 0:  # Random move
        #     next_move = agent.choose_random_move(N)
        #     move_type = "random"
        # else:  # Best move
        #     next_move = agent.choose_best_move(pit_probs)
        #     move_type = "best"
        
        next_move = agent.choose_best_move(pit_probs) #choosing best move
        move_type = "best"
        
        print(f"Step {step}: Moving from {agent.position} to {next_move} ({move_type} move)")
        
        # Check if move is safe
        if world.is_pit(*next_move):
            print(f"Fell into pit at {next_move}! Returning to {agent.last_safe_position}")
            agent.position = agent.last_safe_position
        else:
            agent.position = next_move
            agent.last_safe_position = next_move
            agent.visited.add(next_move)
        
        # Check if found gold
        if world.is_gold(*agent.position):
            print(f"Found gold at {agent.position} after {step} steps!")
            done = True
        
        step += 1
    
    if not done:
        print(f"Failed to find gold within {max_steps} steps")

if __name__ == "__main__":
    main()