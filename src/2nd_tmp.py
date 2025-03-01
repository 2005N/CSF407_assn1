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
                    plt.text(c, r, 'E', ha='center', va='center')
        
        plt.xlim(-0.5, self.N-0.5)
        plt.ylim(self.N-0.5, -0.5)
        plt.title(f"Wumpus World (Step-{step})")
        plt.savefig(f"Wumpus_World_Step_{step}.png")
        plt.close()

class BayesianPitInference:
    def __init__(self, N, pit_prob=0.15):
        self.N = N
        self.pit_prob = pit_prob
        self.model = BayesianNetwork()
        
        # Create nodes for pits and breezes only (no stench nodes)
        self.pit_vars = []
        self.breeze_vars = []
        
        for r in range(N):
            for c in range(N):
                p_var = f"Pit_{r}_{c}"
                b_var = f"Breeze_{r}_{c}"
                self.pit_vars.append(p_var)
                self.breeze_vars.append(b_var)
        
        # Add all nodes to the model
        self.model.add_nodes_from(self.pit_vars + self.breeze_vars)
        
        # Add edges - breezes are affected only by adjacent pits
        for r in range(N):
            for c in range(N):
                b_var = f"Breeze_{r}_{c}"
                neighbors = self.get_adjacent_neighbors(r, c)
                for nr, nc in neighbors:
                    p_var = f"Pit_{nr}_{nc}"
                    self.model.add_edge(p_var, b_var)
        
        # Define CPDs
        self._define_cpds()
        
        # Create inference object
        self.inference = VariableElimination(self.model)
    
    def get_adjacent_neighbors(self, r, c):
        """Get only directly adjacent neighbors (up, down, left, right)"""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.N and 0 <= nc < self.N:
                neighbors.append((nr, nc))
        return neighbors
    
    def _define_cpds(self):
        # Define pit CPDs
        pit_cpds = []
        for r in range(self.N):
            for c in range(self.N):
                # Special case for (0,0) - we know there's no pit
                if (r, c) == (0, 0):
                    p_var = f"Pit_{r}_{c}"
                    cpd = TabularCPD(
                        variable=p_var,
                        variable_card=2,
                        values=[[1.0], [0.0]]  # [no pit, pit]
                    )
                else:
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
                neighbors = self.get_adjacent_neighbors(r, c)
                parent_vars = [f"Pit_{nr}_{nc}" for nr, nc in neighbors]
                
                if not parent_vars:  # No parents case
                    cpd = TabularCPD(
                        variable=b_var,
                        variable_card=2,
                        values=[[1.0], [0.0]]  # [no breeze, breeze]
                    )
                else:
                    num_parents = len(parent_vars)
                    cpd_size = 2 ** num_parents
                    false_row = []  # probability of no breeze
                    true_row = []   # probability of breeze
                    
                    for combo in range(cpd_size):
                        # Convert combo to binary, representing pit presence
                        bits = [(combo >> i) & 1 for i in range(num_parents)]
                        
                        if any(bits):  # If any adjacent cell has a pit
                            false_row.append(0.0)  # No chance of no breeze
                            true_row.append(1.0)   # Definitely a breeze
                        else:  # No pits in adjacent cells
                            false_row.append(1.0)  # Definitely no breeze
                            true_row.append(0.0)   # No chance of breeze
                    
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
        # Filter evidence to include both breeze and pit variables
        filtered_evidence = {}
        for var, value in evidence_dict.items():
            if (var.startswith("Breeze_") and var in self.breeze_vars) or (var.startswith("Pit_") and var in self.pit_vars):
                filtered_evidence[var] = value

        pit_prob_matrix = np.zeros((self.N, self.N))
        
        # Use filtered evidence for inference
        for r in range(self.N):
            for c in range(self.N):
                p_var = f"Pit_{r}_{c}"
                try:
                    query_res = self.inference.query(
                        variables=[p_var],
                        evidence=filtered_evidence,
                        show_progress=False
                    )
                    pit_prob_matrix[r, c] = query_res.values[1]  # Probability of pit
                except Exception as e:
                    #print(f"Inference error at {r},{c}: {e}")
                    pit_prob_matrix[r, c] = 0.5  # Fallback
                
                # Override with known evidence
                if p_var in filtered_evidence:
                    pit_prob_matrix[r, c] = filtered_evidence[p_var]
        
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
        self.path_history = [start]
        self.visit_counts = {start: 1}
        self.known_pits = set()  # Remember where pits are
        self.known_wumpus = set()

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
        
        safe_moves = [m for m in moves if m not in self.known_pits and m not in self.known_wumpus]
        
        if safe_moves:
            moves = safe_moves
        else:
            # If no safe moves are available, stay put
            return self.position

        N = pit_prob_matrix.shape[0]
        
        # Create a penalty matrix that discourages revisits
        visit_penalty = np.zeros((N, N))
        for pos in self.visited:
            r, c = pos
            # Count occurrences in path_history
            count = self.visit_counts.get(pos, 0)
            # Apply an increasing penalty for revisits, capped at 0.4
            visit_penalty[r, c] = min(0.4, count * 0.1)
        
        # Calculate total cost: pit probability + visit penalty
        total_cost = pit_prob_matrix.copy()
        for r in range(N):
            for c in range(N):
                # Add visit penalty to pit probability
                total_cost[r, c] += visit_penalty[r, c]
        
        # Define safety thresholds based on total cost
        safe_threshold = 0.3  # Consider cells with less than 30% total cost as safe
        danger_threshold = 0.6  # Consider cells with more than 60% total cost as dangerous
        
        # Categorize moves based on total cost
        safe_moves = [(mr, mc) for mr, mc in moves if total_cost[mr, mc] < safe_threshold]
        risky_moves = [(mr, mc) for mr, mc in moves if safe_threshold <= total_cost[mr, mc] < danger_threshold]
        dangerous_moves = [(mr, mc) for mr, mc in moves if total_cost[mr, mc] >= danger_threshold]
        
        # Decision strategy incorporating total cost
        # 1. Prefer unvisited safe moves
        unvisited_safe = [m for m in safe_moves if m not in self.visited]
        if unvisited_safe:
            return min(unvisited_safe, key=lambda m: total_cost[m[0], m[1]])
        
        # 2. Prefer any safe move based on total cost
        if safe_moves:
            return min(safe_moves, key=lambda m: total_cost[m[0], m[1]])
        
        # 3. Try unvisited risky moves
        unvisited_risky = [m for m in risky_moves if m not in self.visited]
        if unvisited_risky:
            return min(unvisited_risky, key=lambda m: total_cost[m[0], m[1]])
        
        # 4. Use any risky move with lowest total cost
        if risky_moves:
            return min(risky_moves, key=lambda m: total_cost[m[0], m[1]])
        
        # 5. If forced to use dangerous moves, choose the least dangerous
        if dangerous_moves:
            return min(dangerous_moves, key=lambda m: total_cost[m[0], m[1]])
        
        # Fallback: just choose move with lowest total cost
        return min(moves, key=lambda m: total_cost[m[0], m[1]])
    
    def update_position(self, new_position, world):
        """Update position and handle consequences"""
        r, c = new_position
        
        # Check if move is safe
        if world.is_pit(r, c):
            print(f"Fell into pit at {new_position}! Returning to {self.last_safe_position}")
            self.known_pits.add(new_position)
            self.position = self.last_safe_position
            self.path_history.append(self.last_safe_position)  # Record the return to safety
            return False
        elif world.is_wumpus(r, c):
            print(f"Encountered Wumpus at {new_position}! Returning to {self.last_safe_position}")
            self.known_wumpus.add(new_position)
            self.position = self.last_safe_position
            self.path_history.append(self.last_safe_position)  # Record the return to safety
            return False
        else:
            # Safe move
            self.position = new_position
            self.last_safe_position = new_position
            self.visited.add(new_position)
            self.path_history.append(new_position)  # Record the move
            
            # Update visit count
            self.visit_counts[new_position] = self.visit_counts.get(new_position, 0) + 1
            
            # Check for gold
            if world.is_gold(r, c):
                print(f"Found gold at {new_position}!")
                return True  # Success!
            return False  # Continue exploring


def run_strategy(world, strategy, N, max_steps=None):
    if max_steps is None:
        max_steps = 50*N
    
    # Initialize agents and inference
    bayes_infer = BayesianPitInference(N=N, pit_prob=0.15)
    agent = Agent(start=world.agent_start)
    
    step = 0
    evidence = {}
    done = False
    
    while not done and step < max_steps:
        r, c = agent.position
        
        # Sense environment
        breeze = world.sense_breeze(r, c)
        stench = world.sense_stench(r, c)
        
        # Update evidence - only breeze for now since we don't model stench
        evidence[f"Breeze_{r}_{c}"] = 1 if breeze else 0
        for pos in agent.visited:
            evidence[f"Pit_{pos[0]}_{pos[1]}"] = 0  # Visited and safe
        for pos in agent.known_pits:
            evidence[f"Pit_{pos[0]}_{pos[1]}"] = 1  # Known pits
        
        # Update probabilities
        try:
            pit_probs = bayes_infer.update_inference(evidence)
            # Create plot
            plot_pit_probabilities(pit_probs, step)
        except Exception as e:
            print(f"Error in inference: {e}")
            # Use a simple probability matrix with fixed values
            pit_probs = np.ones((N, N)) * 0.15
            pit_probs[0, 0] = 0.0  # Starting point is safe
            for visited_pos in agent.visited:
                pit_probs[visited_pos[0], visited_pos[1]] = 0.0  # Visited cells are safe
        
        # Choose next move based on strategy
        if strategy == "random":
            next_move = agent.choose_random_move(N)
        else:  # bayesian strategy
            next_move = agent.choose_best_move(pit_probs)
        
        print(f"Step {step}: Moving from {agent.position} to {next_move} ({strategy} move)")
        
        # Update position and check if done
        done = agent.update_position(next_move, world)
        
        step += 1
    
    if done:
        print(f"SUCCESS! {strategy.title()} strategy found gold in {step} steps.")
        return {"success": True, "steps": step}
    else:
        print(f"FAILED! {strategy.title()} strategy did not find gold within {max_steps} steps.")
        return {"success": False, "steps": max_steps}

def main():
    # Get world size input
    while True:
        try:
            N = int(input("Enter the size of the Wumpus World (N>=4): "))
            if N >= 4:
                break
            else:
                print("Please enter a value greater than or equal to 4.")
        except ValueError:
            print("Please enter a valid integer.")
    
    # Set seed for reproducibility
    seed = 42
    random.seed(seed)
    
    # Create world
    world = WumpusWorld(N=N, pit_probability=0.15, seed=seed)
    
    # Print actual world
    print("World layout:")
    for r in range(N):
        row_str = ""
        for c in range(N):
            row_str += world.grid[r][c] + " "
        print(row_str)
    
    # # Run random strategy
    # print("\n=== Running RANDOM strategy ===")
    # random_result = run_strategy(world, "random", N)
    
    # Run Bayesian strategy
    print("\n=== Running BAYESIAN strategy ===")
    bayesian_result = run_strategy(world, "bayesian", N)
    
    # # Compare results
    # print("\n=== RESULTS COMPARISON ===")
    # for strategy, result in [("Random", random_result), ("Bayesian", bayesian_result)]:
    #     outcome = "SUCCESS" if result["success"] else "FAILURE"
    #     print(f"{strategy} strategy: {outcome} ({result['steps']} steps)")

if __name__ == "__main__":
    main()