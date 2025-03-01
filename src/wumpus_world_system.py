import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

class WumpusWorld:
    def __init__(self, N=4, pit_probability=0.2, seed=42):
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


class BayesianInference:
    def __init__(self, N, pit_prob=0.2, seed=42):
        self.N = N
        self.pit_prob = pit_prob
        self.wumpus_prior = 1.0 / (N*N - 1) 
        self.model = BayesianNetwork()
        
        self.pit_vars = []
        self.breeze_vars = []
        self.wumpus_vars = []
        self.stench_vars = []
        
        for r in range(N):
            for c in range(N):
                p_var = f"Pit_{r}_{c}"
                b_var = f"Breeze_{r}_{c}"
                s_var = f"Stench_{r}_{c}"
                self.pit_vars.append(p_var)
                self.breeze_vars.append(b_var)
                self.stench_vars.append(s_var)
                if (r, c) != (0, 0):
                    w_var = f"Wumpus_{r}_{c}"
                    self.wumpus_vars.append(w_var)
        
        self.model.add_nodes_from(self.pit_vars + self.breeze_vars + self.wumpus_vars + self.stench_vars)
        
        for r in range(N):
            for c in range(N):
                b_var = f"Breeze_{r}_{c}"
                s_var = f"Stench_{r}_{c}"
                for nr, nc in self.get_adjacent_neighbors(r, c):
                    self.model.add_edge(f"Pit_{nr}_{nc}", b_var)
                for nr, nc in self.get_adjacent_neighbors(r, c):
                    if (nr, nc) != (0, 0):
                        self.model.add_edge(f"Wumpus_{nr}_{nc}", s_var)
        
        self._define_cpds()
        self.inference = VariableElimination(self.model)
    
    def get_adjacent_neighbors(self, r, c):
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.N and 0 <= nc < self.N:
                neighbors.append((nr, nc))
        return neighbors
    
    def _define_cpds(self):
        pit_cpds = []
        for r in range(self.N):
            for c in range(self.N):
                if (r, c) == (0, 0):
                    cpd = TabularCPD(f"Pit_{r}_{c}", 2, [[1.0], [0.0]])
                else:
                    cpd = TabularCPD(f"Pit_{r}_{c}", 2, [[1-self.pit_prob], [self.pit_prob]])
                pit_cpds.append(cpd)
        
        wumpus_cpds = []
        for var in self.wumpus_vars:
            cpd = TabularCPD(var, 2, [[1 - self.wumpus_prior], [self.wumpus_prior]])
            wumpus_cpds.append(cpd)
        
        breeze_cpds = []
        for r in range(self.N):
            for c in range(self.N):
                b_var = f"Breeze_{r}_{c}"
                parents = [f"Pit_{nr}_{nc}" for nr, nc in self.get_adjacent_neighbors(r, c)]
                if not parents:
                    cpd = TabularCPD(b_var, 2, [[1.0], [0.0]])
                else:
                    values = [[0.0 if any(combo) else 1.0 for combo in itertools.product([0,1], repeat=len(parents))],
                              [1.0 if any(combo) else 0.0 for combo in itertools.product([0,1], repeat=len(parents))]]
                    cpd = TabularCPD(b_var, 2, values, parents, [2]*len(parents))
                breeze_cpds.append(cpd)
        
        stench_cpds = []
        for r in range(self.N):
            for c in range(self.N):
                s_var = f"Stench_{r}_{c}"
                parents = [f"Wumpus_{nr}_{nc}" for nr, nc in self.get_adjacent_neighbors(r, c) if (nr, nc) != (0,0)]
                if not parents:
                    cpd = TabularCPD(s_var, 2, [[1.0], [0.0]])
                else:
                    values = [[0.0 if any(combo) else 1.0 for combo in itertools.product([0,1], repeat=len(parents))],
                              [1.0 if any(combo) else 0.0 for combo in itertools.product([0,1], repeat=len(parents))]]
                    cpd = TabularCPD(s_var, 2, values, parents, [2]*len(parents))
                stench_cpds.append(cpd)
        
        self.model.add_cpds(*(pit_cpds + wumpus_cpds + breeze_cpds + stench_cpds))
        self.model.check_model()
    
    def update_inference(self, evidence_dict):
        filtered_evidence = {k: v for k, v in evidence_dict.items() if k in self.model.nodes()}
        pit_probs = np.zeros((self.N, self.N))
        wumpus_probs = np.zeros((self.N, self.N))
        
        for r in range(self.N):
            for c in range(self.N):
                pit_var = f"Pit_{r}_{c}"
                if pit_var not in filtered_evidence:
                    query = self.inference.query(variables=[pit_var], evidence=filtered_evidence)
                    pit_probs[r, c] = query.values[1]
                else:
                    pit_probs[r, c] = filtered_evidence[pit_var]
                    
                if (r, c) != (0, 0):
                    wumpus_var = f"Wumpus_{r}_{c}"
                    if wumpus_var not in filtered_evidence:
                        query = self.inference.query(variables=[wumpus_var], evidence=filtered_evidence)
                        wumpus_probs[r, c] = query.values[1]
                    else:
                        wumpus_probs[r, c] = filtered_evidence[wumpus_var]
        
        return pit_probs, wumpus_probs

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


def plot_wumpus_probabilities(prob_matrix, step):
    """Plot wumpus probabilities"""
    plt.figure(figsize=(5, 4))
    plt.imshow(prob_matrix, cmap='Reds', origin='upper', vmin=0, vmax=1)
    plt.colorbar(label='Probability of Wumpus')
    
    for r in range(prob_matrix.shape[0]):
        for c in range(prob_matrix.shape[1]):
            plt.text(c, r, f"{prob_matrix[r,c]:.2f}",
                     ha="center", va="center", color="black")
    
    plt.title(f"Wumpus Probability Heatmap\nIn Step-{step}")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.savefig(f"Wumpus_Probability_Heatmap_Step_{step}.png")
    plt.close()


def plot_danger_probabilities(pit_matrix, wumpus_matrix, step):
    """Plot combined danger probabilities (pits + wumpus)"""
    # P(dangerous) = P(pit OR wumpus) = P(pit) + P(wumpus) - P(pit AND wumpus)
    danger_matrix = pit_matrix + wumpus_matrix - (pit_matrix * wumpus_matrix)
    
    # plt.figure(figsize=(5, 4))
    # plt.imshow(danger_matrix, cmap='RdYlGn_r', origin='upper', vmin=0, vmax=1)
    # plt.colorbar(label='Probability of Danger')
    
    # for r in range(danger_matrix.shape[0]):
    #     for c in range(danger_matrix.shape[1]):
    #         plt.text(c, r, f"{danger_matrix[r,c]:.2f}",
    #                  ha="center", va="center", color="black")
    
    # plt.title(f"Combined Danger Probability\nIn Step-{step}")
    # plt.xlabel("Column")
    # plt.ylabel("Row")
    # plt.tight_layout()
    # plt.savefig(f"Danger_Probability_Heatmap_Step_{step}.png")
    # plt.close()
    
    return danger_matrix


class Agent:
    def __init__(self, start=(0,0)):
        self.position = start
        self.visited = {start}
        self.last_safe_position = start
        self.path_history = [start]
        self.visit_counts = {start: 1}
        self.known_pits = set()
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
            return self.position

        N = pit_prob_matrix.shape[0]
        
        # Create a penalty matrix that discourages revisits
        visit_penalty = np.zeros((N, N))
        for pos in self.visited:
            r, c = pos
            count = self.visit_counts.get(pos, 0)
            visit_penalty[r, c] = min(0.4, count * 0.1)
        
        total_cost = pit_prob_matrix.copy()
        for r in range(N):
            for c in range(N):
                total_cost[r, c] += visit_penalty[r, c]
        
        safe_threshold = 0.3  # Consider cells with less than 30% total cost as safe
        danger_threshold = 0.6  # Consider cells with more than 60% total cost as dangerous
        
        safe_moves = [(mr, mc) for mr, mc in moves if total_cost[mr, mc] < safe_threshold]
        risky_moves = [(mr, mc) for mr, mc in moves if safe_threshold <= total_cost[mr, mc] < danger_threshold]
        dangerous_moves = [(mr, mc) for mr, mc in moves if total_cost[mr, mc] >= danger_threshold]
        
        unvisited_safe = [m for m in safe_moves if m not in self.visited]
        if unvisited_safe:
            return min(unvisited_safe, key=lambda m: total_cost[m[0], m[1]])
        
        if safe_moves:
            return min(safe_moves, key=lambda m: total_cost[m[0], m[1]])
        
        unvisited_risky = [m for m in risky_moves if m not in self.visited]
        if unvisited_risky:
            return min(unvisited_risky, key=lambda m: total_cost[m[0], m[1]])
        
        if risky_moves:
            return min(risky_moves, key=lambda m: total_cost[m[0], m[1]])
        
        if dangerous_moves:
            return min(dangerous_moves, key=lambda m: total_cost[m[0], m[1]])
        
        # Fallback: just choose move with lowest total cost
        return min(moves, key=lambda m: total_cost[m[0], m[1]])
    
    def update_position(self, new_position, world):
        """Update position and handle consequences"""
        r, c = new_position
        
        if world.is_pit(r, c):
            print(f"Fell into pit at {new_position}! Returning to {self.last_safe_position}")
            self.known_pits.add(new_position)
            self.position = self.last_safe_position
            self.path_history.append(self.last_safe_position)  
            return False
        elif world.is_wumpus(r, c):
            print(f"Encountered Wumpus at {new_position}! Returning to {self.last_safe_position}")
            self.known_wumpus.add(new_position)
            self.position = self.last_safe_position
            self.path_history.append(self.last_safe_position)
            return False
        else:
            self.position = new_position
            self.last_safe_position = new_position
            self.visited.add(new_position)
            self.path_history.append(new_position) 
            
            self.visit_counts[new_position] = self.visit_counts.get(new_position, 0) + 1
            
            if world.is_gold(r, c):
                print(f"Found gold at {new_position}!")
                return True 
            return False  


def run_strategy(world, strategy, N, max_steps=500):
    agent = Agent()
    bayes_infer = BayesianInference(N)
    evidence = {}
    
    for step in range(max_steps):
        r, c = agent.position
        evidence.update({
            f"Breeze_{r}_{c}": int(world.sense_breeze(r, c)),
            f"Stench_{r}_{c}": int(world.sense_stench(r, c))
        })
        for pos in agent.visited:
            evidence[f"Pit_{pos[0]}_{pos[1]}"] = 0
            if pos != (0,0):
                evidence[f"Wumpus_{pos[0]}_{pos[1]}"] = 0
        
        pit_probs, wumpus_probs = bayes_infer.update_inference(evidence)
        plot_pit_probabilities(pit_probs, step)
        #plot_wumpus_probabilities(wumpus_probs, step)
        
        if strategy == "random":
            next_move = agent.choose_random_move(N)
        else: 
            next_move = agent.choose_best_move(pit_probs)

        done = agent.update_position(next_move, world)
        
        if done:
            print(f"Gold found in {step} steps!")
            return True
    return False


def main():
    while True:
        try:
            N = int(input("Enter the size of the Wumpus World (N>=4): "))
            if N >= 4:
                break
            else:
                print("Please enter a value greater than or equal to 4.")
        except ValueError:
            print("Please enter a valid integer.")
    
    seed = 42
    random.seed(seed)
    
    world = WumpusWorld(N=N, pit_probability=0.2, seed=seed)
    
    print("World layout:")
    for r in range(N):
        row_str = ""
        for c in range(N):
            row_str += world.grid[r][c] + " "
        print(row_str)
    
    # # Uncomment to run random strategy
    # print("\n=== Running RANDOM strategy ===")
    # random_result = run_strategy(world, "random", N)
    
    # # Run Bayesian strategy
    print("\n=== Running BAYESIAN strategy ===")
    bayesian_result = run_strategy(world, "bayesian", N)
    
    # Uncomment to compare results
    # print("\n=== RESULTS COMPARISON ===")
    # for strategy, result in [("Random", random_result), ("Bayesian", bayesian_result)]:
    #     outcome = "SUCCESS" if result["success"] else "FAILURE"
    #     print(f"{strategy} strategy: {outcome} ({result['steps']} steps)")

if __name__ == "__main__":
    main()