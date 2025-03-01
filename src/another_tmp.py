import numpy as np
import matplotlib.pyplot as plt
import random
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import os

class WumpusWorld:
    def __init__(self, size):
        """
        Initialize the Wumpus World with a given size.
       
        Args:
            size (int): The size of the square grid (NxN).
        """
        # Ensure size is at least 4x4
        if size < 4:
            print("Minimum size is 4x4. Setting size to 4.")
            size = 4
       
        self.size = size
        self.agent_position = (0, 0)
        self.visited = np.zeros((size, size), dtype=bool)
        self.visited[0, 0] = True
       
        # Initialize the world
        # 0: Empty, 1: Pit, 2: Wumpus, 3: Gold, 4: Breeze, 5: Stench
        self.world = np.zeros((size, size), dtype=int)
        self.perceived = np.full((size, size), -1, dtype=int)  # -1 means not perceived yet
        self.perceived[0, 0] = 0  # Starting position is known to be safe
       
        # Create the world
        self._create_world()
       
        # Create the Bayesian Network
        self.bayesian_network = None
        self.create_bayesian_network()
       
        # Initialize variable elimination
        self.infer = VariableElimination(self.bayesian_network)
       
        # Track if the agent is alive and has found gold
        self.alive = True
        self.has_gold = False
       
        # Action history
        self.action_history = []
       
    def _create_world(self):
        """Create the Wumpus World with pits, wumpus, and gold."""
        # Place pits (about 20% of the cells)
        for i in range(self.size):
            for j in range(self.size):
                # Skip the starting position and adjacent cells
                if (i == 0 and j == 0) or (i == 0 and j == 1) or (i == 1 and j == 0):
                    continue
               
                # Place a pit with 20% probability
                if random.random() < 0.2:
                    self.world[i, j] = 1  # Pit
                   
                    # Add breeze to adjacent cells
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size and self.world[ni, nj] != 1:
                            self.world[ni, nj] = 4  # Breeze
       
        # Place the Wumpus (not in the starting position)
        while True:
            i, j = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if (i != 0 or j != 0) and self.world[i, j] == 0:  # Empty cell
                self.world[i, j] = 2  # Wumpus
               
                # Add stench to adjacent cells
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        if self.world[ni, nj] == 0:
                            self.world[ni, nj] = 5  # Stench
                        elif self.world[ni, nj] == 4:
                            self.world[ni, nj] = 6  # Both Breeze and Stench
                break
       
        # Place the gold (not in the starting position, not in a pit, not in the wumpus)
        while True:
            i, j = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if (i != 0 or j != 0) and self.world[i, j] == 0:  # Empty cell
                self.world[i, j] = 3  # Gold
                break
   
    def create_bayesian_network(self):
        """
        Create a Bayesian Network for the Wumpus World to model uncertainty.
        """
        # Create a Bayesian Network
        self.bayesian_network = BayesianNetwork()
       
        # Add nodes for each cell representing pit presence
        for i in range(self.size):
            for j in range(self.size):
                node = f"P_{i}_{j}"  # Pit at (i, j)
                self.bayesian_network.add_node(node)
       
        # Add edges based on breeze relationships
        for i in range(self.size):
            for j in range(self.size):
                breeze_node = f"B_{i}_{j}"  # Breeze at (i, j)
                self.bayesian_network.add_node(breeze_node)
               
                # Connect adjacent cells' pits to this breeze
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        pit_node = f"P_{ni}_{nj}"  # Pit at adjacent cell
                        self.bayesian_network.add_edge(pit_node, breeze_node)
       
        # Set the CPDs for pit nodes (prior probability of a pit)
        for i in range(self.size):
            for j in range(self.size):
                # Skip the starting position
                if i == 0 and j == 0:
                    cpd = TabularCPD(
                        variable=f"P_{i}_{j}",
                        variable_card=2,  # 0: No pit, 1: Pit
                        values=[[1.0], [0.0]]  # 0% chance of pit at (0,0)
                    )
                else:
                    cpd = TabularCPD(
                        variable=f"P_{i}_{j}",
                        variable_card=2,  # 0: No pit, 1: Pit
                        values=[[0.8], [0.2]]  # 20% chance of pit
                    )
                self.bayesian_network.add_cpds(cpd)
       
        # Set the CPDs for breeze nodes
        for i in range(self.size):
            for j in range(self.size):
                breeze_node = f"B_{i}_{j}"
               
                # Get adjacent cells
                adjacent_cells = []
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        adjacent_cells.append(f"P_{ni}_{nj}")
               
                if adjacent_cells:
                    # Evidence variables
                    evidence = adjacent_cells
                    evidence_card = [2] * len(evidence)  # Binary states for each pit
                   
                    # CPD values based on logical OR of pit presence
                    # Breeze is present if at least one adjacent cell has a pit
                    values = []
                    num_configs = 2 ** len(evidence)
                   
                    # For no breeze
                    no_breeze_row = []
                    for config in range(num_configs):
                        bin_config = format(config, f'0{len(evidence)}b')
                        has_pit = '1' in bin_config
                        no_breeze_row.append(0.0 if has_pit else 1.0)
                    values.append(no_breeze_row)
                   
                    # For breeze
                    breeze_row = []
                    for config in range(num_configs):
                        bin_config = format(config, f'0{len(evidence)}b')
                        has_pit = '1' in bin_config
                        breeze_row.append(1.0 if has_pit else 0.0)
                    values.append(breeze_row)
                   
                    cpd = TabularCPD(
                        variable=breeze_node,
                        variable_card=2,  # 0: No breeze, 1: Breeze
                        values=values,
                        evidence=evidence,
                        evidence_card=evidence_card
                    )
                   
                    self.bayesian_network.add_cpds(cpd)
   
    def update_beliefs(self):
        """
        Update the agent's beliefs about pit locations based on observations.
        """
        # Update with evidence from visited cells
        evidence = {}
        for i in range(self.size):
            for j in range(self.size):
                if self.visited[i, j]:
                    # We know there's no pit where we've been
                    evidence[f"P_{i}_{j}"] = 0
                   
                    # We know whether there's a breeze
                    is_breeze = self.world[i, j] == 4 or self.world[i, j] == 6
                    evidence[f"B_{i}_{j}"] = 1 if is_breeze else 0
       
        return evidence
   
    def calculate_pit_probabilities(self, evidence):
        """
        Calculate the probability of pits in unvisited cells using Bayesian inference.
       
        Args:
            evidence (dict): Known states of cells.
           
        Returns:
            numpy.ndarray: Grid of pit probabilities.
        """
        pit_probs = np.zeros((self.size, self.size))
       
        # Calculate probability for each unvisited cell
        for i in range(self.size):
            for j in range(self.size):
                if not self.visited[i, j]:
                    try:
                        query_result = self.infer.query(variables=[f"P_{i}_{j}"], evidence=evidence)
                        pit_probs[i, j] = query_result.values[1]  # Probability of pit
                    except Exception as e:
                        # If there's an issue with inference, use the prior probability
                        pit_probs[i, j] = 0.2
                else:
                    # We know there's no pit where we've been
                    pit_probs[i, j] = 0
       
        return pit_probs
   
    def get_safe_moves(self, pit_probs):
        """
        Get possible safe moves based on pit probabilities.
       
        Args:
            pit_probs (numpy.ndarray): Grid of pit probabilities.
           
        Returns:
            list: List of safe moves as (i, j) coordinates.
        """
        i, j = self.agent_position
        possible_moves = []
       
        # Check adjacent cells
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size and 0 <= nj < self.size:
                if not self.visited[ni, nj] and pit_probs[ni, nj] < 0.3:  # Less than 30% chance of pit
                    possible_moves.append((ni, nj))
                elif self.visited[ni, nj]:  # Already visited and safe
                    possible_moves.append((ni, nj))
       
        return possible_moves
   
    def make_best_move(self, pit_probs):
        """
        Make the best move based on pit probabilities.
       
        Args:
            pit_probs (numpy.ndarray): Grid of pit probabilities.
           
        Returns:
            tuple: New position after move.
        """
        safe_moves = self.get_safe_moves(pit_probs)
       
        if not safe_moves:
            # If no safe moves, take the least risky one
            i, j = self.agent_position
            least_risky = None
            min_risk = 1.0
           
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.size and 0 <= nj < self.size and not self.visited[ni, nj]:
                    if pit_probs[ni, nj] < min_risk:
                        min_risk = pit_probs[ni, nj]
                        least_risky = (ni, nj)
           
            if least_risky:
                return self.move_to(least_risky)
            else:
                # Backtrack to a previously visited cell
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size and self.visited[ni, nj]:
                        return self.move_to((ni, nj))
               
                # If still no move, stay in place
                return self.agent_position
        else:
            # Prioritize unvisited cells over revisits
            unvisited_safe_moves = [move for move in safe_moves if not self.visited[move[0], move[1]]]
           
            if unvisited_safe_moves:
                # Choose the safest unvisited move
                safest = min(unvisited_safe_moves, key=lambda move: pit_probs[move[0], move[1]])
                return self.move_to(safest)
            else:
                # Backtrack to an adjacent visited cell
                return self.move_to(safe_moves[0])
   
    def make_random_move(self):
        """
        Make a random move to an adjacent cell.
       
        Returns:
            tuple: New position after move, or None if move failed.
        """
        i, j = self.agent_position
        possible_moves = []
       
        # Get all possible moves
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size and 0 <= nj < self.size:
                possible_moves.append((ni, nj))
       
        if possible_moves:
            # Choose a random move
            new_pos = random.choice(possible_moves)
            return self.move_to(new_pos)
        else:
            # No possible moves
            return None
   
    def move_to(self, position):
        """
        Move the agent to a new position and update the game state.
       
        Args:
            position (tuple): The target position (i, j).
           
        Returns:
            tuple: The new position.
        """
        i, j = position
       
        # Record the action
        self.action_history.append(f"Move to ({i}, {j})")
       
        # Update the agent's position
        self.agent_position = (i, j)
       
        # Mark as visited
        self.visited[i, j] = True
       
        # Check what's in the cell
        cell_content = self.world[i, j]
       
        if cell_content == 1:  # Pit
            self.alive = False
            self.action_history.append("Fell into a pit! Game over.")
            return position
       
        if cell_content == 2:  # Wumpus
            self.alive = False
            self.action_history.append("Encountered the Wumpus! Game over.")
            return position
       
        if cell_content == 3:  # Gold
            self.has_gold = True
            self.action_history.append("Found the gold! Success!")
       
        # Update perceived state
        self.perceived[i, j] = cell_content
       
        return position
   
    def restart_from_last_position(self):
        """
        Restart the agent from the last position after a failed random move.
        """
        # Set the agent as alive again
        self.alive = True
       
        # Log the restart
        self.action_history.append(f"Restarting from position {self.agent_position}")
   
    def visualize_probabilities(self, pit_probs, step_num):
        """
        Visualize the pit probabilities and agent position with (0,0) at the bottom.
        Display probability values in each cell with improved visibility.
       
        Args:
            pit_probs (numpy.ndarray): Grid of pit probabilities.
            step_num (int): The step number for the filename.
        """
        plt.figure(figsize=(10, 8))
       
        # Flip the array vertically to put (0,0) at the bottom
        pit_probs_flipped = np.flipud(pit_probs)
       
        # Plot pit probabilities as a heatmap similar to the document example
        # Using a custom colormap from white to red
        im = plt.imshow(pit_probs_flipped, cmap='Reds', vmin=0, vmax=1.0,
                       interpolation='nearest', origin='lower')
        cbar = plt.colorbar(im, label='Probability of Pit')
        cbar.set_label('Probability of Pit', fontsize=12)
       
        # Create coordinate grids for the flipped world
        world_flipped = np.flipud(self.world)
        visited_flipped = np.flipud(self.visited)
       
        # Mark the agent's position
        i, j = self.agent_position
        # Convert to flipped coordinates
        i_flipped = self.size - 1 - i
        plt.plot(j, i_flipped, 'go', markersize=12, markeredgecolor='black')  # Green dot for agent
       
        # Display probability values in each cell
        for i in range(self.size):
            for j in range(self.size):
                # Convert to original coordinates for accessing the data
                orig_i = self.size - 1 - i
               
                # Display probability value in each cell
                prob_value = pit_probs[orig_i, j]
                # Only display probabilities in non-visited cells or cells with significant probability
                if not visited_flipped[i, j] or prob_value > 0.05:
                    # Choose text color based on background intensity for better readability
                    text_color = 'white' if prob_value > 0.5 else 'black'
                    # Center the text in the cell by setting ha='center' and va='center'
                    plt.text(j, i, f"{prob_value:.2f}", color=text_color, ha='center', va='center',
                            fontsize=10, fontweight='bold')
               
                # Mark visited cells with a subtle indicator
                if visited_flipped[i, j]:
                    plt.plot(j, i, 'kx', markersize=6, alpha=0.6)  # Black X for visited
       
        # Set proper tick marks and adjust their position to be at the edges of cells
        plt.xticks(np.arange(self.size), np.arange(self.size))
        plt.yticks(np.arange(self.size), np.arange(self.size))
       
        # Shift the ticks to be at the edges of the cells rather than the centers
        plt.gca().set_xticks(np.arange(-0.5, self.size, 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, self.size, 1), minor=True)
        plt.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.tick_params(which='minor', length=0)
       
        # Remove the default grid that might interfere with cell boundaries
        plt.grid(which='major', visible=False)
       
        plt.title(f'Pit Probabilities at Step {step_num}', fontsize=14)
        plt.xlabel('Column', fontsize=12)
        plt.ylabel('Row', fontsize=12)
       
        # Status info at the bottom
        status_text = f"Step: {step_num} | Position: {self.agent_position} | "
        status_text += "Gold Found!" if self.has_gold else "Searching for Gold"
        plt.figtext(0.5, 0.01, status_text, ha='center', fontsize=12)
       
        # Create output directory if it doesn't exist
        output_dir = "wumpus_visualizations"
        os.makedirs(output_dir, exist_ok=True)
       
        # Save the figure
        plt.savefig(os.path.join(output_dir, f'wumpus_step_{step_num}.png'), dpi=150, bbox_inches='tight')
        plt.close()

def run_simulation(world_size, max_steps=50, use_best_move=True):
    """
    Run a simulation of the Wumpus World.
   
    Args:
        world_size (int): Size of the world grid.
        max_steps (int): Maximum number of steps.
        use_best_move (bool): Whether to use the best move strategy or random moves.
       
    Returns:
        dict: Result of the simulation.
    """
    # Initialize the world
    world = WumpusWorld(world_size)
   
    # Print current working directory for reference
    print(f"Images will be saved to: {os.path.join(os.getcwd(), 'wumpus_visualizations')}")
   
    # Run the simulation
    step = 0
    while world.alive and not world.has_gold and step < max_steps:
        # Update beliefs
        evidence = world.update_beliefs()
       
        # Calculate pit probabilities
        pit_probs = world.calculate_pit_probabilities(evidence)
       
        # Visualize
        world.visualize_probabilities(pit_probs, step)
       
        # Make a move
        if use_best_move:
            world.make_best_move(pit_probs)
        else:
            old_position = world.agent_position
            world.make_random_move()
           
            # If the agent died, restart from the last position
            if not world.alive:
                world.agent_position = old_position
                world.restart_from_last_position()
       
        step += 1
   
    # Final visualization
    evidence = world.update_beliefs()
    pit_probs = world.calculate_pit_probabilities(evidence)
    world.visualize_probabilities(pit_probs, step)
   
    # Return results
    return {
        "steps": step,
        "found_gold": world.has_gold,
        "alive": world.alive,
        "visited_cells": np.sum(world.visited),
        "action_history": world.action_history
    }

def main():
    """Main function to run the Wumpus World simulation."""
    # Get user input for world size
    try:
        world_size = int(input("Enter the size of the Wumpus World (must be >= 4): "))
        if world_size < 4:
            print("Minimum size is 4. Setting to 4.")
            world_size = 4
    except ValueError:
        print("Invalid input. Using default size of 4.")
        world_size = 4
   
    # Run with best move strategy
    print("\nRunning simulation with best move strategy...")
    best_move_result = run_simulation(world_size, use_best_move=True)
   
    print("\nBest Move Strategy Results:")
    print(f"Steps taken: {best_move_result['steps']}")
    print(f"Found gold: {best_move_result['found_gold']}")
    print(f"Agent alive: {best_move_result['alive']}")
    print(f"Visited cells: {best_move_result['visited_cells']} out of {world_size * world_size}")
   
    # Run with random move strategy
    print("\nRunning simulation with random move strategy...")
    random_move_result = run_simulation(world_size, use_best_move=False)
   
    print("\nRandom Move Strategy Results:")
    print(f"Steps taken: {random_move_result['steps']}")
    print(f"Found gold: {random_move_result['found_gold']}")
    print(f"Agent alive: {random_move_result['alive']}")
    print(f"Visited cells: {random_move_result['visited_cells']} out of {world_size * world_size}")

if __name__ == "__main__":
    main()