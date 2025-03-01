# CSF407_2025_2022A7PS0207H
AI assignment-1

## Overview
This project implements an integrated simulation environment that combines two classic AI problems:

1.Wumpus World: A logical reasoning problem where an agent navigates a grid-based world containing hazards (pits and a wumpus) to find gold.
2.Tic-Tac-Toe: A strategic game played between two agents, where AI models from Gemini and Mistral compete against each other.

## Key Components
### Wumpus World

* A grid-based environment (configurable NxN size, recommended N≥4)
* Contains:
    Agent (starting at position 0,0)
    Gold (random location)
    Wumpus (random location)
    Pits (randomly placed with probability 0.2)

* The agent can sense:
    Breeze (adjacent to pits)
    Stench (adjacent to wumpus)

* Uses Bayesian inference to calculate probability distributions for hazard locations

### Tic-Tac-Toe

* Configurable board size (typically 3x3)
* Two AI players:
    Gemini (Player X)
    Mistral (Player O)
* Human can also play against an LLM agent

### Integrated Simulation

* For each simulation step:

1. A Tic-Tac-Toe game is played between the LLM agents
2. The winner determines the Wumpus World movement strategy:

    * If Gemini (X) wins: Use Bayesian-informed best move
    * If Mistral (O) wins: Use random movement

3. Agent moves in Wumpus World according to the selected strategy
4. Visualizations are generated showing the current state

## Requirements

* Python 3.8+
* Required packages:
    * numpy
    * matplotlib
    * pgmpy
    * scipy
    * mistralai
    * google-generativeai

## API Keys Setup
The simulation requires API keys for both Gemini and Mistral models:

1. Set your Gemini API key in the code (replace "GEMINI_KEY")
2. Set your Mistral API key in the code (replace "MISTRAL_KEY")

## Tic-Tac-Toe Modes:

* LLM vs LLM: Gemini (X) plays against Mistral (O)
* LLM vs Human: Play against Gemini
* Simulation Mode: Run multiple games automatically (e.g., 50 games)
#### Binomial Distribution for Gemini wins (for 50 games)
![image](https://github.com/user-attachments/assets/56dfa870-75e4-4ded-92b4-6c71a940f68d)

## Wumpus World Strategy Modes:

* Best Move: Uses Bayesian inference to assess danger probabilities and selects the safest path
* Random Move: Selects a random valid adjacent cell to explore
  To use random mode, uncomment the random strategy code part in main function
  
#### Few results for N=4
![image](https://github.com/user-attachments/assets/28126f32-a6af-4efd-af9e-d201d4ac41c6)

![image](https://github.com/user-attachments/assets/077ba80f-925e-43d7-87c5-affcb035a39b)

#### Few results for N=10
![image](https://github.com/user-attachments/assets/7e6f77be-d870-477e-b221-2101d630baeb)
![image](https://github.com/user-attachments/assets/4e26fb1a-4656-4a4d-91c2-eca7435ef4b2)
![image](https://github.com/user-attachments/assets/3d280723-c63c-464b-8bcc-b694e4840d2a)


## In Integrated Simulation:
### Visualization
The simulation generates visualizations for each step showing:
* Current Wumpus World state with agent position
* Heat map of pit probability distributions
* Information about the current move type (best or random)

Output images are saved as "Integrated_Simulation_Step_X.png" where X is the step number.

#### Few results for tic_tac_toe board size 3 and Wumpus board 4
![image](https://github.com/user-attachments/assets/ad8f8b58-cfb0-4b53-a9fa-fa5a12983ffb)
![image](https://github.com/user-attachments/assets/7e98d310-1610-4a41-8292-a70617e63c31)

![image](https://github.com/user-attachments/assets/1e29c25a-7364-4509-8e70-af37e0259148)

## Project Structure

* wumpus_world_system.py: Implementation of the Wumpus World environment, Bayesian inference model, and agent logic
* tic_tac_toe.py: Implementation of the Tic-Tac-Toe game and LLM agent interfaces
* integrated_simulation.py: Main simulation that combines both games
