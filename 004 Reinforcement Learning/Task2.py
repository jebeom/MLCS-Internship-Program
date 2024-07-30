############################################
# Title     : Value Iteration
# Author    : Jebeom Chae 
# Date      : 2024-07-22
############################################

import numpy as np
import gym

env = gym.make('FrozenLake-v1', is_slippery=False)

# parameter
n_states = env.observation_space.n
n_actions = env.action_space.n
gamma = 0.99  # discount factor
threshold = 1e-8  

def visualize_environment(env):
    state_matrix = np.arange(env.observation_space.n).reshape((4, 4))

    state_map = np.full_like(state_matrix, '', dtype=object)
    goal_state = 15
    hole_states = [5, 7, 11, 12]
    
    for i in range(4):
        for j in range(4):
            state = state_matrix[i, j]
            if state == goal_state:
                state_map[i, j] = 'G'  
            elif state in hole_states:
                state_map[i, j] = 'H'  
            else:
                state_map[i, j] = '.'  
    
    print("Environment Map:")
    for row in state_map:
        print(' '.join(row))
    print()

def value_iteration(env, gamma, threshold):
    V = np.zeros(n_states)
    iterations = 0
    
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            q_values = [sum([prob * (reward + gamma * V[next_state]) 
                             for prob, next_state, reward, done in env.P[s][a]])
                        for a in range(n_actions)]
            V[s] = max(q_values)
            delta = max(delta, np.abs(v - V[s]))
        
        iterations += 1
        if delta < threshold:
            break
    
    # Extract the optimal policy
    policy = np.zeros([n_states, n_actions])
    for s in range(n_states):
        q_values = [sum([prob * (reward + gamma * V[next_state]) 
                         for prob, next_state, reward, done in env.P[s][a]])
                    for a in range(n_actions)]
        best_action = np.argmax(q_values)
        policy[s, best_action] = 1.0
    
    return policy, V, iterations

optimal_policy, optimal_value_function, iterations = value_iteration(env, gamma, threshold)

visualize_environment(env)

print("Optimal Policy:")
print(optimal_policy)
print(f"\nValue iteration converged in {iterations} iterations\n")

actions = ["Left", "Down", "Right", "Up"]
optimal_actions = np.asarray([actions[np.argmax(optimal_policy[state])] for state in range(n_states)]).reshape((4, 4))

goal_state = 15
optimal_actions[goal_state // 4, goal_state % 4] = 'Goal'

hole_states = [5, 7, 11, 12]
for hole in hole_states:
    optimal_actions[hole // 4, hole % 4] = 'Hole'

print('at each state, chosen action is :\n{}'.format(optimal_actions))

print('\nOptimal Value Function:')
print(optimal_value_function.reshape((4, 4)))