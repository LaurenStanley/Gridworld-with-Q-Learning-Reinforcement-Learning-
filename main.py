import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld
from q_agent import Q_Agent
import time
import os
import math
import sys

# How often to sample data for plotting
sample_frequency = 0.1  # seconds

# Method for calculating the decay rate of epsilon


def get_epsilon(epsilon_initial, t, grid_size, flag, current_reward, max_time):
    if flag:
        time_remaining = max_time-t
        grid_epsilon = 1-np.exp(-np.sqrt(grid_size)/10)
        time_epsilon = 1-np.exp(-time_remaining/4)
        reward_epsilon = np.exp(-(current_reward+9)/7)
        epsilon = grid_epsilon*time_epsilon*reward_epsilon
        return epsilon
    else:
        decay_rate = 0.99997/(1 + np.exp(-((grid_size)/1.4)))
        return epsilon_initial * decay_rate


# Loop which keeps approaching the terminal state over time
def play(environment, per_action_reward,agent, pt4_flag=False, max_time=10, max_steps_per_episode=1000, learn=True, epsilon=0.9, epsilon_decay=True,decay_rate=0):
    print(epsilon_decay)
    #environment = GridWorld(filename)
    #environment.current_location = [(ind, environment.board[ind].index('S')) for ind in range(len(environment.board)) if 'S' in environment.board[ind]][0]
    reward_per_episode = []  # Initialise performance log
    init = time.time()
    epsilons = []
    mean_rewards = []
    cumulative_reward_array = []
    time_steps = np.arange(init, max_time + init +
                           sample_frequency, sample_frequency)
    current_timestep_index = 1
    agent.epsilon = epsilon
    print(epsilon)
        
    sum_rewards = []
    trial = 0
    cumulative_reward_array_mean = -9

    while (time.time() - init < max_time):
        cumulative_reward = 0  # Initialise values of each game
        step = 0
        game_over = False
        while step < max_steps_per_episode and game_over != True:  # Run until max steps or until game is finished
            old_state = environment.current_location
            action = agent.choose_action(environment.actions)
            reward = environment.make_step(action)
            new_state = environment.current_location
            
            if epsilon_decay == 'Decay':
                    agent.epsilon = get_epsilon(agent.epsilon, time.time(
                    ) - init, environment.height * environment.width, pt4_flag, cumulative_reward_array_mean, max_time)
                
            

            if learn == True:  # Update Q-values if learning is specified
                agent.learn(old_state, reward, new_state, action)

            cumulative_reward += reward
            step += 1
            agent.heat_map[new_state] = agent.heat_map[new_state] + 1

            if environment.check_state() == 'TERMINAL':
                if time.time() > time_steps[current_timestep_index]:
                    cumulative_reward_array_mean = np.mean(
                        cumulative_reward_array)
                    mean_rewards.append(cumulative_reward_array_mean)
                    epsilons.append(agent.epsilon)
                    cumulative_reward_array = []
                    current_timestep_index += 1
                    
                cumulative_reward_array.append(cumulative_reward)
                sum_rewards.append(cumulative_reward)
                trial += 1
                game_over = True
                environment = reset(environment, per_action_reward)

                # print(agent.epsilon)
                if epsilon_decay == 'Decay1':
                    agent.epsilon = agent.epsilon*decay_rate
                    #agent.epsilon = get_epsilon(agent.epsilon, time.time(
                    #) - init, environment.height * environment.width, pt4_flag, cumulative_reward_array_mean, max_time)
                elif epsilon_decay == 'Linear':
                    if agent.epsilon > 0.0001:
                        agent.epsilon = agent.epsilon - 0.0001

                

        # print(cumulative_reward)
        # Append reward for current trial to performance log
        reward_per_episode.append(cumulative_reward)
    #print("trial:",trial, "sum rewards:",sum(sum_rewards))
    avg_reward_per_trial = sum(sum_rewards)/trial
    # Return performance log
    return reward_per_episode, avg_reward_per_trial, mean_rewards, epsilons


def reset(environment,per_action_reward):
    environment.grid = np.zeros((environment.height, environment.width))-per_action_reward
    environment.find_terminal_states()
    environment.current_location = [(ind, environment.board[ind].index(
        'S')) for ind in range(len(environment.board)) if 'S' in environment.board[ind]][0]
    return environment


def showHeatMap(environment, heatmap):
    total = 0
    for i in range(environment.height):
        for j in range(environment.width):
            total += heatmap[(i, j)]

    for i in range(0, environment.height):
        print('-'*(9*environment.width + 1))
        out = '| '
        for j in range(0, environment.width):
            val = (i, j)
            if val in environment.barriers:
                out += 'X'.ljust(6) + ' | '
            elif val in environment.terminal_states:
                out += str(environment.board[i]
                           [j]).ljust(6) + ' | '
            else:
                out += str(round(100 *
                                 heatmap[(i, j)]/total, 2)).ljust(6) + ' | '
        print(out)
    print('-'*(9*len(environment.board[0])+1))


def showPolicy(environment, d):
    for i in range(0, environment.height):
        print('-'*(9*environment.width+1))
        out = '| '
        for j in range(0, environment.width):
            token = " "
            val = (i, j)
            utes = d[i, j]
            action = max(utes, key=utes.get)
            if val in environment.barriers:
                token = 'X'
            elif val in environment.terminal_states:
                token = str(environment.board[i][j])
            else:
                if action == "UP":
                    token = "^"
                if action == "DOWN":
                    token = "V"
                if action == "RIGHT":
                    token = ">"
                if action == "LEFT":
                    token = "<"

            out += token.ljust(6) + ' | '
        print(out)
    print('-'*(9*environment.width+1))


def main():
    # Initialize
    os.system('cls')
    # Initialize environment and agent

    # filename = './test1.txt'
    # max_time = 20
    # per_action_reward = -0.1
    # transition_success = 0.7

    filename = sys.argv[1]
    max_time = float(sys.argv[2])
    per_action_reward = float(sys.argv[3])
    transition_success = float(sys.argv[4])
    ignore_time = str(sys.argv[5])
    
    if ignore_time == 'True' or ignore_time == 'true':
        ignore_time = True
    elif ignore_time == 'False' or ignore_time == 'false':
        ignore_time = False

   
    environment = GridWorld(filename, per_action_reward, transition_success)
    

    #epsilons = [[0.9, 'Decay1',0.999],[0.9, 'Linear',1],[0.01,'Static',1],[0.1,'Static',1],[0.3,'Static',1],[0.9,'Static',1]]
    #epsilons = [[0.9, 'Decay1',.9999],[0.9,'Decay1',.999],[0.9,'Decay1',.99]]
    #epsilons = [[0.9,'Decay1',.999],[0.01,'Static',1]]

    #epsilons = [[0.9,'Decay1',.999],[0.9,'Decay',1]]

    epsilons = [[0.9, 'Decay',1],[0.01,'Static',1],[0.1,'Static',1],[0.3,'Static',1]]

    results = []
    epsilon_lists = []
    for epsilon in epsilons:
        # print(epsilon)
        # max time : how long the agent will explore the environment
        agentQ = Q_Agent(environment)
        reward_per_episode, avg_reward_per_trial, mean_rewards, epsilon_list = play(
            environment, per_action_reward,agentQ, pt4_flag, max_time, epsilon=epsilon[0], epsilon_decay=epsilon[1],decay_rate=epsilon[2])
        # print(reward_per_episode)
        results.append(mean_rewards)
        print("mean reward per trial:", avg_reward_per_trial)
        epsilon_lists.append(epsilon_list)
    # Simple learning curve
    for result in results:
        time = np.arange(0, len(result)*sample_frequency, sample_frequency)
        #plt.scatter(time, result)
        plt.plot(time, result, '.-')

    # plt.scatter(range(0,len(mean_rewards)),mean_rewards[::-1])
    plt.xlabel("Time (s)")
    plt.ylabel("Average Reward")
    plt.legend(epsilons)
    plt.title('Reward over Time')
    plt.savefig('reward.png')

    for epsilon_data in epsilon_lists:
        time = np.arange(0, len(epsilon_data) *
                         sample_frequency, sample_frequency)
        #plt.scatter(time, epsilon_data)
        plt.plot(time, epsilon_data, '.-')
    plt.xlabel("Time (s)")
    plt.ylabel("Epsilon")
    plt.title('Epsilon over Time')
    plt.legend(epsilons)
    plt.savefig('epsilon.png')

    print("HeatMap : ")
    showHeatMap(environment, agentQ.heat_map)
    print("Policy :")
    showPolicy(environment, agentQ.q_table)


if __name__ == "__main__":
    main()
