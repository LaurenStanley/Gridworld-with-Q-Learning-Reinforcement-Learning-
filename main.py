import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld
from q_agent import Q_Agent
import time

 
# How often to sample data for plotting
sample_frequency = 0.25 # seconds

# Method for calculating the decay rate of epsilon
def get_epsilon(epsilon_initial,t, grid_size, flag):
    if flag: # based on time
        decay_rate = 1 / (1 + np.exp(-(t*10 / (grid_size))))
        return epsilon_initial * decay_rate
    else: 
        decay_rate =  1/ (1 + np.exp(-(grid_size/2)))
        print(decay_rate)
        return epsilon_initial * decay_rate

#  All the code functionality is hidden in here
# Loop which keeps approaching the terminal state over time
def play(environment, agent, max_time =5, max_steps_per_episode=1000, learn=True, epsilon = 0.9, epsilon_decay = True):
    print(epsilon_decay)
    #environment = GridWorld(filename)
    #environment.current_location = [(ind, environment.board[ind].index('S')) for ind in range(len(environment.board)) if 'S' in environment.board[ind]][0]
    reward_per_episode = [] # Initialise performance log
    init = time.time()
    epsilons = []
    mean_rewards = []
    cumulative_reward_array = []
    time_steps = np.arange(init,max_time + init+sample_frequency,sample_frequency)
    current_timestep_index = 1
    agent.epsilon = epsilon
    while (time.time() - init < max_time):
        cumulative_reward = 0 # Initialise values of each game
        step = 0
        game_over = False
        while step < max_steps_per_episode and game_over != True: # Run until max steps or until game is finished
            old_state = environment.current_location
            action = agent.choose_action(environment.actions) 
            reward = environment.make_step(action)
            new_state = environment.current_location
            #print(agent.epsilon)
            if epsilon_decay == 'Decay':
                # print('time:',time.time() - init)
                decay_rate = 0.99997/(1+ np.exp(-((environment.height*environment.width)/1.4)))
                agent.epsilon = agent.epsilon* decay_rate
                # agent.epsilon = get_epsilon(agent.epsilon,time.time() - init, environment.height * environment.width, False)
            elif epsilon_decay == 'Linear':
                if agent.epsilon > 0.000005:
                    agent.epsilon = agent.epsilon - 0.000005
                
            if learn == True: # Update Q-values if learning is specified
                agent.learn(old_state, reward, new_state, action)
                
            cumulative_reward += reward
            step += 1
            agent.heat_map[new_state] = agent.heat_map[new_state] + 1
            
            if environment.check_state() == 'TERMINAL': # If game is in terminal state, game over and start next trial
                if time.time() > time_steps[current_timestep_index]:
                    mean_rewards.append(np.mean(cumulative_reward_array))
                    epsilons.append(agent.epsilon)
                    cumulative_reward_array = []
                    current_timestep_index += 1
                    
                cumulative_reward_array.append(cumulative_reward)
                game_over = True
                environment = reset(environment)
                #trials += 1     
        #print(cumulative_reward)
        reward_per_episode.append(cumulative_reward) # Append reward for current trial to performance log
        
    return reward_per_episode, mean_rewards, epsilons # Return performance log

def reset(environment):
    environment.grid = np.zeros((environment.height, environment.width))-1
    environment.find_terminal_states()
    environment.current_location = [(ind, environment.board[ind].index('S')) for ind in range(len(environment.board)) if 'S' in environment.board[ind]][0]    
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
            val = (i,j)
            utes = d[i,j]
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
    # Initialize environment and agent
    filename = './test0.txt'
    environment = GridWorld(filename)
    agentQ = Q_Agent(environment)
    # epsilons = [[0.01, False],[0.1, False],[0.3, False],[0.99, False]]
    epsilons = [[0.9, 'Decay'],[0.9, 'Static'],[0.9,'Linear']]

    results = []
    epsilon_lists = []
    for epsilon in epsilons:
        # print(epsilon)
        # max time : how long the agent will explor the environment
        reward_per_episode, mean_rewards, epsilon_list = play(environment, agentQ, max_time= 10, epsilon = epsilon[0], epsilon_decay = epsilon[1] )
        results.append(mean_rewards)
        epsilon_lists.append(epsilon_list)
    # Simple learning curve
    
   
    for result in results:
        time = np.arange(0,len(result)*sample_frequency,sample_frequency)
        plt.scatter(time,result)

    #plt.scatter(range(0,len(mean_rewards)),mean_rewards[::-1])
    plt.xlabel("Time (s)")
    plt.ylabel("Average Reward")
    plt.legend(epsilons)
    plt.title('Reward over Time')
    plt.show()

    for epsilon_data in epsilon_lists:
        time = np.arange(0,len(epsilon_data)*sample_frequency,sample_frequency)
        plt.scatter(time, epsilon_data)
    plt.xlabel("Time (s)")
    plt.ylabel("Epsilon")
    plt.title('Epsilon over Time')
    plt.legend(epsilons)
   
    plt.show()
    print("HeatMap : ")
    showHeatMap(environment, agentQ.heat_map)
    print("Policy :")
    showPolicy(environment,agentQ.q_table)

if __name__ == "__main__":
    main()