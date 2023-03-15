import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld
from q_agent import Q_Agent
import time


def get_epsilon(epsilon_initial,t, grid_size, flag):
    if flag: # based on time
        decay_rate = 1 / (1 + np.exp(-(t*10 / (grid_size))))
        return epsilon_initial * decay_rate
    else: 
        decay_rate =  1/ (1 + np.exp(-(grid_size / 10)))
        return epsilon_initial * decay_rate

def play(environment, agent, max_time =1, max_steps_per_episode=1000, learn=True, epsilon = 0.9, epsilon_decay = True):
    """The play function runs iterations and updates Q-values if desired."""
    #environment = GridWorld(filename)
    #environment.current_location = [(ind, environment.board[ind].index('S')) for ind in range(len(environment.board)) if 'S' in environment.board[ind]][0]
    reward_per_episode = [] # Initialise performance log
    init = time.time()
    
    mean_rewards = []
    mean_reward = []
    time_steps = np.arange(init,max_time + init+.1,.1)
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
            if epsilon_decay:
                print('time:',time.time() - init)
                agent.epsilon = get_epsilon(agent.epsilon,time.time() - init, environment.height * environment.width, True)

            if learn == True: # Update Q-values if learning is specified
                agent.learn(old_state, reward, new_state, action)
                
            cumulative_reward += reward
            step += 1
            agent.heat_map[new_state] = agent.heat_map[new_state] + 1
            
            if environment.check_state() == 'TERMINAL': # If game is in terminal state, game over and start next trial
                if time.time() > time_steps[current_timestep_index]:
                    mean_rewards.append(np.mean(mean_reward))
                    mean_reward = []
                    current_timestep_index += 1
                    #for i in range(0, environment.height):
                    #    for j in range(0, environment.width):
                    #        utes = agent.q_table[i,j]
                    #        action = max(utes, key=utes.get)
                    #        mean_reward.append(agent.q_table[i,j].get(action))
                    #                environment = reset(environment)
                mean_reward.append(cumulative_reward)
                game_over = True
                #trials += 1     
        #print(cumulative_reward)
        reward_per_episode.append(cumulative_reward) # Append reward for current trial to performance log
        
    return reward_per_episode, mean_rewards # Return performance log

def reset(environment):
    environment.grid = np.zeros((environment.height, environment.width))-1
    environment.find_terminal_states()
    environment.current_location = [(ind, environment.board[ind].index('S')) for ind in range(len(environment.board)) if 'S' in environment.board[ind]][0]    
    return environment

def pretty(d, indent=0):
        for key, value in d.items():
            print('\t' * indent + str(key))
            if isinstance(value, dict):
                pretty(value, indent+1)
            else:
                print('\t' * (indent+1) + str(value))

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
    epsilons = [[0.9, False]]

    results = []
    for epsilon in epsilons:
        # print(epsilon)
        # max time : how long the agent will explor the environment
        reward_per_episode, mean_rewards = play(environment, agentQ, max_time= 5, epsilon = epsilon[0], epsilon_decay = [1] )
        results.append(mean_rewards)
    # Simple learning curve
    
   
    for result in results:
        time = np.arange(0,len(result)/10,.1)
        plt.scatter(time,result)

    #plt.scatter(range(0,len(mean_rewards)),mean_rewards[::-1])
    plt.xlabel("Time (s)")
    plt.ylabel("Average Reward")
    plt.legend(epsilons)

    plt.show()
    # print(agentQ.q_table)
    # pretty(agentQ.q_table)
    print("HeatMap : ")
    showHeatMap(environment, agentQ.heat_map)
    print("Policy :")
    showPolicy(environment,agentQ.q_table)

if __name__ == "__main__":
    main()