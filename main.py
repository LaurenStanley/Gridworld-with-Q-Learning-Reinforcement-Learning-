import numpy as np
import operator
import matplotlib.pyplot as plt
from gridworld import GridWorld
from randomagent import RandomAgent
from q_agent import Q_Agent


def play(environment, agent, trials=500, max_steps_per_episode=1000, learn=False):
    """The play function runs iterations and updates Q-values if desired."""
    reward_per_episode = [] # Initialise performance log
    
    for trial in range(trials): # Run trials
        cumulative_reward = 0 # Initialise values of each game
        step = 0
        game_over = False
        while step < max_steps_per_episode and game_over != True: # Run until max steps or until game is finished
            old_state = environment.current_location
            action = agent.choose_action(environment.actions) 
            reward = environment.make_step(action)
            new_state = environment.current_location
            
            if learn == True: # Update Q-values if learning is specified
                agent.learn(old_state, reward, new_state, action)
                
            cumulative_reward += reward
            step += 1
            
            if environment.check_state() == 'TERMINAL': # If game is in terminal state, game over and start next trial
                environment.__init__()
                game_over = True     
                
        reward_per_episode.append(cumulative_reward) # Append reward for current trial to performance log
        
    return reward_per_episode # Return performance log

def pretty(d, indent=0):
        for key, value in d.items():
            print('\t' * indent + str(key))
            if isinstance(value, dict):
                pretty(value, indent+1)
            else:
                print('\t' * (indent+1) + str(value))

def main():
    env = GridWorld()
    agent = RandomAgent()

    print("Current position of the agent =", env.current_location)
    print(env.agent_on_map())
    available_actions = env.get_available_actions()
    print("Available_actions =", available_actions)
    chosen_action = agent.choose_action(available_actions)
    print("Randomly chosen action =", chosen_action)
    reward = env.make_step(chosen_action)
    print("Reward obtained =", reward)
    print("Current position of the agent =", env.current_location)
    print(env.agent_on_map())

    # Initialize environment and agent
    environment = GridWorld()
    random_agent = RandomAgent()

    reward_per_episode = play(environment, random_agent, trials=500)

    # Simple learning curve
    plt.plot(reward_per_episode)

    environment = GridWorld()
    agentQ = Q_Agent(environment)

    # Note the learn=True argument!
    reward_per_episode = play(environment, agentQ, trials=500, learn=True)

    # Simple learning curve
    plt.plot(reward_per_episode)

    
    pretty(agentQ.q_table)

if __name__ == "__main__":
    main()