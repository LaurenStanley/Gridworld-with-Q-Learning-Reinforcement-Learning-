import numpy as np
import csv 

class GridWorld:
    ## Initialise starting data
    def __init__(self, filename):
        # Set information about the gridworld
        self.board = self.read(filename)
        self.height = len(self.board)
        self.width = len(self.board[0])
        self.grid = np.zeros(( self.height, self.width)) - 1
        
        self.current_location = [(ind, self.board[ind].index('S')) for ind in range(len(self.board)) if 'S' in self.board[ind]][0]
     
        self.terminal_states = self.find_terminal_states()

        # Set available actions
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    def read(self, board_file):
        board_array = []
        with open(board_file,newline = '') as board:
            board_data = csv.reader(board, delimiter='\t')
            for row in board_data:
                board_array.append(row)
        return board_array
        
    ## Put methods here:
    def get_available_actions(self):
        """Returns possible actions"""
        return self.actions
    
    def agent_on_map(self):
        """Prints out current location of the agent on the grid (used for debugging)"""
        grid = np.zeros(( self.height, self.width))
        grid[ self.current_location[0], self.current_location[1]] = 1
        return grid
    
    def get_reward(self, new_location):
        """Returns the reward for an input position"""
        return self.grid[ new_location[0]][new_location[1]]
        
    
    def make_step(self, action):
        """Moves the agent in the specified direction. If agent is at a border, agent stays still
        but takes negative reward. Function returns the reward for the move."""
        # Store previous location
        last_location = self.current_location
        
        # UP
        if action == 'UP':
            # If agent is at the top, stay still, collect reward
            if last_location[0] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0] - 1, self.current_location[1])
                reward = self.get_reward(self.current_location)
        
        # DOWN
        elif action == 'DOWN':
            # If agent is at bottom, stay still, collect reward
            if last_location[0] == self.height - 1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0] + 1, self.current_location[1])
                reward = self.get_reward(self.current_location)
            
        # LEFT
        elif action == 'LEFT':
            # If agent is at the left, stay still, collect reward
            if last_location[1] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0], self.current_location[1] - 1)
                reward = self.get_reward(self.current_location)

        # RIGHT
        elif action == 'RIGHT':
            # If agent is at the right, stay still, collect reward
            if last_location[1] == self.width - 1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0], self.current_location[1] + 1)
                reward = self.get_reward(self.current_location)
                
        return reward
    
    def check_state(self):
        """Check if the agent is in a terminal state (gold or bomb), if so return 'TERMINAL'"""
        if self.current_location in self.terminal_states:
            return 'TERMINAL'
    
    def find_terminal_states(self):
        terminal_states = []
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                try:
                    if float(self.board[i][j]) != 0:
                        terminal_states.append((i,j))
                        self.grid[i][j] = float(self.board[i][j])
                except ValueError:
                    continue
        return terminal_states