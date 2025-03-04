import numpy as np

# Create a 40x41 array of zeros
grid_world = np.zeros((40, 41), dtype=object)
grid_world[0, 0] = 'S'

# Define the coordinates of the terminal states and the barrier states
terminals = [(3, 5), (20, 10), (35, 30), (15, 35),
(10, 5), (26, 11), (30, 30), (25, 25),
(3, 5), (20, 10), (17, 15), (14, 40)]
barriers = [(2, 40), (6, 5), (10, 20), (18, 13)]

# Set random rewards for the terminal states and mark the barrier states with 'X'
for block in terminals:
    reward = np.random.randint(-9, 9)
    grid_world[block[0], block[1]] = reward
for block in barriers:
    grid_world[block[0], block[1]] = 'X'

# Print the grid world array
for row in grid_world:
    print(' '.join(str(elem) for elem in row))

with open('test2.txt', 'w') as f:
    for row in grid_world:
        f.write('	'.join(str(elem) for elem in row))
        f.write('\n')