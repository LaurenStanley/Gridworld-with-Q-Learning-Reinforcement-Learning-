import time
import numpy as np

init = time.time()
max_time = 1
time_steps = np.arange(init,max_time + init+.1,.1)
current_timestep_index = 1
print(time_steps)
while time.time() - init < max_time:
    if time.time() > time_steps[current_timestep_index]:
        print(current_timestep_index)
        current_timestep_index += 1
