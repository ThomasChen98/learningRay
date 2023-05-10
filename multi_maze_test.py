import time
from multi_maze import MultiAgentMaze

env = MultiAgentMaze()

while True:
    obs, rew, done, info = env.step(
        {1:env.action_space.sample(), 2:env.action_space.sample()}
    )
    time.sleep(0.1)
    env.render()
    if any(done.values()):
