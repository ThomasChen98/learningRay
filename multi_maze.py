from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Discrete
import os

class MultiAgentMaze(MultiAgentEnv):
    def __init__(self, *args, **kwargs):
        self.action_space = Discrete(4)
        self.observation_space = Discrete(5*5)
        self.agents = {1:(4,0), 2:(0,4)}
        self.goal = (4,4)
        self.info = {1:(4,0), 2:(0,4)}

    def reset(self):
        self.agents = {1:(4,0), 2:(0,4)}

        return {1: self.get_observation(1), 2: self.get_observation(2)}

    def get_observation(self, agent_id):
        seeker = self.agents[agent_id]
        return 5*seeker[0]+seeker[1]

    def get_reward(self, agent_id):
        return 1 if self.agents[agent_id] == self.goal else 0

    def is_done(self, agent_id):
        return self.agents[agent_id] == self.goal

    def step(self, action):
        agent_ids = action.keys()

        for agent_id in agent_ids:
            seeker = self.agents[agent_id]
            if action[agent_id] == 0:  # move down
                seeker = (min(seeker[0] + 1, 4), seeker[1])
            elif action[agent_id] == 1:  # move left
                seeker = (seeker[0], max(seeker[1] - 1, 0))
            elif action[agent_id] == 2:  # move up
                seeker = (max(seeker[0] - 1, 0), seeker[1])
            elif action[agent_id] == 3:  # move right
                seeker = (seeker[0], min(seeker[1] + 1, 4))
            else:
                raise ValueError("Invalid action")
            self.agents[agent_id] = seeker

        observations = {i: self.get_observation(i) for i in agent_ids}
        rewards = {i: self.get_reward(i) for i in agent_ids}
        done = {i: self.is_done(i) for i in agent_ids}

        done["__all__"] = all(done.values())

        return observations, rewards, done, self.info

    def render(self, *args, **kwargs):
        os.system('cls' if os.name == 'nt' else 'clear')
        grid = [['| ' for _ in range(5)] + ["|\n"] for _ in range(5)]
        grid[self.goal[0]][self.goal[1]] = '|G'
        grid[self.agents[1][0]][self.agents[1][1]] = '|1'
        grid[self.agents[2][0]][self.agents[2][1]] = '|2'
        grid[self.agents[2][0]][self.agents[2][1]] = '|2'
        print(''.join([''.join(grid_row) for grid_row in grid]))