import time
import numpy as np
import ray

from maze_policy import Policy
from maze_simulation import Simulation
from maze_gym_env import Environment
# random sample check environment
environment = Environment()

### while not environment.is_done():
#     random_action = environment.action_space.sample()
#     environment.step(random_action)
#     time.sleep(0.1)
#     environment.render()

### untrained policy test simulation
# untrained_policy = Policy(environment)
# sim = Simulation(environment)
#
# exp = sim.rollout(untrained_policy, render=True, epsilon=1.0)
# for row in untrained_policy.state_action_table:
#     print(row)
### Train policy
def update_policy(policy, experiences, weight=0.1, discount_factor=0.9):
    """Updates a given policy with a list of (state, action, reward, state)
    experiences."""
    for state, action, reward, next_state in experiences:
        next_max = np.max(policy.state_action_table[next_state])
        value = policy.state_action_table[state][action]
        new_value = (1 - weight) * value + weight * \
                    (reward + discount_factor * next_max)
        policy.state_action_table[state][action] = new_value

def train_policy(env, num_episodes=10000, weight=0.1, discount_factor=0.9):
    """Training a policy by updating it with rollout experiences."""
    policy = Policy(env)
    sim = Simulation(env)
    for _ in range(num_episodes):
        experiences = sim.rollout(policy)
        update_policy(policy, experiences, weight, discount_factor)

    return policy

def evaluate_policy(env, policy, num_episodes=10):
    """Evaluate a trained policy through rollouts."""
    simulation = Simulation(env)
    steps = 0

    for _ in range(num_episodes):
        experiences = simulation.rollout(policy, render=True, explore=False)
        steps += len(experiences)

    print(f"{steps / num_episodes} steps on average "
          f"for a total of {num_episodes} episodes.")

    return steps / num_episodes

# trained_policy = train_policy(environment)
# evaluate_policy(environment, trained_policy)

### Distributed Ray App
ray.init()

@ray.remote
class SimulationActor(Simulation):
    """Ray actor for a Simulation."""
    def __init__(self):
        env = Environment()
        super().__init__(env)

def train_policy_parallel(env, num_episodes=1000, num_simulations=4):
    """Parallel policy training function."""
    policy = Policy(env)
    simulations = [SimulationActor.remote() for _ in range(num_simulations)]

    policy_ref = ray.put(policy)
    for _ in range(num_episodes):
        experiences = [sim.rollout.remote(policy_ref) for sim in simulations]

        while len(experiences) > 0:
            finished, experiences = ray.wait(experiences)
            for xp in ray.get(finished):
                update_policy(policy, xp)

    return policy

parallel_policy = train_policy_parallel(environment)
evaluate_policy(environment, parallel_policy)