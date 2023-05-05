from ray.tune.logger import pretty_print
from maze_gym_env import GymEnvironment, Environment
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.algorithm import Algorithm
import time

# config = (DQNConfig().environment(GymEnvironment)
#           .rollouts(num_rollout_workers=2, create_env_on_local_worker=True))
#
# pretty_print(config.to_dict())
#
# algo = config.build()
#
# for i in range(10):
#     result = algo.train()
#
# print(pretty_print(result))

checkpoint = '/Users/yuxinchen/ray_results/DQN_GymEnvironment_2023-05-04_17-58-55lisvow_q/checkpoint_000010'
restored_algo = Algorithm.from_checkpoint(checkpoint)

print(checkpoint)

evaluation = restored_algo.evaluate()

print(pretty_print(evaluation))

# algo.stop()

algo = restored_algo

env = Environment()
done = False
total_reward = 0
observations = env.reset()

while not done:
    action = algo.compute_single_action(observations)
    observations, reward, done, info = env.step(action)
    total_reward += reward
    time.sleep(0.05)
    env.render()

policy = algo.get_policy()
print(policy.get_weights())

model = policy.model

workers = algo.workers
workers.foreach_worker(
    lambda remote_trainer: remote_trainer.get_policy().get_weights()
)

model.base_model.summary()

model.q_value_head.summary()

from ray.rllib.models.preprocessors import get_preprocessor

env = GymEnvironment()
obs_space = env.observation_space
preprocessor = get_preprocessor(obs_space)(obs_space)

observations = env.reset()
transformed = preprocessor.transform(observations).reshape(1, -1)

model_output, _ = model({"obs": transformed})

q_values = model.get_q_value_distributions(model_output)
print(q_values)

action_distribution = policy.dist_class(model_output, model)
sample = action_distribution.sample()
print(sample)