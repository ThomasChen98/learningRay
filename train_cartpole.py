from ray.tune.logger import pretty_print
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.algorithm import Algorithm
### configuration
config = (DQNConfig().resources(num_gpus=1, num_cpus_per_worker=2, num_gpus_per_worker=0.25,)
          .rollouts(num_rollout_workers=4, num_envs_per_worker=1, create_env_on_local_worker=True,)
          .environment(env="CartPole-v1", render_env=True))
pretty_print(config.to_dict())
### build algo
algo = config.build()
### train
for i in range(10):
    result = algo.train()
pretty_print(result)
### save checkpoint
checkpoint = algo.save()
print(checkpoint)
### evaluate
evaluation = algo.evaluate()
print(pretty_print(evaluation))
### stop algo
algo.stop()
### load checkpoints
restored_algo = Algorithm.from_checkpoint(checkpoint)
algo = restored_algo
### evaluate checkpoint
print(pretty_print(evaluation))
### policy weights
policy = algo.get_policy()
print(policy.get_weights())
### worker info
workers = algo.workers
workers.foreach_worker(
    lambda remote_trainer: remote_trainer.get_policy().get_weights()
)
### model info
model = policy.model
model.base_model.summary()
