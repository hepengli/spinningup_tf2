"""Example of running SpinUp Bis algorithms."""

import os

import gym
import tensorflow as tf

from spinup_bis import ppo_tf2 as agent  # pylint: disable=import-only-modules

seed = 1
alg = 'ppo'
env_id = 'HalfCheetah-v2'
output_dir = 'out/{}/{}/exp-{}'.format(env_id, alg, seed)
save_path = output_dir + '/checkpoint'

def env_fn():
    return gym.make(env_id)


if 'NEPTUNE_PROJECT_NAME' in os.environ:
    neptune_kwargs = dict(project=os.environ['NEPTUNE_PROJECT_NAME'])
else:
    neptune_kwargs = None


ac_kwargs = dict(hidden_sizes=[256, 256],
                 activation=tf.nn.relu)

logger_kwargs = dict(output_dir=output_dir,
                     exp_name='{}'.format(env_id+'-'+alg),
                     neptune_kwargs=neptune_kwargs)

agent(env_fn=env_fn,
      ac_kwargs=ac_kwargs,
      total_steps=3_000_000,
      log_every=2000,
      train_every=2000,
      max_ep_len=1000,
      save_freq=200_000,
      seed=seed,
      logger_kwargs=logger_kwargs,
      save_path=save_path)
