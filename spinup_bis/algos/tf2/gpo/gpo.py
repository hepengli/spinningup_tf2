"""GPO algorithm implementation."""

import random
import time

import numpy as np
import tensorflow as tf

from spinup_bis.algos.tf2.gpo import core
from spinup_bis.utils import logx

EPS = 1e-8

class ReplayBuffer:
    """A simple FIFO experience replay buffer for GPO agents."""

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros(core.combined_shape(size, obs_dim),
                                dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim),
                                dtype=np.float32)
        self.acts_buf = np.zeros(core.combined_shape(size, act_dim),
                                dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs1=tf.convert_to_tensor(self.obs1_buf[idxs]),
                    obs2=tf.convert_to_tensor(self.obs2_buf[idxs]),
                    acts=tf.convert_to_tensor(self.acts_buf[idxs]),
                    rews=tf.convert_to_tensor(self.rews_buf[idxs]),
                    done=tf.convert_to_tensor(self.done_buf[idxs]))

        return batch

def gpo(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=None, seed=0,
        total_steps=1e6, log_every=10_000, replay_size=100_000, gamma=0.99, 
        polyak=0.995, lr=1e-3, batch_size=256, update_after=1000, update_every=50, 
        sample_size=20, max_ep_len=1000, num_test_episodes=10, save_freq=int(1e4), 
        logger_kwargs=None, save_path=None):
    """General Policy Optimization.

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in `action_space` and
            `observation_space` kwargs, and returns actor and critic
            tf.keras.Model-s.

            Actor should take an observation in and output:
            ===========  ================  =====================================
            Symbol       Shape             Description
            ===========  ================  =====================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled
                                           | by ``pi``. Critical: must be
                                           | differentiable with respect to
                                           | policy parameters all the way
                                           | through action sampling.
            ===========  ================  =====================================

            Critic should take an observation and an action in and output:
            ===========  ================  =====================================
            Symbol       Shape             Description
            ===========  ================  =====================================
            ``q``        (batch,)          | Gives one estimate of Q* for
                                           | states and actions in the input.
            ===========  ================  =====================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to SAC.

        seed (int): Seed for random number generators.

        total_steps (int): Number of environment interactions to run and train
            the agent.

        log_every (int): Number of environment interactions that should elapse
            between dumping logs.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate for policy and Q network

        batch_size (int): Minibatch size for SGD.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        epochs_train (int): Number of training epochs at each update

        sample_size (int): Number of action to sample for expectation estimation.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of environment iterations) to save
            the current policy.

        save_path (str): The path specifying where to save the trained actor
            model (note: path needs to point to a directory). Setting the value
            to None turns off the saving.
    """
    config = locals()
    logger = logx.EpochLogger(**(logger_kwargs or {}))
    logger.save_config(config)

    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = np.prod(env.observation_space.shape)
    act_dim = env.action_space.shape

    # Share information about observation and action spaces with policy.
    ac_kwargs = ac_kwargs or {}
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['observation_space'] = env.observation_space

    # Experience buffer.
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,
                                 size=replay_size)

    # Network
    actor, critic1, critic2 = actor_critic(**ac_kwargs)

    actor.build(input_shape=(None, obs_dim))
    critic1.build(input_shape=(None, obs_dim + act_dim))
    critic2.build(input_shape=(None, obs_dim + act_dim))

    # Target networks
    old_actor, critic1_targ, critic2_targ = actor_critic(**ac_kwargs)

    old_actor.build(input_shape=(None, obs_dim))
    critic1_targ.build(input_shape=(None, obs_dim + act_dim))
    critic2_targ.build(input_shape=(None, obs_dim + act_dim))

    # Copy weights
    old_actor.set_weights(actor.get_weights())
    critic1_targ.set_weights(critic1.get_weights())
    critic2_targ.set_weights(critic2.get_weights())

    # Separate train ops for pi, q
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def learn_on_batch(obs1, obs2, acts, rews, done):
        shape, n = (sample_size, batch_size), sample_size
        with tf.GradientTape(persistent=True) as g:
            # Sample actions
            n_acts, n_logpold = old_actor.sample_logprob(obs1, n)
            n_logp = actor.action_logprob(obs1, n_acts)

            # Calculate Q values and advantages
            n_obs1 = tf.tile(obs1, [sample_size, 1])
            n_acts = tf.reshape(n_acts, (np.prod(shape),)+act_dim)
            n_q1 = critic1([n_obs1, n_acts])
            n_q2 = critic2([n_obs1, n_acts])
            q = tf.reshape(tf.minimum(n_q1, n_q2), shape)

            # Estimate C and Z
            adv = q - tf.reduce_mean(q, axis=0, keepdims=True)
            max_abs_adv = tf.reduce_mean(tf.abs(adv))
            C = max_abs_adv * (gamma**2) / ((1-gamma)**1)
            alpha = adv / (C + EPS)
            Z = tf.reduce_mean(tf.exp(alpha), axis=0)
            ratio = tf.stop_gradient(tf.exp(alpha) / Z)

            # Actor loss
            # n_logptarg = n_logpold + alpha - tf.math.log(Z)
            # error = n_logp - tf.stop_gradient(n_logptarg)
            error = tf.exp(n_logp - n_logpold) - ratio
            loss = 0.5 * tf.reduce_mean(error ** 2)

            # Main outputs from computation graph.
            q1 = critic1([obs1, acts])
            q2 = critic2([obs1, acts])

            # Get actions and log probs of actions for next states.
            act_next = actor.action(obs2)

            # Target Q-values, using actions from *current* policy.
            target_q1 = critic1_targ([obs2, act_next])
            target_q2 = critic2_targ([obs2, act_next])
            target_q = tf.minimum(target_q1, target_q2)

            # Entropy-regularized Bellman backup for Q functions.
            # Using Clipped Double-Q targets.
            td_target = tf.stop_gradient(rews + gamma * (1 - done) * target_q)

            # Soft actor-critic losses.
            q1_loss = 0.5 * tf.reduce_mean((td_target - q1) ** 2)
            q2_loss = 0.5 * tf.reduce_mean((td_target - q2) ** 2)
            value_loss = q1_loss + q2_loss

        # Compute gradients and do updates.
        actor_gradients = g.gradient(loss, actor.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_gradients, actor.trainable_variables))
        critic_variables = critic1.trainable_variables + \
                           critic2.trainable_variables
        critic_gradients = g.gradient(value_loss, critic_variables)
        critic_optimizer.apply_gradients(
            zip(critic_gradients, critic_variables))
        del g

        # Polyak averaging for target variables.
        for a, old_a in zip(actor.trainable_variables, 
                            old_actor.trainable_variables):
            old_a.assign(a)
        for v, target_v in zip(critic1.trainable_variables,
                               critic1_targ.trainable_variables):
            target_v.assign(polyak * target_v + (1 - polyak) * v)
        for v, target_v in zip(critic2.trainable_variables,
                               critic2_targ.trainable_variables):
            target_v.assign(polyak * target_v + (1 - polyak) * v)

        return dict(loss=loss, 
                    q1_loss=q1_loss,
                    q2_loss=q2_loss,
                    q1=q1,
                    q2=q2,
                    alpha=alpha, 
                    C=C, 
                    Z=Z)

    @tf.function
    def actor_train_step(obs1, obs2, acts, rews, done):
        shape, n = (sample_size, batch_size), sample_size
        with tf.GradientTape(persistent=True) as g:
            # Sample actions
            n_acts, n_logpold = old_actor.sample_logprob(obs1, n)
            n_logp = actor.action_logprob(obs1, n_acts)

            # Calculate Q values and advantages
            n_obs1 = tf.tile(obs1, [sample_size, 1])
            n_acts = tf.reshape(n_acts, (np.prod(shape),)+act_dim)
            q1 = critic1([n_obs1, n_acts])
            q2 = critic2([n_obs1, n_acts])
            q = tf.reshape(tf.minimum(q1, q2), shape)

            # Estimate C and Z
            adv = q - tf.reduce_mean(q, axis=0, keepdims=True)
            max_abs_adv = tf.reduce_mean(tf.abs(adv))
            C = max_abs_adv * (gamma**2) / ((1-gamma)**1)
            alpha = adv / (C + EPS)
            Z = tf.reduce_mean(tf.exp(alpha), axis=0)
            ratio = tf.exp(alpha) / Z

            # Actor loss
            n_logptarg = n_logpold + alpha - tf.math.log(Z)
            error = n_logp - n_logptarg
            loss = 0.5 * tf.reduce_mean(error ** 2)

        # Compute gradients and do updates.
        actor_gradients = g.gradient(loss, actor.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_gradients, actor.trainable_variables))
        del g

        return dict(loss=loss, logp=n_logp, alpha=alpha, C=C, Z=Z)

    @tf.function
    def critic_train_step(obs1, obs2, acts, rews, done):
        shape, n = (sample_size, batch_size), sample_size
        with tf.GradientTape(persistent=True) as g:
            # Main outputs from computation graph.
            q1 = critic1([obs1, acts])
            q2 = critic2([obs1, acts])

            # Get actions and log probs of actions for next states.
            act_next = actor.action(obs2)

            # Target Q-values, using actions from *current* policy.
            target_q1 = critic1_targ([obs2, act_next])
            target_q2 = critic2_targ([obs2, act_next])
            target_q = tf.minimum(target_q1, target_q2)

            # Bellman backup for Q function
            td_target = tf.stop_gradient(rews + gamma * (1 - done) * target_q)

            # Soft actor-critic losses.
            q1_loss = 0.5 * tf.reduce_mean((td_target - q1) ** 2)
            q2_loss = 0.5 * tf.reduce_mean((td_target - q2) ** 2)
            value_loss = q1_loss + q2_loss

        critic_variables = critic1.trainable_variables + \
                           critic2.trainable_variables
        critic_gradients = g.gradient(value_loss, critic_variables)
        critic_optimizer.apply_gradients(
            zip(critic_gradients, critic_variables))
        del g

        # Polyak averaging for target variables.
        for v, target_v in zip(critic1.trainable_variables,
                               critic1_targ.trainable_variables):
            target_v.assign(polyak * target_v + (1 - polyak) * v)
        for v, target_v in zip(critic2.trainable_variables,
                               critic2_targ.trainable_variables):
            target_v.assign(polyak * target_v + (1 - polyak) * v)

        return dict(q1_loss=q1_loss, q2_loss=q2_loss, q1=q1, q2=q2)

    def assign_old_eq_new():
        # update old actor
        for a, old_a in zip(actor.trainable_variables + \
                            actor.non_trainable_weights, 
                            old_actor.trainable_variables + \
                            old_actor.non_trainable_weights):
            old_a.assign(a)

    def get_action(observation, deterministic=False):
        return actor.action(np.array([observation]), 
                            deterministic).numpy()[0]

    def test_agent():
        for _ in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time.
                o, r, d, _ = test_env.step(
                    get_action(tf.convert_to_tensor(o), True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    obs, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        iter_time = time.time()
        act = get_action(obs)

        # Step the env
        next_obs, rew, done, _ = env.step(act)
        ep_ret += rew
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state).
        done = False if ep_len == max_ep_len else done

        # Store experience to replay buffer
        replay_buffer.store(obs, act, rew, next_obs, done)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        obs = next_obs

        # End of trajectory handling
        if done or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            obs, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling.
        if t >= update_after and t % update_every == 0:
            assign_old_eq_new()
            # update critic
            for i in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                results = critic_train_step(**batch)
                logger.store(LossQ1=results['q1_loss'].numpy(),
                            LossQ2=results['q2_loss'].numpy(),
                            Q1Vals=results['q1'].numpy(),
                            Q2Vals=results['q2'].numpy())
            # update actor
            for i in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                results = actor_train_step(**batch)
                logger.store(LossA=results['loss'].numpy(),
                            LogP=results['logp'].numpy(),
                            Alpha=results['alpha'].numpy(),
                            C=results['C'].numpy(),
                            Z=results['Z'].numpy())
            # for _ in range(update_every):
            #     batch = replay_buffer.sample_batch(batch_size)
            #     results = learn_on_batch(**batch)
            #     logger.store(LossQ1=results['q1_loss'].numpy(), 
            #                 LossQ2=results['q2_loss'].numpy(), 
            #                 Q1Vals=results['q1'].numpy(), 
            #                 Q2Vals=results['q2'].numpy(), 
            #                 LossA=results['loss'].numpy(), 
            #                 Alpha=results['alpha'].numpy(), 
            #                 C=results['C'].numpy(), 
            #                 Z=results['Z'].numpy())

        logger.store(StepsPerSecond=(1 / (time.time() - iter_time)))

        # End of epoch wrap-up.
        if ((t > update_after) and ((t + 1) % log_every == 0)) or \
            (t + 1 == total_steps):
            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch.
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t + 1)
            logger.log_tabular('Q1Vals', average_only=True)
            logger.log_tabular('Q2Vals', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('LossA', average_only=True)
            logger.log_tabular('LogP', with_min_and_max=True)
            logger.log_tabular('Alpha', with_min_and_max=True)
            logger.log_tabular('Z', with_min_and_max=True)
            logger.log_tabular('C', average_only=True)

            logger.log_tabular('StepsPerSecond', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)

            logger.dump_tabular()

        # Save model.
        if ((t + 1) % save_freq == 0) or (t + 1 == total_steps):
            if save_path is not None:
                tf.keras.models.save_model(actor, save_path)
