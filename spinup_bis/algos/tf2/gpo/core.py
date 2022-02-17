"""Core functions of the GPO algorithm."""
from cv2 import log
import gym
import numpy as np
import scipy.signal
import tensorflow as tf
from tensorflow_probability import distributions as tfd


EPS = 1e-8

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def distribute_value(value, num_proc):
    """Adjusts training parameters for distributed training.

    In case of distributed training frequencies expressed in global steps have
    to be adjusted to local steps, thus divided by the number of processes.
    """
    return max(value // num_proc, 1)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


@tf.function
def gaussian_likelihood(value, mu, log_std):
    """Calculates value's likelihood under Gaussian pdf."""
    pre_sum = -0.5 * (
            ((value - mu) / (tf.exp(log_std) + EPS)) ** 2 +
            2 * log_std + np.log(2 * np.pi)
    )
    return pre_sum


def mlp(hidden_sizes=(64, 32), activation='relu', output_activation=None,
        layer_norm=False):
    """Creates MLP with the specified parameters."""
    model = tf.keras.Sequential()

    for h in hidden_sizes[:-1]:
        model.add(tf.keras.layers.Dense(units=h, activation=None))
        if layer_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(activation))

    model.add(tf.keras.layers.Dense(units=hidden_sizes[-1], activation=None))
    if layer_norm:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(output_activation))

    return model


def make_actor_discrete(observation_space, action_space, hidden_sizes,
                        activation, layer_norm):
    """Creates actor tf.keras.Model.

    This function can be used only in environments with discrete action space.
    """

    class DiscreteActor(tf.keras.Model):
        """Actor model for discrete action space."""

        def __init__(self, observation_space, action_space, hidden_sizes, 
                     activation):
            super().__init__()
            self._act_dim = action_space.n

            obs_input = tf.keras.Input(shape=observation_space.shape)
            actor = mlp(hidden_sizes=list(hidden_sizes) + [action_space.n],
                activation=activation, layer_norm=layer_norm)(obs_input)

            self._network = tf.keras.Model(inputs=obs_input, outputs=actor)

        @tf.function
        def call(self, inputs, training=False, mask=None):
            return tf.nn.log_softmax(self._network(inputs=inputs, training=training))

        @tf.function
        def action(self, observations):
            return tf.squeeze(tf.random.categorical(self(observations), 1),
                              axis=1)

        @tf.function
        def action_logprob(self, observations, actions, training=False):
            return tf.reduce_sum(
                tf.math.multiply(self(observations, training),
                                 tf.one_hot(tf.cast(actions, tf.int32),
                                            depth=self._act_dim)), axis=-1)

    return DiscreteActor(observation_space, action_space, hidden_sizes, activation)


def make_actor_continuous(action_space, hidden_sizes, activation, layer_norm):
    """Creates actor tf.keras.Model.

    This function can be used only in environments with continuous action space.
    """

    class ContinuousActor(tf.keras.Model):
        """Actor model for continuous action space."""

        def __init__(self, action_space, hidden_sizes, activation):
            super().__init__()
            self._action_space = action_space
            self._action_dim = action_space.shape

            self._body = mlp(
                hidden_sizes=list(hidden_sizes),
                activation=activation,
                output_activation=activation,
                layer_norm=layer_norm)

            self._mu = tf.keras.layers.Dense(
                self._action_dim[0], 
                activation=tf.keras.activations.sigmoid, 
                name='mean')
            self._log_std = tf.keras.layers.Dense(
                action_space.shape[0], 
                kernel_initializer=tf.initializers.Orthogonal(0.01), 
                name='log_std_dev')

        @tf.function
        def call(self, inputs, training=None, mask=None):
            x = self._body(inputs=inputs, training=training)
            mu = self._mu(x)
            log_std = self._log_std(x) - 0.5

            log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

            return mu, tf.exp(log_std)

        @tf.function
        def action(self, observations, deterministic=False):
            mu, std = self(observations)
            dist = tfd.TruncatedNormal(loc=mu, scale=std, low=0.0, high=1.0)

            low, high = self._action_space.low, self._action_space.high
            if deterministic:
                return low + (high - low) * mu
            else:
                return low + (high - low) * dist.sample()

        @tf.function
        def sample(self, observations, n):
            mu, std = self(observations)
            dist = tfd.TruncatedNormal(loc=mu, scale=std, low=0.0, high=1.0)

            low, high = self._action_space.low, self._action_space.high
            if deterministic:
                return low + (high - low) * mu
            else:
                return low + (high - low) * dist.sample(n)

        @tf.function
        def action_logprob(self, observations, actions, training=False):
            mu, std = self(observations, training)
            low, high = self._action_space.low, self._action_space.high
            dist = tfd.TruncatedNormal(loc=mu, scale=std, low=0.0, high=1.0)

            # Make sure actions are in correct range
            low, high = self._action_space.low, self._action_space.high
            actions = (actions - low) / (high - low)
            log_prob = tf.clip_by_value(dist.log_prob(actions), -20., 10.)

            return tf.reduce_mean(log_prob, axis=-1)

        @tf.function
        def sample_logprob(self, observations, n_samples):
            mu, std = self(observations)
            dist = tfd.TruncatedNormal(loc=mu, scale=std, low=0.0, high=1.0)

            actions = dist.sample(n_samples)
            log_prob = tf.clip_by_value(dist.log_prob(actions), -20., 10.)

            # Make sure actions are in correct range
            low, high = self._action_space.low, self._action_space.high
            actions = low + (high - low) * actions

            return actions, tf.reduce_mean(log_prob, -1)

    return ContinuousActor(action_space, hidden_sizes, activation)


def make_critic(observation_space, action_space, hidden_sizes, activation):
    """Creates critic tf.keras.Model"""
    obs_input = tf.keras.Input(shape=observation_space.shape)
    act_input = tf.keras.Input(shape=action_space.shape)
    concat_input = tf.keras.layers.Concatenate(axis=-1)([obs_input, act_input])

    critic = tf.keras.Sequential([
        mlp(hidden_sizes=list(hidden_sizes) + [1],
            activation=activation),
        tf.keras.layers.Reshape([]),
    ])(concat_input)

    return tf.keras.Model(inputs=[obs_input, act_input], outputs=critic)


def mlp_actor_critic(observation_space, action_space, hidden_sizes=(256, 256),
                     activation=tf.nn.relu, layer_norm=False):
    """Creates actor and critic tf.keras.Model-s."""
    actor = None

    # default policy builder depends on action space
    if isinstance(action_space, gym.spaces.Discrete):
        actor = make_actor_discrete(observation_space, action_space,
                                    hidden_sizes, activation, layer_norm)
    elif isinstance(action_space, gym.spaces.Box):
        actor = make_actor_continuous(action_space, hidden_sizes, 
                                      activation, layer_norm)

    critic1 = make_critic(observation_space, action_space, hidden_sizes, activation)
    critic2 = make_critic(observation_space, action_space, hidden_sizes, activation)

    return actor, critic1, critic2