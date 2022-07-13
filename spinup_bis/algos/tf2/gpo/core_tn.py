"""Core functions of the GPO algorithm."""
import gym
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd


EPS = 1e-8

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(hidden_sizes=(64, 32), activation='relu', output_activation=None):
    """Creates MLP with the specified parameters."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.BatchNormalization())

    for h in hidden_sizes[:-1]:
        model.add(tf.keras.layers.Dense(units=h, activation=activation))
        model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(
        units=hidden_sizes[-1], activation=output_activation))
    model.add(tf.keras.layers.BatchNormalization())

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


def make_actor_continuous(action_space, hidden_sizes, activation):
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
                output_activation=activation)

            self._mu = tf.keras.layers.Dense(
                self._action_dim[0], 
                activation=tf.tanh,
                name='mean')
            self._log_std = tf.keras.layers.Dense(
                self._action_dim[0], 
                name='log_std_dev')

        @tf.function
        def call(self, inputs):
            x = self._body(inputs=inputs)
            mu = self._mu(x)
            log_std = self._log_std(x)

            log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

            return mu, tf.exp(log_std)

        @tf.function
        def action(self, observations, deterministic=False):
            mu, std = self(observations)
            dist = tfd.TruncatedNormal(loc=mu, scale=std, low=-1, high=1)

            low, high = self._action_space.low, self._action_space.high
            if deterministic:
                return low + (high - low) * (mu + 1) / 2
            else:
                return low + (high - low) * (dist.sample() + 1) / 2

        @tf.function
        def logprob(self, observations, actions):
            mu, std = self(observations)
            low, high = self._action_space.low, self._action_space.high
            dist = tfd.TruncatedNormal(loc=mu, scale=std, low=-1, high=1)

            # Make sure actions are in correct range
            low, high = self._action_space.low, self._action_space.high
            actions = 2 * (actions - low) / (high - low) - 1
            log_prob = dist.log_prob(actions)

            return tf.reduce_sum(log_prob, axis=-1)

        @tf.function
        def sample_logprob(self, observations, n_samples):
            mu, std = self(observations)
            dist = tfd.TruncatedNormal(loc=mu, scale=std, low=-1, high=1)

            actions = dist.sample(n_samples)
            log_prob = dist.log_prob(actions)

            # Make sure actions are in correct range
            low, high = self._action_space.low, self._action_space.high
            actions = low + (high - low) * (actions + 1) / 2

            return actions, tf.reduce_sum(log_prob, axis=-1)

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
                     activation=tf.nn.relu):
    """Creates actor and critic tf.keras.Model-s."""
    actor = None

    # default policy builder depends on action space
    if isinstance(action_space, gym.spaces.Discrete):
        actor = make_actor_discrete(observation_space, action_space,
                                    hidden_sizes, activation)
    elif isinstance(action_space, gym.spaces.Box):
        actor = make_actor_continuous(action_space, hidden_sizes, activation)

    critic1 = make_critic(observation_space, action_space, hidden_sizes, activation)
    critic2 = make_critic(observation_space, action_space, hidden_sizes, activation)

    return actor, critic1, critic2