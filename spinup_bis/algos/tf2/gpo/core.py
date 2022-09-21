"""Core functions of the GPO algorithm."""
import gym
import numpy as np
import tensorflow as tf


EPS = 1e-8

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def log_likelihood(value, mu, log_std):
    """Calculates value's log likelihood under squashed Gaussian pdf."""
    logp = -0.5 * (
        ((value - mu) / (tf.exp(log_std + EPS))) ** 2 +
        2 * log_std + np.log(2 * np.pi)) - \
        (-2 * tf.nn.softplus(-value) - value)
        # 2 * (np.log(2) - value - tf.nn.softplus(-2 * value))

    return tf.reduce_sum(logp, axis=-1)


def mlp(hidden_sizes=(64, 32), activation='relu', output_activation=None):
    """Creates MLP with the specified parameters."""
    model = tf.keras.Sequential()

    for h in hidden_sizes[:-1]:
        model.add(tf.keras.layers.Dense(units=h, activation=activation))

    model.add(tf.keras.layers.Dense(
        units=hidden_sizes[-1], activation=output_activation))

    return model


def make_actor_discrete(observation_space, action_space, hidden_sizes, activation):
    """Creates actor tf.keras.Model.

    This function can be used only in environments with discrete action space.
    """

    class DiscreteActor(tf.keras.Model):
        """Actor model for discrete action space."""

        def __init__(self, observation_space, action_space, hidden_sizes, activation):
            super().__init__()
            self._act_dim = action_space.n

            obs_input = tf.keras.Input(shape=observation_space.shape)
            actor = mlp(hidden_sizes=list(hidden_sizes) + [action_space.n],
                activation=activation)(obs_input)

            self._network = tf.keras.Model(inputs=obs_input, outputs=actor)

        @tf.function
        def call(self, inputs, training=False, mask=None):
            return tf.nn.log_softmax(self._network(inputs=inputs, training=training))

        @tf.function
        def action(self, observations):
            return tf.squeeze(tf.random.categorical(self(observations), 1), axis=1)

        @tf.function
        def logprob(self, observations, actions):
            return tf.reduce_sum(
                tf.math.multiply(self(observations),
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
                name='mean')
            self._log_std = tf.Variable(
                initial_value=-0.5 * np.ones(
                    shape=(1,) + self._action_dim,
                    dtype=np.float32), 
                trainable=True,
                name='log_std_dev')

        @tf.function
        def call(self, inputs):
            x = self._body(inputs=inputs)
            mu = self._mu(x)
            log_std = self._log_std

            return mu, log_std

        @tf.function
        def action(self, observations, deterministic=False):
            mu, log_std = self(observations)
            std = tf.exp(log_std)

            log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
            low, high = self._action_space.low, self._action_space.high

            value = mu if deterministic else mu + tf.random.normal(mu.shape) * std
            action = low + (high - low) * tf.nn.sigmoid(value)

            return action

        @tf.function
        def logp(self, observations, u):
            mu, log_std = self(observations)
            log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
            return log_likelihood(u, mu, log_std)

        @tf.function
        def sample_logp(self, observations, n_samples):
            mu, log_std = self(observations)
            log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = tf.exp(log_std)
            low, high = self._action_space.low, self._action_space.high

            value = mu + tf.random.normal((n_samples,) + mu.shape) * std
            action = low + (high - low) * tf.nn.sigmoid(value)
            logp = log_likelihood(value, mu, log_std)

            return logp, action, value

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