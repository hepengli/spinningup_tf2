"""Core functions of the GPO algorithm."""
import gym
import numpy as np
import tensorflow as tf


EPS = 1e-8

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def gaussian_likelihood(value, mu, log_std):
    """Calculates value's likelihood under Gaussian pdf."""
    pre_sum = -0.5 * (
        ((value - mu) / (tf.exp(log_std) + EPS)) ** 2 +
        2 * log_std + np.log(2 * np.pi)) - \
        2 * (np.log(2) - value - tf.nn.softplus(-2 * value))

    return tf.reduce_sum(pre_sum, axis=-1)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(hidden_sizes=(64, 32), activation='relu', output_activation=None):
    """Creates MLP with the specified parameters."""
    model = tf.keras.Sequential()
    for h in hidden_sizes[:-1]:
        model.add(tf.keras.layers.Dense(units=h, activation=activation))

    model.add(tf.keras.layers.Dense(
        units=hidden_sizes[-1], activation=output_activation))

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
            return self._network(inputs=inputs)

        @tf.function
        def action(self, observations, deterministic=False):
            logprob = tf.nn.log_softmax(self(observations))
            if deterministic:
                return tf.argmax(logprob, axis=-1)
            else:
                return tf.squeeze(tf.random.categorical(logprob, 1), axis=-1)

        @tf.function
        def sample(self, observations, n):
            logprob = tf.nn.log_softmax(self(observations))
            return tf.random.categorical(logprob, n)

        @tf.function
        def kl(self, other, observations):
            self_logits = self(observations)
            other_logits = other(observations)
            a0 = self_logits - tf.reduce_max(self_logits, axis=-1, keepdims=True)
            a1 = other_logits - tf.reduce_max(other_logits, axis=-1, keepdims=True)
            ea0 = tf.exp(a0)
            ea1 = tf.exp(a1)
            z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
            z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(
                p0 * (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)), axis=-1)

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
                output_activation=activation)

            self._mu = tf.keras.layers.Dense(
                self._action_dim[0], 
                name='mean')
            # self._log_std = tf.keras.layers.Dense(
            #     self._action_dim[0], 
            #     name='log_std_dev')
            self._log_std = tf.Variable(
                initial_value=-0.5 * np.ones(shape=(1,) + self._action_dim,
                                             dtype=np.float32), trainable=True,
                name='log_std_dev')

        @tf.function
        def call(self, inputs, training=None, mask=None):
            x = self._body(inputs=inputs)
            mu = self._mu(x)
            log_std = self._log_std

            log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

            return mu, log_std

        @tf.function
        def action(self, observations, deterministic=False, n=None):
            mu, log_std = self(observations)
            std = tf.exp(log_std)
            low, high = self._action_space.low, self._action_space.high

            shape = mu.shape if n is None else (n,) + mu.shape
            value = mu if deterministic else mu + tf.random.normal(shape) * std
            action = low + (high - low) * (tf.nn.tanh(value) + 1) / 2
            log_prob = gaussian_likelihood(value, mu, log_std)

            return action, log_prob

        @tf.function
        def kl(self, other, observations):
            self_mean, self_logstd = self(observations)
            other_mean, other_logstd = other(observations)
            self_std, other_std = tf.exp(self_logstd), tf.exp(other_logstd)
            kl = other_logstd - self_logstd + (tf.square(self_std) + \
                tf.square(self_mean - other_mean)) / (2.0 * tf.square(other_std)) - 0.5

            return tf.reduce_sum(kl, axis=-1)

    return ContinuousActor(action_space, hidden_sizes, activation)


def make_critic_continuous(observation_space, action_space, hidden_sizes, activation):
    """Creates critic tf.keras.Model"""
    obs_input = tf.keras.Input(shape=observation_space.shape)
    act_input = tf.keras.Input(shape=action_space.shape)
    concat_input = tf.keras.layers.Concatenate(axis=-1)([obs_input, act_input])

    critic = tf.keras.Sequential([
        mlp(hidden_sizes=list(hidden_sizes) + [1],
            activation=activation),
        tf.keras.layers.Reshape([]),
    ])(concat_input)

    return tf.keras.Model(inputs=[obs_input, act_input], outputs=[critic])


def make_critic_discrete(observation_space, action_space, hidden_sizes, activation):
    """Creates critic tf.keras.Model"""
    obs_input = tf.keras.Input(shape=observation_space.shape)
    act_input = tf.keras.Input(shape=action_space.shape)

    critic = mlp(hidden_sizes=list(hidden_sizes) + [action_space.n],
            activation=activation)(obs_input)

    onehot_act = tf.one_hot(tf.cast(act_input, tf.int32), action_space.n)
    critic_selected = tf.reduce_sum(critic * onehot_act, axis=-1)

    return tf.keras.Model(inputs=[obs_input, act_input], outputs=[critic_selected, critic])


def mlp_actor_critic(observation_space, action_space, hidden_sizes=(256, 256),
                     activation=tf.nn.relu, layer_norm=False):
    """Creates actor and critic tf.keras.Model-s."""
    actor = None

    # default policy builder depends on action space
    if isinstance(action_space, gym.spaces.Discrete):
        actor = make_actor_discrete(observation_space, action_space,
                                    hidden_sizes, activation, layer_norm)
        critic1 = make_critic_discrete(observation_space, action_space, 
                                         hidden_sizes, activation)
        critic2 = make_critic_discrete(observation_space, action_space, 
                                         hidden_sizes, activation)
    elif isinstance(action_space, gym.spaces.Box):
        actor = make_actor_continuous(action_space, hidden_sizes, 
                                      activation, layer_norm)
        critic1 = make_critic_continuous(observation_space, action_space, 
                                         hidden_sizes, activation)
        critic2 = make_critic_continuous(observation_space, action_space, 
                                         hidden_sizes, activation)

    return actor, critic1, critic2