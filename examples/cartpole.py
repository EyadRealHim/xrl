from xrl.envs.cartpole import CartPole
from xrl.algorithms import PPOTrainer
from xrl.networks import ActorLike, CriticLike

import equinox as eqx

import optax
import jax


class CartPoleCritic(CriticLike):
    critic: eqx.nn.MLP

    def __init__(self, key, in_features):
        self.critic = eqx.nn.MLP(in_features, 1, width_size=2, depth=3, key=key)

    def __call__(self, obs):
        return self.critic(obs)


class CartPoleActor(ActorLike):
    actor: eqx.nn.MLP

    def __init__(self, key, in_features):
        self.actor = eqx.nn.MLP(in_features, 4, width_size=2, depth=1, key=key)

    def __call__(self, obs):
        return self.actor(obs)


key = jax.random.key(1)
env = CartPole()

key, subk = jax.random.split(key)
trainer = PPOTrainer(env=env, optim=optax.adam(1e-4), env_n=4)

agent = trainer.make_agent(subk, CartPoleActor, CartPoleCritic)
agent = trainer.train(key, agent, iterations=128)

trainer.record(
    "/gehaz/cartpole.mp4",
    trainer.capture(key=jax.random.key(817), agent=agent, max_steps=300),
)
