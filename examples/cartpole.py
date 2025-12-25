from xrl.envs.cartpole import CartPole
from xrl.algorithms import PPOTrainer
from xrl.networks import ActorLike, CriticLike

from tqdm import tqdm

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


key = jax.random.key(0)
env = CartPole()

key, subk = jax.random.split(key)
trainer = PPOTrainer(env=env, optim=optax.adam(1e-3))

agent = trainer.make_agent(subk, CartPoleActor, CartPoleCritic)
agent = trainer.train(key, agent, iterations=32)

frames = [
    env.render(state)
    for state in tqdm(
        trainer.capture(
            key=jax.random.key(817),
            agent=agent,
            max_steps=300,
        )
    )
]

duration = int(1000 / 30)
frames[0].save(
    "cartpole.gif",
    save_all=True,
    append_images=frames[1:],
    duration=duration,
    loop=0,
)
