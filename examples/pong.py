from xrl.envs.pong import Pong
from xrl.algorithms import PPOTrainer
from xrl.networks import ActorLike, CriticLike

from tqdm import tqdm

import equinox as eqx

import optax
import jax


class PongCritic(CriticLike):
    critic: eqx.nn.MLP

    def __init__(self, key, in_features):
        self.critic = eqx.nn.MLP(in_features, 1, width_size=4, depth=3, key=key)

    def __call__(self, obs):
        return self.critic(obs)


class PongActor(ActorLike):
    actor: eqx.nn.MLP

    def __init__(self, key, in_features):
        self.actor = eqx.nn.MLP(in_features, 4, width_size=4, depth=1, key=key)

    def __call__(self, obs):
        return self.actor(obs)


key = jax.random.key(0)
env = Pong()

key, subk = jax.random.split(key)
trainer = PPOTrainer(env=env, optim=optax.adam(1e-3), env_n=4)

agent = trainer.make_agent(subk, PongActor, PongCritic)
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
    "pong.gif",
    save_all=True,
    append_images=frames[1:],
    duration=duration,
    loop=0,
)
