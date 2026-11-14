import equinox as eqx
import jax
import optax  # type: ignore[import-untyped]

from xrl.algorithm.PPO import PPO, PPOAgentConfig
from xrl.envs.CartPole import CartPoleEnv
from xrl.networks import ActorLike, CriticLike
from xrl.trainer import Trainer


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


key = jax.random.key(5)
env = CartPoleEnv()
algo = PPO()
trainer = Trainer(algo, env)


key, subk = jax.random.split(key)
agent = trainer.make_agent(
    subk,
    env,
    PPOAgentConfig(actor=CartPoleActor, critic=CartPoleCritic, optim=optax.adam(1e-3)),
)

key, subkey = jax.random.split(key)
state, obs = env.reset(subkey)

agent, _ = trainer.train(key, agent, 64)

trainer.record("test.mp4", trainer.episode(key, agent, step_n=1024))
