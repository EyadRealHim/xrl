from .algorithm import _RLAgent, RLTrainer, Transition, UpdatedPkg
from ..environment import ParallelEnvironment, TState
from ..networks import (
    ActorLike,
    CriticLike,
    Actor,
    Critic,
    ActorContainer,
    CriticContainer,
)

from typing import Generic, Type, Tuple
from jaxtyping import PyTree, Array, Float

from distrax import Categorical

import jax.numpy as jnp
import equinox as eqx

import jax


class PPOAgent(_RLAgent):
    pass


class PPOTrainer(RLTrainer[TState, PPOAgent], Generic[TState]):
    discount: float = eqx.field(static=True, default=0.96)
    lambda_: float = eqx.field(static=True, default=0.95)

    epoch_n: int = 10
    clip_coef: float = 0.2
    entropy_coef: float = 0.01

    def make_agent(
        self, key: Array, mactor: Type[ActorLike], mcritic: Type[CriticLike]
    ) -> PPOAgent:
        ActorConstuct = Actor
        CriticConstruct = Critic

        if isinstance(self.env, ParallelEnvironment):
            ActorConstuct = ActorContainer.create
            CriticConstruct = CriticContainer.create

        a, b = jax.random.split(key)

        actor = ActorConstuct(
            a, mactor, self.env.observation_space(), self.env.action_space()
        )
        critic = CriticConstruct(b, mcritic, self.env.observation_space())

        return PPOAgent(
            actor=actor,
            critic=critic,
            opt={
                "actor": actor.opt_state(self.optim),
                "critic": critic.opt_state(self.optim),
            },
        )

    def update(self, pkg):
        obs, transition, actor, critic, optactor, optcritic = pkg

        advantage, returns = jax.vmap(self.compute_advantage_and_returns)(
            transition, jax.vmap(critic)(obs)
        )

        def act(actor: Actor, obs, action):
            logits = jax.vmap(jax.vmap(actor))(obs)

            def fun(logits, action):
                dist = Categorical(logits=logits)

                return dist.log_prob(action), dist.entropy()

            acttree = jax.tree.map(fun, logits, action)
            acttree = jnp.squeeze(jnp.array(jax.tree.leaves(acttree)))

            return acttree[0], acttree[1]

        def values(critic: Critic, obs):
            return jnp.squeeze(jax.vmap(jax.vmap(critic))(obs))

        init_log_p, _ = act(actor, transition.observation, transition.action)
        init_value = values(critic, transition.observation)

        @eqx.filter_grad
        def actor_loss(actor: Actor, obs, action: PyTree, advantage):
            log_p, entropy = act(actor, obs, action)

            ratio = jnp.exp(log_p - init_log_p)

            return (
                -jnp.minimum(
                    ratio * advantage,
                    jnp.clip(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * advantage,
                ).mean()
                - self.entropy_coef * entropy.mean()
            )

        @eqx.filter_grad
        def critic_loss(critic: Critic, obs, returns):
            value = values(critic, obs)

            loss1 = (value - returns) ** 2

            clipped = init_value + jnp.clip(
                value - init_value, -self.clip_coef, self.clip_coef
            )
            loss2 = (clipped - returns) ** 2

            return 0.5 * jnp.maximum(loss1, loss2).mean()

        params, static = eqx.partition((actor, critic), eqx.is_array)

        def epoch(_, epoch_state):
            params, optactor, optcritic = epoch_state

            actor, critic = eqx.combine(params, static)

            actor_grad = actor_loss(
                actor, transition.observation, transition.action, advantage
            )
            critic_grad = critic_loss(critic, transition.observation, returns)

            actor, optactor = actor.update(actor_grad, optactor, self.optim)
            critic, optcritic = critic.update(critic_grad, optcritic, self.optim)

            params = eqx.filter((actor, critic), eqx.is_array)

            return params, optactor, optcritic

        params, optactor, optcritic = jax.lax.fori_loop(
            0, self.epoch_n, epoch, (params, optactor, optcritic)
        )
        actor, critic = eqx.combine(params, static)

        return UpdatedPkg(
            actor=actor,
            critic=critic,
            opt_actor=optactor,
            opt_critic=optcritic,
        )

    def compute_advantage_and_returns(
        self, transition: Transition, bootstrap: Float[Array, "1"]
    ) -> Tuple[Float[Array, "step_n"], Float[Array, "step_n"]]:
        discount = (1.0 - transition.done.astype(jnp.float32)) * self.discount
        values = jnp.concatenate([jnp.squeeze(transition.value), bootstrap])
        deltas = transition.reward + discount * values[1:] - values[:-1]

        def compute(prev, pair):
            delta, value, discount = pair

            gae = delta + discount * self.lambda_ * prev

            return gae, (gae, gae + value)

        _, (advantage, returns) = jax.lax.scan(
            compute, 0, (deltas, jnp.squeeze(transition.value), discount), reverse=True
        )

        return advantage, returns
