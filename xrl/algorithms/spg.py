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


class SimplePolicyGradientAgent(_RLAgent):
    pass


class SimplePolicyGradientTrainer(
    RLTrainer[TState, SimplePolicyGradientAgent], Generic[TState]
):
    discount: float = eqx.field(static=True, default=0.96)
    lambda_: float = eqx.field(static=True, default=0.95)

    def make_agent(
        self, key: Array, mactor: Type[ActorLike], mcritic: Type[CriticLike]
    ) -> SimplePolicyGradientAgent:
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

        return SimplePolicyGradientAgent(
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

        @eqx.filter_grad
        def actor_loss(actor: Actor, obs, action: PyTree, advantage):
            logits = jax.vmap(jax.vmap(actor))(obs)
            log_p = jax.tree.map(
                lambda logits, action: Categorical(logits=logits).log_prob(action),
                logits,
                action,
            )

            adv = jnp.expand_dims(advantage, axis=-1)
            losses = jax.tree.map(lambda x: x * adv, log_p)

            return -jnp.array(jax.tree.leaves(losses)).mean()

        @eqx.filter_grad
        def critic_loss(critic: Critic, obs, returns):
            values = jnp.squeeze(jax.vmap(jax.vmap(critic))(obs))

            return ((values - returns) ** 2).mean()

        grad = actor_loss(actor, transition.observation, transition.action, advantage)
        actor, optactor = actor.update(grad, optactor, self.optim)

        grad = critic_loss(critic, transition.observation, returns)
        critic, optcritic = critic.update(grad, optcritic, self.optim)

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
