from .algorithm import RLTrainer, Transition, UpdatePkg
from ..environment import TState

from typing import Generic, Tuple
from jaxtyping import Array, Float

from distrax import Categorical

import jax.numpy as jnp
import equinox as eqx

import jax


class SimplePolicyGradientTrainer(RLTrainer[TState], Generic[TState]):
    discount: float = eqx.field(static=True, default=0.96)
    lambda_: float = eqx.field(static=True, default=0.95)

    def update(self, actor, critic, optactor, optcritic, transition, bootstraps):
        advantage, returns = jax.vmap(self.compute_advantage_and_returns)(
            transition, jax.vmap(critic)(bootstraps)
        )

        @eqx.filter_grad
        def actor_loss(actor, obs, action, advantage):
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
        def critic_loss(critic, obs, returns):
            values = jnp.squeeze(jax.vmap(jax.vmap(critic))(obs))

            return ((values - returns) ** 2).mean()

        grad = actor_loss(actor, transition.observation, transition.action, advantage)
        actor, optactor = actor.update(grad, optactor, self.optim)

        grad = critic_loss(critic, transition.observation, returns)
        critic, optcritic = critic.update(grad, optcritic, self.optim)

        return UpdatePkg(
            actor=actor,
            critic=critic,
            optactor=optactor,
            optcritic=optcritic,
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
