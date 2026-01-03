from xrl.networks import Actor, ActorLike, Critic, CriticLike
from .algorithm import RLTrainer, Transition, RLAgent
from ..environment import GroupEnv, TState
from ..xrl_tree import of_instance

from typing import Generic, Tuple, NamedTuple, Type
from jaxtyping import Array, Float

from distrax import Categorical

import jax.numpy as jnp
import equinox as eqx

import jax


class PPOData(NamedTuple):
    transition: Transition
    advantage: Float[Array, "n"]
    returns: Float[Array, "n"]

    logp: Float[Array, "..."]
    value: Float[Array, "..."]


class SelfPlayPPOTrainer(RLTrainer[TState, PPOData], Generic[TState]):
    discount: float = eqx.field(static=True, default=0.96)
    lambda_: float = eqx.field(static=True, default=0.95)

    epoch_n: int = 10
    clip_coef: float = 0.2
    entropy_coef: float = 0.01

    def make_agent(
        self, key: Array, actor: Type[ActorLike], critic: Type[CriticLike]
    ) -> RLAgent:
        observation = self.env.observation_space()
        action = self.env.action_space()

        assert isinstance(self.env, GroupEnv), "only multi agent is support"

        def is_leaf(x):
            return id(x) != id(observation)

        assert jax.tree.all(
            jax.tree.map(
                lambda a, b: type(a) is type(b)
                and a.shape == b.shape
                and a.dtype == b.dtype,
                *jax.tree.leaves(observation, lambda x: id(x) != id(observation)),
            )
        ), "Agents observation space must be the same"
        assert jax.tree.all(
            jax.tree.map(
                lambda a, b: type(a) is type(b)
                and a.shape == b.shape
                and a.dtype == b.dtype,
                *jax.tree.leaves(action, lambda x: id(x) != id(action)),
            )
        ), "Agents action space must be the same"

        obs = jax.tree.leaves(observation, lambda x: id(x) != id(observation))[0]
        act = jax.tree.leaves(action, lambda x: id(x) != id(action))[0]

        actork, critick = jax.random.split(key)
        actors = Actor(actork, actor, obs, act)
        critics = Critic(critick, critic, obs)

        return RLAgent(
            actor=actors,
            critic=critics,
            optactor=jax.tree.map(
                lambda actor: actor.opt_state(self.optim),
                actors,
                is_leaf=of_instance(Actor),
            ),
            optcritic=jax.tree.map(
                lambda critic: critic.opt_state(self.optim),
                critics,
                is_leaf=of_instance(Critic),
            ),
        )

    def map(self, fn, agent: RLAgent, *trees):
        tree, *trees = trees
        return jax.tree.map(
            lambda *x: fn(agent.actor, agent.critic, *x),
            tree,
            *trees,
            is_leaf=lambda x: id(x) != id(tree),
        )

    def _update(self, agent: RLAgent, data: PPOData) -> RLAgent:
        gradient = self.map(self.gradient, agent, data)

        updates = jax.tree.leaves(gradient, is_leaf=lambda x: id(x) != id(gradient))
        actor_grad, critic_grad = jax.tree.map(lambda x, y: x + y, *updates)

        actor, optactor = agent.actor.update(actor_grad, agent.optactor, self.optim)
        critic, optcritic = agent.critic.update(
            critic_grad, agent.optcritic, self.optim
        )

        return RLAgent(
            actor=actor, critic=critic, optactor=optactor, optcritic=optcritic
        )

    def update(self, agent, data):
        params, static = eqx.partition(agent, eqx.is_array)

        def epoch(_, params):
            agent = eqx.combine(params, static)
            agent = self._update(agent, data)

            params = eqx.filter(agent, eqx.is_array)

            return params

        params = jax.lax.fori_loop(0, self.epoch_n, epoch, params)

        return eqx.combine(params, static)

    def compute_data(self, agent, rs):
        rs, transition = jax.vmap(self.rollout, in_axes=(0, None))(rs, agent)

        def lambda_(actor, critic, transition: Transition, obsv):
            advantage, returns = jax.vmap(self.compute_advantage_and_returns)(
                transition, jax.vmap(critic)(obsv)
            )

            logp, _ = self.logp_entropy(
                actor, transition.observation, transition.action
            )
            value = self.values(critic, transition.observation)

            return PPOData(
                transition=transition,
                advantage=advantage,
                returns=returns,
                logp=logp,
                value=value,
            )

        return rs, self.map(lambda_, agent, transition, rs.obs)

    def gradient(self, actor, critic, data):
        tran, advantage, returns, init_logp, init_value = data

        @eqx.filter_grad
        def actor_loss(actor, obs, action, advantage):
            logp, entropy = self.logp_entropy(actor, obs, action)

            ratio = jnp.exp(logp - init_logp)
            advantage = jnp.expand_dims(advantage, axis=-1)

            return (
                -jnp.minimum(
                    ratio * advantage,
                    jnp.clip(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * advantage,
                ).mean()
                - self.entropy_coef * entropy.mean()
            )

        @eqx.filter_grad
        def critic_loss(critic, obs, returns):
            value = self.values(critic, obs)
            returns = jnp.expand_dims(returns, axis=-1)

            loss1 = (value - returns) ** 2

            clipped = init_value + jnp.clip(
                value - init_value, -self.clip_coef, self.clip_coef
            )
            loss2 = (clipped - returns) ** 2

            return 0.5 * jnp.maximum(loss1, loss2).mean()

        actor_grad = actor_loss(actor, tran.observation, tran.action, advantage)
        critic_grad = critic_loss(critic, tran.observation, returns)

        return actor_grad, critic_grad

    @staticmethod
    def logp_entropy(actor, obsv, action):
        logits = jax.vmap(jax.vmap(actor))(obsv)

        log_probs = jax.tree.map(
            lambda logits, action: Categorical(logits=logits).log_prob(action),
            logits,
            action,
        )
        log_probs = jnp.concat(jax.tree.leaves(log_probs), axis=-1)

        entropies = jax.tree.map(
            lambda logits: Categorical(logits=logits).entropy(), logits
        )
        entropies = jnp.concat(jax.tree.leaves(entropies), axis=-1)

        return log_probs, entropies

    @staticmethod
    def values(critic, obsv):
        return jax.vmap(jax.vmap(critic))(obsv)

    # def update(self, actor, critic, optactor, optcritic, transition, bootstraps):
    #     advantage, returns = jax.vmap(self.compute_advantage_and_returns)(
    #         transition, jax.vmap(critic)(bootstraps)
    #     )

    #     def act(actor, obs, action):
    #         logits = jax.vmap(jax.vmap(actor))(obs)

    #         log_probs = jax.tree.map(
    #             lambda logits, action: Categorical(logits=logits).log_prob(action),
    #             logits,
    #             action,
    #         )
    #         log_probs = jnp.concat(jax.tree.leaves(log_probs), axis=-1)

    #         entropies = jax.tree.map(
    #             lambda logits: Categorical(logits=logits).entropy(), logits
    #         )
    #         entropies = jnp.concat(jax.tree.leaves(entropies), axis=-1)

    #         return log_probs, entropies

    #     def values(critic, obs):
    #         return jax.vmap(jax.vmap(critic))(obs)

    #     init_log_p, _ = act(actor, transition.observation, transition.action)
    #     init_value = values(critic, transition.observation)

    #     @eqx.filter_grad
    #     def actor_loss(actor, obs, action, advantage):
    #         log_p, entropy = act(actor, obs, action)

    #         ratio = jnp.exp(log_p - init_log_p)
    #         advantage = jnp.expand_dims(advantage, axis=-1)

    #         return (
    #             -jnp.minimum(
    #                 ratio * advantage,
    #                 jnp.clip(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * advantage,
    #             ).mean()
    #             - self.entropy_coef * entropy.mean()
    #         )

    #     @eqx.filter_grad
    #     def critic_loss(critic, obs, returns):
    #         value = values(critic, obs)
    #         returns = jnp.expand_dims(returns, axis=-1)

    #         loss1 = (value - returns) ** 2

    #         clipped = init_value + jnp.clip(
    #             value - init_value, -self.clip_coef, self.clip_coef
    #         )
    #         loss2 = (clipped - returns) ** 2

    #         return 0.5 * jnp.maximum(loss1, loss2).mean()

    #     params, static = eqx.partition((actor, critic), eqx.is_array)

    #     def epoch(_, epoch_state):
    #         params, optactor, optcritic = epoch_state

    #         actor, critic = eqx.combine(params, static)

    #         actor_grad = actor_loss(
    #             actor, transition.observation, transition.action, advantage
    #         )
    #         critic_grad = critic_loss(critic, transition.observation, returns)

    #         actor, optactor = actor.update(actor_grad, optactor, self.optim)
    #         critic, optcritic = critic.update(critic_grad, optcritic, self.optim)

    #         params = eqx.filter((actor, critic), eqx.is_array)

    #         return params, optactor, optcritic

    #     params, optactor, optcritic = jax.lax.fori_loop(
    #         0, self.epoch_n, epoch, (params, optactor, optcritic)
    #     )
    #     actor, critic = eqx.combine(params, static)

    #     return UpdatePkg(
    #         actor=actor,
    #         critic=critic,
    #         optactor=optactor,
    #         optcritic=optcritic,
    #     )

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
