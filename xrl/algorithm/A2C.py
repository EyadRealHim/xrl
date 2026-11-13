from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import optax  # type: ignore[import-untyped]
from beartype import beartype
from distrax import Distribution  # type: ignore[import-untyped]
from jaxtyping import Array, Bool, Float, jaxtyped

from xrl.tree import keys_like

from ..environment import TERMINAL_OBSERVATION, Observation
from ..networks import ActionInterpreter, ActorLike, CriticLike, ObservationInterpreter
from .algorithm import Algorithm, Transition


@jaxtyped(typechecker=beartype)
class A2CAgentCritic(eqx.Module):
    obs_interpreter: ObservationInterpreter
    network: CriticLike
    value: eqx.nn.Linear


@jaxtyped(typechecker=beartype)
class A2CAgentActor(eqx.Module):
    act_interpreter: ActionInterpreter = eqx.field(static=True)
    obs_interpreter: ObservationInterpreter

    network: ActorLike
    logits: eqx.nn.Linear


@jaxtyped(typechecker=beartype)
class A2CAgent(eqx.Module):
    optim: optax.GradientTransformation = eqx.field(static=True)

    critic: A2CAgentCritic
    actor: A2CAgentActor

    optcritic: optax.OptState
    optactor: optax.OptState


@jaxtyped(typechecker=beartype)
class A2CAgentConfig(eqx.Module):
    optim: optax.GradientTransformation

    actor: type[ActorLike]
    critic: type[CriticLike]


@jaxtyped(typechecker=beartype)
def estimate_advantage_and_returns(
    reward: Float[Array, " step_n"],
    value: Float[Array, " step_n"],
    next_value: Float[Array, " step_n"],
    terminated: Bool[Array, " step_n"],
    truncated: Bool[Array, " step_n"],
    discount: float = 0.98,
    lambda_: float = 0.95,
):
    deltas = reward + discount * (1 - terminated) * next_value - value

    def compute(prev, pair):
        delta, value, mask = pair

        gae = delta + mask * discount * lambda_ * prev

        return gae, (gae, gae + value)

    return jax.lax.scan(
        compute,
        0,
        (deltas, value, 1.0 - jnp.logical_or(terminated, truncated)),
        reverse=True,
    )[1]


class A2C(Algorithm):
    @jaxtyped(typechecker=beartype)
    def train(self, agent: A2CAgent, transition: Transition):
        @jaxtyped(typechecker=beartype)
        def critic_forward(critic: A2CAgentCritic):
            network = jax.vmap(self.__value, in_axes=(None, 0))
            return jax.vmap(network, in_axes=(None, 0))(
                critic, transition.observation
            ).squeeze(-1)

        value, critic_vjp = eqx.filter_vjp(critic_forward, agent.critic)
        critic_network = jax.vmap(self.value, in_axes=(None, 0))
        next_value = jax.vmap(critic_network, in_axes=(None, 0))(
            agent, transition.info[TERMINAL_OBSERVATION]
        ).squeeze(-1)

        advantage, returns = jax.vmap(estimate_advantage_and_returns)(
            transition.reward,
            value,
            next_value,
            transition.terminated,
            transition.truncated,
        )

        @eqx.filter_grad
        @jaxtyped(typechecker=beartype)
        def actor_loss(
            actor: A2CAgentActor,
            obs: Observation,
            advantage: Float[Array, "..."],
        ):
            network = jax.vmap(self.__act, in_axes=(None, 0))
            dists = jax.vmap(network, in_axes=(None, 0))(actor, obs)
            log_p = jax.tree.map(
                lambda d, a: d.log_prob(a),
                dists,
                transition.action,
                is_leaf=lambda x: isinstance(x, Distribution),
            )

            log_p = sum(jax.tree.leaves(log_p))

            return -(advantage * log_p).mean()

        @jaxtyped(typechecker=beartype)
        def critic_loss(value: Float[Array, "..."]):
            return ((value - returns) ** 2).mean()

        grads = actor_loss(agent.actor, transition.observation, advantage)
        updates, optactor = agent.optim.update(
            grads, agent.optactor, eqx.filter(agent.actor, eqx.is_inexact_array)
        )
        actor = eqx.apply_updates(agent.actor, updates)

        (grads,) = critic_vjp(jax.grad(critic_loss)(value))
        updates, optcritic = agent.optim.update(
            grads, agent.optcritic, eqx.filter(agent.critic, eqx.is_inexact_array)
        )
        critic = eqx.apply_updates(agent.critic, updates)

        return replace(
            agent, critic=critic, actor=actor, optcritic=optcritic, optactor=optactor
        )

    @jaxtyped(typechecker=beartype)
    def make_agent(
        self,
        key,
        observation_space,
        action_space,
        config: A2CAgentConfig,
        transformations=None,
    ) -> A2CAgent:
        key, subkey = jax.random.split(key)
        obs_interpreter = ObservationInterpreter(
            subkey, observation_space, transformations=transformations
        )
        act_interpreter = ActionInterpreter(action_space)

        out_features = obs_interpreter.out_features
        obs_shape = jax.ShapeDtypeStruct((out_features,), jnp.float32)

        # Making actor + logits layer
        key, actork, logitsk = jax.random.split(key, 3)
        actor = config.actor(key=actork, in_features=out_features)
        output = jax.eval_shape(actor, obs_shape)

        logits = eqx.nn.Linear(output.size, act_interpreter.features, key=logitsk)
        logits = eqx.tree_at(
            lambda layer: layer.weight, logits, jnp.zeros_like(logits.weight)
        )
        logits = eqx.tree_at(
            lambda layer: layer.bias, logits, jnp.zeros_like(logits.bias)
        )

        # Making an critic + value layer
        key, critick = jax.random.split(key)
        critic = config.critic(key=critick, in_features=out_features)
        output = jax.eval_shape(critic, obs_shape)

        value = eqx.nn.Linear(output.size, 1, key=key)
        value = eqx.tree_at(
            lambda layer: layer.weight, value, jnp.zeros_like(value.weight)
        )
        value = eqx.tree_at(lambda layer: layer.bias, value, jnp.zeros_like(value.bias))

        pi = A2CAgentActor(act_interpreter, obs_interpreter, actor, logits)
        v = A2CAgentCritic(obs_interpreter, critic, value)

        return A2CAgent(
            actor=pi,
            critic=v,
            optim=config.optim,
            optactor=config.optim.init(eqx.filter(pi, eqx.is_inexact_array)),
            optcritic=config.optim.init(eqx.filter(v, eqx.is_inexact_array)),
        )

    @jaxtyped(typechecker=beartype)
    def act(self, key, agent: A2CAgent, obs):
        dists = self.__act(agent.actor, obs)
        keys = keys_like(
            key,
            jax.tree.structure(
                dists, is_leaf=lambda dist: isinstance(dist, Distribution)
            ),
        )

        return jax.tree.map(lambda key, dist: dist.sample(seed=key), keys, dists)

    @jaxtyped(typechecker=beartype)
    def value(self, agent: A2CAgent, obs: Observation):
        return self.__value(agent.critic, obs)

    @jaxtyped(typechecker=beartype)
    def __act(self, actor: A2CAgentActor, obs):
        obsv = actor.obs_interpreter.interpret(obs)
        logits = actor.logits(actor.network(obsv))
        dists = actor.act_interpreter.interpret(logits)

        return dists

    @jaxtyped(typechecker=beartype)
    def __value(self, critic: A2CAgentCritic, obs: Observation):
        obsv = critic.obs_interpreter.interpret(obs)
        return critic.value(critic.network(obsv))
