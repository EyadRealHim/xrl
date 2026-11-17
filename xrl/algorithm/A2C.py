from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import optax  # type: ignore[import-untyped]
from beartype import beartype
from distrax import Distribution, Transformed  # type: ignore[import-untyped]
from jaxtyping import Array, Bool, Float, Key, jaxtyped

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
) -> tuple[Float[Array, " step_n"], Float[Array, " step_n"]]:
    """Generalised Advantage Estimation over one environment's time axis.

    `next_value` is the critic's value of the observation that *followed* each step
    (the terminal observation when an episode ended), so truncated episodes are
    bootstrapped while terminated ones are not. The GAE recursion is cut at both
    kinds of episode boundary.

    Returns `(advantage, returns)`, where `returns = advantage + value` is the
    lambda-return used as the critic's regression target.
    """
    nonterminal = 1.0 - terminated.astype(value.dtype)
    alive = 1.0 - jnp.logical_or(terminated, truncated).astype(value.dtype)

    deltas = reward + discount * nonterminal * next_value - value

    def compute(carry, xs):
        delta, value, alive = xs

        advantage = delta + alive * discount * lambda_ * carry

        return advantage, (advantage, advantage + value)

    _, (advantage, returns) = jax.lax.scan(
        compute,
        jnp.zeros((), value.dtype),
        (deltas, value, alive),
        reverse=True,
    )

    return advantage, returns


def action_entropy(dist: Distribution, key: Key[Array, ""]) -> Float[Array, ""]:
    """Entropy (in nats) of the distribution over a single action.

    `Box` actions are tanh-squashed (`distrax.Transformed`), and distrax has no closed
    form for a bijector whose Jacobian isn't constant. There we use
    `H(Y) = H(X) + E[log|det dY/dX|]`, estimating the expectation with one
    reparameterised sample from the base distribution.
    """
    if isinstance(dist, Transformed):
        base = dist.distribution
        x = base.sample(seed=key)

        return base.entropy() + dist.bijector.forward_log_det_jacobian(x)

    return dist.entropy()


def zero_linear(key, in_features: int, out_features: int) -> eqx.nn.Linear:
    """Linear layer with zero weights and bias.

    Used for the output heads: a zero `logits` head gives a uniform starting policy
    and a zero `value` head gives a zero initial value estimate.
    """
    layer = eqx.nn.Linear(in_features, out_features, use_bias=True, key=key)
    assert layer.bias is not None, "How Did We Get Here?"

    return eqx.tree_at(
        lambda lin: (lin.weight, lin.bias),
        layer,
        (jnp.zeros_like(layer.weight), jnp.zeros_like(layer.bias)),
    )


def apply_gradients(optim: optax.GradientTransformation, params, opt_state, grads):
    """One optimizer step on the inexact-array leaves of `params`."""
    updates, opt_state = optim.update(
        grads, opt_state, eqx.filter(params, eqx.is_inexact_array)
    )

    return eqx.apply_updates(params, updates), opt_state


@jaxtyped(typechecker=beartype)
class A2C(Algorithm):
    """Advantage Actor-Critic with GAE.

    Each `Trainer` step rolls out `env_n * step_n` samples with the current policy and
    makes a single gradient step on them, for the actor and the critic separately.

    Actor and critic keep separate parameters and optimizer states, so there is no
    shared-trunk loss to balance and therefore no value-loss coefficient.

    Only transitions flagged `online` (the agent was active) contribute to the
    losses and to the advantage normalisation statistics.

    Args:
        discount: Discount factor used for GAE.
        lambda_: GAE lambda.
        ent_coef: Weight of the entropy bonus (`0.0` disables it).
        normalize_advantage: Standardise advantages over the online transitions of
            the rollout before they weight the policy gradient. The critic's targets
            are unaffected.
    """

    discount: float = eqx.field(static=True, default=0.98)
    lambda_: float = eqx.field(static=True, default=0.95)
    ent_coef: float = eqx.field(static=True, default=0.0)
    normalize_advantage: bool = eqx.field(static=True, default=False)

    def __check_init__(self):
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError(f"A2C `discount` must be in [0, 1], got {self.discount}.")

        if not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError(f"A2C `lambda_` must be in [0, 1], got {self.lambda_}.")

        if self.ent_coef < 0.0:
            raise ValueError(f"A2C `ent_coef` must be >= 0, got {self.ent_coef}.")

    @jaxtyped(typechecker=beartype)
    def train(self, key, agent: A2CAgent, transition: Transition) -> A2CAgent:
        env_n, step_n = transition.reward.shape
        batch_n = env_n * step_n

        # (env_n, step_n, ...) -> (batch_n, ...); GAE below needs the time axis, the
        # losses don't.
        def flatten(tree):
            return jax.tree.map(lambda x: x.reshape((batch_n,) + x.shape[2:]), tree)

        obs, action, next_obs = flatten(
            (
                transition.observation,
                transition.action,
                transition.info[TERMINAL_OBSERVATION],
            )
        )

        # Train only on transitions where the agent was active. Shapes stay static
        # for jit, so offline entries get zero weight instead of being indexed out.
        online = transition.online.reshape(batch_n)
        n_online = jnp.maximum(online.sum(), 1)  # fully offline batch -> zero loss

        def masked_mean(x):
            return jnp.where(online, x, 0.0).sum() / n_online

        values = jax.vmap(self.__value, in_axes=(None, 0))

        # GAE needs the critic's values anyway, so keep that forward pass as a VJP and
        # reuse it for the critic update instead of running the network again.
        # `next_value` stays outside the VJP: bootstrap targets carry no gradient.
        value, critic_vjp = eqx.filter_vjp(
            lambda critic: values(critic, obs).squeeze(-1), agent.critic
        )
        next_value = values(agent.critic, next_obs).squeeze(-1)

        # An online step followed by an offline one must not inherit advantage from
        # it: cut the GAE trace there so it bootstraps from V(next_obs) instead.
        next_offline = jnp.pad(~transition.online[:, 1:], ((0, 0), (0, 1)))
        truncated = transition.truncated | next_offline

        def gae(reward, value, next_value, terminated, truncated):
            return estimate_advantage_and_returns(
                reward,
                value,
                next_value,
                terminated,
                truncated,
                self.discount,
                self.lambda_,
            )

        advantage, returns = jax.vmap(gae)(
            transition.reward,
            value.reshape(env_n, step_n),
            next_value.reshape(env_n, step_n),
            transition.terminated,
            truncated,
        )
        advantage, returns = advantage.reshape(batch_n), returns.reshape(batch_n)

        if self.normalize_advantage:
            mean = masked_mean(advantage)
            std = jnp.sqrt(masked_mean((advantage - mean) ** 2))
            advantage = (advantage - mean) / (std + 1e-8)

        def actor_loss(actor, obs, action, advantage, key):
            log_p = jax.vmap(self.__log_prob, in_axes=(None, 0, 0))(actor, obs, action)
            loss = -masked_mean(advantage * log_p)

            if self.ent_coef != 0.0:
                entropy = jax.vmap(self.__entropy, in_axes=(None, 0, 0))(
                    actor, obs, jax.random.split(key, batch_n)
                )
                loss = loss - self.ent_coef * masked_mean(entropy)

            return loss

        def critic_loss(value):
            return masked_mean((value - returns) ** 2)

        actor_grads = eqx.filter_grad(actor_loss)(
            agent.actor, obs, action, advantage, key
        )
        (critic_grads,) = critic_vjp(jax.grad(critic_loss)(value))

        actor, optactor = apply_gradients(
            agent.optim, agent.actor, agent.optactor, actor_grads
        )
        critic, optcritic = apply_gradients(
            agent.optim, agent.critic, agent.optcritic, critic_grads
        )

        return replace(
            agent,
            actor=actor,
            critic=critic,
            optactor=optactor,
            optcritic=optcritic,
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
        obsk, actork, logitsk, critick, valuek = jax.random.split(key, 5)

        # Actor and critic start from the same interpreter weights but are trained
        # independently from then on.
        obs_interpreter = ObservationInterpreter(
            obsk, observation_space, transformations=transformations
        )
        act_interpreter = ActionInterpreter(action_space)

        out_features = obs_interpreter.out_features
        obs_shape = jax.ShapeDtypeStruct((out_features,), jnp.float32)

        # Making an actor + logits layer (zero-initialised: a uniform starting policy)
        actor = config.actor(key=actork, in_features=out_features)
        logits = zero_linear(
            logitsk, jax.eval_shape(actor, obs_shape).size, act_interpreter.features
        )

        # Making a critic + value layer
        critic = config.critic(key=critick, in_features=out_features)
        value = zero_linear(valuek, jax.eval_shape(critic, obs_shape).size, 1)

        pi = A2CAgentActor(
            act_interpreter=act_interpreter,
            obs_interpreter=obs_interpreter,
            network=actor,
            logits=logits,
        )
        v = A2CAgentCritic(obs_interpreter=obs_interpreter, network=critic, value=value)

        return A2CAgent(
            optim=config.optim,
            critic=v,
            actor=pi,
            optcritic=config.optim.init(eqx.filter(v, eqx.is_inexact_array)),
            optactor=config.optim.init(eqx.filter(pi, eqx.is_inexact_array)),
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

    @jaxtyped(typechecker=beartype)
    def __log_prob(self, actor: A2CAgentActor, obs: Observation, action):
        dists = self.__act(actor, obs)
        log_p = jax.tree.map(
            lambda dist, a: dist.log_prob(a),
            dists,
            action,
            is_leaf=lambda x: isinstance(x, Distribution),
        )

        return sum(jax.tree.leaves(log_p))

    @jaxtyped(typechecker=beartype)
    def __entropy(self, actor: A2CAgentActor, obs: Observation, key: Key[Array, ""]):
        dists = self.__act(actor, obs)
        keys = keys_like(
            key,
            jax.tree.structure(
                dists, is_leaf=lambda dist: isinstance(dist, Distribution)
            ),
        )
        entropy = jax.tree.map(
            action_entropy,
            dists,
            keys,
            is_leaf=lambda x: isinstance(x, Distribution),
        )

        return sum(jax.tree.leaves(entropy))
