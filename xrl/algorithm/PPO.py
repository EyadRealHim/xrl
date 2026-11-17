from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import optax  # type: ignore[import-untyped]
from beartype import beartype
from distrax import Distribution, Transformed  # type: ignore[import-untyped]
from jaxtyping import Array, Float, Key, jaxtyped

from xrl.tree import keys_like

from ..environment import TERMINAL_OBSERVATION, Observation
from ..networks import ActionInterpreter, ActorLike, CriticLike, ObservationInterpreter
from .A2C import estimate_advantage_and_returns
from .algorithm import Algorithm, Transition


@jaxtyped(typechecker=beartype)
class PPOAgentCritic(eqx.Module):
    obs_interpreter: ObservationInterpreter
    network: CriticLike
    value: eqx.nn.Linear


@jaxtyped(typechecker=beartype)
class PPOAgentActor(eqx.Module):
    act_interpreter: ActionInterpreter = eqx.field(static=True)
    obs_interpreter: ObservationInterpreter

    network: ActorLike
    logits: eqx.nn.Linear


@jaxtyped(typechecker=beartype)
class PPOAgent(eqx.Module):
    optim: optax.GradientTransformation = eqx.field(static=True)

    critic: PPOAgentCritic
    actor: PPOAgentActor

    optcritic: optax.OptState
    optactor: optax.OptState


@jaxtyped(typechecker=beartype)
class PPOAgentConfig(eqx.Module):
    optim: optax.GradientTransformation

    actor: type[ActorLike]
    critic: type[CriticLike]


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


@jaxtyped(typechecker=beartype)
class PPO(Algorithm):
    """Proximal Policy Optimization: clipped surrogate objective with GAE.

    Each `Trainer` step rolls out `env_n * step_n` samples with the current policy and
    then makes `epochs` passes over them, in `minibatches` shuffled minibatches per
    pass. The rollout is always on-policy (the agent being trained collected it), so
    the behaviour log-probs are recomputed from the pre-update actor instead of being
    stored in `Transition`.

    Actor and critic keep separate parameters and optimizer states.

    Only transitions flagged `online` (the agent was active) contribute to the
    losses and to the advantage normalisation statistics.

    Args:
        discount: Discount factor used for GAE.
        lambda_: GAE lambda.
        clip_eps: The probability ratio is clipped to `[1 - clip_eps, 1 + clip_eps]`.
        epochs: Passes over each rollout.
        minibatches: Minibatches per pass; `env_n * step_n` must be divisible by it.
        ent_coef: Weight of the entropy bonus (`0.0` disables it).
        normalize_advantage: Standardise advantages over the online samples of each
            minibatch.
    """

    discount: float = eqx.field(static=True, default=0.99)
    lambda_: float = eqx.field(static=True, default=0.95)
    clip_eps: float = eqx.field(static=True, default=0.2)
    epochs: int = eqx.field(static=True, default=4)
    minibatches: int = eqx.field(static=True, default=4)
    ent_coef: float = eqx.field(static=True, default=0.0)
    normalize_advantage: bool = eqx.field(static=True, default=True)

    def __check_init__(self):
        if self.epochs < 1 or self.minibatches < 1:
            raise ValueError(
                "PPO needs `epochs` >= 1 and `minibatches` >= 1, "
                f"got epochs={self.epochs}, minibatches={self.minibatches}."
            )

        if self.clip_eps <= 0:
            raise ValueError(f"PPO `clip_eps` must be positive, got {self.clip_eps}.")

    @jaxtyped(typechecker=beartype)
    def train(self, key, agent: PPOAgent, transition: Transition) -> PPOAgent:
        env_n, step_n = transition.reward.shape
        batch_n = env_n * step_n

        if batch_n % self.minibatches != 0:
            raise ValueError(
                f"PPO cannot split {batch_n} samples ({env_n} envs x {step_n} steps) "
                f"into {self.minibatches} equal minibatches."
            )

        minibatch_n = batch_n // self.minibatches
        optim = agent.optim

        # Train only on transitions where the agent was active. Shapes stay static
        # for jit, so offline entries get zero weight instead of being indexed out.
        def masked_mean(x, mask):
            # An all-offline minibatch gives a zero loss rather than 0/0.
            return jnp.where(mask, x, 0.0).sum() / jnp.maximum(mask.sum(), 1)

        # Advantages and the ratio's denominator both come from the pre-update agent.
        def values(obs: Observation):
            network = jax.vmap(self.__value, in_axes=(None, 0))

            return jax.vmap(network, in_axes=(None, 0))(agent.critic, obs).squeeze(-1)

        value = values(transition.observation)
        next_value = values(transition.info[TERMINAL_OBSERVATION])

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
            value,
            next_value,
            transition.terminated,
            truncated,
        )

        log_prob = jax.vmap(
            jax.vmap(self.__log_prob, in_axes=(None, 0, 0)), in_axes=(None, 0, 0)
        )
        old_log_p = log_prob(agent.actor, transition.observation, transition.action)

        # (env_n, step_n, ...) -> (batch_n, ...); GAE above needed the time axis. The
        # online mask travels with the samples through the shuffle below.
        batch = jax.tree.map(
            lambda x: x.reshape((batch_n,) + x.shape[2:]),
            (
                transition.observation,
                transition.action,
                old_log_p,
                advantage,
                returns,
                transition.online,
            ),
        )

        def actor_loss(actor, obs, action, old_lp, adv, online, key):
            log_p = jax.vmap(self.__log_prob, in_axes=(None, 0, 0))(actor, obs, action)

            if self.normalize_advantage:
                mean = masked_mean(adv, online)
                std = jnp.sqrt(masked_mean((adv - mean) ** 2, online))
                adv = (adv - mean) / (std + 1e-8)

            ratio = jnp.exp(log_p - old_lp)
            clipped = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            loss = -masked_mean(jnp.minimum(ratio * adv, clipped * adv), online)

            if self.ent_coef != 0.0:
                entropy = jax.vmap(self.__entropy, in_axes=(None, 0, 0))(
                    actor, obs, jax.random.split(key, minibatch_n)
                )
                loss = loss - self.ent_coef * masked_mean(entropy, online)

            return loss

        def critic_loss(critic, obs, target, online):
            value = jax.vmap(self.__value, in_axes=(None, 0))(critic, obs).squeeze(-1)

            return masked_mean((value - target) ** 2, online)

        # Only arrays travel through the scans; modules also hold non-array leaves
        # (activation functions and the like), which stay fixed in `static`.
        state, static = eqx.partition(
            (agent.actor, agent.critic, agent.optactor, agent.optcritic), eqx.is_array
        )

        def minibatch(state, xs):
            actor, critic, optactor, optcritic = eqx.combine(state, static)
            (obs, action, old_lp, adv, target, online), key = xs

            grads = eqx.filter_grad(actor_loss)(
                actor, obs, action, old_lp, adv, online, key
            )
            updates, optactor = optim.update(
                grads, optactor, eqx.filter(actor, eqx.is_inexact_array)
            )
            actor = eqx.apply_updates(actor, updates)

            grads = eqx.filter_grad(critic_loss)(critic, obs, target, online)
            updates, optcritic = optim.update(
                grads, optcritic, eqx.filter(critic, eqx.is_inexact_array)
            )
            critic = eqx.apply_updates(critic, updates)

            return eqx.filter((actor, critic, optactor, optcritic), eqx.is_array), None

        def epoch(state, key):
            permk, stepk = jax.random.split(key)
            perm = jax.random.permutation(permk, batch_n)

            shuffled = jax.tree.map(
                lambda x: x[perm].reshape(
                    (self.minibatches, minibatch_n) + x.shape[1:]
                ),
                batch,
            )
            keys = jax.random.split(stepk, self.minibatches)

            return jax.lax.scan(minibatch, state, (shuffled, keys))[0], None

        state, _ = jax.lax.scan(epoch, state, jax.random.split(key, self.epochs))

        actor, critic, optactor, optcritic = eqx.combine(state, static)

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
        config: PPOAgentConfig,
        transformations=None,
    ) -> PPOAgent:
        obsk, actork, logitsk, critick, valuek = jax.random.split(key, 5)

        obs_interpreter = ObservationInterpreter(
            obsk, observation_space, transformations=transformations
        )
        act_interpreter = ActionInterpreter(action_space)

        out_features = obs_interpreter.out_features
        obs_shape = jax.ShapeDtypeStruct((out_features,), jnp.float32)

        # Making an actor + logits layer (zero-initialised: a uniform starting policy)
        actor = config.actor(key=actork, in_features=out_features)
        output = jax.eval_shape(actor, obs_shape)

        logits = eqx.nn.Linear(output.size, act_interpreter.features, key=logitsk)
        logits = eqx.tree_at(
            lambda layer: layer.weight, logits, jnp.zeros_like(logits.weight)
        )
        logits = eqx.tree_at(
            lambda layer: layer.bias, logits, jnp.zeros_like(logits.bias)
        )

        # Making a critic + value layer
        critic = config.critic(key=critick, in_features=out_features)
        output = jax.eval_shape(critic, obs_shape)

        value = eqx.nn.Linear(output.size, 1, key=valuek)
        value = eqx.tree_at(
            lambda layer: layer.weight, value, jnp.zeros_like(value.weight)
        )
        value = eqx.tree_at(lambda layer: layer.bias, value, jnp.zeros_like(value.bias))

        pi = PPOAgentActor(
            act_interpreter=act_interpreter,
            obs_interpreter=obs_interpreter,
            network=actor,
            logits=logits,
        )
        v = PPOAgentCritic(obs_interpreter=obs_interpreter, network=critic, value=value)

        return PPOAgent(
            optim=config.optim,
            critic=v,
            actor=pi,
            optcritic=config.optim.init(eqx.filter(v, eqx.is_inexact_array)),
            optactor=config.optim.init(eqx.filter(pi, eqx.is_inexact_array)),
        )

    @jaxtyped(typechecker=beartype)
    def act(self, key, agent: PPOAgent, obs):
        dists = self.__act(agent.actor, obs)
        keys = keys_like(
            key,
            jax.tree.structure(
                dists, is_leaf=lambda dist: isinstance(dist, Distribution)
            ),
        )

        return jax.tree.map(lambda key, dist: dist.sample(seed=key), keys, dists)

    @jaxtyped(typechecker=beartype)
    def value(self, agent: PPOAgent, obs: Observation):
        return self.__value(agent.critic, obs)

    @jaxtyped(typechecker=beartype)
    def __act(self, actor: PPOAgentActor, obs):
        obsv = actor.obs_interpreter.interpret(obs)
        logits = actor.logits(actor.network(obsv))
        dists = actor.act_interpreter.interpret(logits)

        return dists

    @jaxtyped(typechecker=beartype)
    def __value(self, critic: PPOAgentCritic, obs: Observation):
        obsv = critic.obs_interpreter.interpret(obs)
        return critic.value(critic.network(obsv))

    @jaxtyped(typechecker=beartype)
    def __log_prob(self, actor: PPOAgentActor, obs: Observation, action):
        dists = self.__act(actor, obs)
        log_p = jax.tree.map(
            lambda dist, a: dist.log_prob(a),
            dists,
            action,
            is_leaf=lambda x: isinstance(x, Distribution),
        )

        return sum(jax.tree.leaves(log_p))

    @jaxtyped(typechecker=beartype)
    def __entropy(self, actor: PPOAgentActor, obs: Observation, key: Key[Array, ""]):
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
