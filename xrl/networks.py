from .environment import Observation, Space, Box, Discrete, MultiDiscrete

from typing import Type, Sequence, Mapping
from jaxtyping import PyTree, Array, Float
from optax import GradientTransformation, OptState

import jax.numpy as jnp
import equinox as eqx
import numpy as np

import jax


class ActorLike(eqx.Module):
    def __init__(self, key: Array, in_features: int):
        pass

    def __call__(self, obs: Observation) -> Float[Array, "n"]:
        raise NotImplementedError


class CriticLike(eqx.Module):
    def __init__(self, key: Array, in_features: int):
        pass

    def __call__(self, obs: Observation) -> Float[Array, "1"]:
        raise NotImplementedError


class ObservationInterpreter(eqx.Module):
    out_features: int = eqx.field(static=True)

    def __init__(self, obs: PyTree[Space]):
        assert jax.tree.all(
            jax.tree.map(lambda s: isinstance(s, Box) and len(s.shape) == 1, obs)
        )

        self.out_features = sum(sum(x.shape) for x in jax.tree.leaves(obs))

    def interpret(self, obs: PyTree[Observation]) -> Float[Array, "..."]:
        return jnp.squeeze(
            jnp.concat(jax.tree.leaves(jax.tree.map(jnp.atleast_1d, obs)))
        )


class ActionInterpreter(eqx.Module):
    out_features: int = eqx.field(static=True)

    nvec: Sequence[Sequence[int]] = eqx.field(static=True)
    treedef: dict = eqx.field(static=True)

    def __init__(self, action: PyTree[Space]):
        assert jax.tree.all(
            jax.tree.map(
                lambda s: isinstance(s, Discrete) or isinstance(s, MultiDiscrete),
                action,
            )
        )

        leaves, _ = jax.tree.flatten(action)
        nvec = jax.tree.map(
            lambda x: [x.n] if isinstance(x, Discrete) else x.nvec, leaves
        )

        self.out_features = sum(sum(x) for x in nvec)
        self.nvec = nvec
        self.treedef = dict(action)

    def interpret(self, logits: Float[Array, "n"]) -> PyTree[Float[Array, "n"]]:  # noqa: F821
        assert logits.shape[0] == self.out_features

        leaves = []
        logits_ = jax.lax.split(logits, [sum(x) for x in self.nvec])

        for v, logits in zip(self.nvec, logits_):
            nvec = np.array(v)
            y, x = len(nvec), nvec.max()
            grid = jnp.full((y, x), -jnp.inf, dtype=jnp.float32)

            rows = np.repeat(np.arange(y), nvec)
            cols = np.arange(nvec.sum()) - np.repeat(np.cumsum(nvec) - nvec, nvec)

            logits = grid.at[rows, cols].set(logits)

            leaves.append(logits)

        return jax.tree.unflatten(jax.tree.structure(self.treedef), leaves)


class Actor(eqx.Module):
    observation: ObservationInterpreter = eqx.field(static=True)
    action: ActionInterpreter = eqx.field(static=True)

    actor: ActorLike
    head: eqx.nn.Linear

    def __init__(
        self,
        key: Array,
        mactor: Type[ActorLike],
        observation_space: PyTree[Space],
        action_space: PyTree[Space],
    ):
        self.observation = ObservationInterpreter(observation_space)
        self.action = ActionInterpreter(action_space)

        self.actor = mactor(key=key, in_features=self.observation.out_features)

        output = jax.eval_shape(
            self.actor,
            jax.ShapeDtypeStruct((self.observation.out_features,), jnp.float32),
        )

        head = eqx.nn.Linear(output.size, self.action.out_features, key=key)

        head = eqx.tree_at(
            lambda layer: layer.weight, head, jnp.zeros_like(head.weight)
        )
        self.head = eqx.tree_at(
            lambda layer: layer.bias, head, jnp.zeros_like(head.bias)
        )

    def opt_state(self, optim: GradientTransformation):
        return optim.init(eqx.filter(self, eqx.is_inexact_array))

    def update(self, grad, opt: OptState, optim: GradientTransformation):
        updates, opt = optim.update(grad, opt, eqx.filter(self, eqx.is_inexact_array))

        return eqx.apply_updates(self, updates), opt

    def __call__(self, obs: Float[Array, "..."]):
        obs = self.observation.interpret(obs)
        logits = self.head(self.actor(obs))

        return self.action.interpret(logits)


class Critic(eqx.Module):
    observation: ObservationInterpreter = eqx.field(static=True)
    critic: CriticLike

    def __init__(
        self, key: Array, mcritic: Type[CriticLike], observation_space: PyTree[Space]
    ):
        self.observation = ObservationInterpreter(observation_space)
        self.critic = mcritic(key=key, in_features=self.observation.out_features)

    def opt_state(self, optim: GradientTransformation):
        return optim.init(eqx.filter(self, eqx.is_inexact_array))

    def update(self, grad, opt: OptState, optim: GradientTransformation):
        updates, opt = optim.update(grad, opt, eqx.filter(self, eqx.is_inexact_array))

        return eqx.apply_updates(self, updates), opt

    def __call__(self, obs: Float[Array, "..."]):
        obs = self.observation.interpret(obs)

        return self.critic(obs)


class ActorContainer(eqx.Module):
    actors: Mapping[str, Actor]

    @staticmethod
    def create(
        key: Array,
        mactor: Type[ActorLike],
        observation_space: Mapping[str, PyTree[Space]],
        action_space: Mapping[str, PyTree[Space]],
    ):
        actors = {}

        for name, act_space, obs_space in zip(
            action_space.keys(), action_space.values(), observation_space.values()
        ):
            key, subkey = jax.random.split(key)
            actors[name] = Actor(subkey, mactor, obs_space, act_space)

        return ActorContainer(actors=actors)

    def opt_state(self, optim: GradientTransformation):
        return {k: actor.opt_state(optim) for k, actor in self.actors.items()}

    def update(self, grad, opt: dict[str, OptState], optim: GradientTransformation):
        return {
            k: actor.update(grad[k], opt[k], optim) for k, actor in self.actors.items()
        }

    def __call__(self, obs: Mapping[str, Float[Array, "..."]]):
        return {k: actor(obs[k]) for k, actor in self.actors.items()}


class CriticContainer(eqx.Module):
    critics: Mapping[str, Critic]

    @staticmethod
    def create(
        key: Array,
        mcritic: Type[CriticLike],
        observation_space: Mapping[str, PyTree[Space]],
    ):
        critics = {}

        for name, obs_space in zip(
            observation_space.keys(), observation_space.values()
        ):
            key, subkey = jax.random.split(key)
            critics[name] = Critic(subkey, mcritic, obs_space)

        return CriticContainer(critics=critics)

    def opt_state(self, optim: GradientTransformation):
        return {k: critic.opt_state(optim) for k, critic in self.critics.items()}

    def update(self, grad, opt: dict[str, OptState], optim: GradientTransformation):
        return {
            k: critic.update(grad[k], opt[k], optim)
            for k, critic in self.critics.items()
        }

    def __call__(self, obs: Mapping[str, Float[Array, "..."]]):
        return {k: critic(obs[k]) for k, critic in self.critics.items()}
