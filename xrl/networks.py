from .environment import Observation, Space, Box, Discrete, MultiDiscrete, Map

from typing import Type, Sequence
from jaxtyping import PyTree, Array, Float
from optax import GradientTransformation, OptState

import jax.numpy as jnp
import equinox as eqx
import numpy as np

import jax


class ActorLike(eqx.Module):
    def __init__(self, key: Array, in_features: int):
        pass

    def __call__(self, obs: Float[Array, "n"]) -> Float[Array, "m"]:
        raise NotImplementedError


class CriticLike(eqx.Module):
    def __init__(self, key: Array, in_features: int):
        pass

    def __call__(self, obs: Float[Array, "n"]) -> Float[Array, "m"]:
        raise NotImplementedError


class ObservationInterpreter(eqx.Module):
    out_features: int = eqx.field(static=True)

    # for asserations
    fields: Map[Sequence[int]] = eqx.field(static=True)

    def __init__(self, obs: Map[Space]):
        assert all(isinstance(x, Box) and len(x.shape) == 1 for x in obs.values())

        self.out_features = sum(sum(x.shape) for x in obs.values())
        self.fields = {k: space.shape for k, space in obs.items()}

    def interpret(self, obs: Observation) -> Float[Array, "n"]:
        assert isinstance(obs, dict), (
            f"observation must be of type dict, instead got '{obs}' of type '{type(obs)}'"
        )

        for k, shape in self.fields.items():
            assert k in obs, f"field '{k}' is missing in observation: {obs}"
            assert obs[k].shape == shape, (
                f"field '{k}' in observation must be of shape '{shape}' as described by the environment, but instead got '{obs[k].shape}'"
            )

        for k in obs.keys():
            assert k in self.fields, (
                f"field '{k}' is not described by the environment observation"
            )

        return jnp.concatenate(jax.tree.leaves(obs), axis=0)


class ActionInterpreter(eqx.Module):
    out_features: int = eqx.field(static=True)

    nvec: Sequence[Sequence[int]] = eqx.field(static=True)
    treedef: dict = eqx.field(static=True)

    def __init__(self, action: Map[Space]):
        assert all(
            isinstance(x, Discrete) or isinstance(x, MultiDiscrete)
            for x in action.values()
        )

        self.nvec = [
            [x.n] if isinstance(x, Discrete) else x.nvec for x in action.values()
        ]
        self.out_features = sum(sum(x) for x in self.nvec)
        self.treedef = dict(action)

    def interpret(self, logits: Float[Array, "n"]) -> PyTree[Float[Array, "n"]]:  # noqa: F821
        assert logits.shape[0] == self.out_features and len(logits.shape) == 1

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
        observation_space: Map[Space],
        action_space: Map[Space],
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

    def __call__(self, obs: Observation):
        obsv = self.observation.interpret(obs)
        logits = self.head(self.actor(obsv))

        return self.action.interpret(logits)


class Critic(eqx.Module):
    observation: ObservationInterpreter = eqx.field(static=True)
    critic: CriticLike
    head: eqx.nn.Linear

    def __init__(
        self, key: Array, mcritic: Type[CriticLike], observation_space: Map[Space]
    ):
        self.observation = ObservationInterpreter(observation_space)
        self.critic = mcritic(key=key, in_features=self.observation.out_features)

        output = jax.eval_shape(
            self.critic,
            jax.ShapeDtypeStruct((self.observation.out_features,), jnp.float32),
        )

        head = eqx.nn.Linear(output.size, 1, key=key)

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

    def __call__(self, obs: Observation):
        obsv = self.observation.interpret(obs)

        return self.head(self.critic(obsv))
