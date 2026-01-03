from typing import (
    TypeAlias,
    Sequence,
    TypeVar,
    Tuple,
    Mapping,
    Generic,
    NamedTuple,
)

from jaxtyping import Array, Float, Int, Bool, DTypeLike

from .xrl_tree import prefix, of_instance
from PIL import Image

import jax.numpy as jnp
import equinox as eqx

import jax


T = TypeVar("T")
Map: TypeAlias = Mapping[str, T]

Observation: TypeAlias = Map[Float[Array, "..."]]
Action: TypeAlias = Map[Int[Array, "n"]]


class Space:
    shape: Sequence[int]
    dtype: DTypeLike


class Box(Space):
    def __init__(
        self,
        low,
        high,
        shape: Sequence[int],
    ):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = jnp.float32


class MultiDiscrete(Space):
    def __init__(
        self,
        nvec: Sequence[int],
    ):
        self.nvec = nvec
        self.shape = (len(nvec),)
        self.dtype = jnp.int32


class Discrete(Space):
    def __init__(self, n: int):
        self.n = n
        self.shape = (1,)
        self.dtype = jnp.int32


TState = TypeVar("TState")


class TimeStep(NamedTuple):
    reward: Float[Array, "1"]
    done: Bool[Array, "1"]


class SoloEnv(eqx.Module, Generic[TState]):
    """
    an envitonment for a single-agent
    """

    def autostep(
        self, key: Array, state: TState, action: Action
    ) -> Tuple[TState, TimeStep, Observation]:
        key, stepk = jax.random.split(key)
        state, time, obs = self.step(stepk, state, action)

        assert time.done.shape == (), (
            f"TimeStep.done must be a scalar (shape=()), but got shape={time.done.shape}"
        )
        assert time.done.dtype == jnp.bool_, (
            f"TimeStep.done must be boolean, but got dtype={time.done.dtype}"
        )
        assert time.reward.shape == (), (
            f"TimeStep.reward must be a scalar (shape=()), but got shape={time.reward.shape}"
        )
        assert time.reward.dtype == jnp.float32, (
            f"TimeStep.reward must be float32, but got dtype={time.reward.dtype}"
        )

        state, obs = jax.lax.cond(
            time.done,
            self.reset,
            lambda _: (state, obs),
            key,
        )

        return state, time, obs

    def reset(self, key: Array) -> Tuple[TState, Observation]:
        raise NotImplementedError

    def step(
        self, key: Array, state: TState, action: Action
    ) -> Tuple[TState, TimeStep, Observation]:
        raise NotImplementedError

    def observation_space(self) -> Map[Space]:
        raise NotImplementedError

    def action_space(self) -> Map[Space]:
        raise NotImplementedError

    def render(self, state: TState) -> Image.Image:
        raise NotImplementedError


class GroupEnv(eqx.Module, Generic[TState]):
    def autostep(
        self, key: Array, state: TState, action: Map[Action]
    ) -> Tuple[TState, Map[TimeStep], Map[Observation]]:
        key, stepk = jax.random.split(key)
        state, time, obs = self.step(stepk, state, action)

        assert prefix(action) == prefix(time), (
            f"Agent mismatch: The TimeStep dict keys do not match the Action dict keys.\n"
            f"Expected keys: {list(action.keys())}\n"
            f"Received keys: {list(time.keys())}"
        )

        def check(path, time):
            key = path[-1].key

            if time.reward.shape != ():
                yield f"Agent {key}: reward.shape must be (), got {time.reward.shape}"
            if time.reward.dtype != jnp.float32:
                yield f"Agent {key}: reward.dtype must be float32, got {time.reward.dtype}"

            if time.done.shape != ():
                yield f"Agent {key}: done.shape must be (), got {time.done.shape}"
            if time.done.dtype != jnp.bool_:
                yield f"Agent {key}: done.dtype must be bool, got {time.done.dtype}"

        errors_g = jax.tree.map_with_path(check, time, is_leaf=of_instance(TimeStep))
        errors = [e for errs in jax.tree.leaves(errors_g) for e in errs]

        assert not errors, "\n".join(errors)

        done = jax.tree.reduce(
            lambda a, b: jnp.logical_or(a, b.done),
            time,
            False,
            is_leaf=lambda x: isinstance(x, TimeStep),
        )

        state, obs = jax.lax.cond(done, self.reset, lambda _: (state, obs), key)

        return state, time, obs

    def reset(self, key: Array) -> Tuple[TState, Map[Observation]]:
        raise NotImplementedError

    def step(
        self, key: Array, state: TState, action: Map[Action]
    ) -> Tuple[TState, Map[TimeStep], Map[Observation]]:
        raise NotImplementedError

    def observation_space(self) -> Map[Map[Space]]:
        raise NotImplementedError

    def action_space(self) -> Map[Map[Space]]:
        raise NotImplementedError

    def render(self, state: TState) -> Image.Image:
        raise NotImplementedError
