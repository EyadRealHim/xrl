from typing import (
    TypeAlias,
    Sequence,
    TypeVar,
    Tuple,
    Mapping,
    Generic,
    Union,
    NamedTuple,
)

from jaxtyping import PyTree, Array, Float, Int, Bool, DTypeLike

from PIL import Image

import jax.numpy as jnp
import equinox as eqx

import jax

Observation: TypeAlias = PyTree[Float[Array, "..."]]  # TypeVar("Observation")
Action: TypeAlias = PyTree[Int[Array, "..."]]


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
    reward: Union[PyTree[Float[Array, "..."]], Float[Array, "..."]]
    done: Bool[Array, "..."]


class Environment(eqx.Module, Generic[TState]):
    def autostep(
        self, key: Array, state: TState, action: Action
    ) -> Tuple[TState, TimeStep, Observation]:
        key, stepk = jax.random.split(key)
        state, step, obs = self.step(stepk, state, action)

        state, obs = jax.lax.cond(
            step.done,
            self.reset,
            lambda _: (state, obs),
            key,
        )

        return state, step, obs

    def reset(self, key: Array) -> Tuple[TState, Observation]:
        raise NotImplementedError

    def step(
        self, key: Array, state: TState, action: Action
    ) -> Tuple[TState, TimeStep, Observation]:
        raise NotImplementedError

    def observation_space(self) -> PyTree[Space]:
        raise NotImplementedError

    def action_space(self) -> PyTree[Space]:
        raise NotImplementedError

    def render(self, state: TState) -> Image.Image:
        raise NotImplementedError


class ParallelEnvironment(Environment[TState]):
    def autostep(
        self, key: Array, state: TState, action: Mapping[str, Action]
    ) -> Tuple[TState, TimeStep, Mapping[str, Observation]]:
        return super().autostep(key=key, state=state, action=action)

    def reset(self, key: Array) -> Tuple[TState, Mapping[str, Observation]]:
        raise NotImplementedError

    def step(
        self, key: Array, state: TState, action: Mapping[str, Action]
    ) -> Tuple[TState, TimeStep, Mapping[str, Observation]]:
        raise NotImplementedError

    def observation_space(self) -> Mapping[str, PyTree[Space]]:
        raise NotImplementedError

    def action_space(self) -> Mapping[str, PyTree[Space]]:
        raise NotImplementedError
