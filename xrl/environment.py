from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, DTypeLike, Float, Int, Key, PyTree, UInt8, jaxtyped


class Space(eqx.Module):
    """Base class representing a state or action space."""

    shape: Sequence[int]
    dtype: DTypeLike


class Discrete(Space):
    """A finite discrete space running from 0 to n-1."""

    n: int

    @jaxtyped(typechecker=beartype)
    def __init__(self, n: int):
        """Initializes a Discrete space.

        Args:
            n: Number of discrete outcomes (must be strictly positive).
        """
        if n <= 0:
            raise ValueError(
                f"Discrete space size 'n' must be a positive integer, got {n}."
            )

        object.__setattr__(self, "n", n)
        object.__setattr__(self, "shape", ())
        object.__setattr__(self, "dtype", jnp.int32)


class MultiDiscrete(Space):
    """A set of discrete spaces with independent dimensions."""

    nvec: Sequence[int]

    @jaxtyped(typechecker=beartype)
    def __init__(self, nvec: Sequence[int]):
        """Initializes a MultiDiscrete space.

        Args:
            nvec: Sequence specifying the number of categories for each dimension.
        """
        if len(nvec) == 0:
            raise ValueError("MultiDiscrete space 'nvec' cannot be empty.")

        if any(n < 1 for n in nvec):
            raise ValueError(
                f"All elements in MultiDiscrete 'nvec' must be positive integers, got {nvec}."
            )

        object.__setattr__(self, "nvec", nvec)
        object.__setattr__(self, "shape", (len(nvec),))
        object.__setattr__(self, "dtype", jnp.int32)


class Box(Space):
    """A continuous space defined by lower and upper bounds."""

    low: float
    high: float

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        low: float,
        high: float,
        shape: Sequence[int],
    ):
        """Initializes a Box space.

        Args:
            low: Lower bound scalar.
            high: Upper bound scalar.
            shape: Shape of the box space.
        """
        if low > high:
            raise ValueError(
                f"Box 'low' must be less than or equal to 'high'. Got low={low}, high={high}."
            )

        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "shape", tuple(shape))
        object.__setattr__(self, "dtype", jnp.float32)


type Observation = PyTree[Float[Array, "..."]]
type Action = PyTree[Float[Array, "..."] | Int[Array, ""] | Int[Array, " nvec"]]
type EnvInfo = dict

TERMINAL_OBSERVATION = "TERMINAL_OBSERVATION_DO_NOT_OVERWRITE"


@jaxtyped(typechecker=beartype)
class MonoTimeStep(eqx.Module):
    reward: Float[Array, ""]
    terminated: Bool[Array, ""]
    truncated: Bool[Array, ""]


class MonoEnv[TState](eqx.Module):
    """
    an envitonment for a single-agent
    """

    @jaxtyped(typechecker=beartype)
    def autostep(
        self, key: Key[Array, ""], state: TState, action: Action
    ) -> tuple[TState, MonoTimeStep, Observation, EnvInfo]:
        key, stepk = jax.random.split(key)
        state, time, obs, info = self.step(stepk, state, action)

        info[TERMINAL_OBSERVATION] = obs

        state, obs = jax.lax.cond(
            jnp.logical_or(time.terminated, time.truncated),
            self.reset,
            lambda _: (state, obs),
            key,
        )

        return state, time, obs, info

    @jaxtyped(typechecker=beartype)
    def reset(self, key: Key[Array, ""]) -> tuple[TState, Observation]:
        raise NotImplementedError

    @jaxtyped(typechecker=beartype)
    def step(
        self, key: Array, state: TState, action: Action
    ) -> tuple[TState, MonoTimeStep, Observation, EnvInfo]:
        raise NotImplementedError

    @jaxtyped(typechecker=beartype)
    def observation_space(self) -> PyTree[Space]:
        raise NotImplementedError

    @jaxtyped(typechecker=beartype)
    def action_space(self) -> PyTree[Space]:
        raise NotImplementedError

    @jaxtyped(typechecker=beartype)
    def render(self, state: TState) -> UInt8[Array, "height width 3"]:
        raise NotImplementedError
