from collections.abc import Sequence
from typing import Any

import distrax  # type: ignore[import-untyped]
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Array, Float, Key, PyTree, jaxtyped

from .environment import Action, Box, Discrete, MultiDiscrete, Observation, Space
from .tree import keys_like


def space_shape(x: Space):
    if isinstance(x, Discrete):
        return np.sum((x.n,))
    if isinstance(x, MultiDiscrete):
        return np.sum(x.nvec)
    if isinstance(x, Box):
        return np.prod(x.shape) * 2  # mean and logstd for each entry

    raise NotImplementedError()


class ActorLike(eqx.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self, key: Key[Array, ""], in_features: int):
        pass

    @jaxtyped(typechecker=beartype)
    def __call__(self, obs: Float[Array, " n"]) -> Float[Array, " m"]:
        raise NotImplementedError


class CriticLike(eqx.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self, key: Key[Array, ""], in_features: int):
        pass

    @jaxtyped(typechecker=beartype)
    def __call__(self, obs: Float[Array, " n"]) -> Float[Array, " m"]:
        raise NotImplementedError


class ObservationTransformation(eqx.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self, key: Key[Array, ""], inshape: Sequence[int]):
        raise NotImplementedError

    @jaxtyped(typechecker=beartype)
    def __call__(self, xs: Array):
        raise NotImplementedError


class Flatten(ObservationTransformation):
    def __init__(self, *args: Any, **kwargs: Any):
        """Consumes arbitrary `*args` and `**kwargs` but ignores them."""

    def __call__(self, x: Array) -> Array:
        return jnp.ravel(x)


class ObservationInterpreter(eqx.Module):
    spaces: PyTree[Space] = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    transformations: PyTree[ObservationTransformation]

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        key: Key[Array, ""],
        spaces: PyTree[Space],
        transformations: PyTree[type[ObservationTransformation]] | None,
    ):
        self.spaces = spaces

        transform_pairs, _ = jax.tree_util.tree_flatten_with_path(
            transformations,
            is_leaf=lambda x: (
                isinstance(x, type) and issubclass(x, ObservationTransformation)
            ),
        )
        space_pairs, space_structure = jax.tree_util.tree_flatten_with_path(
            spaces, is_leaf=lambda x: isinstance(x, Space)
        )

        transform_by_path = dict(transform_pairs)

        extra = transform_by_path.keys() - {path for path, _ in space_pairs}
        if extra:
            raise ValueError(
                f"Transformations contain paths outside spaces: "
                f"{[jax.tree_util.keystr(p) for p in extra]}"
            )

        leaves = [transform_by_path.get(path, Flatten) for path, _ in space_pairs]
        self.transformations = jax.tree.map(
            lambda key, space, module: module(key=key, inshape=space.shape),
            keys_like(key, space_structure),
            spaces,
            jax.tree_util.tree_unflatten(space_structure, leaves),
        )

        out_features = jax.tree.map(
            lambda space, transformation: np.prod(
                jax.eval_shape(
                    transformation, jax.ShapeDtypeStruct(space.shape, dtype=space.dtype)
                ).shape
            ),
            self.spaces,
            self.transformations,
            is_leaf=lambda x: isinstance(x, Space),
        )
        out_features = jax.tree.leaves(out_features)
        self.out_features = int(np.sum(out_features).item())

    @jaxtyped(typechecker=beartype)
    def interpret(self, observation: Observation) -> Float[Array, " n"]:
        is_space = lambda x: isinstance(x, Space)

        from operator import eq  # mypy needs this :D

        assert eq(
            jax.tree.structure(self.spaces, is_leaf=is_space),
            jax.tree.structure(observation),
        ), f"Observation structure mismatch: expected `{
            jax.tree.structure(self.spaces, is_leaf=is_space)
        }` for the observation space but found `{jax.tree.structure(observation)}`"

        @jaxtyped(typechecker=beartype)
        def fn(
            space: Space,
            y: Float[Array, "..."],
            transform: ObservationTransformation,
        ):
            assert space.shape == y.shape, (
                f"Observation Shape mismatch: expect `{space.__class__.__name__}` with shape {space.shape} but got {y.shape}"
            )

            return transform(y)

        data = jax.tree.leaves(
            jax.tree.map(
                fn, self.spaces, observation, self.transformations, is_leaf=is_space
            )
        )

        return jnp.concatenate(data, axis=0)


class ActionInterpreter(eqx.Module):
    spaces: PyTree[Space] = eqx.field(static=True)
    features: int = eqx.field(static=True)

    @jaxtyped(typechecker=beartype)
    def __init__(self, spaces: PyTree[Space]):
        self.spaces = spaces

        self.features = int(
            np.sum(
                jax.tree.leaves(
                    jax.tree.map(
                        space_shape,
                        self.spaces,
                        is_leaf=lambda x: isinstance(x, Space),
                    )
                )
            ).item()
        )

    @jaxtyped(typechecker=beartype)
    def interpret(self, logits: Float[Array, " n"]) -> Action:
        assert len(logits.shape) == 1
        assert logits.shape[-1] == self.features

        leaves, treedef = jax.tree.flatten(
            jax.tree.map(
                lambda x: np.prod(space_shape(x)),
                self.spaces,
                is_leaf=lambda x: isinstance(x, Space),
            )
        )
        logits = jax.tree.unflatten(treedef, jax.lax.split(logits, leaves))

        @jaxtyped(typechecker=beartype)
        def map(logits: Float[Array, "..."], space: Space):
            if isinstance(space, Discrete):
                return distrax.Categorical(
                    logits=self.remap((space.n,), logits).reshape((space.n,))
                )
            if isinstance(space, MultiDiscrete):
                return distrax.Independent(
                    distrax.Categorical(logits=self.remap(space.nvec, logits)),
                    reinterpreted_batch_ndims=1,
                )
            if isinstance(space, Box):
                logits = jnp.reshape(logits, (2, -1))
                mean, logstd = jnp.split(logits, 2, axis=0)

                mean = mean.reshape(space.shape)
                logstd = logstd.reshape(space.shape)

                base = distrax.Independent(
                    distrax.Normal(
                        loc=mean,
                        scale=jnp.exp(logstd),
                    ),
                    reinterpreted_batch_ndims=len(space.shape),
                )

                scale = (space.high - space.low) / 2
                bias = (space.high + space.low) / 2

                transform = distrax.Chain(
                    [
                        distrax.Block(
                            distrax.ScalarAffine(shift=bias, scale=scale),
                            ndims=len(space.shape),
                        ),
                        distrax.Block(
                            distrax.Tanh(),
                            ndims=len(space.shape),
                        ),
                    ]
                )

                return distrax.Transformed(base, transform)

        return jax.tree.map(map, logits, self.spaces)

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def remap(nvec_list: Sequence[int], logits: Float[Array, " n"]):
        nvec = np.array(nvec_list)
        y, x = len(nvec), nvec.max()
        grid = jnp.full((y, x), -jnp.inf, dtype=jnp.float32)

        rows = np.repeat(np.arange(y), nvec)
        cols = np.arange(nvec.sum()) - np.repeat(np.cumsum(nvec) - nvec, nvec)

        return grid.at[rows, cols].set(logits)
