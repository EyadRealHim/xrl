from jaxtyping import PyTree, Array
from typing import Any

import jax


def prefix(tree: PyTree):
    return jax.tree.structure(tree, is_leaf=lambda x: id(x) != id(tree))


def of_instance(object: Any):
    return lambda x: isinstance(x, object)


def keys_like(key: Array, treedef):
    return jax.tree.unflatten(treedef, jax.random.split(key, num=treedef.num_leaves))
