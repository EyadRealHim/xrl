import jax
from beartype import beartype
from jaxtyping import Array, Key, PyTree, jaxtyped


def prefix(tree: PyTree):
    return jax.tree.structure(tree, is_leaf=lambda x: id(x) != id(tree))


@jaxtyped(typechecker=beartype)
def keys_like(key: Key[Array, ""], treedef):
    return jax.tree.unflatten(treedef, jax.random.split(key, num=treedef.num_leaves))
