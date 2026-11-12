import jax
from beartype import beartype
from jaxtyping import Array, Key, jaxtyped


@jaxtyped(typechecker=beartype)
def keys_like(key: Key[Array, ""], treedef):
    return jax.tree.unflatten(treedef, jax.random.split(key, num=treedef.num_leaves))
