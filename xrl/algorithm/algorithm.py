import equinox as eqx
from beartype import beartype
from jaxtyping import Array, Bool, Float, Key, PyTree, jaxtyped

from xrl.networks import ObservationTransformation

from ..environment import Action, EnvInfo, Observation, Space


class Transition(eqx.Module):
    observation: PyTree[Observation]
    action: PyTree[Action]
    reward: PyTree[Float[Array, ""]]
    terminated: PyTree[Bool[Array, ""]]
    truncated: PyTree[Bool[Array, ""]]
    info: PyTree[EnvInfo]


@jaxtyped(typechecker=beartype)
class Algorithm[Agent, AgentConfig: eqx.Module](eqx.Module):
    def make_agent(
        self,
        key: Key[Array, ""],
        observation_space: PyTree[Space],
        action_space: PyTree[Space],
        config: AgentConfig,
        transformations: PyTree[type[ObservationTransformation]] | None = None,
    ) -> Agent:
        raise NotImplementedError()

    def act(self, key: Key[Array, ""], agent: Agent, obs: Observation) -> Action:
        raise NotImplementedError()

    def train(self, key: Key[Array, ""], agent: Agent, transition: Transition) -> Agent:
        raise NotImplementedError()
