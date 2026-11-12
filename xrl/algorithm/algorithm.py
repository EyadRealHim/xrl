import equinox as eqx
from beartype import beartype
from jaxtyping import Array, Bool, Float, Key, PyTree, jaxtyped

from xrl.networks import ObservationTransformation

from ..environment import Action, EnvInfo, MonoEnv, Observation


@jaxtyped(typechecker=beartype)
class Transition(eqx.Module):
    observation: Observation
    action: Action
    reward: Float[Array, ""]
    terminated: Bool[Array, ""]
    truncated: Bool[Array, ""]
    info: EnvInfo


@jaxtyped(typechecker=beartype)
class Algorithm[Agent, AgentConfig: eqx.Module](eqx.Module):
    @jaxtyped(typechecker=beartype)
    def make_agent(
        self,
        key: Key[Array, ""],
        env: MonoEnv,
        config: AgentConfig,
        transformations: PyTree[type[ObservationTransformation]] | None = None,
    ) -> Agent:
        raise NotImplementedError()

    @jaxtyped(typechecker=beartype)
    def act(self, key: Key[Array, ""], agent: Agent, obs: Observation) -> Action:
        raise NotImplementedError()

    @jaxtyped(typechecker=beartype)
    def train(self, agent: Agent, transition: Transition) -> Agent:
        raise NotImplementedError()
