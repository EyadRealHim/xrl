from ..environment import SoloEnv, GroupEnv, TState, Observation, Action, Map, TimeStep
from ..networks import Actor, ActorLike, Critic, CriticLike

from typing import Union, Generic, NamedTuple, Tuple, Type
from jaxtyping import PyTree, Array, Float, Bool
from optax import OptState, GradientTransformation

from ..xrl_tree import of_instance, keys_like
from distrax import Categorical
from tqdm import tqdm

import jax.numpy as jnp
import equinox as eqx

import jax


class Transition(NamedTuple):
    observation: Observation
    reward: Float[Array, "1"]
    action: Action
    value: Float[Array, "1"]
    done: Bool[Array, "1"]


class RolloutState(NamedTuple, Generic[TState]):
    key: Array
    state: TState
    obs: Union[Map[Observation], Observation]


class RLAgent(eqx.Module):
    actor: PyTree[Actor]
    critic: PyTree[Critic]

    optactor: PyTree[OptState]
    optcritic: PyTree[OptState]


class UpdatePkg(NamedTuple):
    actor: Actor
    critic: Critic

    optactor: OptState
    optcritic: OptState


class RLTrainer(eqx.Module, Generic[TState]):
    env: Union[SoloEnv[TState], GroupEnv[TState]] = eqx.field(static=True)
    optim: GradientTransformation = eqx.field(static=True)

    env_n: int = eqx.field(static=True, default=8)
    cycle_n: int = eqx.field(static=True, default=64)
    step_n: int = eqx.field(static=True, default=128)

    def update(
        self,
        actor: Actor,
        critic: Critic,
        optactor: OptState,
        optcritic: OptState,
        transition: Transition,
        bootstraps: Observation,
    ) -> UpdatePkg:
        raise NotImplementedError

    def make_agent(
        self, key: Array, actor: Type[ActorLike], critic: Type[CriticLike]
    ) -> RLAgent:
        observation = self.env.observation_space()
        action = self.env.action_space()

        is_multi_agent = isinstance(self.env, GroupEnv)

        def is_leaf(x):
            return not is_multi_agent or id(x) != id(observation)

        actork, critick = jax.random.split(key)
        actors = jax.tree.map(
            lambda obs, action, key: Actor(key, actor, obs, action),
            observation,
            action,
            keys_like(actork, jax.tree.structure(observation, is_leaf=is_leaf)),
            is_leaf=is_leaf,
        )
        critics = jax.tree.map(
            lambda obs, key: Critic(key, critic, obs),
            observation,
            keys_like(critick, jax.tree.structure(observation, is_leaf=is_leaf)),
            is_leaf=is_leaf,
        )

        return RLAgent(
            actor=actors,
            critic=critics,
            optactor=jax.tree.map(
                lambda actor: actor.opt_state(self.optim),
                actors,
                is_leaf=of_instance(Actor),
            ),
            optcritic=jax.tree.map(
                lambda critic: critic.opt_state(self.optim),
                critics,
                is_leaf=of_instance(Critic),
            ),
        )

    def train_step(
        self, rs: RolloutState[TState], agent: RLAgent
    ) -> Tuple[RolloutState[TState], RLAgent]:
        rs, transition = jax.vmap(self.rollout, in_axes=(0, None))(rs, agent)

        pkgs = jax.tree.map(
            self.update,
            agent.actor,
            agent.critic,
            agent.optactor,
            agent.optcritic,
            transition,
            rs.obs,
            is_leaf=of_instance(Actor),
        )

        def isleaf(x):
            return isinstance(x, UpdatePkg)

        return rs, RLAgent(
            actor=jax.tree.map(lambda x: x.actor, pkgs, is_leaf=isleaf),
            critic=jax.tree.map(lambda x: x.critic, pkgs, is_leaf=isleaf),
            optactor=jax.tree.map(lambda x: x.optactor, pkgs, is_leaf=isleaf),
            optcritic=jax.tree.map(lambda x: x.optcritic, pkgs, is_leaf=isleaf),
        )

    def train(
        self,
        key: Array,
        agent: RLAgent,
        iterations: int = 128,
    ) -> RLAgent:
        key, traink = jax.random.split(key)

        state, obs = jax.vmap(self.env.reset)(jax.random.split(traink, self.env_n))

        rs = RolloutState(key=jax.random.split(key, self.env_n), state=state, obs=obs)

        print("Compiling training cycle...")
        train_cycle = eqx.filter_jit(self.train_cycle)
        _ = jax.block_until_ready(train_cycle(rs, agent))

        for _ in tqdm(range(iterations)):
            rs, agent = train_cycle(rs, agent)

        return agent

    def train_cycle(
        self, rs: RolloutState[TState], agent: RLAgent
    ) -> Tuple[RolloutState[TState], RLAgent]:
        params, static = eqx.partition(agent, eqx.is_array)

        def body(_, pair):
            rs, params = pair

            agent = eqx.combine(params, static)
            rs, agent = self.train_step(rs, agent)
            params = eqx.filter(agent, eqx.is_array)

            return rs, params

        rs, params = jax.lax.fori_loop(0, self.cycle_n, body, (rs, params))
        agent = eqx.combine(params, static)

        return rs, agent

    def rollout(
        self, rs: RolloutState[TState], agent: RLAgent
    ) -> Tuple[RolloutState[TState], Union[Map[Transition], Transition]]:
        def step(rs: RolloutState[TState], _):
            key, state, obs = rs

            logits = jax.tree.map(
                lambda actor, obsv: actor(obsv),
                agent.actor,
                obs,
                is_leaf=of_instance(Actor),
            )
            value = jax.tree.map(
                lambda critic, obsv: critic(obsv),
                agent.critic,
                obs,
                is_leaf=of_instance(Critic),
            )

            key, action_key = jax.random.split(key)
            action = jax.tree.map(
                lambda key, logits: Categorical(logits=logits).sample(seed=key),
                keys_like(key, jax.tree.structure(logits)),
                logits,
            )

            key, step_key = jax.random.split(key)
            state, time, obsv = self.env.autostep(step_key, state, action)

            assert jax.tree.structure(obsv) == jax.tree.structure(obs), (
                "Observations differ"
            )

            transition = jax.tree.map(
                lambda time, value, action, obs: Transition(
                    observation=obs,
                    action=action,
                    value=value,
                    reward=time.reward,
                    done=time.done,
                ),
                time,
                value,
                action,
                obs,
                is_leaf=lambda x: isinstance(x, TimeStep),
            )

            return RolloutState(key=key, state=state, obs=obsv), transition

        return jax.lax.scan(step, rs, length=self.step_n)

    def capture(
        self, key: Array, agent: RLAgent, max_steps=999, deterministic: bool = False
    ):
        key, subkey = jax.random.split(key)
        state, obs = self.env.reset(subkey)

        yield state

        for _ in range(max_steps - 1):
            key, actionk = jax.random.split(key)

            logits = jax.tree.map(
                lambda actor, obsv: actor(obsv),
                agent.actor,
                obs,
                is_leaf=of_instance(Actor),
            )
            if not deterministic:
                action = jax.tree.map(
                    lambda key, logits: Categorical(logits=logits).sample(seed=key),
                    keys_like(key, jax.tree.structure(logits)),
                    logits,
                )
            else:
                action = jax.tree.map(
                    lambda logits: jnp.argmax(logits, axis=-1),
                    logits,
                )

            key, stepk = jax.random.split(key)
            state, time, obs = self.env.step(stepk, state, action)

            yield state

            done = jax.tree.reduce(
                lambda a, b: jnp.logical_or(a, b.done),
                time,
                False,
                is_leaf=lambda x: isinstance(x, TimeStep),
            )

            if done:
                break
