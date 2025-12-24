from ..environment import Environment, TState, Observation, Action
from ..networks import Actor, Critic, ActorContainer, CriticContainer

from typing import TypeVar, Union, Generic, NamedTuple, Tuple, Mapping
from jaxtyping import PyTree, Array, Float, Bool
from optax import OptState, GradientTransformation

from dataclasses import replace
from distrax import Categorical
from tqdm import tqdm

import equinox as eqx

import jax


class Transition(NamedTuple):
    observation: Observation
    reward: Float[Array, "..."]
    action: Action
    value: Float[Array, "..."]
    done: Bool[Array, "..."]


class _RLAgent(eqx.Module):
    actor: Union[Actor, ActorContainer]
    critic: Union[Critic, CriticContainer]

    opt: PyTree[OptState]


RLAgent = TypeVar("RLAgent", bound=_RLAgent)


class RolloutState(NamedTuple, Generic[TState]):
    key: Array
    state: TState
    obs: Observation


class RLTrainer(eqx.Module, Generic[TState, RLAgent]):
    env: Environment[TState] = eqx.field(static=True)
    optim: GradientTransformation = eqx.field(static=True)

    env_n: int = eqx.field(static=True, default=8)
    cycle_n: int = eqx.field(static=True, default=64)
    step_n: int = eqx.field(static=True, default=128)

    def make_agent(self, key: Array, *args, **kwargs) -> RLAgent:
        raise NotImplementedError

    def gradient(
        self,
        obs: Observation,
        transition: Transition,
        actor: Actor,
        critic: Critic,
    ) -> PyTree:
        raise NotImplementedError

    def train_step(
        self, rs: RolloutState[TState], agent: RLAgent
    ) -> Tuple[RolloutState[TState], RLAgent]:
        rs, transition = jax.vmap(self.rollout, in_axes=(0, None))(rs, agent)

        if not isinstance(transition, Transition):
            assert isinstance(agent.actor, ActorContainer)
            assert isinstance(agent.critic, CriticContainer)

            gradient = {"actor": {}, "critic": {}}
            for k in transition.keys():
                critic = agent.critic.critics[k]
                actor = agent.actor.actors[k]

                t = transition[k]
                obs = rs.obs[k]

                grad = self.gradient(obs, t, actor, critic)
                for i in gradient.keys():
                    gradient[i][k] = grad[i]
        else:
            assert isinstance(agent.actor, Actor)
            assert isinstance(agent.critic, Critic)

            gradient = self.gradient(rs.obs, transition, agent.actor, agent.critic)

        result = {
            k: getattr(agent, k).update(gradient[k], agent.opt[k], self.optim)
            for k in ["actor", "critic"]
        }

        ac = jax.tree.map(
            lambda x: x[0], result, is_leaf=lambda x: isinstance(x, tuple)
        )
        opt = jax.tree.map(
            lambda x: x[1], result, is_leaf=lambda x: isinstance(x, tuple)
        )

        if isinstance(agent.actor, ActorContainer):
            ac = {
                "actor": ActorContainer(ac["actor"]),
                "critic": CriticContainer(ac["critic"]),
            }

        return rs, replace(agent, **ac, opt=opt)

    def rollout(
        self, rs: RolloutState[TState], agent: RLAgent
    ) -> Tuple[RolloutState[TState], Union[Mapping[str, Transition], Transition]]:
        def step(rs: RolloutState[TState], _):
            key, state, obs = rs
            key, actionk, stepk = jax.random.split(key, 3)

            logits = agent.actor(obs)
            value = agent.critic(obs)

            actkey = jax.tree.unflatten(
                jax.tree.structure(logits),
                jax.random.split(actionk, len(jax.tree.leaves(logits))),
            )

            action = jax.tree.map(
                lambda logits, key: Categorical(logits=logits).sample(seed=key),
                logits,
                actkey,
            )

            state, (reward, done), obsv = self.env.autostep(stepk, state, action)

            return RolloutState(key=key, state=state, obs=obsv), (
                obs,
                action,
                value,
                reward,
                done,
            )

        rs, data = jax.lax.scan(step, rs, length=self.step_n)

        reward = data[3]
        keys = reward.keys() if isinstance(reward, dict) else None

        if keys is None:
            assert not isinstance(data[2], dict), "Unreachable"
            return rs, Transition(
                observation=data[0],
                action=data[1],
                value=data[2],
                reward=data[3],
                done=data[4],
            )

        return rs, {
            k: Transition(
                observation=data[0][k],
                action=data[1][k],
                value=data[2][k],
                reward=data[3][k],
                done=data[4],
            )
            for k in keys
        }

    def train_cycle(
        self, rs: RolloutState[TState], agent: RLAgent
    ) -> Tuple[RolloutState[TState], RLAgent]:
        params, static = eqx.partition(agent, eqx.is_array)

        def body(pair, _):
            rs, params = pair

            agent = eqx.combine(params, static)
            rs, agent = self.train_step(rs, agent)
            params = eqx.filter(agent, eqx.is_array)

            return (rs, params), None

        (rs, params), _ = jax.lax.scan(body, (rs, params), length=self.cycle_n)
        agent = eqx.combine(params, static)

        return rs, agent

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

        for i in tqdm(range(iterations)):
            rs, agent = train_cycle(rs, agent)

        return agent

    def render_episode(self, key: Array, agent: RLAgent, max_steps: int = 300):
        key, subkey = jax.random.split(key)
        state, obs = self.env.reset(subkey)

        try:
            yield self.env.render(state)
        except NotImplementedError:
            print(
                f"[WARNING]: enviornment '{self.env.__name__}' does not implement .render"
            )
            return

        for _ in range(max_steps):
            logits = agent.actor(obs)

            key, actionk = jax.random.split(key)
            keys = jax.random.split(actionk, len(logits.keys()))
            keys = jax.tree.unflatten(jax.tree.structure(logits), keys)

            action = jax.tree.map(
                lambda logits, key: Categorical(logits=logits).sample(seed=key),
                logits,
                keys,
            )

            key, stepk = jax.random.split(key)
            state, (reward, done), obs = self.env.step(stepk, state, action)

            yield self.env.render(state)

            if done.item():
                break
