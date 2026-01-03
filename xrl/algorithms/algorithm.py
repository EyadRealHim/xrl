from ..environment import SoloEnv, GroupEnv, TState, Observation, Action, Map, TimeStep
from ..networks import Actor, ActorLike, Critic, CriticLike

from typing import Optional, Union, Generic, NamedTuple, Tuple, Type, TypeVar, Iterable
from jaxtyping import PyTree, Array, Float, Bool
from optax import OptState, GradientTransformation, Updates

from ..xrl_tree import of_instance, keys_like, prefix
from rich.console import Console
from rich.progress import track
from distrax import Categorical
from itertools import chain
from pathlib import Path

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


TData = TypeVar("TData")


class RLTrainer(eqx.Module, Generic[TState, TData]):
    env: Union[SoloEnv[TState], GroupEnv[TState]] = eqx.field(static=True)
    optim: GradientTransformation = eqx.field(static=True)

    env_n: int = eqx.field(static=True, default=8)
    cycle_n: int = eqx.field(static=True, default=64)
    step_n: int = eqx.field(static=True, default=128)

    def gradient(
        self, actor: Actor, critic: Critic, data: TData
    ) -> Tuple[Updates, Updates]:
        raise NotImplementedError

    def compute_data(
        self, agent: RLAgent, rs: RolloutState[TState]
    ) -> Tuple[RolloutState[TState], PyTree[TData]]:
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

    def map(self, fn, agent: RLAgent, *trees):
        assert len(trees) > 0, "need at least one tree to map over"

        return jax.tree.map(
            fn,
            agent.actor,
            agent.critic,
            *trees,
            is_leaf=of_instance(Actor),
        )

    def update(self, agent: RLAgent, data: TData) -> RLAgent:
        gradient = self.map(self.gradient, agent, data)

        def lambda_(actor, critic, opta, optc, grad):
            actor_grad, critic_grad = grad

            actor, opta = actor.update(actor_grad, opta, self.optim)
            critic, optc = critic.update(critic_grad, optc, self.optim)

            return actor, critic, opta, optc

        pkgs = self.map(lambda_, agent, agent.optactor, agent.optcritic, gradient)
        isleaf = of_instance(tuple)

        return RLAgent(
            actor=jax.tree.map(lambda x: x[0], pkgs, is_leaf=isleaf),
            critic=jax.tree.map(lambda x: x[1], pkgs, is_leaf=isleaf),
            optactor=jax.tree.map(lambda x: x[2], pkgs, is_leaf=isleaf),
            optcritic=jax.tree.map(lambda x: x[3], pkgs, is_leaf=isleaf),
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

        train_cycle = eqx.filter_jit(self.train_cycle)
        with Console().status("[bold orchid] Compiling training cycle", spinner="dots"):
            _ = jax.block_until_ready(train_cycle(rs, agent))

        for _ in track(range(iterations), "Training"):
            rs, agent = train_cycle(rs, agent)

        return agent

    def train_cycle(
        self, rs: RolloutState[TState], agent: RLAgent
    ) -> Tuple[RolloutState[TState], RLAgent]:
        params, static = eqx.partition(agent, eqx.is_array)

        def body(_, pair):
            rs, params = pair

            agent = eqx.combine(params, static)

            rs, data = self.compute_data(agent, rs)
            agent = self.update(agent, data)

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

            pairs = self.map(
                lambda actor, critic, obsv: (actor(obsv), critic(obsv)),
                agent,
                obs,
            )

            logits = jax.tree.map(lambda p: p[0], pairs, is_leaf=of_instance(tuple))
            value = jax.tree.map(lambda p: p[1], pairs, is_leaf=of_instance(tuple))

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

            logits = self.map(lambda actor, _, obsv: actor(obsv), agent, obs)

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

    def record(
        self,
        filename: Union[Path, str],
        states: Iterable[TState],
        total: Optional[int] = None,
        fps: Optional[int] = 30,
    ):
        import numpy as np
        import av

        it = iter(states)
        try:
            first_state = next(it)
        except StopIteration:
            print("[WARING]: No video is recorded, the iterable is emtpy.")
            return

        full_it = chain([first_state], it)

        if total is None:
            if hasattr(states, "__len__"):
                total = len(states)

        container = av.open(str(filename), mode="w")

        first_img = self.env.render(first_state).convert("RGB")

        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = first_img.size
        stream.pix_fmt = "yuv420p"

        for state in track(full_it, description="Rendering", total=total):
            img = self.env.render(state).convert("RGB")
            frame = av.VideoFrame.from_ndarray(np.array(img), format="rgb24")

            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        container.close()
