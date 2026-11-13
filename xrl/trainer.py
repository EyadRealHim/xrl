from pathlib import Path

import av
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Array, Bool, Key, PyTree, jaxtyped
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track,
)

from .networks import ObservationTransformation

from .algorithm.algorithm import Algorithm, Transition
from .environment import MonoEnv, Observation, PolyEnv


@jaxtyped(typechecker=beartype)
class RolloutState[EnvState](eqx.Module):
    key: Key[Array, "..."]  # Uh, man
    state: EnvState
    obs: Observation


@jaxtyped(typechecker=beartype)
class Trainer[EnvState, Agent, AgentConfig: eqx.Module](eqx.Module):
    algo: Algorithm[Agent, AgentConfig] = eqx.field(static=True)
    env: MonoEnv[EnvState] | PolyEnv[EnvState] = eqx.field(static=True)

    env_n: int = eqx.field(static=True, default=16)
    step_n: int = eqx.field(static=True, default=128)
    cycle_n: int = eqx.field(static=True, default=32)

    @jaxtyped(typechecker=beartype)
    def make_agent(
        self,
        key: Key[Array, ""],
        env: MonoEnv,
        config: AgentConfig,
        transformations: PyTree[type[ObservationTransformation]] | None = None,
    ) -> Agent:
        return self.algo.make_agent(key, env, config, transformations)

    @jaxtyped(typechecker=beartype)
    def train(
        self,
        key: Array,
        agent: Agent,
        iterations: int = 32,
        rs: RolloutState[EnvState] | None = None,
    ) -> tuple[Agent, RolloutState[EnvState]]:
        key, traink = jax.random.split(key)

        state, obs = jax.vmap(self.env.reset)(jax.random.split(traink, self.env_n))

        rs = rs or RolloutState(
            key=jax.random.split(key, self.env_n), state=state, obs=obs
        )

        train_cycle = eqx.filter_jit(self.train_cycle)
        with Console().status("[bold orchid] Compiling training cycle", spinner="arc"):
            rs, agent = jax.block_until_ready(train_cycle(rs, agent))

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TextColumn("EnvStep: {task.fields[steps]}"),
        ) as progress:
            task = progress.add_task(
                "Training",
                total=iterations,
                steps=0,
            )

            progress.update(
                task,
                advance=1,
                steps=f"{(self.step_n * self.env_n * self.cycle_n):,}",
            )

            for i in range(iterations - 1):
                rs, agent = jax.block_until_ready(train_cycle(rs, agent))
                steps = self.step_n * self.env_n * self.cycle_n * (i + 2)

                progress.update(task, advance=1, steps=f"{steps:,}")
        return agent, rs

    @jaxtyped(typechecker=beartype)
    def train_cycle(
        self, rs: RolloutState[EnvState], agent: Agent
    ) -> tuple[RolloutState[EnvState], Agent]:
        params, static = eqx.partition(agent, eqx.is_array)

        def body(_, pair):
            rs, params = pair

            agent = eqx.combine(params, static)
            rs, transitions = eqx.filter_vmap(self.rollout, in_axes=(0, None))(
                rs, agent
            )
            agent = self.algo.train(agent, transitions)

            params = eqx.filter(agent, eqx.is_array)

            return rs, params

        rs, params = jax.lax.fori_loop(0, self.cycle_n, body, (rs, params))
        agent = eqx.combine(params, static)

        return rs, agent

    @jaxtyped(typechecker=beartype)
    def rollout(self, rs: RolloutState[EnvState], agent: Agent):
        @jaxtyped(typechecker=beartype)
        def step(rs: RolloutState[EnvState], _):
            key, actionk = jax.random.split(rs.key)
            action = self.algo.act(actionk, agent, rs.obs)

            key, step_key = jax.random.split(key)

            state, time, obsv, info = self.env.autostep(step_key, rs.state, action)

            return RolloutState(key=key, state=state, obs=obsv), Transition(
                observation=rs.obs,
                action=action,
                reward=time.reward,
                truncated=time.truncated,
                terminated=time.terminated,
                info=info,
            )

        return jax.lax.scan(step, rs, length=self.step_n)

    @jaxtyped(typechecker=beartype)
    def episode(
        self, key: Key[Array, ""], agent: Agent, step_n=128
    ) -> tuple[EnvState, Bool[Array, " step_n"]]:
        @jaxtyped(typechecker=beartype)
        def step(rs: RolloutState[EnvState], _):
            key, actionk = jax.random.split(rs.key)
            action = self.algo.act(actionk, agent, rs.obs)

            key, step_key = jax.random.split(key)

            state, time, obsv, _info = self.env.step(step_key, rs.state, action)

            return RolloutState(key=key, state=state, obs=obsv), (
                state,
                jax.numpy.logical_or(time.truncated, time.terminated),
            )

        key, resetk = jax.random.split(key)
        istate, obs = self.env.reset(resetk)

        _, (state, mask) = jax.lax.scan(
            jax.jit(step), RolloutState(key, istate, obs), length=step_n
        )

        mask = jnp.concat((jnp.array([False]), mask))
        state = jax.tree.map(
            lambda a, b: jnp.concat(
                (jnp.expand_dims(b, axis=0), a),
                axis=0,
            ),
            state,
            istate,
        )

        return state, mask

    @jaxtyped(typechecker=beartype)
    def record(
        self,
        filename: Path | str,
        states_and_mask: tuple[EnvState, Bool[Array, " step_n"]],
        fps: int = 30,
    ):
        states, mask = states_and_mask
        frames = jax.lax.map(eqx.filter_jit(self.env.render), states, batch_size=32)
        idx = int(jax.numpy.argmax(mask) + 1) if mask.any() else len(mask)

        container = av.open(str(filename), mode="w")
        stream = container.add_stream("libx264", rate=fps)
        _, stream.height, stream.width, _ = frames.shape
        stream.pix_fmt = "yuv420p"

        for frame in track(list(frames)[:idx], description="Writing video"):
            frame = av.VideoFrame.from_ndarray(np.array(frame), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        container.close()
