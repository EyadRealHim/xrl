import jax
import equinox as eqx
import jax.numpy as jnp

from tqdm import tqdm

from distrax import Categorical

import optax

from jaxtyping import Array, Float, Bool, Int, DTypeLike
from typing import TypeVar, NamedTuple, Tuple, Any

from dataclasses import replace


Observation = TypeVar("Observation")


class Space:
    shape: Tuple[int, ...]
    dtype: DTypeLike


class Box(Space):
    def __init__(
        self,
        low,
        high,
        shape: Tuple[int, ...],
    ):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = jnp.float32


class MultiDiscrete(Space):
    def __init__(
        self,
        nvec: Tuple[int, ...],
    ):
        self.nvec = nvec
        self.shape = (len(nvec),)
        self.dtype = jnp.int32


class State(NamedTuple):
    theta: Float[Array, "..."]


class TimeStep(NamedTuple):
    reward: Float[Array, "..."]
    done: Bool[Array, "..."]


class Environment(eqx.Module):
    def autostep(
        self, key: Array, state: State, action: Any
    ) -> Tuple[State, TimeStep, Observation]:
        key, stepk = jax.random.split(key)
        state, step, obs = self.step(stepk, state, action)

        state, obs = jax.lax.cond(
            step.done,
            self.reset,
            lambda _: (state, obs),
            key,
        )

        return state, step, obs

    def reset(self, key: Array) -> Tuple[State, Observation]:
        raise NotImplementedError

    def step(
        self, key: Array, state: State, action: Any
    ) -> Tuple[State, TimeStep, Observation]:
        raise NotImplementedError

    def observation_space(self) -> Space:
        raise NotImplementedError

    def action_space(self) -> Space:
        raise NotImplementedError


class MyGame(Environment):
    def reset(self, key: Array):
        theta = jax.random.uniform(
            key=key,
            shape=(1,),
            maxval=jnp.pi / 2 + jnp.pi / 12,
            minval=jnp.pi / 2 - jnp.pi / 12,
            dtype=jnp.float32,
        )

        state = State(theta=theta)

        return state, theta

    def step(self, key: Array, state: State, action):
        iota = state.theta - jnp.pi / 12 + action * jnp.pi / 6
        iota = jnp.clip(iota, 0.0, jnp.pi)

        state = State(theta=iota)
        step = TimeStep(
            reward=jnp.squeeze(self.reward(state.theta)),
            done=jnp.squeeze(self.done(state.theta)),
        )

        return state, step, iota

    def reward(self, theta: Array):
        return 1 - jnp.abs(jnp.cos(theta))

    def done(self, theta: Array):
        return jnp.logical_or(theta <= 0, theta >= jnp.pi)

    def observation_space(self):
        return Box(low=0, high=jnp.pi, shape=(1,))

    def action_space(self):
        return MultiDiscrete([2])


class Transition(NamedTuple):
    observation: Observation
    reward: Float[Array, "..."]
    action: Int[Array, "..."]
    value: Float[Array, "..."]
    done: Bool[Array, "..."]


class Actor(eqx.Module):
    def logits(self, obs: Observation) -> Float[Array, "n"]:
        raise NotImplementedError


class GameActor(Actor):
    policy: eqx.nn.MLP

    def __init__(self, key: Array):
        self.policy = eqx.nn.MLP(1, 2, width_size=4, depth=1, key=key)

    def logits(self, obs):
        return self.policy(obs)


class SimplePolicyGradientAgent(eqx.Module):
    actor: Actor
    optim: optax.GradientTransformationExtraArgs
    opt_state: optax.OptState


class RolloutState(NamedTuple):
    key: Array
    state: State
    obs: Observation


class SimplePolicyGradientTrainer(eqx.Module):
    env: Environment = eqx.field(static=True)

    env_n: int = eqx.field(static=True, default=8)
    step_n: int = eqx.field(static=True, default=128)

    lr: float = eqx.field(static=True, default=1e-3)
    discount: float = eqx.field(static=True, default=0.96)

    def make_agent(self, actor: Actor) -> SimplePolicyGradientAgent:
        optim = optax.adam(self.lr)

        return SimplePolicyGradientAgent(
            optim=optim,
            actor=actor,
            opt_state=optim.init(eqx.filter(actor, eqx.is_array)),
        )

    def rollout(
        self, rs: RolloutState, actor: Actor
    ) -> Tuple[RolloutState, Transition]:
        def step(rs: RolloutState, _):
            key, state, obs = rs
            key, actionk, stepk = jax.random.split(key, 3)

            logits = actor.logits(obs)

            dist = Categorical(logits=logits)
            action = dist.sample(seed=actionk)

            state, (reward, done), obsv = self.env.autostep(stepk, state, action)

            return RolloutState(key=key, state=state, obs=obsv), Transition(
                observation=obs,
                action=action,
                value=None,
                reward=reward,
                done=done,
            )

        return jax.lax.scan(step, rs, length=self.step_n)

    def collect(
        self,
        rs: RolloutState,
        agent: SimplePolicyGradientAgent,
    ) -> Tuple[RolloutState, Transition, Float[Array, "step_n"]]:
        rs, transition = self.rollout(rs, agent.actor)

        def discounted_reward(prev, pair):
            reward, cut = pair
            reward = reward + cut * self.discount * prev

            return reward, reward

        _, returns = jax.lax.scan(
            discounted_reward,
            0.0,
            (
                transition.reward,
                1.0 - transition.done.astype(jnp.float32),
            ),
            reverse=True,
        )

        return rs, transition, returns

    def train_step(
        self,
        rs: RolloutState,
        agent: SimplePolicyGradientAgent,
    ) -> Tuple[
        SimplePolicyGradientAgent,
        RolloutState,
        Float[Array, "..."],
    ]:
        rs, transition, returns = jax.vmap(self.collect, in_axes=(0, None))(rs, agent)

        def compute_loss(actor, obs, action, reward):
            logits = jax.vmap(jax.vmap(actor.logits))(obs)
            log_p = Categorical(logits).log_prob(action)

            return -(log_p * reward).mean(axis=-1).mean()

        actor = agent.actor
        value, grad = eqx.filter_value_and_grad(compute_loss)(
            actor,
            transition.observation,
            transition.action,
            returns,
        )

        update, opt_state = agent.optim.update(
            grad, agent.opt_state, eqx.filter(actor, eqx.is_array)
        )
        actor = eqx.apply_updates(actor, update)

        return (
            replace(
                agent,
                actor=actor,
                opt_state=opt_state,
            ),
            rs,
            returns,
        )

    def train(
        self, key: Array, agent: SimplePolicyGradientAgent
    ) -> SimplePolicyGradientAgent:
        key, traink = jax.random.split(key)

        state, obs = jax.vmap(self.env.reset)(jax.random.split(traink, self.env_n))

        rs = RolloutState(key=jax.random.split(key, self.env_n), state=state, obs=obs)

        train_step = eqx.filter_jit(self.train_step)
        for i in (bar := tqdm(range(1000))):
            agent, rs, r = train_step(rs, agent)

            bar.set_postfix({"returns": r.mean()})

        return agent


if __name__ == "__main__":
    key = jax.random.key(0)
    env = MyGame()

    key, actork = jax.random.split(key)
    trainer = SimplePolicyGradientTrainer(env=env)
    agent = trainer.make_agent(actor=GameActor(key=actork))

    agent = trainer.train(key, agent)
