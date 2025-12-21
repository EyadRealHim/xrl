import jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np

from tqdm import tqdm

from distrax import Categorical

import optax

from jaxtyping import Array, Float, Bool, Int, DTypeLike
from typing import Generic, TypeVar, Callable, Sequence, NamedTuple, Tuple, Any

from dataclasses import replace


Observation = Array  # TypeVar("Observation")


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
        nvec: Sequence[int],
    ):
        self.nvec = nvec
        self.shape = (len(nvec),)
        self.dtype = jnp.int32


class Discrete(Space):
    def __init__(self, n: int):
        self.n = n
        self.shape = ()
        self.dtype = jnp.int32


TState = TypeVar("TState")


class TimeStep(NamedTuple):
    reward: Float[Array, "..."]
    done: Bool[Array, "..."]


class Environment(eqx.Module, Generic[TState]):
    def autostep(
        self, key: Array, state: TState, action: Any
    ) -> Tuple[TState, TimeStep, Observation]:
        key, stepk = jax.random.split(key)
        state, step, obs = self.step(stepk, state, action)

        state, obs = jax.lax.cond(
            step.done,
            self.reset,
            lambda _: (state, obs),
            key,
        )

        return state, step, obs

    def reset(self, key: Array) -> Tuple[TState, Observation]:
        raise NotImplementedError

    def step(
        self, key: Array, state: TState, action: Any
    ) -> Tuple[TState, TimeStep, Observation]:
        raise NotImplementedError

    def observation_space(self) -> Space:
        raise NotImplementedError

    def action_space(self) -> Space:
        raise NotImplementedError


class GameState(NamedTuple):
    theta: Float[Array, "..."]


class MyGame(Environment[GameState]):
    def reset(self, key):
        theta = jax.random.uniform(
            key=key,
            shape=(1,),
            maxval=jnp.pi / 2 + jnp.pi / 12,
            minval=jnp.pi / 2 - jnp.pi / 12,
            dtype=jnp.float32,
        )

        state = GameState(theta=theta)

        return state, theta

    def step(self, key, state, action):
        iota = state.theta - jnp.pi / 12 + action * jnp.pi / 6
        iota = jnp.clip(iota, 0.0, jnp.pi)

        state = GameState(theta=iota)
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
        return Discrete(2)


class Transition(NamedTuple):
    observation: Observation
    reward: Float[Array, "..."]
    action: Int[Array, "..."]
    value: Float[Array, "..."]
    done: Bool[Array, "..."]


ActorLogits = Float[Array, "n"]


class Actor(eqx.Module):
    def __call__(self, obs: Observation) -> ActorLogits:
        raise NotImplementedError


class Critic(eqx.Module):
    def value(self, obs: Observation) -> Float[Array, "1"]:
        raise NotImplementedError


class ActorAutomatic(eqx.Module):
    nvec: Sequence[int] = eqx.field(static=True)
    actor: Actor
    head: eqx.nn.Linear

    def __init__(
        self, key: Array, actor: Actor, space: Space, obs_shape: Sequence[int]
    ):
        assert isinstance(space, Discrete) or isinstance(space, MultiDiscrete)

        self.actor = actor

        output = jax.eval_shape(
            self.actor, jax.ShapeDtypeStruct(obs_shape, jnp.float32)
        )
        self.nvec = [space.n] if isinstance(space, Discrete) else space.nvec

        self.head = eqx.nn.Linear(output.size, sum(self.nvec), key=key)

    def __call__(self, obs: Observation):
        logits = self.head(self.actor(obs))

        nvec = np.array(self.nvec)
        y, x = len(nvec), nvec.max()
        zeros = jnp.zeros((y, x))

        rows = jnp.repeat(jnp.arange(y), nvec)
        cols = jnp.arange(nvec.sum()) - jnp.repeat(jnp.cumsum(nvec) - nvec, nvec)

        logits = zeros.at[rows, cols].set(logits)

        return logits, Categorical(logits=logits)


class GameCritic(Critic):
    critic: eqx.nn.MLP

    def __init__(self, key: Array):
        self.critic = eqx.nn.MLP(1, 1, width_size=4, depth=2, key=key)

    def value(self, obs):
        return self.critic(obs)


class GameActor(Actor):
    actor: eqx.nn.MLP

    def __init__(self, key: Array):
        self.actor = eqx.nn.MLP(1, 4, width_size=4, depth=1, key=key)

    def __call__(self, obs):
        return self.actor(obs)


class ActorCritic(NamedTuple):
    actor: ActorAutomatic
    critic: Critic


class SimplePolicyGradientAgent(eqx.Module):
    ac: ActorCritic

    opt_state: Tuple[optax.OptState, optax.OptState]


class RolloutState(NamedTuple, Generic[TState]):
    key: Array
    state: TState
    obs: Observation


class SimplePolicyGradientTrainer(eqx.Module, Generic[TState]):
    optim: optax.GradientTransformation = eqx.field(static=True)
    env: Environment[TState] = eqx.field(static=True)

    env_n: int = eqx.field(static=True, default=8)
    cycle_n: int = eqx.field(static=True, default=64)
    step_n: int = eqx.field(static=True, default=128)

    lr: float = eqx.field(static=True, default=1e-3)

    discount: float = eqx.field(static=True, default=0.96)
    lambda_: float = eqx.field(static=True, default=0.95)

    def __init__(self, env: Environment[TState], **kwargs):
        self.env = env

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.optim = optax.adam(self.lr)

    def make_agent(
        self,
        key: Array,
        actor: Callable[[Array], Actor],
        critic: Callable[[Array], Critic],
    ) -> SimplePolicyGradientAgent:
        a, b = jax.random.split(key)
        a, c = jax.random.split(a)

        assert isinstance(self.env.observation_space(), Box)

        ac = ActorCritic(
            actor=ActorAutomatic(
                key=c,
                actor=actor(a),
                space=self.env.action_space(),
                obs_shape=self.env.observation_space().shape,
            ),
            critic=critic(b),
        )
        return SimplePolicyGradientAgent(
            opt_state=(
                self.optim.init(eqx.filter(ac.actor, eqx.is_inexact_array)),
                self.optim.init(eqx.filter(ac.critic, eqx.is_inexact_array)),
            ),
            ac=ac,
        )

    def rollout(
        self, rs: RolloutState[TState], ac: ActorCritic
    ) -> Tuple[RolloutState[TState], Transition]:
        def step(rs: RolloutState[TState], _):
            key, state, obs = rs
            key, actionk, stepk = jax.random.split(key, 3)

            logits, dist = ac.actor(obs)
            action = dist.sample(seed=actionk)
            value = ac.critic.value(obs)

            state, (reward, done), obsv = self.env.autostep(stepk, state, action)

            return RolloutState(key=key, state=state, obs=obsv), Transition(
                observation=obs,
                action=action,
                value=value,
                reward=reward,
                done=done,
            )

        return jax.lax.scan(step, rs, length=self.step_n)

    def collect(
        self,
        rs: RolloutState[TState],
        agent: SimplePolicyGradientAgent,
    ) -> Tuple[
        RolloutState[TState], Transition, Float[Array, "step_n"], Float[Array, "step_n"]
    ]:
        rs, transition = self.rollout(rs, agent.ac)

        discount = (1.0 - transition.done.astype(jnp.float32)) * self.discount
        values = jnp.concatenate(
            [jnp.squeeze(transition.value), agent.ac.critic.value(rs.obs)]
        )
        deltas = transition.reward + discount * values[1:] - values[:-1]

        def compute(prev, pair):
            delta, value, discount = pair

            gae = delta + discount * self.lambda_ * prev

            return gae, (gae, gae + value)

        _, (advantage, returns) = jax.lax.scan(
            compute, 0, (deltas, jnp.squeeze(transition.value), discount), reverse=True
        )

        return rs, transition, advantage, returns

    def train_step(
        self,
        rs: RolloutState[TState],
        agent: SimplePolicyGradientAgent,
    ) -> Tuple[
        SimplePolicyGradientAgent,
        RolloutState[TState],
        Float[Array, "..."],
    ]:
        rs, transition, advantage, returns = jax.vmap(self.collect, in_axes=(0, None))(
            rs, agent
        )

        def actor_loss(actor: ActorAutomatic, obs, action, advantage):
            _, dist = jax.vmap(jax.vmap(actor))(obs)
            log_p = dist.log_prob(action)

            return -(log_p * jnp.expand_dims(advantage, axis=-1)).mean(axis=-1).mean()

        def critic_loss(critic: Critic, obs, returns):
            values = jnp.squeeze(jax.vmap(jax.vmap(critic.value))(obs))

            return ((values - returns) ** 2).mean()

        actor_grad = eqx.filter_grad(actor_loss)(
            agent.ac.actor,
            transition.observation,
            transition.action,
            advantage,
        )

        critic_grad = eqx.filter_grad(critic_loss)(
            agent.ac.critic, transition.observation, returns
        )

        opt1, opt2 = agent.opt_state

        actor_update, opt1 = self.optim.update(
            actor_grad, opt1, eqx.filter(agent.ac.actor, eqx.is_inexact_array)
        )
        actor = eqx.apply_updates(agent.ac.actor, actor_update)

        critic_update, opt2 = self.optim.update(
            critic_grad, opt2, eqx.filter(agent.ac.critic, eqx.is_inexact_array)
        )
        critic = eqx.apply_updates(agent.ac.critic, critic_update)

        return (
            replace(
                agent,
                ac=ActorCritic(actor=actor, critic=critic),
                opt_state=(opt1, opt2),
            ),
            rs,
            transition.reward,
        )

    def train_cycle(
        self, agent: SimplePolicyGradientAgent, rs: RolloutState[TState]
    ) -> Tuple[SimplePolicyGradientAgent, RolloutState[TState], Float[Array, "..."]]:
        params, static = eqx.partition(agent, eqx.is_array)

        def body(pair, _):
            params, rs = pair

            agent = eqx.combine(params, static)
            agent, rs, r = self.train_step(rs, agent)
            params = eqx.filter(agent, eqx.is_array)

            return (params, rs), r

        (params, rs), r = jax.lax.scan(body, (params, rs), length=self.cycle_n)
        agent = eqx.combine(params, static)

        return agent, rs, r

    def train(
        self, key: Array, agent: SimplePolicyGradientAgent, iterations: int = 128
    ) -> SimplePolicyGradientAgent:
        key, traink = jax.random.split(key)

        state, obs = jax.vmap(self.env.reset)(jax.random.split(traink, self.env_n))

        rs = RolloutState(key=jax.random.split(key, self.env_n), state=state, obs=obs)

        train_cycle = eqx.filter_jit(self.train_cycle)

        mean = 0
        for i in (bar := tqdm(range(0, iterations))):
            agent, rs, r = train_cycle(agent, rs)

            mean = mean + (r.mean() - mean) / (i + 1)

            bar.set_postfix({"avg_rewards": f"{mean:.3f}"})

        return agent


if __name__ == "__main__":
    key = jax.random.key(0)
    env = MyGame()

    key, subk = jax.random.split(key)
    trainer = SimplePolicyGradientTrainer(env=env)
    agent = trainer.make_agent(subk, actor=GameActor, critic=GameCritic)

    agent = trainer.train(key, agent, iterations=128 * 2)
