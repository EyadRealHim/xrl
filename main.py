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


from PIL import Image, ImageDraw


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


class CartPoleState(NamedTuple):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: int = 0


class CartPole(Environment[CartPoleState]):
    """
    CartPole environment from OpenAI Gym.
    """

    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    force_mag: float = 10.0
    tau: float = 0.02

    theta_threshold_radians: float = 12 * 2 * np.pi / 360
    x_threshold: float = 2.4

    max_episode_steps: int = 500

    @property
    def total_mass(self):
        return self.masscart + self.masspole

    @property
    def polemass_length(self):
        return self.masspole * self.length

    def step(self, key, state, action):
        action = jnp.squeeze(action)
        force = self.force_mag * action - self.force_mag * (1 - action)
        costheta = jnp.cos(state.theta)
        sintheta = jnp.sin(state.theta)

        temp = (
            force + self.polemass_length * state.theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = state.x + self.tau * state.x_dot
        x_dot = state.x_dot + self.tau * xacc
        theta = state.theta + self.tau * state.theta_dot
        theta_dot = state.theta_dot + self.tau * thetaacc

        state = CartPoleState(
            x=x,
            x_dot=x_dot,
            theta=theta,
            theta_dot=theta_dot,
            time=state.time + 1,
        )

        timestep = TimeStep(
            reward=self.get_reward(),
            done=jnp.logical_or(self.get_terminated(state), self.get_truncated(state)),
        )

        return state, timestep, self.get_observation(state)

    def reset(self, key):
        state_variables = jax.random.uniform(key, shape=(4,), minval=-0.05, maxval=0.05)
        state = CartPoleState(
            x=state_variables[0],
            x_dot=state_variables[1],
            theta=state_variables[2],
            theta_dot=state_variables[3],
        )
        observation = self.get_observation(state)
        return state, observation

    def get_observation(self, state: CartPoleState) -> Array:
        return jnp.array(
            [state.x, state.x_dot, state.theta, state.theta_dot], dtype=jnp.float32
        )

    def get_reward(self):
        return jnp.array(1.0)

    def get_terminated(self, state: CartPoleState) -> Array:
        return jnp.logical_or(
            jnp.abs(state.x) > self.x_threshold,
            jnp.abs(state.theta) > self.theta_threshold_radians,
        )

    def get_truncated(self, state: CartPoleState) -> bool:
        return state.time >= self.max_episode_steps

    def observation_space(self) -> Box:
        high = jnp.array(
            [
                self.x_threshold * 2,
                np.finfo(jnp.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(jnp.float32).max,
            ]
        )
        return Box(
            low=-high,
            high=high,
            shape=(4,),
        )

    def action_space(self) -> Discrete:
        return Discrete(2)

    def render(self, state: CartPoleState):
        """
        Render the CartPole environment state as a PIL Image.

        Args:
            state: CartPoleState containing x, x_dot, theta, theta_dot

        Returns:
            PIL Image of the rendered cart-pole system
        """
        screen_width = 600
        screen_height = 400

        # Create white background
        img = Image.new("RGB", (screen_width, screen_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Calculate scaling and dimensions
        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        # Extract state values (convert from JAX arrays to floats)
        x = float(state.x)
        theta = float(state.theta)

        # Cart position
        cartx = x * scale + screen_width / 2.0
        carty = 100  # Fixed y position for cart

        # Draw the track (horizontal line)
        draw.line([(0, carty), (screen_width, carty)], fill=(0, 0, 0), width=2)

        # Draw cart
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        cart_coords = [
            (l + cartx, b + carty),
            (l + cartx, t + carty),
            (r + cartx, t + carty),
            (r + cartx, b + carty),
        ]
        draw.polygon(cart_coords, fill=(0, 0, 0), outline=(0, 0, 0))

        # Draw pole
        axleoffset = cartheight / 4.0
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        # Rotate pole coordinates
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            # Rotate around origin
            cos_theta = np.cos(-theta)
            sin_theta = np.sin(-theta)
            rotated_x = coord[0] * cos_theta - coord[1] * sin_theta
            rotated_y = coord[0] * sin_theta + coord[1] * cos_theta
            # Translate to cart position
            final_x = rotated_x + cartx
            final_y = rotated_y + carty + axleoffset
            pole_coords.append((final_x, final_y))

        draw.polygon(pole_coords, fill=(202, 152, 101), outline=(202, 152, 101))

        # Draw axle (circle at joint)
        axle_radius = int(polewidth / 2)
        axle_center = (int(cartx), int(carty + axleoffset))
        draw.ellipse(
            [
                (axle_center[0] - axle_radius, axle_center[1] - axle_radius),
                (axle_center[0] + axle_radius, axle_center[1] + axle_radius),
            ],
            fill=(129, 132, 203),
            outline=(129, 132, 203),
        )

        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # Flip vertically to match pygame's coordinate system
        # img = img.transpose(Image.FLIP_TOP_BOTTOM)

        return img


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
        self.critic = eqx.nn.MLP(4, 1, width_size=4, depth=3, key=key)

    def value(self, obs):
        return self.critic(obs)


class GameActor(Actor):
    actor: eqx.nn.MLP

    def __init__(self, key: Array):
        self.actor = eqx.nn.MLP(4, 4, width_size=4, depth=1, key=key)

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
        Transition,
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
            transition,
        )

    def train_cycle(
        self,
        agent: SimplePolicyGradientAgent,
        rs: RolloutState[TState],
    ) -> Tuple[SimplePolicyGradientAgent, RolloutState[TState], Transition]:
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
        self,
        key: Array,
        agent: SimplePolicyGradientAgent,
        compute_metric: Callable[[Transition], dict],
        iterations: int = 128,
    ) -> SimplePolicyGradientAgent:
        key, traink = jax.random.split(key)

        state, obs = jax.vmap(self.env.reset)(jax.random.split(traink, self.env_n))

        rs = RolloutState(key=jax.random.split(key, self.env_n), state=state, obs=obs)

        train_cycle = eqx.filter_jit(self.train_cycle)

        for i in (bar := tqdm(range(0, iterations))):
            agent, rs, transition = train_cycle(agent, rs)

            bar.set_postfix(compute_metric(transition))

        return agent


if __name__ == "__main__":
    key = jax.random.key(0)
    env = CartPole()

    key, subk = jax.random.split(key)
    trainer = SimplePolicyGradientTrainer(env=env)
    agent = trainer.make_agent(subk, actor=GameActor, critic=GameCritic)

    def compute_metric(t: Transition):
        return {"early": t.done.sum()}

    agent = trainer.train(key, agent, iterations=128, compute_metric=compute_metric)

    if True:
        frames = []
        key = jax.random.key(67)
        key, resetk = jax.random.split(key)
        state, obs = env.reset(resetk)

        done = jnp.array(False)

        frames.append(state)
        while not done.item():
            _, dist = agent.ac.actor(obs)

            key, actionk, stepk = jax.random.split(key, 3)
            action = dist.sample(seed=actionk)
            state, (reward, done), obs = env.step(stepk, state, action)

            frames.append(state)

        frames = [env.render(state) for state in frames]

        duration = int(1000 / 60)
        frames[0].save(
            "/sdcard/ff.gif",
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
