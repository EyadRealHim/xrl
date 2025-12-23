import jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np

from tqdm import tqdm

from distrax import Categorical

import optax

from jaxtyping import PyTree, Array, Float, Bool, Int, DTypeLike
from typing import (
    Mapping,
    Generic,
    TypeVar,
    Callable,
    Sequence,
    TypeAlias,
    NamedTuple,
    Type,
    Tuple,
)

from dataclasses import replace


from PIL import Image, ImageDraw


Observation: TypeAlias = PyTree[Float[Array, "..."]]  # TypeVar("Observation")
Action: TypeAlias = PyTree[Int[Array, "..."]]


class Space:
    shape: Sequence[int]
    dtype: DTypeLike


class Box(Space):
    def __init__(
        self,
        low,
        high,
        shape: Sequence[int],
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
        self.shape = (1,)
        self.dtype = jnp.int32


TState = TypeVar("TState")


class TimeStep(NamedTuple):
    reward: Float[Array, "..."]
    done: Bool[Array, "..."]


class Environment(eqx.Module, Generic[TState]):
    def autostep(
        self, key: Array, state: TState, action: Action
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
        self, key: Array, state: TState, action: Action
    ) -> Tuple[TState, TimeStep, Observation]:
        raise NotImplementedError

    def observation_space(self) -> PyTree[Space]:
        raise NotImplementedError

    def action_space(self) -> PyTree[Space]:
        raise NotImplementedError


class ParallelEnvironment(Environment[TState]):
    def autostep(
        self, key: Array, state: TState, action: Mapping[str, Action]
    ) -> Tuple[TState, TimeStep, Mapping[str, Observation]]:
        return super().autostep(key=key, state=state, action=action)

    def reset(self, key: Array) -> Tuple[TState, Mapping[str, Observation]]:
        raise NotImplementedError

    def step(
        self, key: Array, state: TState, action: Mapping[str, Action]
    ) -> Tuple[TState, TimeStep, Mapping[str, Observation]]:
        raise NotImplementedError

    def observation_space(self) -> Mapping[str, PyTree[Space]]:
        raise NotImplementedError

    def action_space(self) -> Mapping[str, PyTree[Space]]:
        raise NotImplementedError


class PongState(NamedTuple):
    ys: Float[Array, "2"]
    ball_velocity: Float[Array, "2"]
    ball_position: Float[Array, "2"]


class Pong(ParallelEnvironment[PongState]):
    width: int = 800
    height: int = 600

    paddle_height: float = 40.0
    paddle_width: float = 10.0

    ball_radius: float = 5.0
    ball_speed: float = 10.0

    paddle_speed: float = 20.0

    xoff = 20

    def observation_space(self):
        return {
            "alpha": {
                "enemy": Box(-1, 1, shape=(1,)),
                "mine": Box(-1, 1, shape=(1,)),
                "ball": Box(-1, 1, shape=(2,)),
                "ballv": Box(-1, 1, shape=(2,)),
            },
            "beta": {
                "enemy": Box(-1, 1, shape=(1,)),
                "mine": Box(-1, 1, shape=(1,)),
                "ball": Box(-1, 1, shape=(2,)),
                "ballv": Box(-1, 1, shape=(2,)),
            },
        }

    def action_space(self):
        return {
            "alpha": {"move": Discrete(3)},
            "beta": {"move": Discrete(3)},
        }

    def reset(self, key):
        posk, velk = jax.random.split(key)

        state = PongState(
            ys=jax.random.uniform(
                key=posk,
                shape=(2,),
                minval=self.paddle_height,
                maxval=self.height - self.paddle_height,
            ),
            ball_velocity=jax.random.uniform(key=velk, shape=(2,)) * 2.0 - 1.0,
            ball_position=jnp.array(
                [self.width / 2, self.height / 2], dtype=jnp.float32
            ),
        )

        return state, self.get_obs(state)

    def step(self, key, state, action):
        ys = state.ys + self.paddle_speed * (
            jnp.array([action["alpha"]["move"], action["beta"]["move"]]) - 1
        )

        ys = jnp.clip(ys, self.paddle_height, self.height - self.paddle_height)
        norm = jnp.linalg.norm(state.ball_velocity)
        v = state.ball_velocity / (norm + 1e-6)

        bp = state.ball_position + v * self.ball_speed

        upper = (
            jnp.array([self.width, self.height], dtype=jnp.float32) - self.ball_radius
        )
        lower = jnp.full_like(upper, self.ball_radius)

        # collsion left:
        v = v * jnp.where(
            jnp.logical_and(
                bp[0] - self.ball_radius < self.xoff + self.paddle_width,
                jnp.abs(bp[1] - ys[0]) <= self.ball_radius + self.paddle_height,
            ),
            -1,
            1,
        )

        # collsion right
        v = v * jnp.where(
            jnp.logical_and(
                bp[0] + self.ball_radius > self.width - (self.xoff + self.paddle_width),
                jnp.abs(bp[1] - ys[1]) <= self.ball_radius + self.paddle_height,
            ),
            -1,
            1,
        )

        # collsion walls:
        v = v * jnp.where(upper < bp, -1, 1)
        v = v * jnp.where(lower > bp, -1, 1)

        # is it done?
        alpha_score = jnp.where(upper[0] < bp[0], 1, 0)
        beta_score = jnp.where(lower[0] > bp[0], 1, 0)

        done = jnp.logical_or(alpha_score > 0, beta_score > 0)

        state = PongState(ys=ys, ball_velocity=v, ball_position=bp)
        timestep = TimeStep(reward=jnp.array(1.0), done=done)

        return state, timestep, self.get_obs(state)

    def get_obs(self, state: PongState):
        return {
            "alpha": {
                "enemy": state.ys[1][:, None],
                "mine": state.ys[0][:, None],
                "ball": state.ball_position,
                "ballv": state.ball_velocity,
            },
            "beta": {
                "enemy": state.ys[0][:, None],
                "mine": state.ys[1][:, None],
                "ball": state.ball_position,
                "ballv": state.ball_velocity,
            },
        }

    def render(self, state: PongState):
        img = Image.new("RGB", (self.width, self.height), color=(33, 77, 33))
        draw = ImageDraw.Draw(img)

        pos = [
            [self.width * i + (-1 if i else 1) * self.xoff, y]
            for i, y in enumerate(state.ys.tolist())
        ]

        for x, y in pos:
            draw.rectangle(
                (
                    x,
                    y - self.paddle_height,
                    x + self.paddle_width,
                    y + self.paddle_height,
                ),
                fill=(255, 255, 255),
            )

        x, y = state.ball_position.tolist()
        draw.ellipse(
            (
                x - self.ball_radius,
                y - self.ball_radius,
                x + self.ball_radius,
                y + self.ball_radius,
            ),
            fill=(255, 255, 255),
        )

        return img


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
        action = jnp.squeeze(action["swing"])
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

    def get_observation(self, state: CartPoleState) -> Observation:
        return {
            "a1": state.x,
            "a2": state.x_dot,
            "a3": state.theta,
            "a4": state.theta_dot,
        }

    def get_reward(self):
        return jnp.array(1.0)

    def get_terminated(self, state: CartPoleState) -> Array:
        return jnp.logical_or(
            jnp.abs(state.x) > self.x_threshold,
            jnp.abs(state.theta) > self.theta_threshold_radians,
        )

    def get_truncated(self, state: CartPoleState) -> bool:
        return state.time >= self.max_episode_steps

    def observation_space(self):
        return {
            "obs": {
                "a1": Box(0, 0, shape=(1,)),
                "a2": Box(0, 0, shape=(1,)),
                "a3": Box(0, 0, shape=(1,)),
                "a4": Box(0, 0, shape=(1,)),
            }
        }

    def action_space(self):
        return {"swing": Discrete(2)}

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
    action: Action
    value: Float[Array, "..."]
    done: Bool[Array, "..."]


class Actor(eqx.Module):
    def __init__(self, key: Array, in_features: int):
        pass

    def __call__(self, obs: Observation) -> Float[Array, "n"]:
        raise NotImplementedError


class Critic(eqx.Module):
    def __init__(self, key: Array, in_features: int):
        pass

    def __call__(self, obs: Observation) -> Float[Array, "1"]:
        raise NotImplementedError


class ObservationInterpreter(eqx.Module):
    out_features: int = eqx.field(static=True)

    def __init__(self, obs: PyTree[Space]):
        assert jax.tree.all(
            jax.tree.map(lambda s: isinstance(s, Box) and len(s.shape) == 1, obs)
        )

        self.out_features = sum(sum(x.shape) for x in jax.tree.leaves(obs))

    def interpret(self, obs: PyTree[Observation]) -> Float[Array, "..."]:
        return jnp.squeeze(jnp.stack(jax.tree.leaves(obs)))


class ActionInterpreter(eqx.Module):
    out_features: int = eqx.field(static=True)

    nvec: Sequence[Sequence[int]] = eqx.field(static=True)
    treedef: dict = eqx.field(static=True)

    def __init__(self, action: PyTree[Space]):
        assert jax.tree.all(
            jax.tree.map(
                lambda s: isinstance(s, Discrete) or isinstance(s, MultiDiscrete),
                action,
            )
        )

        leaves, _ = jax.tree.flatten(action)
        nvec = jax.tree.map(
            lambda x: [x.n] if isinstance(x, Discrete) else x.nvec, leaves
        )

        self.out_features = sum(sum(x) for x in nvec)
        self.nvec = nvec
        self.treedef = dict(action)

    def interpret(self, logits: Float[Array, "n"]) -> PyTree[Float[Array, "n"]]:  # noqa: F821
        assert logits.shape[0] == self.out_features

        leaves = []
        logits_ = jax.lax.split(logits, [sum(x) for x in self.nvec])

        for v, logits in zip(self.nvec, logits_):
            nvec = np.array(v)
            y, x = len(nvec), nvec.max()
            grid = jnp.full((y, x), -jnp.inf, dtype=jnp.float32)

            rows = np.repeat(np.arange(y), nvec)
            cols = np.arange(nvec.sum()) - np.repeat(np.cumsum(nvec) - nvec, nvec)

            logits = grid.at[rows, cols].set(logits)

            leaves.append(logits)

        return jax.tree.unflatten(jax.tree.structure(self.treedef), leaves)


class PongCritic(Critic):
    critic: eqx.nn.MLP

    def __init__(self, key: Array, in_features: int):
        self.critic = eqx.nn.MLP(in_features, 1, width_size=4, depth=3, key=key)

    def __call__(self, obs):
        return self.critic(obs)


class PongActor(Actor):
    actor: eqx.nn.MLP

    def __init__(self, key: Array, in_features: int):
        self.actor = eqx.nn.MLP(in_features, 4, width_size=4, depth=1, key=key)

    def __call__(self, obs):
        return self.actor(obs)


class ActorContainer(eqx.Module):
    observation: ObservationInterpreter = eqx.field(static=True)
    action: ActionInterpreter = eqx.field(static=True)

    actor: Actor
    head: eqx.nn.Linear

    def __init__(
        self,
        key: Array,
        mactor: Type[Actor],
        observation_space: PyTree[Space],
        action_space: PyTree[Space],
    ):
        self.observation = ObservationInterpreter(observation_space)
        self.action = ActionInterpreter(action_space)

        self.actor = mactor(key=key, in_features=self.observation.out_features)

        output = jax.eval_shape(
            self.actor,
            jax.ShapeDtypeStruct((self.observation.out_features,), jnp.float32),
        )

        self.head = eqx.nn.Linear(output.size, self.action.out_features, key=key)

    def __call__(self, obs: Float[Array, "..."]):
        obs = self.observation.interpret(obs)
        logits = self.head(self.actor(obs))

        return self.action.interpret(logits)


class CriticContainer(eqx.Module):
    observation: ObservationInterpreter = eqx.field(static=True)
    critic: Critic

    def __init__(
        self, key: Array, mcritic: Type[Critic], observation_space: PyTree[Space]
    ):
        self.observation = ObservationInterpreter(observation_space)
        self.critic = mcritic(key=key, in_features=self.observation.out_features)

    def __call__(self, obs: Float[Array, "..."]):
        obs = self.observation.interpret(obs)

        return self.critic(obs)


class _RLAgent(eqx.Module):
    actor: ActorContainer
    critic: CriticContainer

    opt: PyTree[optax.OptState]


RLAgent = TypeVar("RLAgent", bound=_RLAgent)


class RolloutState(NamedTuple, Generic[TState]):
    key: Array
    state: TState
    obs: Observation


class RLTrainer(eqx.Module, Generic[TState, RLAgent]):
    env: Environment[TState] = eqx.field(static=True)
    optim: optax.GradientTransformation = eqx.field(static=True)

    env_n: int = eqx.field(static=True, default=8)
    cycle_n: int = eqx.field(static=True, default=64)
    step_n: int = eqx.field(static=True, default=128)

    def make_agent(self, key: Array, *args, **kwargs) -> RLAgent:
        raise NotImplementedError

    def gradient(
        self, rs: RolloutState[TState], transition: Transition, agent: RLAgent
    ) -> PyTree:
        raise NotImplementedError

    def train_step(
        self, rs: RolloutState[TState], agent: RLAgent
    ) -> Tuple[RolloutState[TState], RLAgent]:
        rs, transition = jax.vmap(self.rollout, in_axes=(0, None))(rs, agent)

        gradient = self.gradient(rs, transition, agent)

        def update(model, opt, grad):
            updates, opt = self.optim.update(
                grad, opt, eqx.filter(model, eqx.is_inexact_array)
            )

            return eqx.apply_updates(model, updates), opt

        result = {
            k: update(getattr(agent, k), agent.opt[k], gradient[k])
            for k in ["actor", "critic"]
        }

        ac = {k: a for k, (a, _) in result.items()}
        opt = {k: opt for k, (_, opt) in result.items()}

        return rs, replace(agent, **ac, opt=opt)

    def rollout(
        self, rs: RolloutState[TState], agent: RLAgent
    ) -> Tuple[RolloutState[TState], Transition]:
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

            return RolloutState(key=key, state=state, obs=obsv), Transition(
                observation=obs,
                action=action,
                value=value,
                reward=reward,
                done=done,
            )

        return jax.lax.scan(step, rs, length=self.step_n)

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

        train_cycle = eqx.filter_jit(self.train_cycle)
        for i in tqdm(range(iterations)):
            rs, agent = train_cycle(rs, agent)

        return agent


class SPGAgent(_RLAgent):
    pass


class SimplePolicyGradientTrainer(RLTrainer[TState, SPGAgent], Generic[TState]):
    discount: float = eqx.field(static=True, default=0.96)
    lambda_: float = eqx.field(static=True, default=0.95)

    def make_agent(
        self, key: Array, mactor: Type[Actor], mcritic: Type[Critic]
    ) -> SPGAgent:
        a, b = jax.random.split(key)

        actor = ActorContainer(
            a, mactor, self.env.observation_space(), self.env.action_space()
        )
        critic = CriticContainer(a, mcritic, self.env.observation_space())

        return SPGAgent(
            actor=actor,
            critic=critic,
            opt={
                "actor": self.optim.init(eqx.filter(actor, eqx.is_inexact_array)),
                "critic": self.optim.init(eqx.filter(critic, eqx.is_inexact_array)),
            },
        )

    def gradient(self, rs, transition, agent):
        advantage, returns = jax.vmap(self.compute_advantage_and_returns)(
            transition, jax.vmap(agent.critic)(rs.obs)
        )

        @eqx.filter_grad
        def actor_loss(actor: ActorContainer, obs, action: PyTree, advantage):
            logits = jax.vmap(jax.vmap(actor))(obs)
            log_p = jax.tree.map(
                lambda logits, action: Categorical(logits=logits).log_prob(action),
                logits,
                action,
            )

            adv = jnp.expand_dims(advantage, axis=-1)
            losses = jax.tree.map(lambda x: x * adv, log_p)

            return -jnp.array(jax.tree.leaves(losses)).mean()

        @eqx.filter_grad
        def critic_loss(critic: CriticContainer, obs, returns):
            values = jnp.squeeze(jax.vmap(jax.vmap(critic))(obs))

            return ((values - returns) ** 2).mean()

        actor_grad = actor_loss(
            agent.actor, transition.observation, transition.action, advantage
        )
        critic_grad = critic_loss(agent.critic, transition.observation, returns)

        return {"actor": actor_grad, "critic": critic_grad}

    def compute_advantage_and_returns(
        self, transition: Transition, bootstrap: Float[Array, "1"]
    ) -> Tuple[Float[Array, "step_n"], Float[Array, "step_n"]]:
        discount = (1.0 - transition.done.astype(jnp.float32)) * self.discount
        values = jnp.concatenate([jnp.squeeze(transition.value), bootstrap])
        deltas = transition.reward + discount * values[1:] - values[:-1]

        def compute(prev, pair):
            delta, value, discount = pair

            gae = delta + discount * self.lambda_ * prev

            return gae, (gae, gae + value)

        _, (advantage, returns) = jax.lax.scan(
            compute, 0, (deltas, jnp.squeeze(transition.value), discount), reverse=True
        )

        return advantage, returns


if __name__ == "__main__":
    key = jax.random.key(0)
    env = CartPole()

    key, subk = jax.random.split(key)
    trainer = SimplePolicyGradientTrainer(env=env, optim=optax.adam(1e-3))
    agent = trainer.make_agent(subk, PongActor, PongCritic)

    agent = trainer.train(key, agent, iterations=64)

    frames = []
    key = jax.random.key(67)
    key, resetk = jax.random.split(key)
    state, obs = env.reset(resetk)

    done = jnp.array(False)

    frames.append(state)
    while not done.item():
        logits = agent.actor(obs)
        action = jax.tree.map(jnp.argmax, logits)

        key, stepk = jax.random.split(key)
        state, (reward, done), obs = env.step(stepk, state, action)

        frames.append(state)

    frames = [env.render(state) for state in frames]

    duration = int(1000 / 30)
    frames[0].save(
        "/sdcard/ff.gif",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
