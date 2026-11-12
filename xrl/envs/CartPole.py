from typing import NamedTuple

import jax
import jax.numpy as jnp
import jaxgraphic as jg  # type: ignore[import-untyped]
from jaxtyping import Array, Float, Int, Key, UInt8

from ..environment import Action, Box, Discrete, EnvInfo, MonoEnv, MonoTimeStep


class CartPoleState(NamedTuple):
    x: Float[Array, ""]
    x_dot: Float[Array, ""]
    theta: Float[Array, ""]
    theta_dot: Float[Array, ""]
    steps: Int[Array, ""]


class CartPoleEnv(MonoEnv[CartPoleState]):
    """Classic CartPole environment implemented with JAX."""

    # Physical constants
    gravity = 9.8
    mass_cart = 1.0
    mass_pole = 0.1
    total_mass = mass_cart + mass_pole

    # Half the pole length
    length = 0.5
    pole_mass_length = mass_pole * length

    force_mag = 10.0
    tau = 0.02

    # Termination thresholds
    x_threshold = 2.4
    theta_threshold = 12.0 * jnp.pi / 180.0

    max_episode_steps = 500

    def reset(self, key: Key[Array, ""]) -> tuple[CartPoleState, Float[Array, "4"]]:
        values = jax.random.uniform(
            key,
            shape=(4,),
            minval=-0.05,
            maxval=0.05,
            dtype=jnp.float32,
        )

        state = CartPoleState(
            x=values[0],
            x_dot=values[1],
            theta=values[2],
            theta_dot=values[3],
            steps=jnp.array(0, dtype=jnp.int32),
        )

        return state, jnp.stack(
            [
                state.x,
                state.x_dot,
                state.theta,
                state.theta_dot,
            ]
        )

    def step(
        self,
        key: Key[Array, ""],
        state: CartPoleState,
        action: Action,
    ) -> tuple[CartPoleState, MonoTimeStep, Float[Array, "4"], EnvInfo]:
        action = jnp.asarray(action, dtype=jnp.int32)

        # Action 0: push left.
        # Action 1: push right.
        force = jnp.where(
            action == 1,
            self.force_mag,
            -self.force_mag,
        )

        x = state.x
        x_dot = state.x_dot

        theta = state.theta
        theta_dot = state.theta_dot

        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)

        # CartPole equations of motion.
        temp = (
            force + self.pole_mass_length * theta_dot**2 * sin_theta
        ) / self.total_mass

        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length * (4.0 / 3.0 - self.mass_pole * cos_theta**2 / self.total_mass)
        )

        x_acc = temp - self.pole_mass_length * theta_acc * cos_theta / self.total_mass

        # Euler integration.
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc

        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        steps = state.steps + 1

        next_state = CartPoleState(
            x=x,
            x_dot=x_dot,
            theta=theta,
            theta_dot=theta_dot,
            steps=steps,
        )

        # Terminate when the cart or pole exceeds its limits,
        # or when the episode reaches its time limit.

        # Standard CartPole reward.
        reward = jnp.array(1.0, dtype=jnp.float32)

        timestep = MonoTimeStep(
            reward=reward,
            terminated=(jnp.abs(x) > self.x_threshold)
            | (jnp.abs(theta) > self.theta_threshold),
            truncated=steps >= self.max_episode_steps,
        )

        obs = jnp.stack(
            [
                next_state.x,
                next_state.x_dot,
                next_state.theta,
                next_state.theta_dot,
            ]
        )

        return next_state, timestep, obs, {}

    def observation_space(self) -> Box:
        return Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(4,),
        )

    def action_space(self):
        return Discrete(2)

    def render(self, state: CartPoleState) -> UInt8[Array, "400 600 3"]:
        screen_width = 600
        screen_height = 400

        # Canvas and scaling
        img = jg.new_canvas(screen_height, screen_width, "#ffffff")

        world_width = self.x_threshold * 2
        scale = screen_width / world_width

        polewidth = 10.0
        polelen = scale * (2 * self.length)

        cartwidth = 50.0
        cartheight = 30.0

        # Cart position
        cartx = state.x * scale + screen_width / 2.0

        # Account for the vertical flip in the original PIL renderer
        carty = screen_height - 1.0 - 100.0

        # Track
        img = jg.fill(
            img,
            jg.box(
                center=(screen_width / 2.0, carty),
                size=(screen_width, 2.0),
            ),
            "#000000",
        )

        # Cart
        img = jg.fill(
            img,
            jg.box(
                center=(cartx, carty),
                size=(cartwidth, cartheight),
            ),
            "#000000",
        )

        # Pole geometry
        axleoffset = cartheight / 4.0
        axle_y = carty - axleoffset

        # Original PIL pole bounds:
        # y ranges from -polewidth/2 to polelen - polewidth/2
        pole_center_offset = polelen / 2.0 - polewidth / 2.0

        # Pole center after rotation and vertical flip
        sin_theta = jnp.sin(state.theta)
        cos_theta = jnp.cos(state.theta)

        pole_center = (
            cartx + pole_center_offset * sin_theta,
            axle_y - pole_center_offset * cos_theta,
        )

        # Pole
        img = jg.fill(
            img,
            jg.box(
                center=pole_center,
                size=(polewidth, polelen),
                angle=state.theta,
            ),
            "#ca9865",
        )

        # Axle
        img = jg.fill(
            img,
            jg.circle(
                center=(cartx, axle_y),
                radius=polewidth / 2.0,
            ),
            "#8184cb",
        )

        return (img * 255).astype(jnp.uint8)
