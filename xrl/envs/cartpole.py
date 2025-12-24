from ..environment import Environment, Observation, Box, Discrete, TimeStep

from typing import NamedTuple
from jaxtyping import Array

from PIL import Image, ImageDraw
import jax.numpy as jnp
import numpy as np

import jax


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
        lef, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        cart_coords = [
            (lef + cartx, b + carty),
            (lef + cartx, t + carty),
            (r + cartx, t + carty),
            (r + cartx, b + carty),
        ]
        draw.polygon(cart_coords, fill=(0, 0, 0), outline=(0, 0, 0))

        # Draw pole
        axleoffset = cartheight / 4.0
        lef, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        # Rotate pole coordinates
        pole_coords = []
        for coord in [(lef, b), (lef, t), (r, t), (r, b)]:
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

        return img
