from ..environment import ParallelEnvironment, Box, Discrete, TimeStep

from typing import NamedTuple
from jaxtyping import Array, Float

from PIL import Image, ImageDraw
import jax.numpy as jnp
import numpy as np

import jax


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
            jnp.squeeze(jnp.array([action["alpha"]["move"], action["beta"]["move"]]))
            - 1
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
        v = v.at[0].multiply(
            jnp.where(
                jnp.logical_and(
                    bp[0] - self.ball_radius < self.xoff + self.paddle_width,
                    jnp.abs(bp[1] - ys[0]) <= self.ball_radius + self.paddle_height,
                ),
                -1,
                1,
            )
        )

        # collsion right
        v = v.at[0].multiply(
            jnp.where(
                jnp.logical_and(
                    bp[0] + self.ball_radius
                    > self.width - (self.xoff + self.paddle_width),
                    jnp.abs(bp[1] - ys[1]) <= self.ball_radius + self.paddle_height,
                ),
                -1,
                1,
            )
        )

        # collsion walls:
        v = v * jnp.where(upper < bp, -1, 1)
        v = v * jnp.where(lower > bp, -1, 1)

        # is it done?
        alpha_score = jnp.where(upper[0] < bp[0], 1, 0)
        beta_score = jnp.where(lower[0] > bp[0], 1, 0)

        done = jnp.logical_or(alpha_score > 0, beta_score > 0)
        bp = jnp.clip(
            bp,
            np.array([self.ball_radius] * 2),
            np.array([self.width - self.ball_radius, self.height - self.ball_radius]),
        )

        def reward(y):
            return 1 - jnp.abs(bp[1] - y) / self.height

        state = PongState(ys=ys, ball_velocity=v, ball_position=bp)
        timestep = TimeStep(
            reward={"alpha": reward(ys[0]), "beta": reward(ys[1])},
            done=done,
        )

        return state, timestep, self.get_obs(state)

    def get_obs(self, state: PongState):
        return {
            "alpha": {
                "enemy": jnp.atleast_1d(state.ys[1]),
                "mine": jnp.atleast_1d(state.ys[0]),
                "ball": state.ball_position,
                "ballv": state.ball_velocity,
            },
            "beta": {
                "enemy": jnp.atleast_1d(state.ys[0]),
                "mine": jnp.atleast_1d(state.ys[1]),
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
