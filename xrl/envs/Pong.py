import equinox as eqx
import jax
import jax.numpy as jnp
import jaxgraphic as jg  # type: ignore[import-untyped]
import numpy as np
from beartype import beartype
from jaxtyping import Array, Float, Int, jaxtyped

from ..environment import Box, Discrete, PolyEnv, PolyTimeStep


@jaxtyped(typechecker=beartype)
class PongState(eqx.Module):
    ys: Float[Array, "2"]
    ball_velocity: Float[Array, "2"]
    ball_position: Float[Array, "2"]
    steps: Int[Array, ""]


class PongEnv(PolyEnv):
    width: int = 800
    height: int = 600

    paddle_height: float = 40.0
    paddle_width: float = 10.0

    ball_radius: float = 5.0
    ball_speed: float = 10.0

    paddle_speed: float = 20.0

    max_episode_steps: int = 1000

    xoff: int = 20

    def step(self, key, state, action):
        ys = state.ys + self.paddle_speed * (
            jnp.squeeze(jnp.array([action["left"], action["right"]])) - 1
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
        left_score = jnp.where(upper[0] < bp[0], 1, 0)
        right_score = jnp.where(lower[0] > bp[0], 1, 0)

        bp = jnp.clip(
            bp,
            np.array([self.ball_radius] * 2),
            np.array([self.width - self.ball_radius, self.height - self.ball_radius]),
        )

        def reward(y):
            return 1 - jnp.abs(bp[1] - y) / self.height

        state = PongState(
            ys=ys, ball_velocity=v, ball_position=bp, steps=state.steps + 1
        )
        timestep = PolyTimeStep(
            reward={"left": reward(ys[0]), "right": reward(ys[1])},
            inactive={"left": jnp.array(False), "right": jnp.array(False)},
            terminated=jnp.logical_or(left_score > 0, right_score > 0),
            truncated=state.steps >= self.max_episode_steps,
        )

        return state, timestep, self.get_obs(state), {}

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
            steps=jnp.array(0),
        )

        return state, self.get_obs(state)

    def observation_space(self):
        return {
            "left": {
                "self": Box(-1.0, 1.0, shape=(1,)),
                "other": Box(-1.0, 1.0, shape=(1,)),
                "ball": Box(-1.0, 1.0, shape=(2,)),
                "ball_velocity": Box(-1.0, 1.0, shape=(2,)),
            },
            "right": {
                "self": Box(-1.0, 1.0, shape=(1,)),
                "other": Box(-1.0, 1.0, shape=(1,)),
                "ball": Box(-1.0, 1.0, shape=(2,)),
                "ball_velocity": Box(-1.0, 1.0, shape=(2,)),
            },
        }

    def action_space(self):
        return {
            "left": Discrete(3),
            "right": Discrete(3),
        }

    def get_obs(self, state: PongState):
        ys = state.ys / self.height * 2.0 - 1.0
        ball = state.ball_position / jnp.array([self.width, self.height]) * 2.0 - 1.0
        return {
            "left": {
                "other": ys[1:2],
                "self": ys[0:1],
                "ball": ball,
                "ball_velocity": state.ball_velocity,
            },
            "right": {
                "other": ys[0:1],
                "self": ys[1:2],
                "ball": ball,
                "ball_velocity": state.ball_velocity,
            },
        }

    def render(self, state: PongState):
        pos = jnp.array(
            [
                [self.xoff, state.ys[0]],
                [self.width - self.xoff, state.ys[1]],
            ]
        )
        sizes = jnp.array([[self.paddle_width, self.paddle_height * 2]] * 2)

        return jg.to_hwc(
            jg.to_uint8(
                jg.Canvas.new(self.height, self.width, bg="#214d21")
                .fill_many(
                    jg.Box(pos, sizes),
                    colors=["#fff", "#fff"],
                )
                .fill(jg.Circle(state.ball_position, self.ball_radius), color="#fff")
                .img
            )
        )
