"""Two-player Pong for XRL, written as pure JAX (Ultra-compact, Double Speed)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxgraphic as jg
import numpy as np
from beartype import beartype
from jaxtyping import Array, Float, Int, jaxtyped

from ..environment import Box, Discrete, PolyEnv, PolyTimeStep

__all__ = ["PongEnv", "PongState"]


@jaxtyped(typechecker=beartype)
class PongState(eqx.Module):
    ys: Float[Array, "2"]
    ball_position: Float[Array, "2"]
    ball_velocity: Float[Array, "2"]
    steps: Int[Array, ""]


class PongEnv(PolyEnv):
    width: int = 800
    height: int = 600
    xoff: int = 30
    paddle_width: float = 12.0
    paddle_height: float = 80.0
    ball_radius: float = 8.0
    paddle_speed: float = 20.0
    ball_speed: float = 20.0
    max_episode_steps: int = 1000

    def step(self, key, state: PongState, action):
        hp, r = self.paddle_height / 2, self.ball_radius
        move = jnp.stack([action["left"], action["right"]]).astype(jnp.float32) - 1.0
        ys = jnp.clip(state.ys + move * self.paddle_speed, hp, self.height - hp)

        bp, bv = state.ball_position + state.ball_velocity, state.ball_velocity

        # Wall collisions
        hit_y = (bp[1] - r <= 0.0) | (bp[1] + r >= self.height)
        bp = bp.at[1].set(jnp.clip(bp[1], r, self.height - r))
        bv = bv.at[1].set(jnp.where(hit_y, -bv[1], bv[1]))

        # Paddle collisions
        rx = jnp.array([self.xoff, self.width - self.xoff])
        hit_x = (jnp.abs(bp[0] - rx) <= self.paddle_width / 2 + r) & (
            (bv[0] < 0) == (jnp.arange(2) == 0)
        )
        hit_y_pad = (bp[1] >= ys - hp) & (bp[1] <= ys + hp)
        hit = jnp.any(hit_x & hit_y_pad)
        bv = bv.at[0].set(jnp.where(hit, -bv[0] * 1.02, bv[0]))

        # Scoring & Termination
        scored = (bp[0] > self.width) | (bp[0] < 0.0)
        rew = jnp.where(bp[0] > self.width, 1.0, jnp.where(bp[0] < 0.0, -1.0, 0.0))

        new_state = PongState(
            ys=ys, ball_position=bp, ball_velocity=bv, steps=state.steps + 1
        )
        timestep = PolyTimeStep(
            reward={"left": rew, "right": -rew},
            inactive={"left": jnp.array(False), "right": jnp.array(False)},
            terminated=scored,
            truncated=new_state.steps >= self.max_episode_steps,
        )
        return (
            new_state,
            timestep,
            self.get_obs(new_state),
            {"ball_speed": jnp.linalg.norm(bv)},
        )

    def reset(self, key):
        pk, sk = jax.random.split(key)
        hp = self.paddle_height / 2
        ys = jax.random.uniform(pk, (2,), minval=hp, maxval=self.height - hp)
        angle = jax.random.uniform(sk, (), minval=-0.3, maxval=0.3)
        dir_sign = jnp.where(jax.random.bernoulli(sk), 1.0, -1.0)
        initial_ball_vel = self.ball_speed * jnp.stack(
            [dir_sign * jnp.cos(angle), jnp.sin(angle)]
        )
        initial_pos = jnp.array([self.width / 2, self.height / 2], dtype=jnp.float32)
        initial_steps = jnp.array(0, jnp.int32)

        state = PongState(
            ys=ys,
            ball_position=initial_pos,
            ball_velocity=initial_ball_vel,
            steps=initial_steps,
        )
        return state, self.get_obs(state)

    def observation_space(self):
        p = {
            "self": Box(-1.0, 1.0, (1,)),
            "other": Box(-1.0, 1.0, (1,)),
            "ball": Box(-1.0, 1.0, (2,)),
            "ball_velocity": Box(-1.0, 1.0, (2,)),
        }
        return {"left": p, "right": p}

    def action_space(self):
        return {"left": Discrete(3), "right": Discrete(3)}

    def get_obs(self, state: PongState):
        ys = state.ys / self.height * 2.0 - 1.0
        ball = state.ball_position / jnp.array([self.width, self.height]) * 2.0 - 1.0
        vel = state.ball_velocity / (self.ball_speed * 2.5)
        m = jnp.array([-1.0, 1.0])
        view = lambda i, f: {
            "self": ys[i : i + 1],
            "other": ys[1 - i : 2 - i],
            "ball": jnp.clip(ball * (m if f else 1.0), -1.0, 1.0),
            "ball_velocity": jnp.clip(vel * (m if f else 1.0), -1.0, 1.0),
        }
        return {"left": view(0, False), "right": view(1, True)}

    def render(self, state: PongState):
        W, H, r = self.width, self.height, self.ball_radius
        canvas = jg.Canvas.new(H, W, bg="#0b1226")

        # Center net
        n = 20
        dy = (np.arange(n, dtype=np.float32) + 0.5) * (H / n)
        canvas = canvas.fill_many(
            jg.Box(
                np.stack([np.full(n, W / 2, np.float32), dy], axis=1),
                np.tile([4.0, H / (n * 2)], (n, 1)),
            ),
            colors=["#94a3b8"] * n,
            alphas=0.3,
            at=np.stack([np.full(n, W / 2, np.float32), dy], axis=1),
            window=(10, int(H / n)),
        )

        # Ball & Paddles
        canvas = canvas.fill(jg.Circle(state.ball_position, r), color="#f8fafc")
        for i, (x, col) in enumerate(
            ((self.xoff, "#38bdf8"), (W - self.xoff, "#fb7185"))
        ):
            canvas = canvas.fill(
                jg.Box(
                    jnp.stack([jnp.float32(x), state.ys[i]]),
                    (self.paddle_width, self.paddle_height),
                    corner=self.paddle_width / 2,
                ),
                color=col,
            )

        return jg.to_hwc(jg.to_uint8(canvas.img))
