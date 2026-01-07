
# Building Custom Environments

In **xRL**, environments are designed to be purely functional and stateless. This design allows JAX to jit-compile the environment logic, enabling massive vectorization and hardware acceleration on GPUs and TPUs.

To build a custom environment, you must implement the `SoloEnv` (single agent) or `GroupEnv` (multi-agent) interface.

---

## 1. Defining the State

Unlike standard Python classes where state is stored in `self`, xRL environments pass the state explicitly to every function. You should define your state as a `NamedTuple`. This ensures it is a valid JAX PyTree.

```python
from typing import NamedTuple
from jaxtyping import Array, Float
import jax.numpy as jnp

class CartPoleState(NamedTuple):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: int = 0

```

---

## 2. Creating a Single-Agent Environment (`SoloEnv`)

Inherit from `SoloEnv` and implement the core methods. The logic must be written using `jax.numpy` to support compilation.

### The `CartPole` Example

```python
from xrl.environment import SoloEnv, Observation, Box, Discrete, TimeStep
from PIL import Image, ImageDraw
import jax

class CartPole(SoloEnv[CartPoleState]):
    # Environment Constants
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    force_mag: float = 10.0
    tau: float = 0.02
    max_episode_steps: int = 500

    # 1. Define Observation & Action Spaces
    def observation_space(self):
        return {
            "a1": Box(0, 0, shape=(1,)),
            "a2": Box(0, 0, shape=(1,)),
            "a3": Box(0, 0, shape=(1,)),
            "a4": Box(0, 0, shape=(1,)),
        }

    def action_space(self):
        return {"swing": Discrete(2)}

    # 2. Reset: Returns (State, Observation)
    def reset(self, key):
        state_variables = jax.random.uniform(key, shape=(4,), minval=-0.05, maxval=0.05)
        state = CartPoleState(
            x=state_variables[0],
            x_dot=state_variables[1],
            theta=state_variables[2],
            theta_dot=state_variables[3],
        )
        return state, self.get_observation(state)

    # 3. Step: Returns (NewState, TimeStep, Observation)
    def step(self, key, state, action):
        # ... Physics calculation omitted for brevity ...
        
        # Update State
        state = CartPoleState(
            x=x, x_dot=x_dot, theta=theta, theta_dot=theta_dot, time=state.time + 1
        )

        # Define Timestep (Reward + Termination)
        timestep = TimeStep(
            reward=jnp.array(1.0),
            done=jnp.logical_or(self.get_terminated(state), self.get_truncated(state)),
        )

        return state, timestep, self.get_observation(state)

    # Helper methods
    def get_observation(self, state):
        return {
            "a1": jnp.atleast_1d(state.x),
            "a2": jnp.atleast_1d(state.x_dot),
            # ...
        }

    # 4. Render: Pure python/PIL logic (not JIT compiled)
    def render(self, state: CartPoleState):
        img = Image.new("RGB", (600, 400), color=(255, 255, 255))
        # ... drawing logic ...
        return img

```

---

## 3. Creating a Multi-Agent Environment (`GroupEnv`)

For multi-agent scenarios, inherit from `GroupEnv`. The key difference is that observations and actions become **dictionaries** keyed by agent IDs (e.g., `"alpha"`, `"beta"`).

### The `Pong` Example

```python
from xrl.environment import GroupEnv

class PongState(NamedTuple):
    ys: Float[Array, "2"]
    ball_velocity: Float[Array, "2"]
    ball_position: Float[Array, "2"]

class Pong(GroupEnv[PongState]):
    
    # 1. Spaces are nested dictionaries
    def observation_space(self):
        return {
            "alpha": { ... },
            "beta":  { ... },
        }

    def action_space(self):
        return {
            "alpha": {"move": Discrete(3)},
            "beta": {"move": Discrete(3)},
        }

    # 2. Reset
    def reset(self, key):
        # ... random initialization ...
        state = PongState(...)
        return state, self.get_obs(state)

    # 3. Step handles physics for ALL agents simultaneously
    def step(self, key, state, action):
        # Extract actions for specific agents
        move_alpha = action["alpha"]["move"]
        move_beta = action["beta"]["move"]

        # ... Physics Logic ...
        
        # Rewards are distributed per agent
        timestep = {
            "alpha": TimeStep(reward=reward_alpha, done=done),
            "beta":  TimeStep(reward=reward_beta,  done=done),
        }

        return state, timestep, self.get_obs(state)

```

---

## Core Rules for Environments

1. **Pure Functions**: `step` and `reset` must not have side effects. Do not modify `self` variables during execution.
2. **JAX Arrays**: All numerical computations in `step` must use `jax.numpy`, not standard `numpy`.
3. **Key Handling**: Randomness is handled by passing a PRNG `key` explicitly. Split keys if you need multiple random operations.
4. **Vectorization**: Do not worry about batch dimensions. xRL trainers automatically `vmap` your environment to run many instances in parallel. Write your logic for a *single* instance.
