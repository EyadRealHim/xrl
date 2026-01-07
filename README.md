
# xRL

**xRL** is a high-performance reinforcement learning framework built on top of **JAX**, designed for scalability and functional purity.

## Philosophy

The core philosophy of xRL is **separation of concerns**: you provide the environment, and we provide the optimized training logic.

Unlike other libraries, xRL is not a collection of environments. It is a robust collection of **Trainers** designed to ingest user-defined environments and execute distributed, hardware-accelerated RL loops.

---

## Getting Started

### 1. Import Modules

xRL integrates seamlessly with **Equinox** for model definition and **Optax** for optimization.

```python
import jax
import optax
import equinox as eqx

from xrl.envs.cartpole import CartPole
from xrl.algorithms import PPOTrainer
from xrl.networks import ActorLike, CriticLike
```

### 2. Define Network Architectures

Define your Actor and Critic by subclassing the xRL base modules.

```python
class CartPoleCritic(CriticLike):
    critic: eqx.nn.MLP

    def __init__(self, key, in_features):
        self.critic = eqx.nn.MLP(
            in_size=in_features, 
            out_size=1, 
            width_size=2, 
            depth=3, 
            key=key
        )

    def __call__(self, obs):
        return self.critic(obs)


class CartPoleActor(ActorLike):
    actor: eqx.nn.MLP

    def __init__(self, key, in_features):
        self.actor = eqx.nn.MLP(
            in_size=in_features, 
            out_size=4, 
            width_size=2, 
            depth=1, 
            key=key
        )

    def __call__(self, obs):
        return self.actor(obs)

```

### 3. Initialization

Initialize your environment and the `PPOTrainer`

```python
key = jax.random.key(1)
env = CartPole()

# Initialize the trainer with an Optax optimizer
trainer = PPOTrainer(
    env=env, 
    optim=optax.adam(1e-4), 
    env_n=4
)

```

### 4. Agent Creation & Training

```python
# Create the agent
key, subkey = jax.random.split(key)
agent = trainer.make_agent(subkey, CartPoleActor, CartPoleCritic)

# Execute the training loop
# Training is JIT-compiled for maximum performance on GPU/TPU
agent = trainer.train(key, agent, iterations=128)

```

### 5. Visualization (Optional)

Record an episode to evaluate the agent's performance visually.

```python
trainer.record(
    path="cartpole.mp4",
    frames=trainer.capture(
        key=jax.random.key(817), 
        agent=agent, 
        max_steps=300
    ),
)

```

<video src="examples/cartpole.mp4" controls width="600">
  Your browser does not support the video tag.
</video>


## Unified Multi-Agent Support

A key feature of xRL is its **unified interface** for both single and multi-agent scenarios. Every trainer in xRL is polymorphic, supporting both `SoloEnv` and `GroupEnv` architectures without requiring code changes to the training loop.

* **SoloEnv:** Optimized for single-agent tasks (e.g., `CartPole`).
* **GroupEnv:** Designed for multi-agent competition or cooperation (e.g., `Pong` from `xrl.envs.pong`).

Because xRL abstracts the agent-to-environment mapping, you can scale from training a single pole-balancer to a multi-player arena by simply swapping the environment instance.

walkthrough on implementing the environment for xRL, see our  **[Environment Creation Guide](environment_guide.md)**
