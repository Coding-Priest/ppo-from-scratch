# Proximal Policy Optimization (PPO) from Scratch

A clean, modular implementation of Proximal Policy Optimization (PPO) using PyTorch. This project implements the PPO-Clip algorithm to solve the `Pendulum-v1` continuous control environment.

## Project Structure

```text
├── main.py              # Entry point: Configuration and training loop
├── ppo_model-v1.pth     # Saved model weights
├── pyproject.toml       # Project dependencies
├── src/
│   ├── agent.py         # Actor-Critic Neural Network
│   ├── buffer.py        # RolloutBuffer with GAE
│   ├── trainer.py       # PPO loss calculation and optimization
│   └── utils.py         # Utilities (Seeding, device handling)
````

## Installation

This project requires Python 3.13+ and the dependencies listed in `pyproject.toml`.

1.  Clone the repository:

    ```bash
    git clone https://github.com/Coding-Priest/ppo-from-scratch.git
    cd ppo
    ```

2.  Install dependencies:

    ```bash
    pip install gymnasium[mujoco] numpy torch swig
    ```

## Usage

To start training the agent, run the main script:

```bash
python main.py
```

### Configuration

Hyperparameters are defined in `main.py`:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `env_id` | `Pendulum-v1` | Gymnasium Environment ID |
| `total_steps` | `1,000,000` | Total timesteps for training |
| `num_steps` | `2048` | Steps per rollout (buffer size) |
| `batch_size` | `64` | Optimization batch size |
| `num_epochs` | `10` | Optimization epochs per rollout |
| `learning_rate`| `3e-4` | Adam optimizer learning rate |
| `gamma` | `0.99` | Discount factor |
| `gae_lambda` | `0.95` | GAE smoothing parameter |

## Implementation Details

  * **Algorithm:** PPO-Clip with Generalized Advantage Estimation (GAE).
  * **Network:** Shared backbone with separate Actor and Critic heads.
      * **Backbone:** Linear(64) -\> Tanh -\> Linear(64) -\> Tanh.
      * **Actor:** Outputs mean action; learns log standard deviation separately.
  * **Optimization:** Adam optimizer with orthogonal layer initialization.

