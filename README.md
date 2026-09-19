# Flappy Bird DQN

A deep reinforcement learning project that trains a DQN agent to play the Flappy Bird environment from Gymnasium using PyTorch.

## Overview

This project implements a classic Deep Q-Network (DQN) agent with:

- Experience replay memory
- Target network stabilization
- Epsilon-greedy exploration
- Reward tracking and checkpointing
- Config-based hyperparameter tuning

The model is trained against the `FlappyBird-v0` environment and can be run in training or inference mode.

## Project Structure

- `agent.py` — main training/testing entry point and agent logic
- `dqn.py` — Q-network implementation
- `experience_replay.py` — replay buffer for off-policy learning
- `parameters.yaml` — hyperparameters for the training setup
- `runs/` — saved model checkpoints and training logs

## Requirements

- Python 3.9+
- PyTorch
- Gymnasium
- flappy-bird-gymnasium
- PyYAML

## Installation

Install the required packages:

```bash
pip install torch gymnasium flappy-bird-gymnasium pyyaml
```

If you are using a GPU-enabled machine, PyTorch should automatically use CUDA when available.

## Training

Train the agent with the default configuration:

```bash
python agent.py flappybirdv0 --train
```

This will:

- load the configuration from `parameters.yaml`
- initialize the DQN and replay buffer
- train the model until it reaches the reward threshold or continues indefinitely in training mode
- save the best model to `runs/flappybirdv0.pt`
- log best reward statistics to `runs/flappybirdv0.log`

## Testing / Playing

Run the trained model in inference mode:

```bash
python agent.py flappybirdv0
```

This loads the saved checkpoint and renders the game using the environment.

## Hyperparameters

The training settings are defined in `parameters.yaml`:

```yaml
flappybirdv0:
  envid: FlappyBird-v0
  epsilon_init: 1
  epsilon_min: 0.05
  epsilon_decay: 0.995
  replay_memory_size: 100000
  mini_batch_size: 64
  network_sync_rate: 1500
  alpha: 0.0005
  gamma: 0.99
  reward_threshold: 1000
```

### Parameter Notes

- `epsilon_init`: starting exploration rate
- `epsilon_min`: minimum exploration rate
- `epsilon_decay`: reduces exploration over time
- `replay_memory_size`: number of transitions kept in memory
- `mini_batch_size`: batch size used for optimization
- `network_sync_rate`: how often the target network is synced
- `alpha`: learning rate
- `gamma`: discount factor
- `reward_threshold`: stopping threshold used during training episodes

## How It Works

The agent observes the game state, selects actions using an epsilon-greedy policy, stores transitions in replay memory, and trains the DQN using sampled batches from that memory. The target network helps stabilize training by reducing moving-target issues common in Q-learning.

## Example Outputs

The training code prints each episode's score and epsilon value, such as:

```text
for episode=1 total rewards=24 and epsilon=0.995
best reward=42 for episode=3
```

## Notes

- Model checkpoints and logs are stored under the `runs/` directory.
- You can customize a new parameter set by adding a new section to `parameters.yaml`.
- If the environment is not installed properly, install the `flappy-bird-gymnasium` package before running training.

## License

This project is provided as a learning/demo project for reinforcement learning experimentation.
