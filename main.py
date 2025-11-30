import gymnasium as gym
import torch
import torch.optim as optim

from src.agent import PPO
from src.buffer import RolloutBuffer
from src.trainer import train_ppo
from src.utils import get_device, seed_everything

def main():
    config = {
        "model_id": "model-v1",
        "env_id": "Pendulum-v1",
        "total_steps": 1_000_000,
        "num_steps": 2048,     
        "batch_size": 64,
        "num_epochs": 10,
        "hidden_dims": 64,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "seed": 42,
    }
    
    config["num_updates"] = int(config["total_steps"] / config["num_steps"])
    config["device"] = get_device()
    
    print(f"Running PPO on {config['env_id']} using {config['device']}")
    
    seed_everything(config["seed"])
    env = gym.make(config["env_id"])
    
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    
    agent = PPO(state_dim, config["hidden_dims"], action_dim).to(config["device"])
    buffer = RolloutBuffer(
        config["num_steps"], 
        state_dim, 
        action_dim, 
        config["batch_size"], 
        device=config["device"]
    )
    optimizer = optim.Adam(agent.parameters(), lr=config["learning_rate"], eps=1e-5)
    
    train_ppo(env, agent, buffer, optimizer, config)

    save_path = f"ppo_{config['model_id']}.pth"
    torch.save(agent.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    env.close()

if __name__ == "__main__":
    main()