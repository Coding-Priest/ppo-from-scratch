import torch
import torch.nn as nn
import numpy as np

def train_ppo(env, agent, buffer, optimizer, config):
    device = config['device']
    prev_state, _ = env.reset()
    
    episode_rewards = []
    current_ep_reward = 0

    for update in range(config['num_updates']):
        buffer.reset()
        
        for step in range(config['num_steps']):
            state_tensor = torch.as_tensor(prev_state, dtype=torch.float32, device=device)
            state_tensor = state_tensor.reshape(1, -1)
            
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(state_tensor)

            action_numpy = action.cpu().numpy()
            clipped_action = np.clip(action_numpy, -2, 2).flatten()
            
            next_state, reward, terminated, truncated, _ = env.step(clipped_action)
            done = terminated or truncated
            
            if isinstance(reward, np.ndarray):
                reward = reward.item()
            
            current_ep_reward += reward

            buffer.populate(
                state_tensor, action, 
                torch.as_tensor(reward, dtype=torch.float32, device=device),
                log_prob, value, 
                torch.as_tensor(int(done), dtype=torch.float32, device=device)
            )

            prev_state = next_state
            if done:
                prev_state, _ = env.reset()
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0

        with torch.no_grad():
            next_state_tensor = torch.as_tensor(prev_state, dtype=torch.float32, device=device)
            next_state_tensor = next_state_tensor.reshape(1, -1)
            next_value = agent.get_value(next_state_tensor)
        
        buffer.compute_returns_and_advantages(
            next_value, config['gamma'], config['gae_lambda']
        )

        b_v_losses = []
        
        for epoch in range(config['num_epochs']):
            for batch in buffer.sample():
                b_states, b_actions, b_log_probs, b_returns, b_advantages = batch
            
                _, new_log_prob, entropy, new_value = agent.get_action_and_value(b_states, action=b_actions)
                
                log_ratio = new_log_prob - b_log_probs
                ratio = log_ratio.exp()

                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                pg_loss1 = -b_advantages * ratio
                pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_value - b_returns) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss + v_loss * 0.5 - entropy_loss * 0.01

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                
                b_v_losses.append(v_loss.item())

        if (update + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            print(f"Update {update+1}/{config['num_updates']} | Avg Reward: {avg_reward:.2f} | Value Loss: {np.mean(b_v_losses):.4f}")