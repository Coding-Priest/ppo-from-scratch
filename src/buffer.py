import torch

class RolloutBuffer():
    def __init__(self, num_steps, state_dim, action_dim, batch_size, device='cuda'):
        self.states = torch.zeros((num_steps, state_dim), device = device)
        self.actions = torch.zeros((num_steps, action_dim), device=device)
        self.rewards = torch.zeros(num_steps, device=device)
        self.log_probs = torch.zeros(num_steps, device=device)
        self.values = torch.zeros(num_steps, device=device)
        self.dones = torch.zeros(num_steps, device=device)

        self.advantages = torch.zeros(num_steps, device=device)
        self.returns = torch.zeros(num_steps, device=device)

        self.num_steps = num_steps
        self.batch_size = batch_size
        self.ptr = 0

    def populate(self, state, action, reward, log_prob, value, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.dones[self.ptr] = done

        self.ptr += 1

    def reset(self):
        self.ptr = 0
        
    def sample(self):
        indices = torch.randperm(self.num_steps)

        for start in range(0, self.num_steps, self.batch_size):
            end = start + self.batch_size
            batch_indices = indices[start:end]

            yield (
                self.states[batch_indices],
                self.actions[batch_indices],
                self.log_probs[batch_indices],
                self.returns[batch_indices],
                self.advantages[batch_indices]
            )

    def compute_returns_and_advantages(self, next_value, gamma=0.99, gae_lambda=0.95):
        for t in reversed(range(len(self.advantages))):
            if t == len(self.advantages) - 1:
                delta = self.rewards[t] + gamma*next_value*(1 - self.dones[t]) - self.values[t]
                self.advantages[t] = delta
                self.returns[t] = self.advantages[t] + self.values[t]
            else:
                delta = self.rewards[t] + gamma*self.values[t + 1]*(1 - self.dones[t]) - self.values[t]
                self.advantages[t] = delta + gamma*gae_lambda*self.advantages[t + 1]*(1 - self.dones[t])
                self.returns[t] = self.advantages[t] + self.values[t]
